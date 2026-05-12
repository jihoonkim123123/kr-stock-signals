"""
한국주식 매매 신호 대시보드 v2.0 (Full Stack)
=====================================================
기존 v1 + 4가지 신규 모듈 통합

신규 기능:
1. DART 재무제표 9개 섹션 (financials.py)
2. 1개월 개인/기관/외국인 수급 분석 (supply_demand.py)
3. 네이버 뉴스 감성 분석 (naver_news_sentiment.py)
4. 자동 매수 전략 생성 (buy_strategy.py)

종합 점수 = 0.5 × 기술 + 0.2 × 수급 + 0.2 × 뉴스 + 0.1 × 모멘텀

사용법:
    export DART_API_KEY="your_key"  # opendart.fss.or.kr에서 발급
    python kr_stock_signals_v2.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# 신규 모듈 import
from dart_financials import (
    get_full_financials, get_corp_code_map,
    calculate_valuation, get_recent_filings,
)
from supply_demand import (
    fetch_supply_demand_1m, get_priority_signal,
)
from naver_news_sentiment import fetch_and_analyze
from buy_strategy import generate_buy_strategy

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent
LOOKBACK_DAYS = 260
WORKERS = 8

# 설정
SCORE_THRESHOLD = 75
ENABLE_DART = True  # API 키 필요
ENABLE_SUPPLY_DEMAND = True
ENABLE_NEWS_SENTIMENT = True
ENABLE_BUY_STRATEGY = True

TOP_N_DETAIL = 30  # 상위 N개만 상세 분석 (DART/수급/뉴스)

# 종합 점수 가중치
WEIGHTS = {
    "technical": 0.50,
    "supply_demand": 0.20,
    "sentiment": 0.20,
    "momentum": 0.10,
}


# ---------------------------------------------------------------------------
# 1) 기존 기술적 분석 (kr_stock_signals.py에서 가져옴)
# ---------------------------------------------------------------------------
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in (5, 20, 60, 120):
        df[f"MA{n}"] = df["Close"].rolling(n).mean()
    
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    
    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift()).abs()
    l_c = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    
    df["Pct5"] = (df["Close"] / df["Close"].shift(5) - 1) * 100
    df["Pct20"] = (df["Close"] / df["Close"].shift(20) - 1) * 100
    df["Pct60"] = (df["Close"] / df["Close"].shift(60) - 1) * 100
    
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / df["Vol_MA20"].replace(0, np.nan)
    
    return df


def score_trend(r: pd.Series):
    needed = ["MA5", "MA20", "MA60", "MA120", "RSI", "MACD", "MACD_sig"]
    if any(pd.isna(r[c]) for c in needed):
        return 0, []
    
    score, reasons = 0, []
    
    if r["MA5"] > r["MA20"] > r["MA60"] > r["MA120"]:
        score += 35
        reasons.append("완전정배열")
    elif r["MA20"] > r["MA60"] > r["MA120"]:
        score += 22
        reasons.append("중장기정배열")
    
    if r["Close"] > r["MA20"]:
        score += 12
        reasons.append("MA20위")
    
    if r["MACD"] > 0 and r["MACD"] > r["MACD_sig"]:
        score += 18
        reasons.append("MACD매수")
    
    if 50 <= r["RSI"] <= 70:
        score += 15
        reasons.append(f"RSI {r['RSI']:.0f}")
    elif r["RSI"] > 80:
        score -= 10
    
    if not pd.isna(r["Pct60"]) and r["Pct60"] > 0:
        score += min(15, int(r["Pct60"] / 2))
        reasons.append(f"60일+{r['Pct60']:.0f}%")
    
    return max(0, min(100, score)), reasons


def analyze_one(code: str, name: str, market: str):
    import FinanceDataReader as fdr
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=LOOKBACK_DAYS)
        df = fdr.DataReader(code, start, end)
        if len(df) < 130:
            return None
        
        df = calc_indicators(df)
        last = df.iloc[-1]
        t_score, t_reasons = score_trend(last)
        
        atr = last["ATR"] if not pd.isna(last["ATR"]) else None
        close = float(last["Close"])
        
        return {
            "code": code,
            "name": name,
            "market": market,
            "close": close,
            "chg1d": float((close / df["Close"].iloc[-2] - 1) * 100),
            "chg20d": float(last["Pct20"]) if not pd.isna(last["Pct20"]) else 0.0,
            "chg60d": float(last["Pct60"]) if not pd.isna(last["Pct60"]) else 0.0,
            "rsi": float(last["RSI"]) if not pd.isna(last["RSI"]) else None,
            "vol_ratio": float(last["Vol_ratio"]) if not pd.isna(last["Vol_ratio"]) else None,
            "ma20": float(last["MA20"]) if not pd.isna(last["MA20"]) else None,
            "ma60": float(last["MA60"]) if not pd.isna(last["MA60"]) else None,
            "macd_hist": float(last["MACD_hist"]) if not pd.isna(last["MACD_hist"]) else None,
            "atr": float(atr) if atr is not None else None,
            "stop": float(close - 2 * atr) if atr else None,
            "target": float(close + 3 * atr) if atr else None,
            "trend": t_score,
            "trend_why": ", ".join(t_reasons),
            # 신규 필드 (나중에 채움)
            "supply_demand": None,
            "sentiment": None,
            "financials": None,
            "buy_strategy": None,
            "combined": t_score,
            "priority_tier": None,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2) 종합 점수 계산
# ---------------------------------------------------------------------------
def calculate_combined_score(stock: dict) -> dict:
    """4개 모듈 점수 통합."""
    tech = stock["trend"]
    
    # 수급 점수 (0~100)
    sd = stock.get("supply_demand") or {}
    supply_score = sd.get("smart_money_score", 50)
    
    # 감성 점수 (-100~+100 → 0~100)
    sent = stock.get("sentiment") or {}
    sent_score = (sent.get("score", 0) + 100) / 2 if sent.get("total", 0) > 0 else 50
    
    # 모멘텀 점수 (60일 수익률 기반)
    chg60 = stock.get("chg60d", 0)
    if chg60 >= 30:
        mom_score = 90
    elif chg60 >= 15:
        mom_score = 75
    elif chg60 >= 5:
        mom_score = 60
    elif chg60 >= 0:
        mom_score = 50
    elif chg60 >= -10:
        mom_score = 30
    else:
        mom_score = 15
    
    # 가중 평균
    combined = (
        tech * WEIGHTS["technical"]
        + supply_score * WEIGHTS["supply_demand"]
        + sent_score * WEIGHTS["sentiment"]
        + mom_score * WEIGHTS["momentum"]
    )
    
    return {
        "combined": round(combined, 1),
        "technical_score": tech,
        "supply_score": supply_score,
        "sentiment_score": round(sent_score, 1),
        "momentum_score": mom_score,
    }


# ---------------------------------------------------------------------------
# 3) 종목 유니버스
# ---------------------------------------------------------------------------
def get_universe() -> pd.DataFrame:
    import FinanceDataReader as fdr
    listing = fdr.StockListing("KRX")
    cap_col = next(
        (c for c in ("Marcap", "MarketCap", "marcap") if c in listing.columns),
        None,
    )
    if cap_col is None:
        raise RuntimeError("StockListing 결과에서 시가총액 컬럼을 찾지 못했어요.")
    
    listing = listing.dropna(subset=[cap_col, "Market", "Name"])
    listing = listing[~listing["Name"].str.contains("스팩|우$|우B|우C", regex=True, na=False)]
    
    kospi = listing[listing["Market"] == "KOSPI"].sort_values(cap_col, ascending=False).head(200)
    kosdaq = listing[listing["Market"] == "KOSDAQ"].sort_values(cap_col, ascending=False).head(150)
    
    rows = [{"code": r["Code"], "name": r["Name"], "market": "KOSPI200"} for _, r in kospi.iterrows()]
    rows += [{"code": r["Code"], "name": r["Name"], "market": "KOSDAQ150"} for _, r in kosdaq.iterrows()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4) 메인 파이프라인
# ---------------------------------------------------------------------------
def main():
    print("🚀 한국주식 매매 신호 대시보드 v2.0\n")
    
    # 1. 종목 리스트
    print("📋 종목 리스트 조회 중...")
    universe = get_universe()
    print(f"   코스피200 + 코스닥150 = {len(universe)}개 종목\n")
    
    # 2. 기술적 분석 (전 종목)
    print("📊 기술적 분석 중 (5~10분 소요)...")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(analyze_one, code, name, market): code
            for code, name, market in universe[["code", "name", "market"]]
            .itertuples(index=False, name=None)
        }
        done = 0
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
            done += 1
            if done % 25 == 0:
                print(f"   {done}/{len(universe)} 처리됨")
    
    print(f"\n✓ 기술적 분석 완료: {len(results)}개\n")
    
    # 상위 N개 추출 (상세 분석 대상)
    top_candidates = sorted(results, key=lambda x: -x["trend"])[:TOP_N_DETAIL]
    top_codes = [r["code"] for r in top_candidates]
    
    # 3. 1개월 수급 분석 (상위 30개)
    if ENABLE_SUPPLY_DEMAND:
        print(f"💰 1개월 개인/기관/외국인 수급 분석 (상위 {len(top_codes)}개)...")
        from supply_demand import analyze_supply_demand_batch
        sd_results = analyze_supply_demand_batch(top_codes, workers=4)
        
        for r in results:
            if r["code"] in sd_results:
                r["supply_demand"] = sd_results[r["code"]]
                signal = get_priority_signal(r["supply_demand"])
                r["priority_tier"] = signal["priority"]
                r["priority_icon"] = signal["icon"]
                r["priority_reason"] = signal["reason"]
        
        tier1_count = sum(1 for r in results if r.get("priority_tier") == "TIER1")
        tier2_count = sum(1 for r in results if r.get("priority_tier") == "TIER2")
        print(f"✓ 수급 분석 완료: TIER1 {tier1_count}개, TIER2 {tier2_count}개\n")
    
    # 4. 네이버 뉴스 감성 분석
    if ENABLE_NEWS_SENTIMENT:
        print(f"📰 네이버 뉴스 감성 분석 (상위 {len(top_codes)}개)...")
        from naver_news_sentiment import batch_sentiment_analysis
        news_results = batch_sentiment_analysis(top_codes, workers=4)
        
        for r in results:
            if r["code"] in news_results:
                r["sentiment"] = news_results[r["code"]]
        
        analyzed = sum(1 for r in results if r.get("sentiment") and r["sentiment"].get("total", 0) > 0)
        print(f"✓ 뉴스 분석 완료: {analyzed}개 종목\n")
    
    # 5. DART 재무 데이터 (상위 30개)
    if ENABLE_DART and os.environ.get("DART_API_KEY"):
        print(f"📊 DART 재무 데이터 조회 (상위 {len(top_codes)}개)...")
        for code in top_codes:
            stock = next((r for r in results if r["code"] == code), None)
            if stock:
                fin = get_full_financials(code)
                if fin:
                    stock["financials"] = fin
                    # 밸류에이션 계산 (시가총액은 별도 조회 필요)
                    # market_cap = stock["close"] * 발행주식수
                    # stock["valuation"] = calculate_valuation(code, market_cap, fin)
        print(f"✓ 재무 분석 완료\n")
    
    # 6. 종합 점수 계산
    print("🎯 종합 점수 계산...")
    for r in results:
        scores = calculate_combined_score(r)
        r.update(scores)
    
    # 7. 매수 전략 자동 생성 (상위 20개)
    if ENABLE_BUY_STRATEGY:
        print("📝 매수 전략 자동 생성...")
        # 종합 점수 상위 20개만
        top_combined = sorted(results, key=lambda x: -x["combined"])[:20]
        for r in top_combined:
            r["buy_strategy"] = generate_buy_strategy(
                r,
                portfolio_pct=5.0,
                total_budget=60_000_000,  # 예시
            )
    
    # 8. 대시보드 생성
    print("🎨 대시보드 생성 중...")
    today = dt.date.today().strftime("%Y-%m-%d")
    
    # HTML 템플릿은 별도 파일에서 로드
    template_path = Path(__file__).parent / "dashboard_template.html"
    if template_path.exists():
        html_template = template_path.read_text(encoding="utf-8")
    else:
        html_template = "<!-- dashboard_template.html 파일 필요 -->"
    
    html = (
        html_template
        .replace("__DATE__", today)
        .replace("__DATA__", json.dumps(results, ensure_ascii=False, default=str))
    )
    
    out_path = OUTPUT_DIR / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✅ 대시보드 → {out_path.resolve()}")
    
    # TOP 10 출력
    top10 = sorted(results, key=lambda x: -x["combined"])[:10]
    print(f"\n🏆 종합 매수 우선순위 TOP 10")
    print("=" * 80)
    for i, r in enumerate(top10, 1):
        tier = r.get("priority_tier", "N/A")
        icon = r.get("priority_icon", "")
        print(
            f"{i:2d}. [{r['combined']:5.1f}] {icon} {r['name']:<12s} ({r['code']}) "
            f"₩{r['close']:>9,.0f} | T:{r.get('technical_score', 0):3.0f} "
            f"S:{r.get('supply_score', 50):3.0f} N:{r.get('sentiment_score', 50):3.0f} "
            f"M:{r.get('momentum_score', 50):3.0f} | {tier}"
        )
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(1)
