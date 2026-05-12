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



# ---------------------------------------------------------------------------
# 5) HTML 대시보드 (Robinhood-style + News Sentiment)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Dashboard v2.0 — Full Stack Analysis</title>
<link href="https://cdn.jsdelivr.net/npm/gridjs/dist/theme/mermaid.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/gridjs/dist/gridjs.umd.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>
:root {
  --bg-body: #060d17;
  --bg-card: #121a24;
  --bg-modal: #0b141d;
  --primary: #00ff8a;
  --gold: #fbbf24;
  --up-color: #00ff8a;
  --down-color: #ff3b3b;
  --text-main: #ffffff;
  --text-sub: #8a94a6;
  --border: #232d39;
  --hover-row: #1b2633;
  --tier1: #00ff8a;
  --tier2: #fbbf24;
  --tier3: #60a5fa;
  --avoid: #ff3b3b;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Pretendard', -apple-system, sans-serif;
  background: var(--bg-body);
  color: var(--text-main);
  padding: 20px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.mono { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
.container-max { max-width: 1600px; margin: 0 auto; }

/* === Header === */
.header-section {
  margin-bottom: 32px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 20px;
}
h1 {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--primary) 0%, var(--gold) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.sub { color: var(--text-sub); font-size: 14px; margin-top: 6px; }

/* === Tabs === */
.tabs { display: flex; gap: 12px; margin: 24px 0; flex-wrap: wrap; }
.tab {
  padding: 12px 24px;
  background: transparent;
  border-radius: 100px;
  cursor: pointer;
  font-weight: 700;
  font-size: 14px;
  border: 1.5px solid var(--border);
  color: var(--text-sub);
  transition: all 0.2s ease;
}
.tab.on {
  background: var(--primary);
  color: #000;
  border-color: var(--primary);
}

/* === Priority Bar === */
.priority-bar {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.priority-pill {
  padding: 10px 18px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 700;
  border: 1px solid var(--border);
}
.priority-pill.tier1 {
  background: rgba(0, 255, 138, 0.1);
  color: var(--tier1);
  border-color: rgba(0, 255, 138, 0.3);
}
.priority-pill.tier2 {
  background: rgba(251, 191, 36, 0.1);
  color: var(--tier2);
  border-color: rgba(251, 191, 36, 0.3);
}

/* === Grid Table === */
.table-wrap {
  background: var(--bg-card);
  border-radius: 20px;
  padding: 12px;
  border: 1px solid var(--border);
  overflow-x: auto;
}
.gridjs-table { background: transparent !important; min-width: 1600px; }
.gridjs-th {
  background-color: #0b141d !important;
  color: var(--text-sub) !important;
  padding: 16px 10px !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  text-transform: uppercase;
}
.gridjs-td {
  padding: 14px 10px !important;
  border-bottom: 1px solid var(--border) !important;
  background-color: transparent !important;
  color: #fff !important;
  font-size: 13px !important;
  cursor: pointer;
}
.gridjs-tr { background-color: var(--bg-card) !important; }
.gridjs-tr:hover .gridjs-td { background-color: var(--hover-row) !important; }
.row-tier1 td { background: rgba(0, 255, 138, 0.05) !important; }
.row-tier2 td { background: rgba(251, 191, 36, 0.04) !important; }

.pos { color: var(--up-color); font-weight: 700; }
.neg { color: var(--down-color); font-weight: 700; }

.pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 44px;
  height: 26px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 800;
}
.score-hi { background: var(--up-color); color: #000; }
.score-md { background: #f59e0b; color: #000; }
.score-lo { background: #334155; color: #fff; }

.tier-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 100px;
  font-size: 11px;
  font-weight: 800;
}
.tier-badge.TIER1 { background: rgba(0, 255, 138, 0.15); color: var(--tier1); }
.tier-badge.TIER2 { background: rgba(251, 191, 36, 0.15); color: var(--tier2); }
.tier-badge.TIER3 { background: #2d3748; color: var(--text-sub); }
.tier-badge.AVOID { background: rgba(255, 59, 59, 0.15); color: var(--avoid); }

/* === MODAL === */
.modal-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.85);
  z-index: 1000;
  overflow-y: auto;
  backdrop-filter: blur(8px);
}
.modal-overlay.open { display: block; }
.modal {
  max-width: 1200px;
  margin: 40px auto;
  background: var(--bg-modal);
  border-radius: 24px;
  border: 1px solid var(--border);
  overflow: hidden;
}
.modal-header {
  padding: 28px 32px;
  background: linear-gradient(135deg, rgba(0, 255, 138, 0.05) 0%, rgba(0, 0, 0, 0) 100%);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 16px;
}
.modal-title h2 {
  font-size: 26px;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 4px;
}
.modal-title .code {
  font-family: 'JetBrains Mono', monospace;
  color: var(--text-sub);
  font-size: 13px;
}
.modal-price {
  text-align: right;
}
.modal-price .price {
  font-family: 'JetBrains Mono', monospace;
  font-size: 32px;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.modal-price .change {
  font-size: 14px;
  font-weight: 700;
  margin-top: 4px;
}
.modal-close {
  position: absolute;
  top: 20px;
  right: 24px;
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid var(--border);
  color: #fff;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  cursor: pointer;
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.modal-close:hover { background: var(--down-color); }

.modal-body { padding: 24px 32px; }

/* === Buy Strategy Panel === */
.strategy-panel {
  background: linear-gradient(135deg, rgba(0, 255, 138, 0.08) 0%, var(--bg-card) 100%);
  border: 1px solid var(--primary);
  border-radius: 20px;
  padding: 28px;
  margin-bottom: 24px;
}
.strategy-panel h3 {
  font-size: 18px;
  font-weight: 800;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.strategy-summary {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 20px;
  font-size: 14px;
  color: var(--text-sub);
  line-height: 1.7;
}
.tier-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}
.tier-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  position: relative;
}
.tier-card.tier-1 { border-color: var(--tier1); border-top: 4px solid var(--tier1); }
.tier-card.tier-2 { border-color: var(--tier2); border-top: 4px solid var(--tier2); }
.tier-card.tier-3 { border-color: var(--tier3); border-top: 4px solid var(--tier3); }
.tier-card .label {
  font-size: 13px;
  font-weight: 800;
  margin-bottom: 8px;
}
.tier-card .price-range {
  font-family: 'JetBrains Mono', monospace;
  font-size: 18px;
  font-weight: 800;
  color: var(--text-main);
  margin-bottom: 4px;
}
.tier-card .pct {
  font-size: 11px;
  color: var(--text-sub);
  margin-bottom: 8px;
}
.tier-card .trigger {
  font-size: 12px;
  color: var(--text-sub);
  padding-top: 8px;
  border-top: 1px dashed var(--border);
  margin-top: 8px;
  line-height: 1.6;
}
.tier-card .shares {
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 700;
  color: var(--primary);
}

.strategy-meta {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin-top: 20px;
}
.meta-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
}
.meta-card.stop { border-left: 4px solid var(--down-color); }
.meta-card.profit { border-left: 4px solid var(--up-color); }
.meta-card .label {
  font-size: 11px;
  color: var(--text-sub);
  letter-spacing: 0.05em;
  margin-bottom: 6px;
  font-weight: 700;
  text-transform: uppercase;
}
.meta-card .value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 20px;
  font-weight: 800;
}
.meta-card.stop .value { color: var(--down-color); }
.meta-card.profit .value { color: var(--up-color); }

.tp-ladder {
  list-style: none;
  margin-top: 12px;
}
.tp-ladder li {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px dashed var(--border);
  font-size: 13px;
}
.tp-ladder li:last-child { border-bottom: none; }

/* === 9 Section Cards === */
.section-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}
.section-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
}
.section-card h4 {
  font-size: 14px;
  font-weight: 800;
  margin-bottom: 12px;
  color: var(--text-main);
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-card .section-num {
  background: var(--primary);
  color: #000;
  font-size: 10px;
  font-weight: 800;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.chart-container {
  position: relative;
  height: 200px;
  margin-top: 12px;
}
.metric-table {
  width: 100%;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  margin-top: 8px;
}
.metric-table td {
  padding: 6px 0;
  border-bottom: 1px dashed var(--border);
  color: var(--text-sub);
}
.metric-table td:last-child {
  text-align: right;
  color: var(--text-main);
  font-weight: 700;
}

/* === Supply Demand Card === */
.supply-demand-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 24px;
  margin-bottom: 24px;
}
.supply-demand-card h3 {
  font-size: 18px;
  font-weight: 800;
  margin-bottom: 16px;
}
.sd-summary {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}
.sd-stat {
  background: var(--bg-modal);
  border-radius: 12px;
  padding: 16px;
  text-align: center;
}
.sd-stat .label {
  font-size: 10px;
  color: var(--text-sub);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 700;
  margin-bottom: 6px;
}
.sd-stat .value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 18px;
  font-weight: 800;
}

/* === News Card === */
.news-section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 24px;
  margin-bottom: 24px;
}
.news-section h3 {
  font-size: 18px;
  font-weight: 800;
  margin-bottom: 16px;
}
.news-list { list-style: none; }
.news-item {
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  display: flex;
  gap: 12px;
  align-items: flex-start;
}
.news-item:last-child { border-bottom: none; }
.news-item .tag { font-size: 18px; }
.news-item .info { flex: 1; }
.news-item .title {
  font-size: 14px;
  color: var(--text-main);
  margin-bottom: 4px;
  font-weight: 600;
}
.news-item .meta {
  font-size: 11px;
  color: var(--text-sub);
  font-family: 'JetBrains Mono', monospace;
}
.news-item a {
  color: inherit;
  text-decoration: none;
}
.news-item a:hover .title { color: var(--primary); }

/* === Tooltip === */
.tooltip {
  position: relative;
  display: inline-block;
}
.tooltip .tooltiptext {
  visibility: hidden;
  background: #000;
  color: #fff;
  text-align: center;
  border-radius: 6px;
  padding: 6px 10px;
  position: absolute;
  z-index: 1;
  bottom: 125%;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  white-space: nowrap;
}
.tooltip:hover .tooltiptext { visibility: visible; }

/* === Responsive === */
@media (max-width: 768px) {
  body { padding: 12px; }
  .modal { margin: 16px; }
  .modal-body { padding: 16px; }
  .tier-grid { grid-template-columns: 1fr; }
  .section-grid { grid-template-columns: 1fr; }
  .sd-summary { grid-template-columns: 1fr 1fr; }
  .strategy-meta { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<div class="container-max">
  
  <!-- HEADER -->
  <div class="header-section">
    <h1>📈 Trading Dashboard v2.0</h1>
    <div class="sub">
      Updated: <b>__DATE__</b> · Assets: <b id="cnt"></b> · 
      종합 점수 = 50% 기술 + 20% 수급 + 20% 감성 + 10% 모멘텀
    </div>
    <div id="regimeBox"></div>
  </div>

  <!-- TABS -->
  <div class="tabs">
    <div class="tab on" data-mode="trend">🚀 종합 우선순위</div>
    <div class="tab" data-mode="tier1">⭐ TIER 1 스마트머니</div>
    <div class="tab" data-mode="swing">🎯 스윙 트레이딩</div>
    <div class="tab" data-mode="all">📊 전체 종목</div>
  </div>

  <!-- PRIORITY BAR -->
  <div class="priority-bar" id="priorityBar"></div>

  <!-- TABLE -->
  <div class="table-wrap">
    <div id="grid"></div>
  </div>

  <!-- HINT -->
  <p style="text-align:center; color:var(--text-sub); margin-top:16px; font-size:13px;">
    💡 종목을 클릭하면 <strong style="color:var(--primary);">매수 전략 + 9개 재무 섹션 + 1개월 수급 + 네이버 뉴스</strong>가 표시됩니다
  </p>

</div>

<!-- MODAL -->
<div class="modal-overlay" id="modalOverlay" onclick="if(event.target===this) closeModal()">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div class="modal-header" id="modalHeader"></div>
    <div class="modal-body" id="modalBody"></div>
  </div>
</div>

<script>
const DATA = __DATA__;
document.getElementById("cnt").textContent = DATA.length;

// ===== Helpers =====
const num = (v, d=0) => v == null ? "-" : Number(v).toLocaleString("ko-KR", {minimumFractionDigits: d, maximumFractionDigits: d});
const num100M = (v) => v == null ? "-" : `${(v/1e8).toFixed(1)}억`;
const pct = (v) => {
  if (v == null) return "-";
  return `<span class="${v >= 0 ? 'pos' : 'neg'}">${v >= 0 ? '+' : ''}${v.toFixed(1)}%</span>`;
};
const scoreCell = (v) => {
  const cls = v >= 80 ? "score-hi" : v >= 60 ? "score-md" : "score-lo";
  return `<span class="pill ${cls}">${Math.round(v)}</span>`;
};
const tierBadge = (tier, icon) => {
  if (!tier) return "-";
  return `<span class="tier-badge ${tier}">${icon || ""} ${tier}</span>`;
};

// ===== Priority Bar =====
function renderPriorityBar() {
  const tier1 = DATA.filter(r => r.priority_tier === "TIER1").length;
  const tier2 = DATA.filter(r => r.priority_tier === "TIER2").length;
  const withNews = DATA.filter(r => r.sentiment && r.sentiment.total > 0).length;
  const withFin = DATA.filter(r => r.financials).length;
  
  document.getElementById("priorityBar").innerHTML = `
    <div class="priority-pill tier1">🚀 TIER 1 (스마트머니): ${tier1}개</div>
    <div class="priority-pill tier2">⭐ TIER 2 (외국인/기관): ${tier2}개</div>
    <div class="priority-pill">📰 뉴스 분석: ${withNews}개</div>
    <div class="priority-pill">📊 재무 분석: ${withFin}개</div>
  `;
}

// ===== Grid Rendering =====
let grid = null;
function render(mode) {
  let rows = [...DATA];
  
  if (mode === "tier1") {
    rows = rows.filter(r => r.priority_tier === "TIER1");
  } else if (mode === "swing") {
    rows = rows.filter(r => r.combined >= 70);
  }
  rows.sort((a, b) => (b.combined || 0) - (a.combined || 0));
  
  const cols = [
    { name: "종목", width: "180px", formatter: cell => gridjs.html(cell) },
    { name: "code", hidden: true },
    { name: "종합", width: "65px", formatter: v => gridjs.html(scoreCell(v)) },
    { name: "기술", width: "55px", formatter: v => gridjs.html(scoreCell(v)) },
    { name: "수급", width: "55px", formatter: v => gridjs.html(scoreCell(v)) },
    { name: "감성", width: "55px", formatter: v => gridjs.html(scoreCell(v)) },
    { name: "Tier", width: "100px", formatter: (_, r) => gridjs.html(tierBadge(r.cells[12].data, r.cells[13].data)) },
    { name: "현재가", width: "95px", formatter: v => num(v) },
    { name: "1일", width: "70px", formatter: v => gridjs.html(pct(v)) },
    { name: "20일", width: "70px", formatter: v => gridjs.html(pct(v)) },
    { name: "60일", width: "70px", formatter: v => gridjs.html(pct(v)) },
    { name: "사유", width: "260px", formatter: v => gridjs.html(`<div style="white-space:normal;font-size:12px;color:#cbd5e1;line-height:1.4;">${v || ""}</div>`) },
    { name: "_tier", hidden: true },
    { name: "_icon", hidden: true },
  ];
  
  const data = rows.map(r => [
    `<a class="stock-link" onclick="openModal('${r.code}'); return false;" style="cursor:pointer;color:var(--primary);font-weight:800;text-decoration:none;">${r.name}</a><span style="color:var(--text-sub);font-size:11px;margin-left:6px;">${r.code}</span>`,
    r.code,
    r.combined || r.trend,
    r.technical_score || r.trend,
    r.supply_score || 50,
    r.sentiment_score || 50,
    null, // Tier badge formatter
    r.close,
    r.chg1d,
    r.chg20d,
    r.chg60d,
    r.trend_why,
    r.priority_tier || "TIER3",
    r.priority_icon || "📊",
  ]);
  
  if (grid) grid.destroy();
  grid = new gridjs.Grid({
    columns: cols,
    data,
    sort: true,
    pagination: { limit: 25 },
    search: true,
    resizable: true,
    language: { search: { placeholder: "Search stocks..." } },
    rowAttributes: (row) => {
      const tier = row.cells[12].data;
      if (tier === "TIER1") return { class: "row-tier1" };
      if (tier === "TIER2") return { class: "row-tier2" };
      return {};
    },
  }).render(document.getElementById("grid"));
}

// ===== MODAL =====
let chartInstances = [];
function destroyCharts() {
  chartInstances.forEach(c => { try { c.destroy(); } catch(e){} });
  chartInstances = [];
}

function openModal(code) {
  const stock = DATA.find(r => r.code === code);
  if (!stock) return;
  
  renderModalHeader(stock);
  renderModalBody(stock);
  
  document.getElementById("modalOverlay").classList.add("open");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  document.getElementById("modalOverlay").classList.remove("open");
  document.body.style.overflow = "";
  destroyCharts();
}

function renderModalHeader(s) {
  const changeClass = s.chg1d >= 0 ? "pos" : "neg";
  const sign = s.chg1d >= 0 ? "+" : "";
  
  document.getElementById("modalHeader").innerHTML = `
    <div class="modal-title">
      <h2>${s.name}</h2>
      <div class="code">${s.code} · ${s.market}</div>
      <div style="margin-top:12px;">${tierBadge(s.priority_tier, s.priority_icon)} 
        <span style="color:var(--text-sub);font-size:12px;margin-left:8px;">${s.priority_reason || ""}</span>
      </div>
    </div>
    <div class="modal-price">
      <div class="price">₩${num(s.close)}</div>
      <div class="change ${changeClass}">${sign}${s.chg1d.toFixed(2)}% (1일)</div>
    </div>
  `;
}

function renderModalBody(s) {
  let html = "";
  
  // 1. BUY STRATEGY PANEL
  if (s.buy_strategy) {
    html += renderBuyStrategy(s.buy_strategy);
  } else {
    html += renderBasicStrategy(s);
  }
  
  // 2. SUPPLY DEMAND
  if (s.supply_demand) {
    html += renderSupplyDemand(s.supply_demand);
  }
  
  // 3. NEWS
  if (s.sentiment && s.sentiment.headlines && s.sentiment.headlines.length > 0) {
    html += renderNews(s.sentiment);
  }
  
  // 4. 9 SECTIONS (DART Financials)
  if (s.financials) {
    html += renderFinancials(s.financials);
  } else {
    html += `<div class="news-section">
      <h3>📊 재무 분석</h3>
      <p style="color:var(--text-sub);font-size:13px;line-height:1.6;">
        DART 재무 데이터가 없습니다. <code style="background:var(--bg-modal);padding:2px 6px;border-radius:4px;">DART_API_KEY</code> 환경변수를 설정하면 9개 섹션 분석이 표시됩니다.<br>
        🔗 <a href="https://opendart.fss.or.kr/" target="_blank" style="color:var(--primary);">opendart.fss.or.kr</a>에서 무료 발급 가능.
      </p>
    </div>`;
  }
  
  document.getElementById("modalBody").innerHTML = html;
  
  // Render charts after DOM injection
  setTimeout(() => {
    if (s.supply_demand) renderSupplyChart(s.supply_demand);
    if (s.financials) renderFinancialCharts(s.financials);
  }, 100);
}

function renderBuyStrategy(strategy) {
  const tiers = strategy.entry_tiers.map((t, i) => `
    <div class="tier-card tier-${t.tier}">
      <div class="label">${t.label} (${t.pct}%)</div>
      <div class="price-range">₩${num(t.price_low)} ~ ₩${num(t.price_high)}</div>
      <div class="shares">${t.shares}주 · ₩${num(t.amount)}</div>
      <div class="trigger">${t.trigger}<br><em>${t.condition}</em></div>
    </div>
  `).join("");
  
  const tpLadder = strategy.take_profit.map(tp => `
    <li>
      <span>₩${num(tp.price)} (+${tp.return_pct}%)</span>
      <span style="color:var(--primary);font-weight:700;">${tp.sell_pct}% 매도</span>
    </li>
  `).join("");
  
  return `
    <div class="strategy-panel">
      <h3>💡 8/13 매도 목표 매수 전략 (${strategy.strategy_type})</h3>
      <div class="strategy-summary">${strategy.summary}</div>
      
      <div class="tier-grid">${tiers}</div>
      
      <div style="background:rgba(0,0,0,0.3);border-radius:12px;padding:16px;margin:16px 0;">
        <div style="display:flex;justify-content:space-between;font-size:13px;">
          <span style="color:var(--text-sub);">가중평균 매수가</span>
          <span class="mono" style="font-weight:800;color:var(--primary);">₩${num(strategy.average_buy_price)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:13px;margin-top:6px;">
          <span style="color:var(--text-sub);">총 매수 수량</span>
          <span class="mono">${strategy.total_shares}주</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:13px;margin-top:6px;">
          <span style="color:var(--text-sub);">총 투자금</span>
          <span class="mono">₩${num(strategy.total_investment)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:14px;margin-top:10px;padding-top:10px;border-top:1px dashed var(--border);">
          <span style="color:var(--text-sub);">확률 가중 기대수익률</span>
          <span class="mono" style="font-weight:800;color:var(--up-color);">${strategy.weighted_return >= 0 ? '+' : ''}${strategy.weighted_return}%</span>
        </div>
      </div>
      
      <div class="strategy-meta">
        <div class="meta-card stop">
          <div class="label">🛡️ Stop-Loss</div>
          <div class="value">₩${num(strategy.stop_loss.price)}</div>
          <div style="font-size:11px;color:var(--text-sub);margin-top:4px;">평단 ${strategy.stop_loss.pct_from_avg}%</div>
        </div>
        <div class="meta-card profit">
          <div class="label">💰 차익실현 사다리</div>
          <ul class="tp-ladder">${tpLadder}</ul>
        </div>
      </div>
      
      ${strategy.risk_management.length > 0 ? `
      <div style="margin-top:16px;padding:14px;background:rgba(255,59,59,0.06);border-left:3px solid var(--down-color);border-radius:8px;font-size:12px;">
        <strong style="color:var(--down-color);">⚠️ 리스크 관리</strong>
        <ul style="margin:8px 0 0 16px;color:var(--text-sub);line-height:1.7;">
          ${strategy.risk_management.map(r => `<li>${r}</li>`).join("")}
        </ul>
      </div>` : ""}
    </div>
  `;
}

function renderBasicStrategy(s) {
  return `
    <div class="strategy-panel">
      <h3>💡 매수 전략 (기본)</h3>
      <div class="strategy-summary">
        매수 전략 자동 생성을 위해 상위 후보 분석이 필요합니다. 
        손절가 ₩${num(s.stop)}, 목표가 ₩${num(s.target)} 기준으로 ATR 2배 손절 / 3배 목표 전략을 따르세요.
      </div>
    </div>
  `;
}

function renderSupplyDemand(sd) {
  const forVal = sd.foreign_value;
  const instVal = sd.institution_value;
  const indVal = sd.individual_value;
  const colorFor = forVal >= 0 ? "up-color" : "down-color";
  const colorInst = instVal >= 0 ? "up-color" : "down-color";
  const colorInd = indVal >= 0 ? "up-color" : "down-color";
  
  return `
    <div class="supply-demand-card">
      <h3>💰 1개월 개인/기관/외국인 수급</h3>
      <div class="sd-summary">
        <div class="sd-stat">
          <div class="label">스마트머니 점수</div>
          <div class="value" style="color:var(--primary);">${sd.smart_money_score}/100</div>
        </div>
        <div class="sd-stat">
          <div class="label">외국인</div>
          <div class="value" style="color:var(--${colorFor});">${forVal >= 0 ? '+' : ''}${num100M(forVal)}</div>
        </div>
        <div class="sd-stat">
          <div class="label">기관</div>
          <div class="value" style="color:var(--${colorInst});">${instVal >= 0 ? '+' : ''}${num100M(instVal)}</div>
        </div>
        <div class="sd-stat">
          <div class="label">개인</div>
          <div class="value" style="color:var(--${colorInd});">${indVal >= 0 ? '+' : ''}${num100M(indVal)}</div>
        </div>
      </div>
      <div class="chart-container">
        <canvas id="supplyChart"></canvas>
      </div>
    </div>
  `;
}

function renderSupplyChart(sd) {
  const ctx = document.getElementById("supplyChart");
  if (!ctx || !sd.daily_data || sd.daily_data.length === 0) return;
  
  const labels = sd.daily_data.map(d => d.date.slice(5));  // MM-DD
  const ind = sd.daily_data.map(d => d.individual / 1e8);
  const fore = sd.daily_data.map(d => d.foreign / 1e8);
  const inst = sd.daily_data.map(d => d.institution / 1e8);
  
  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: '외국인 (억원)', data: fore, backgroundColor: 'rgba(0, 255, 138, 0.7)' },
        { label: '기관 (억원)', data: inst, backgroundColor: 'rgba(251, 191, 36, 0.7)' },
        { label: '개인 (억원)', data: ind, backgroundColor: 'rgba(96, 165, 250, 0.5)' },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#8a94a6', font: { size: 10 } }, grid: { color: '#232d39' } },
        y: { ticks: { color: '#8a94a6' }, grid: { color: '#232d39' } },
      },
      plugins: { legend: { labels: { color: '#fff' } } },
    },
  });
  chartInstances.push(chart);
}

function renderNews(sentiment) {
  const items = sentiment.headlines.map(h => `
    <li class="news-item">
      <span class="tag">${h.tag}</span>
      <div class="info">
        <a href="${h.url}" target="_blank">
          <div class="title">${h.title}</div>
        </a>
        <div class="meta">${h.date} · ${h.press}</div>
      </div>
    </li>
  `).join("");
  
  const scoreColor = sentiment.score >= 20 ? "var(--up-color)" : sentiment.score <= -20 ? "var(--down-color)" : "var(--text-sub)";
  const sign = sentiment.score >= 0 ? "+" : "";
  
  return `
    <div class="news-section">
      <h3>📰 네이버 뉴스 감성 분석 
        <span style="color:${scoreColor};margin-left:8px;font-family:'JetBrains Mono';font-size:16px;">${sign}${sentiment.score}</span>
      </h3>
      <div style="display:flex;gap:12px;margin-bottom:16px;font-size:12px;color:var(--text-sub);">
        <span>📈 긍정 ${sentiment.positive_count}건</span>
        <span>📉 부정 ${sentiment.negative_count}건</span>
        <span>· 중립 ${sentiment.neutral_count}건</span>
        <span style="margin-left:auto;">총 ${sentiment.total}건 분석</span>
      </div>
      <ul class="news-list">${items}</ul>
    </div>
  `;
}

function renderFinancials(fin) {
  return `
    <div class="news-section" style="margin-bottom:16px;">
      <h3>📊 DART 재무 분석 (9개 섹션)</h3>
      <p style="font-size:12px;color:var(--text-sub);margin-bottom:16px;">
        ${fin.periods.join(" · ")}
      </p>
    </div>
    <div class="section-grid">
      ${renderFinSection(1, "Growth (성장성)", fin.periods, [
        ["매출", fin.growth.revenue],
        ["영업이익", fin.growth.op_income],
        ["당기순이익", fin.growth.net_income],
      ])}
      ${renderFinSection(2, "Profitability (수익성 %)", fin.periods, [
        ["영업이익률", fin.profitability.op_margin],
        ["순이익률", fin.profitability.net_margin],
        ["ROE", fin.profitability.roe],
      ], true)}
      ${renderFinSection(3, "Balance Sheet (재무상태)", fin.periods, [
        ["자산총계", fin.balance_sheet.total_assets],
        ["부채총계", fin.balance_sheet.total_liab],
        ["자본총계", fin.balance_sheet.total_equity],
      ])}
      ${renderFinSection(4, "Cash Flow (현금흐름)", fin.periods, [
        ["영업CF", fin.cash_flow.op_cf],
        ["투자CF", fin.cash_flow.inv_cf],
        ["재무CF", fin.cash_flow.fin_cf],
      ])}
      ${renderFinSection(5, "Liquidity (유동성)", fin.periods, [
        ["현금", fin.liquidity.cash],
        ["유동부채", fin.liquidity.current_liab],
        ["부채총계", fin.liquidity.total_liab],
      ])}
      ${renderFinSection(6, "Working Capital (운전자본)", fin.periods, [
        ["매출채권", fin.working_capital.receivables],
        ["재고자산", fin.working_capital.inventory],
        ["유동자산", fin.working_capital.current_assets],
      ])}
    </div>
  `;
}

function renderFinSection(num_, title, periods, datasets, isPct = false) {
  const tableId = `chart_sec_${num_}`;
  const rows = datasets.map(([name, values]) => {
    const cells = values.map(v => {
      if (isPct) return `${(v || 0).toFixed(1)}%`;
      const abs = Math.abs(v || 0);
      if (abs >= 1e12) return `${(v/1e12).toFixed(1)}조`;
      if (abs >= 1e8) return `${(v/1e8).toFixed(0)}억`;
      return num(v);
    });
    return `<tr><td>${name}</td><td>${cells[cells.length-1]}</td></tr>`;
  }).join("");
  
  return `
    <div class="section-card">
      <h4><span class="section-num">${num_}</span> ${title}</h4>
      <div class="chart-container">
        <canvas id="${tableId}" data-section='${JSON.stringify({periods, datasets, isPct})}'></canvas>
      </div>
      <table class="metric-table">${rows}</table>
    </div>
  `;
}

function renderFinancialCharts(fin) {
  document.querySelectorAll('[id^="chart_sec_"]').forEach(canvas => {
    const cfg = JSON.parse(canvas.dataset.section);
    const colors = ['#00ff8a', '#fbbf24', '#60a5fa'];
    const datasets = cfg.datasets.map(([name, values], idx) => ({
      label: name,
      data: values.map(v => cfg.isPct ? v : v / 1e8),  // 억원 단위
      borderColor: colors[idx],
      backgroundColor: colors[idx] + "30",
      tension: 0.3,
    }));
    
    const chart = new Chart(canvas, {
      type: 'line',
      data: { labels: cfg.periods, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#8a94a6', font: { size: 10 } }, grid: { color: '#232d39' } },
          y: { ticks: { color: '#8a94a6' }, grid: { color: '#232d39' } },
        },
        plugins: { legend: { labels: { color: '#fff', font: { size: 11 } } } },
      },
    });
    chartInstances.push(chart);
  });
}

// ===== Init =====
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", (e) => {
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("on"));
  e.target.classList.add("on");
  render(e.target.dataset.mode);
}));

renderPriorityBar();
render("trend");

// ESC to close modal
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});
</script>

</body>
</html>
"""
