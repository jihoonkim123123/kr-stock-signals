"""
한국주식 매매 신호 대시보드 v3 (Robinhood-style, Single File)
==============================================================

종합 점수 = 40% 기술 + 30% 수급(개인/기관/외국인) + 30% 모멘텀
뉴스 감성은 별도 표시 (참고용 — 점수 가중에서 제외).

신규:
- 거래량 폭발 + RSI 과매도 탈출 필터
- ATR 기반 변동성 조정 포지션 크기 권장
- 15~20개 포지션 권장 (자본 균등 분할이 아닌 변동성 가중)

사용법:
    pip install -r requirements.txt
    python kr_stock_signals.py
"""
from __future__ import annotations
import datetime as dt
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent
LOOKBACK_DAYS = 260
WORKERS = 8

# 운용 설정
SCORE_THRESHOLD = 75
ENABLE_SWING = False
REGIME_FILTER = True
REGIME_INDEX = "KS11"
REGIME_MA = 200

# 종합 점수 가중치 (총 100%, 감성은 제외)
WEIGHTS = {
    "technical": 0.40,
    "supply":    0.30,
    "momentum":  0.30,
}

# 포지션 크기 권장 (ATR 기반 변동성 조정)
RISK_PER_TRADE = 0.005           # 거래당 자본의 0.5% 손실 한도
MIN_POSITION_PCT = 3.0
MAX_POSITION_PCT = 8.0
TARGET_POSITIONS = 18            # 목표 동시 보유 종목 수 (15~20)

# 수급 분석
SUPPLY_LOOKBACK_DAYS = 25        # 최근 1개월 ≈ 25 거래일

# 뉴스 (참고용)
ENABLE_NEWS = True
NEWS_TOP_N = 30
NEWS_LOOKBACK_DAYS = 14
NEWS_WORKERS = 4

POSITIVE_KW = [
    "beat", "beats", "record", "surge", "rally", "upgrade", "outperform",
    "buyback", "acquire", "acquisition", "strong", "growth", "expansion",
    "all-time high", "breakthrough", "deal", "partnership", "boom", "soar",
    "jumps", "raise", "raised", "wins", "approved", "approval", "launched",
    "어닝서프라이즈", "호실적", "사상최고", "신고가", "상향", "흑자전환",
    "수주", "계약체결", "독점공급", "특허", "신제품", "확대", "급등",
    "매수추천", "목표가상향", "최대실적", "사상최대",
]
NEGATIVE_KW = [
    "miss", "misses", "decline", "drop", "slump", "downgrade", "underperform",
    "lawsuit", "probe", "investigation", "recall", "layoff", "loss", "warning",
    "fraud", "scandal", "below estimate", "weak", "concern", "fall", "plunge",
    "cuts", "slashed", "delay", "delayed", "tumble", "sinks",
    "어닝쇼크", "급락", "하락", "적자", "감원", "하향", "리콜", "수사", "조사",
    "벌금", "소송", "부진", "전망악화", "감소", "철수", "결함", "리스크",
    "매도추천", "목표가하향", "손실",
]


# =============================================================================
# 1) 시장 레짐 — KOSPI vs 200MA
# =============================================================================

def check_regime() -> dict:
    import FinanceDataReader as fdr
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=REGIME_MA * 2 + 60)
        df = fdr.DataReader(REGIME_INDEX, start, end)
        if len(df) < REGIME_MA:
            return {"ok": True, "kospi": None, "ma": None, "diff_pct": 0.0, "warn": False}
        ma = float(df["Close"].rolling(REGIME_MA).mean().iloc[-1])
        last = float(df["Close"].iloc[-1])
        diff = float((last / ma - 1) * 100)
        return {"ok": bool(last > ma), "kospi": last, "ma": ma,
                "diff_pct": diff, "warn": bool(last <= ma)}
    except Exception as e:
        print(f"  ⚠️ 레짐 데이터 조회 실패 ({e}). 강세장 가정.")
        return {"ok": True, "kospi": None, "ma": None, "diff_pct": 0.0, "warn": False}


# =============================================================================
# 2) 수급 분석 — 개인/기관/외국인 (pykrx)
# =============================================================================

def fetch_supply_demand(code: str) -> dict:
    """최근 25 거래일 기관+외국인 누적 순매수 분석.

    smart_money = 기관 + 외국인 (개인의 반대편 = 스마트머니)
    score 50 = 중립, 100 = 강한 매수, 0 = 강한 매도
    """
    out = {"smart_score": 50, "inst_net": 0.0, "foreign_net": 0.0, "indiv_net": 0.0,
           "buying_days": 0, "n_days": 0, "ok": False}
    try:
        from pykrx import stock
        end = dt.date.today()
        start = end - dt.timedelta(days=SUPPLY_LOOKBACK_DAYS + 15)
        df = stock.get_market_trading_value_by_date(
            start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code
        )
        if df.empty:
            return out
        inst_col = next((c for c in df.columns if "기관" in c and "합계" in c), None) or \
                   next((c for c in df.columns if "기관" in c), None)
        foreign_col = next((c for c in df.columns if "외국인" in c and "합계" in c), None) or \
                      next((c for c in df.columns if "외국인" in c), None)
        indiv_col = next((c for c in df.columns if "개인" in c), None)
        total_col = next((c for c in df.columns if "전체" in c or "총" in c), None)
        if not (inst_col and foreign_col):
            return out
        recent = df.tail(SUPPLY_LOOKBACK_DAYS)
        inst_net = float(recent[inst_col].sum())
        foreign_net = float(recent[foreign_col].sum())
        indiv_net = float(recent[indiv_col].sum()) if indiv_col else 0.0
        smart_net = inst_net + foreign_net
        if total_col:
            total_val = float(recent[total_col].abs().sum()) or 1.0
        else:
            total_val = abs(inst_net) + abs(foreign_net) + abs(indiv_net) or 1.0
        ratio = smart_net / total_val
        if ratio > 0.15:    score = 92
        elif ratio > 0.08:  score = 80
        elif ratio > 0.03:  score = 68
        elif ratio > 0:     score = 55
        elif ratio > -0.03: score = 42
        elif ratio > -0.08: score = 28
        else:               score = 15
        buying_days = int(((recent[inst_col] + recent[foreign_col]) > 0).sum())
        if buying_days >= 15:  score = min(100, score + 5)
        elif buying_days <= 8: score = max(0, score - 5)
        out.update({
            "smart_score": score, "inst_net": inst_net, "foreign_net": foreign_net,
            "indiv_net": indiv_net, "buying_days": buying_days,
            "n_days": len(recent), "ok": True,
        })
    except Exception:
        pass
    return out


# =============================================================================
# 3) 뉴스 감성 (참고용 — 점수 가중치에서 제외)
# =============================================================================

def fetch_news_sentiment(code: str, market: str) -> dict:
    try:
        import yfinance as yf
        suffix = ".KS" if market == "KOSPI200" else ".KQ"
        ticker = yf.Ticker(f"{code}{suffix}")
        news = ticker.news or []
    except Exception:
        return {"score": 0.0, "n": 0, "headlines": []}
    if not news:
        return {"score": 0.0, "n": 0, "headlines": []}
    cutoff = time.time() - NEWS_LOOKBACK_DAYS * 86400
    recent = []
    for art in news:
        ts = art.get("providerPublishTime", 0)
        if not ts and isinstance(art.get("content"), dict):
            ts_str = art["content"].get("pubDate", "")
            try:
                ts = dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() if ts_str else 0
            except Exception:
                ts = 0
        if ts and ts > cutoff:
            recent.append(art)
    if not recent:
        recent = news[:8]
    pos, neg = 0, 0
    headlines = []
    for art in recent[:15]:
        title = art.get("title") or ""
        if not title and isinstance(art.get("content"), dict):
            title = art["content"].get("title", "")
        if not title:
            continue
        lower = title.lower()
        p = sum(1 for kw in POSITIVE_KW if kw.lower() in lower)
        n = sum(1 for kw in NEGATIVE_KW if kw.lower() in lower)
        pos += p; neg += n
        tag = "📈" if p > n else "📉" if n > p else "·"
        url = art.get("link", "")
        if not url and isinstance(art.get("content"), dict):
            cu = art["content"].get("canonicalUrl") or {}
            url = cu.get("url", "") if isinstance(cu, dict) else ""
        headlines.append({"title": title, "tag": tag, "url": url})
    total = pos + neg
    sent = round((pos - neg) / total * 100, 1) if total > 0 else 0.0
    return {"score": sent, "n": len(recent), "headlines": headlines[:5]}


# =============================================================================
# 4) 종목 유니버스
# =============================================================================

def get_universe() -> pd.DataFrame:
    import FinanceDataReader as fdr
    listing = fdr.StockListing("KRX")
    cap_col = next((c for c in ("Marcap", "MarketCap", "marcap") if c in listing.columns), None)
    if cap_col is None:
        raise RuntimeError("StockListing 시가총액 컬럼 없음.")
    listing = listing.dropna(subset=[cap_col, "Market", "Name"])
    listing = listing[~listing["Name"].str.contains("스팩|우$|우B|우C", regex=True, na=False)]
    kospi = listing[listing["Market"] == "KOSPI"].sort_values(cap_col, ascending=False).head(200)
    kosdaq = listing[listing["Market"] == "KOSDAQ"].sort_values(cap_col, ascending=False).head(150)
    rows = [{"code": r["Code"], "name": r["Name"], "market": "KOSPI200"} for _, r in kospi.iterrows()]
    rows += [{"code": r["Code"], "name": r["Name"], "market": "KOSDAQ150"} for _, r in kosdaq.iterrows()]
    return pd.DataFrame(rows)


# =============================================================================
# 5) 기술적 지표 + 신규 필터
# =============================================================================

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
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_up"] = bb_mid + 2 * bb_std
    df["BB_lo"] = bb_mid - 2 * bb_std
    df["BB_pct"] = (df["Close"] - df["BB_lo"]) / (df["BB_up"] - df["BB_lo"]).replace(0, np.nan)
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / df["Vol_MA20"].replace(0, np.nan)
    df["Vol_5d"] = df["Volume"].rolling(5).mean()
    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift()).abs()
    l_c = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / df["Close"]
    df["Pct5"] = (df["Close"] / df["Close"].shift(5) - 1) * 100
    df["Pct20"] = (df["Close"] / df["Close"].shift(20) - 1) * 100
    df["Pct60"] = (df["Close"] / df["Close"].shift(60) - 1) * 100
    df["RSI_prev"] = df["RSI"].shift(1)
    df["RSI_5d_min"] = df["RSI"].rolling(5).min()
    df["Hist_prev"] = df["MACD_hist"].shift(1)

    df["price_min_recent"] = df["Close"].rolling(10).min()
    df["price_min_prev"] = df["Close"].rolling(20).min().shift(10)
    df["rsi_min_recent"] = df["RSI"].rolling(10).min()
    df["rsi_min_prev"] = df["RSI"].rolling(20).min().shift(10)
    df["bullish_div"] = (
        (df["price_min_recent"] < df["price_min_prev"]) &
        (df["rsi_min_recent"] > df["rsi_min_prev"]) &
        (df["RSI"] < 60)
    )
    hi20 = df["High"].rolling(20).max()
    lo20 = df["Low"].rolling(20).min()
    df["range_20"] = (hi20 - lo20) / df["Close"]
    df["price_pos_20"] = (df["Close"] - lo20) / (hi20 - lo20).replace(0, np.nan)
    df["consolidation_breakout"] = (
        (df["range_20"] < 0.18) &
        (df["price_pos_20"] > 0.65) &
        (df["Vol_5d"] > df["Vol_MA20"].replace(0, np.nan) * 1.2)
    )

    # 신규 필터
    df["volume_explosion"] = df["Vol_ratio"] >= 3.0
    df["rsi_oversold_exit"] = (df["RSI_5d_min"] < 30) & (df["RSI"] >= 35) & (df["RSI"] <= 60)

    return df


# =============================================================================
# 6) 점수 (필터 강화)
# =============================================================================

def score_swing(r):
    if pd.isna(r["RSI"]) or pd.isna(r["BB_pct"]):
        return 0, []
    score, reasons = 0, []
    if 30 <= r["RSI"] <= 45 and r["RSI"] > r["RSI_prev"]:
        score += 30; reasons.append(f"RSI반등 {r['RSI_prev']:.0f}→{r['RSI']:.0f}")
    elif r["RSI"] < 30:
        score += 22; reasons.append(f"RSI과매도 {r['RSI']:.0f}")
    if r["BB_pct"] < 0.2:
        score += 22; reasons.append("BB하단")
    elif r["BB_pct"] < 0.4:
        score += 10
    if not pd.isna(r["Vol_ratio"]):
        if r["Vol_ratio"] >= 2.0:
            score += 18; reasons.append(f"거래량x{r['Vol_ratio']:.1f}")
        elif r["Vol_ratio"] >= 1.5:
            score += 10
    if r["Close"] > r["Open"] and not pd.isna(r["MA5"]) and r["Close"] >= r["MA5"]:
        score += 15; reasons.append("양봉+MA5↑")
    if not pd.isna(r["Hist_prev"]) and r["MACD_hist"] > r["Hist_prev"] and r["Hist_prev"] < 0:
        score += 10; reasons.append("MACD반등")
    if not pd.isna(r["Pct60"]) and r["Pct60"] < -25:
        score -= 15; reasons.append("⚠️60일급락")
    return max(0, min(100, score)), reasons


def score_trend(r):
    needed = ["MA5", "MA20", "MA60", "MA120", "RSI", "MACD", "MACD_sig"]
    if any(pd.isna(r[c]) for c in needed):
        return 0, []
    score, reasons = 0, []
    if r["MA5"] > r["MA20"] > r["MA60"] > r["MA120"]:
        score += 35; reasons.append("완전정배열")
    elif r["MA20"] > r["MA60"] > r["MA120"]:
        score += 22; reasons.append("중장기정배열")
    if r["Close"] > r["MA20"]:
        score += 12; reasons.append("MA20위")
    if r["MACD"] > 0 and r["MACD"] > r["MACD_sig"]:
        score += 18; reasons.append("MACD매수")
    elif r["MACD"] > r["MACD_sig"]:
        score += 8
    if 50 <= r["RSI"] <= 70:
        score += 15; reasons.append(f"RSI {r['RSI']:.0f}")
    elif 70 < r["RSI"] <= 80:
        score += 5
    elif r["RSI"] > 80:
        score -= 10; reasons.append("⚠️과열")
    if not pd.isna(r["Pct60"]) and r["Pct60"] > 0:
        score += min(15, int(r["Pct60"] / 2)); reasons.append(f"60일+{r['Pct60']:.0f}%")
    bd = r.get("bullish_div", False)
    if bd is True or (bd is not None and not pd.isna(bd) and bool(bd)):
        score += 12; reasons.append("RSI다이버전스↑")
    cb = r.get("consolidation_breakout", False)
    if cb is True or (cb is not None and not pd.isna(cb) and bool(cb)):
        score += 10; reasons.append("횡보+거래량↑")
    ve = r.get("volume_explosion", False)
    if ve is True or (ve is not None and not pd.isna(ve) and bool(ve)):
        score += 8; reasons.append("거래량폭발x3+")
    oe = r.get("rsi_oversold_exit", False)
    if oe is True or (oe is not None and not pd.isna(oe) and bool(oe)):
        score += 10; reasons.append("RSI과매도탈출")
    return max(0, min(100, score)), reasons


# =============================================================================
# 7) 모멘텀 점수 + ATR 포지션 크기
# =============================================================================

def momentum_score(chg60d):
    if chg60d is None or pd.isna(chg60d):
        return 50
    if chg60d >= 30:    return 92
    elif chg60d >= 15:  return 78
    elif chg60d >= 5:   return 62
    elif chg60d >= 0:   return 52
    elif chg60d >= -10: return 35
    elif chg60d >= -25: return 20
    else:               return 8


def position_size_pct(atr_pct):
    """ATR/Close (일일 변동성) 기반 권장 포지션 크기."""
    if atr_pct is None or pd.isna(atr_pct) or atr_pct <= 0:
        return MIN_POSITION_PCT
    raw = RISK_PER_TRADE / (2 * atr_pct) * 100
    return float(max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, raw)))


# =============================================================================
# 8) 종목별 분석
# =============================================================================

def analyze_one(code, name, market):
    import FinanceDataReader as fdr
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=LOOKBACK_DAYS)
        df = fdr.DataReader(code, start, end)
        if len(df) < 130:
            return None
        df = calc_indicators(df)
        last = df.iloc[-1]
        s_score, s_reasons = score_swing(last)
        t_score, t_reasons = score_trend(last)
        atr = last["ATR"] if not pd.isna(last["ATR"]) else None
        atr_pct = float(last["ATR_pct"]) if not pd.isna(last["ATR_pct"]) else None
        close = float(last["Close"])
        pos_pct = position_size_pct(atr_pct)
        return {
            "code": code, "name": name, "market": market, "close": close,
            "chg1d": float((close / df["Close"].iloc[-2] - 1) * 100),
            "chg20d": float(last["Pct20"]) if not pd.isna(last["Pct20"]) else 0.0,
            "chg60d": float(last["Pct60"]) if not pd.isna(last["Pct60"]) else 0.0,
            "rsi": float(last["RSI"]) if not pd.isna(last["RSI"]) else None,
            "vol_ratio": float(last["Vol_ratio"]) if not pd.isna(last["Vol_ratio"]) else None,
            "atr": float(atr) if atr is not None else None,
            "atr_pct": atr_pct,
            "stop": float(close - 2 * atr) if atr else None,
            "target": float(close + 3 * atr) if atr else None,
            "pos_pct": round(pos_pct, 1),
            "swing": s_score, "swing_why": ", ".join(s_reasons),
            "trend": t_score, "trend_why": ", ".join(t_reasons),
            "supply_score": None, "supply_detail": None,
            "momentum": momentum_score(float(last["Pct60"]) if not pd.isna(last["Pct60"]) else None),
            "sentiment": None, "news_count": 0, "headlines": [],
            "combined": t_score,
        }
    except Exception:
        return None


# =============================================================================
# 9) HTML 대시보드 (Robinhood-style v3)
# =============================================================================

HTML = r"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Signals Dashboard v3</title>
<link href="https://cdn.jsdelivr.net/npm/gridjs/dist/theme/mermaid.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/gridjs/dist/gridjs.umd.js"></script>
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
:root { --bg-body:#060d17;--bg-card:#121a24;--primary:#00ff8a;--up:#00ff8a;--down:#ff3b3b;
        --text:#fff;--sub:#8a94a6;--border:#232d39;--hover:#1b2633; }
body { font-family:'Pretendard',system-ui,sans-serif; background:var(--bg-body); color:var(--text);
       margin:0; padding:20px; line-height:1.5; }
.container-max { max-width:1700px; margin:0 auto; }
h1 { font-size:26px; font-weight:800; margin:0; letter-spacing:-0.5px; }
.sub { color:var(--sub); font-size:13px; margin-top:6px; }
.header-section { margin-bottom:28px; border-bottom:1px solid var(--border); padding-bottom:18px; }
.regime { padding:18px; border-radius:14px; margin:18px 0; background:var(--bg-card);
          border:1px solid var(--border); display:flex; align-items:center; gap:14px; }
.regime.bull { border-left:5px solid var(--up); color:var(--up); }
.regime.bear { border-left:5px solid var(--down); color:var(--down); }
.regime-info b { font-size:17px; display:block; margin-bottom:2px; }
.tabs { display:flex; gap:10px; margin:20px 0; overflow-x:auto; }
.tab { white-space:nowrap; padding:11px 22px; border-radius:100px; cursor:pointer;
       font-weight:700; font-size:14px; border:1.5px solid var(--border);
       color:var(--sub); background:transparent; transition:0.2s; }
.tab.on { background:var(--primary); color:#000; border-color:var(--primary);
          box-shadow:0 4px 15px rgba(0,255,138,0.3); }
.summary-bar { margin-bottom:14px; display:flex; gap:10px; flex-wrap:wrap; }
.info-pill { padding:9px 16px; border-radius:10px; font-size:13px; font-weight:600;
             border:1px solid var(--border); }
.candidates-bar { background:rgba(0,255,138,0.1); color:var(--primary);
                  border:1px solid rgba(0,255,138,0.25); }
.table-wrap { background:var(--bg-card); border-radius:18px; padding:12px;
              border:1px solid var(--border); overflow-x:auto; }
.gridjs-table { table-layout:auto !important; width:100% !important;
                min-width:1600px; background:transparent !important; }
.gridjs-th { background:#0b141d !important; color:var(--sub) !important;
             padding:14px 8px !important; font-size:11px !important; font-weight:700 !important;
             text-transform:uppercase; letter-spacing:0.4px;
             border-bottom:2px solid var(--border) !important; }
.gridjs-td { padding:14px 8px !important; border-bottom:1px solid var(--border) !important;
             font-size:13px !important; color:#fff !important; background:transparent !important;
             vertical-align:middle; }
.gridjs-tr { background:var(--bg-card) !important; }
.gridjs-tr:hover .gridjs-td { background:var(--hover) !important; color:var(--primary) !important; }
.row-priority td { background:rgba(0,255,138,0.04) !important; }
.pos { color:var(--up) !important; font-weight:700; }
.neg { color:var(--down) !important; font-weight:700; }
.pill { display:inline-flex; align-items:center; justify-content:center;
        min-width:40px; height:24px; border-radius:6px; font-size:12px; font-weight:800;
        color:#fff !important; }
.score-hi { background:var(--up); color:#000 !important; }
.score-md { background:#f59e0b; color:#000 !important; }
.score-lo { background:#334155; }
.spill { display:inline-flex; align-items:center; justify-content:center;
         min-width:48px; height:24px; border-radius:6px; font-size:11px; font-weight:700; }
.sp-pos { background:rgba(0,255,138,0.15); color:var(--up) !important; }
.sp-neg { background:rgba(255,59,59,0.15); color:var(--down) !important; }
.sp-neu { background:#2d3748; color:var(--sub) !important; }
.sp-na { color:var(--sub) !important; font-size:11px; }
.stock-link { text-decoration:none; color:var(--primary); font-weight:800; font-size:14px; }
.stock-code { color:var(--sub); font-size:11px; margin-left:5px; }
.reason-cell { white-space:normal !important; min-width:230px; font-size:12px;
               color:#cbd5e1 !important; line-height:1.4; }
.pos-pct { font-weight:700; color:var(--primary); }
.gridjs-search-input { background:#1b2633 !important; color:#fff !important;
                       border:1px solid var(--border) !important; padding:10px 14px !important;
                       border-radius:10px !important; width:280px !important; }
.gridjs-pagination .gridjs-summary { color:var(--sub) !important; }
.gridjs-pagination button { color:#fff !important; background:#1b2633 !important;
                            border:1px solid var(--border) !important; border-radius:6px !important; }
.gridjs-pagination button.gridjs-currentPage { background:var(--primary) !important; color:#000 !important; }
details.news-detail { margin-top:18px; }
details.news-detail summary { cursor:pointer; color:var(--sub); font-size:13px;
                               padding:9px 16px; border-radius:8px; background:#0b141d;
                               border:1px solid var(--border); display:inline-block; }
details.news-detail[open] summary { color:var(--primary); }
.news-card { margin-top:10px; padding:12px 16px; background:#0b141d;
             border-radius:10px; border:1px solid var(--border); }
.news-card h4 { margin:0 0 8px; font-size:14px; color:var(--primary); }
.news-card ul { margin:0; padding-left:18px; color:#cbd5e1; font-size:12px; line-height:1.6; }
.news-card ul a { color:#cbd5e1; text-decoration:none; }
.news-card ul a:hover { color:var(--primary); }
</style></head><body>
<div class="container-max">
<div class="header-section">
<h1>Trading Dashboard <span style="font-size:14px;color:var(--sub)">v3</span></h1>
<div class="sub">Updated: <b>__DATE__</b> · Assets: <b id="cnt"></b> ·
Combined = 40% Technical + 30% Supply + 30% Momentum (감성은 참고용)</div>
<div id="regimeBox"></div></div>
<div class="tabs">
<div class="tab on" data-mode="trend">📈 Trend Following</div>
<div class="tab" data-mode="swing">🎯 Swing Trading</div>
<div class="tab" data-mode="all">📋 All Assets</div>
</div>
<div class="summary-bar" id="candidatesBar"></div>
<div class="table-wrap"><div id="grid"></div></div>
<div id="newsDetail"></div>
</div>
<script>
const DATA = __DATA__;
const REGIME = __REGIME__;
const TARGET_POSITIONS = __TARGET_POSITIONS__;
document.getElementById("cnt").textContent = DATA.length;
(function() {
    if (!REGIME || REGIME.kospi == null) return;
    const isBull = REGIME.ok;
    document.getElementById("regimeBox").innerHTML =
        `<div class="regime ${isBull?'bull':'bear'}">
            <div style="font-size:22px">${isBull?'💹':'⚠️'}</div>
            <div class="regime-info">
                <b>Market Regime: ${isBull?'Bullish':'Bearish'}</b>
                <span>KOSPI ${REGIME.kospi.toLocaleString()} (200MA: ${REGIME.diff_pct>=0?'+':''}${REGIME.diff_pct.toFixed(1)}%) — ${isBull?'Favorable':'Caution'}</span>
            </div></div>`;
})();
const num = (v) => v==null?"-":Number(v).toLocaleString("ko-KR");
const pct = v => v==null?"-":`<span class="${v>=0?'pos':'neg'}">${v>=0?'+':''}${v.toFixed(1)}%</span>`;
const scoreCell = v => {
    if (v == null) return "-";
    const cls = v>=80?"score-hi":v>=60?"score-md":"score-lo";
    return `<span class="pill ${cls}">${Math.round(v)}</span>`;
};
const sentimentCell = (v, n) => {
    if (v==null || n===0) return `<span class="sp-na">no news</span>`;
    const cls = v>=20?"sp-pos":v<=-20?"sp-neg":"sp-neu";
    return `<span class="spill ${cls}">${v>0?'+':''}${v.toFixed(0)}</span>`;
};
const supplyCell = v => {
    if (v==null) return `<span class="sp-na">-</span>`;
    const cls = v>=70?"sp-pos":v<=40?"sp-neg":"sp-neu";
    return `<span class="spill ${cls}">${v}</span>`;
};
let grid = null;
function render(mode) {
    let rows = [...DATA];
    if (mode === "swing") {
        rows = rows.filter(r=>r.swing>0).sort((a,b)=>b.swing-a.swing);
    } else {
        rows = rows.filter(r=>r.trend>0).sort((a,b)=>(b.combined-a.combined)||(b.trend-a.trend));
    }
    const TH = 75;
    const candCount = rows.filter(r=>r.combined>=TH).length;
    const supN = rows.filter(r=>r.supply_score!=null).length;
    const newsN = rows.filter(r=>r.news_count>0).length;
    document.getElementById("candidatesBar").innerHTML = `
        ${candCount>0?`<div class="info-pill candidates-bar">✨ Top Picks: ${candCount}개 (Combined ${TH}+)</div>`:""}
        <div class="info-pill">💰 수급분석 ${supN}개 · 📰 뉴스 ${newsN}개 (참고용)</div>
        <div class="info-pill">📊 권장 ${TARGET_POSITIONS}종목 분산 · ATR 변동성 기반 비중</div>`;
    const cols = [
        { name:"종목", width:"170px", formatter:v=>gridjs.html(v) },
        { name:"코드", hidden:true }, { name:"이름", hidden:true },
        { name:"종합", width:"70px", formatter:v=>gridjs.html(scoreCell(v)) },
        { name:"기술", width:"60px", formatter:v=>gridjs.html(scoreCell(v)) },
        { name:"수급", width:"70px", formatter:v=>gridjs.html(supplyCell(v)) },
        { name:"모멘", width:"65px", formatter:v=>gridjs.html(scoreCell(v)) },
        { name:"감성", width:"75px",
          formatter: (_,r)=>gridjs.html(sentimentCell(r.cells[19].data, r.cells[20].data)),
          attributes:{title:'뉴스 감성 (참고용)'} },
        { name:"비중", width:"70px",
          formatter:v=>gridjs.html(`<span class="pos-pct">${v?v.toFixed(1)+'%':'-'}</span>`),
          attributes:{title:'ATR 변동성 기반 권장 포지션'} },
        { name:"시장", width:"85px" },
        { name:"종가", width:"85px", formatter:v=>num(v) },
        { name:"1일", width:"70px", formatter:v=>gridjs.html(pct(v)) },
        { name:"60일", width:"75px", formatter:v=>gridjs.html(pct(v)) },
        { name:"RSI", width:"55px", formatter:v=>v==null?"-":v.toFixed(0) },
        { name:"거래량x", width:"75px", formatter:v=>v==null?"-":v.toFixed(1) },
        { name:"사유", width:"240px", formatter:v=>gridjs.html(`<div class="reason-cell">${v||""}</div>`) },
        { name:"손절가", width:"85px", formatter:v=>num(v) },
        { name:"목표가", width:"85px", formatter:v=>num(v) },
        { name:"_sup_n", hidden:true },
        { name:"_sentiment", hidden:true },
        { name:"_news_count", hidden:true },
    ];
    const data = rows.map(r => [
        `<a class="stock-link" href="https://finance.naver.com/item/main.naver?code=${r.code}" target="_blank">${r.name}</a><span class="stock-code">${r.code}</span>`,
        r.code, r.name, r.combined, r.trend, r.supply_score, r.momentum, r.sentiment,
        r.pos_pct, r.market, r.close, r.chg1d, r.chg60d, r.rsi, r.vol_ratio,
        mode==="trend"?r.trend_why:(r.swing_why||r.trend_why),
        r.stop, r.target,
        r.supply_detail ? r.supply_detail.n_days : 0, r.sentiment, r.news_count,
    ]);
    if (grid) grid.destroy();
    grid = new gridjs.Grid({
        columns: cols, data, sort: true, pagination: { limit: 25 }, search: true, resizable: true,
        language: { search: { placeholder: "검색 (한글/영어/숫자)..." } },
        rowAttributes: row => row.cells[3].data>=TH ? { class: "row-priority" } : {},
    }).render(document.getElementById("grid"));
    renderNewsDetail(rows);
}
function renderNewsDetail(rows) {
    const box = document.getElementById("newsDetail");
    const tops = rows.filter(r=>r.news_count>0 && r.combined>=60).slice(0,5);
    if (!tops.length) { box.innerHTML=""; return; }
    let html = `<details class="news-detail" open><summary>📰 상위 후보 뉴스 헤드라인 (참고용)</summary>`;
    for (const r of tops) {
        html += `<div class="news-card"><h4>${r.name} (${r.code}) · 감성 ${r.sentiment>=0?'+':''}${r.sentiment} / ${r.news_count}건</h4><ul>`;
        for (const h of (r.headlines || [])) {
            const t = (h.title||"").replace(/</g,"&lt;");
            const link = h.url ? `<a href="${h.url}" target="_blank">${t}</a>` : t;
            html += `<li>${h.tag||""} ${link}</li>`;
        }
        html += `</ul></div>`;
    }
    html += `</details>`;
    box.innerHTML = html;
}
document.querySelectorAll(".tab").forEach(t=>t.addEventListener("click",e=>{
    document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));
    e.target.classList.add("on");
    render(e.target.dataset.mode);
}));
render("trend");
</script></body></html>
"""


# =============================================================================
# 10) 메인 파이프라인
# =============================================================================

def main():
    print("🚀 한국주식 매매 신호 v3\n")
    print(f"   가중치: 기술 {WEIGHTS['technical']*100:.0f}% + 수급 {WEIGHTS['supply']*100:.0f}% + "
          f"모멘텀 {WEIGHTS['momentum']*100:.0f}% (감성은 참고용)\n")

    print("📈 시장 레짐 확인...")
    regime = check_regime()
    if regime["kospi"] is not None:
        sign = "✅" if regime["ok"] else "⚠️"
        print(f"   KOSPI {regime['kospi']:,.1f} / 200MA {regime['ma']:,.1f} ({regime['diff_pct']:+.1f}%) {sign}\n")

    print("📋 종목 리스트...")
    universe = get_universe()
    print(f"   {len(universe)}개 종목\n")

    print("📊 기술적 분석 (5~10분)...")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(analyze_one, code, name, market): code
                for code, name, market in universe[["code","name","market"]].itertuples(index=False, name=None)}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            if r: results.append(r)
            done += 1
            if done % 50 == 0: print(f"   {done}/{len(universe)}")
    print(f"✓ 기술 분석 {len(results)}개\n")

    candidates = sorted(results, key=lambda x: -x["trend"])[:NEWS_TOP_N]

    print(f"💰 수급 분석 (상위 {len(candidates)}개)...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        sup_futs = {ex.submit(fetch_supply_demand, r["code"]): r for r in candidates}
        done = 0
        for f in as_completed(sup_futs):
            r = sup_futs[f]
            try:
                sd = f.result()
                if sd["ok"]:
                    r["supply_score"] = sd["smart_score"]
                    r["supply_detail"] = sd
            except Exception: pass
            done += 1
            if done % 10 == 0: print(f"   {done}/{len(candidates)}")
    print()

    if ENABLE_NEWS:
        print(f"📰 뉴스 감성 분석 (참고용, {len(candidates)}개)...")
        with ThreadPoolExecutor(max_workers=NEWS_WORKERS) as ex:
            news_futs = {ex.submit(fetch_news_sentiment, r["code"], r["market"]): r for r in candidates}
            done = 0
            for f in as_completed(news_futs):
                r = news_futs[f]
                try:
                    s = f.result()
                    r["sentiment"] = s["score"]; r["news_count"] = s["n"]; r["headlines"] = s["headlines"]
                except Exception: pass
                done += 1
                if done % 10 == 0: print(f"   {done}/{len(candidates)}")
        print()

    print("🎯 종합 점수 계산...")
    for r in results:
        tech = r["trend"]
        supply = r.get("supply_score") if r.get("supply_score") is not None else 50
        mom = r["momentum"]
        r["combined"] = round(
            tech * WEIGHTS["technical"] + supply * WEIGHTS["supply"] + mom * WEIGHTS["momentum"], 1
        )

    top_picks = sorted(
        [r for r in results if r["trend"] >= 60],
        key=lambda x: (-x["combined"], -x["trend"])
    )[:TARGET_POSITIONS]

    today = dt.date.today().strftime("%Y-%m-%d")
    html = (HTML
            .replace("__DATE__", today)
            .replace("__DATA__", json.dumps(results, ensure_ascii=False, default=str))
            .replace("__REGIME__", json.dumps(regime, ensure_ascii=False))
            .replace("__TARGET_POSITIONS__", str(TARGET_POSITIONS)))

    primary_path = Path.cwd() / "dashboard.html"
    primary_path.write_text(html, encoding="utf-8")
    print(f"\n✅ 대시보드 → {primary_path.resolve()}")
    if OUTPUT_DIR.resolve() != Path.cwd().resolve():
        (OUTPUT_DIR / "dashboard.html").write_text(html, encoding="utf-8")
    sys.stdout.flush()

    # GitHub Issue 마크다운
    pages_url = os.environ.get("PAGES_URL", "")
    md = [f"# 📊 매매 신호 — {today}\n"]
    if regime["kospi"] is not None:
        emoji = "🟢" if regime["ok"] else "🔴"
        label = "강세" if regime["ok"] else "약세"
        md.append(f"### {emoji} 레짐: {label} (KOSPI {regime['kospi']:,.0f} / 200MA {regime['ma']:,.0f}, {regime['diff_pct']:+.1f}%)\n")
        if not regime["ok"]:
            md.append("> ⚠️ **신규 진입 자제. 손절선 관리 우선.**\n")
    md.append(f"**가중치**: 기술 40% + 수급 30% + 모멘텀 30% (감성은 참고용)\n")

    md.append(f"\n## 🏆 매수 우선순위 TOP {TARGET_POSITIONS}\n")
    if top_picks:
        md.append("| # | 종합 | 기술 | 수급 | 모멘 | 종목 | 비중 | 현재가 | 손절 | 목표 | 사유 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(top_picks, 1):
            link = f"[{r['name']}](https://finance.naver.com/item/main.naver?code={r['code']})"
            sup = f"{r.get('supply_score') or 50:.0f}"
            md.append(f"| {i} | **{r['combined']:.0f}** | {r['trend']} | {sup} | {r['momentum']} | "
                      f"{link} | {r['pos_pct']:.1f}% | ₩{r['close']:,.0f} | "
                      f"₩{r['stop'] or 0:,.0f} | ₩{r['target'] or 0:,.0f} | {r['trend_why']} |")
    else:
        md.append("_조건 충족 후보 없음. 오늘은 관망._\n")

    with_news = [r for r in top_picks if r.get("news_count", 0) > 0][:5]
    if with_news:
        md.append("\n## 📰 상위 후보 뉴스 (참고용 — 점수 미반영)\n")
        for r in with_news:
            md.append(f"\n**{r['name']} ({r['code']})** · 감성 {r['sentiment']:+.0f}")
            for h in r.get("headlines", []):
                t = (h.get("title","") or "").replace("|","\\|")
                url = h.get("url","")
                md.append(f"- {h.get('tag','·')} {('['+t+']('+url+')') if url else t}")

    md.append("\n---\n")
    md.append(f"⚠️ 알고리즘 신호 — 매매 책임은 본인. 권장 비중 {MIN_POSITION_PCT}~{MAX_POSITION_PCT}% (ATR 기반).")
    md.append(f"\n💡 운용: 목표 {TARGET_POSITIONS}종목 분산, 같은 종목 30일 쿨다운, 손절 자동 집행.")
    if pages_url:
        md.append(f"\n📊 [대시보드]({pages_url})")
    Path.cwd().joinpath("top.md").write_text("\n".join(md), encoding="utf-8")
    print("📝 top.md 작성")
    sys.stdout.flush()

    print(f"\n🏆 매수 우선순위 TOP {min(10, len(top_picks))}")
    for i, r in enumerate(top_picks[:10], 1):
        sup = f"{r.get('supply_score') or 50:3.0f}"
        print(f"  {i:2d}. [종합 {r['combined']:5.1f}] T:{r['trend']:3d} S:{sup} M:{r['momentum']:3d} | "
              f"{r['name']:<10s}({r['code']}) ₩{r['close']:>9,.0f} | 비중 {r['pos_pct']:.1f}% | {r['trend_why']}")
    if regime["warn"]:
        print("\n⚠️ 시장 약세 — 신규 진입 자제!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(1)
