"""
한국주식 매매 신호 대시보드 (Robinhood-style + News Sentiment)
==============================================================

KOSPI200 + KOSDAQ150 전체 종목을 스캔하여 장기 추세추종 신호를 생성하고,
야후파이낸스 뉴스 감성을 결합한 종합 점수로 매수 우선순위를 매깁니다.

종합 점수 = 0.7 × 기술적 추세 + 0.3 × 뉴스 감성

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

# 백테스트로 검증된 운용 설정
SCORE_THRESHOLD = 80
ENABLE_SWING = False
REGIME_FILTER = True
REGIME_INDEX = "KS11"
REGIME_MA = 200

# 야후파이낸스 뉴스 감성 분석
ENABLE_NEWS_SENTIMENT = True
NEWS_TOP_N = 30               # 기술적 점수 상위 N개만 뉴스 분석 (속도)
NEWS_WEIGHT = 0.30            # 종합 = 0.7×기술 + 0.3×뉴스
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


# ---------------------------------------------------------------------------
# 0) 시장 레짐 필터
# ---------------------------------------------------------------------------

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
        print(f"  ⚠️ 레짐 데이터 조회 실패 ({e}). 필터 비활성 처리.")
        return {"ok": True, "kospi": None, "ma": None, "diff_pct": 0.0, "warn": False}


# ---------------------------------------------------------------------------
# 0-1) 야후파이낸스 뉴스 감성 분석
# ---------------------------------------------------------------------------

def fetch_news_sentiment(code: str, market: str) -> dict:
    """야후파이낸스 헤드라인 감성 점수 (-100 ~ +100)."""
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


# ---------------------------------------------------------------------------
# 1) 종목 유니버스
# ---------------------------------------------------------------------------

def get_universe() -> pd.DataFrame:
    import FinanceDataReader as fdr
    listing = fdr.StockListing("KRX")
    cap_col = next((c for c in ("Marcap", "MarketCap", "marcap") if c in listing.columns), None)
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
# 2) 기술적 지표
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

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_up"] = bb_mid + 2 * bb_std
    df["BB_lo"] = bb_mid - 2 * bb_std
    df["BB_pct"] = (df["Close"] - df["BB_lo"]) / (df["BB_up"] - df["BB_lo"]).replace(0, np.nan)

    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / df["Vol_MA20"].replace(0, np.nan)

    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift()).abs()
    l_c = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["Pct5"] = (df["Close"] / df["Close"].shift(5) - 1) * 100
    df["Pct20"] = (df["Close"] / df["Close"].shift(20) - 1) * 100
    df["Pct60"] = (df["Close"] / df["Close"].shift(60) - 1) * 100

    df["RSI_prev"] = df["RSI"].shift(1)
    df["Hist_prev"] = df["MACD_hist"].shift(1)

    # 강세 RSI 다이버전스
    df["price_min_recent"] = df["Close"].rolling(10).min()
    df["price_min_prev"] = df["Close"].rolling(20).min().shift(10)
    df["rsi_min_recent"] = df["RSI"].rolling(10).min()
    df["rsi_min_prev"] = df["RSI"].rolling(20).min().shift(10)
    df["bullish_div"] = (
        (df["price_min_recent"] < df["price_min_prev"]) &
        (df["rsi_min_recent"] > df["rsi_min_prev"]) &
        (df["RSI"] < 60)
    )

    # 거래량 동반 횡보 돌파
    hi20 = df["High"].rolling(20).max()
    lo20 = df["Low"].rolling(20).min()
    df["range_20"] = (hi20 - lo20) / df["Close"]
    df["price_pos_20"] = (df["Close"] - lo20) / (hi20 - lo20).replace(0, np.nan)
    df["vol_5"] = df["Volume"].rolling(5).mean()
    df["vol_25"] = df["Volume"].rolling(25).mean()
    df["consolidation_breakout"] = (
        (df["range_20"] < 0.18) &
        (df["price_pos_20"] > 0.65) &
        (df["vol_5"] > df["vol_25"].replace(0, np.nan) * 1.2)
    )

    return df


# ---------------------------------------------------------------------------
# 3) 점수
# ---------------------------------------------------------------------------

def score_swing(r: pd.Series):
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


def score_trend(r: pd.Series):
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
        score += 15; reasons.append("RSI다이버전스↑")
    cb = r.get("consolidation_breakout", False)
    if cb is True or (cb is not None and not pd.isna(cb) and bool(cb)):
        score += 12; reasons.append("횡보+거래량↑")
    return max(0, min(100, score)), reasons


# ---------------------------------------------------------------------------
# 4) 종목별 분석
# ---------------------------------------------------------------------------

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
        s_score, s_reasons = score_swing(last)
        t_score, t_reasons = score_trend(last)
        atr = last["ATR"] if not pd.isna(last["ATR"]) else None
        close = float(last["Close"])
        return {
            "code": code, "name": name, "market": market, "close": close,
            "chg1d": float((close / df["Close"].iloc[-2] - 1) * 100),
            "chg20d": float(last["Pct20"]) if not pd.isna(last["Pct20"]) else 0.0,
            "chg60d": float(last["Pct60"]) if not pd.isna(last["Pct60"]) else 0.0,
            "rsi": float(last["RSI"]) if not pd.isna(last["RSI"]) else None,
            "vol_ratio": float(last["Vol_ratio"]) if not pd.isna(last["Vol_ratio"]) else None,
            "ma20": float(last["MA20"]) if not pd.isna(last["MA20"]) else None,
            "ma60": float(last["MA60"]) if not pd.isna(last["MA60"]) else None,
            "bb_pct": float(last["BB_pct"]) if not pd.isna(last["BB_pct"]) else None,
            "macd_hist": float(last["MACD_hist"]) if not pd.isna(last["MACD_hist"]) else None,
            "atr": float(atr) if atr is not None else None,
            "stop": float(close - 2 * atr) if atr else None,
            "target": float(close + 3 * atr) if atr else None,
            "swing": s_score, "swing_why": ", ".join(s_reasons),
            "trend": t_score, "trend_why": ", ".join(t_reasons),
            "sentiment": None, "news_count": 0, "headlines": [],
            "combined": t_score,  # 기본은 추세 점수, 뉴스 분석 후 갱신
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5) HTML 대시보드 (Robinhood-style + News Sentiment)
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Signals Dashboard — Robinhood Style</title>
    <link href="https://cdn.jsdelivr.net/npm/gridjs/dist/theme/mermaid.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/gridjs/dist/gridjs.umd.js"></script>
    <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
        :root {
            --bg-body: #060d17;
            --bg-card: #121a24;
            --primary: #00ff8a;
            --up-color: #00ff8a;
            --down-color: #ff3b3b;
            --text-main: #ffffff;
            --text-sub: #8a94a6;
            --border: #232d39;
            --hover-row: #1b2633;
        }
        body {
            font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
            background-color: var(--bg-body); color: var(--text-main);
            margin: 0; padding: 20px; line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }
        .container-max { max-width: 1600px; margin: 0 auto; }
        .header-section { margin-bottom: 32px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }
        h1 { font-size: 26px; font-weight: 800; margin: 0; letter-spacing: -0.5px; color: #ffffff; }
        .sub { color: var(--text-sub); font-size: 14px; margin-top: 6px; }
        .regime {
            padding: 20px; border-radius: 16px; margin: 20px 0;
            font-size: 15px; background: var(--bg-card); border: 1px solid var(--border);
            display: flex; align-items: center; gap: 15px;
        }
        .regime.bull { border-left: 6px solid var(--up-color); color: var(--up-color); }
        .regime.bear { border-left: 6px solid var(--down-color); color: var(--down-color); }
        .regime-icon { font-size: 24px; }
        .regime-info b { font-size: 18px; display: block; margin-bottom: 2px; }
        .tabs { display: flex; gap: 12px; margin: 24px 0; overflow-x: auto; scrollbar-width: none; }
        .tabs::-webkit-scrollbar { display: none; }
        .tab {
            white-space: nowrap; padding: 12px 24px; background: transparent;
            border-radius: 100px; cursor: pointer; font-weight: 700; font-size: 14px;
            border: 1.5px solid var(--border); color: var(--text-sub); transition: all 0.2s ease;
        }
        .tab.on { background: var(--primary); color: #000; border-color: var(--primary); box-shadow: 0 4px 15px rgba(0, 255, 138, 0.3); }
        .summary-bar { margin-bottom: 16px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
        .candidates-bar {
            display: inline-flex; align-items: center; padding: 10px 18px;
            background: rgba(0, 255, 138, 0.1); color: var(--primary);
            border-radius: 12px; font-size: 14px; font-weight: 700; border: 1px solid rgba(0, 255, 138, 0.2);
        }
        .info-pill {
            display: inline-flex; align-items: center; padding: 8px 14px;
            background: rgba(138, 148, 166, 0.08); color: var(--text-sub);
            border-radius: 10px; font-size: 13px; border: 1px solid var(--border);
        }
        .table-wrap {
            background: var(--bg-card); border-radius: 20px; padding: 12px;
            border: 1px solid var(--border); overflow-x: auto;
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        }
        .gridjs-table { table-layout: auto !important; width: 100% !important; min-width: 1500px; background: transparent !important; }
        .gridjs-th {
            background-color: #0b141d !important; color: var(--text-sub) !important;
            padding: 16px 10px !important; font-size: 12px !important; font-weight: 700 !important;
            text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 2px solid var(--border) !important;
        }
        .gridjs-td {
            padding: 16px 10px !important;
            border-bottom: 1px solid var(--border) !important;
            font-size: 14px !important;
            color: #ffffff !important;
            background-color: transparent !important;
            vertical-align: middle;
        }
        .gridjs-tr { background-color: var(--bg-card) !important; }
        .gridjs-tr:hover .gridjs-td { background-color: var(--hover-row) !important; color: var(--primary) !important; }
        .row-priority td { background: rgba(0, 255, 138, 0.04) !important; }
        .pos { color: var(--up-color) !important; font-weight: 700; }
        .neg { color: var(--down-color) !important; font-weight: 700; }
        .neutral { color: var(--text-sub) !important; }

        .pill {
            display:inline-flex; align-items: center; justify-content: center;
            min-width: 44px; height: 26px; border-radius: 6px;
            font-size: 12px; font-weight: 800; color: #fff !important;
        }
        .score-hi { background: var(--up-color); color: #000 !important; }
        .score-md { background: #f59e0b; color: #000 !important; }
        .score-lo { background: #334155; color: #ffffff !important; }
        .sentiment-pill {
            display: inline-flex; align-items: center; justify-content: center;
            min-width: 56px; height: 26px; border-radius: 6px;
            font-size: 12px; font-weight: 800;
        }
        .sent-pos { background: rgba(0, 255, 138, 0.15); color: var(--up-color) !important; }
        .sent-neg { background: rgba(255, 59, 59, 0.15); color: var(--down-color) !important; }
        .sent-neu { background: #2d3748; color: var(--text-sub) !important; }
        .sent-na { color: var(--text-sub) !important; font-size: 12px; }
        .news-count { color: var(--text-sub); font-size: 12px; }
        .stock-link { text-decoration: none; color: var(--primary); font-weight: 800; font-size: 15px; }
        .stock-code { color: var(--text-sub); font-size: 11px; margin-left: 6px; }
        .reason-cell { white-space: normal !important; min-width: 250px; font-size: 13px; color: #cbd5e1 !important; line-height: 1.4; }
        .gridjs-search-input {
            background: #1b2633 !important; color: #ffffff !important;
            border: 1px solid var(--border) !important; padding: 10px 15px !important;
            border-radius: 10px !important; width: 280px !important;
        }
        .gridjs-pagination .gridjs-summary { color: var(--text-sub) !important; }
        .gridjs-pagination button { color: #ffffff !important; background: #1b2633 !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }
        .gridjs-pagination button.gridjs-currentPage { background: var(--primary) !important; color: #000 !important; }

        /* 헤드라인 디테일 */
        details.news-detail { margin-top: 14px; }
        details.news-detail summary {
            cursor: pointer; color: var(--text-sub); font-size: 13px;
            padding: 8px 14px; border-radius: 8px; background: #0b141d;
            border: 1px solid var(--border); display: inline-block; user-select: none;
        }
        details.news-detail[open] summary { color: var(--primary); }
        .news-card {
            margin-top: 12px; padding: 12px 16px; background: #0b141d;
            border-radius: 10px; border: 1px solid var(--border);
        }
        .news-card h4 { margin: 0 0 8px; font-size: 14px; color: var(--primary); }
        .news-card ul { margin: 0; padding-left: 18px; color: #cbd5e1; font-size: 13px; line-height: 1.6; }
        .news-card ul li { margin: 4px 0; }
        .news-card ul a { color: #cbd5e1; text-decoration: none; }
        .news-card ul a:hover { color: var(--primary); text-decoration: underline; }
    </style>
</head>
<body>
<div class="container-max">
    <div class="header-section">
        <h1>Trading Dashboard</h1>
        <div class="sub">Updated: <b>__DATE__</b> · Assets: <b id="cnt"></b> · Data via KRX/Naver Finance + Yahoo News</div>
        <div id="regimeBox"></div>
    </div>
    <div class="tabs">
        <div class="tab on" data-mode="trend">Trend Following</div>
        <div class="tab" data-mode="swing">Swing Trading</div>
        <div class="tab" data-mode="all">All Assets</div>
    </div>
    <div class="summary-bar" id="candidatesBar"></div>
    <div class="table-wrap">
        <div id="grid"></div>
    </div>
    <div id="newsDetail"></div>
</div>
<script>
    const DATA = __DATA__;
    const REGIME = __REGIME__;
    document.getElementById("cnt").textContent = DATA.length;

    (function() {
        if (!REGIME || REGIME.kospi == null) return;
        const box = document.getElementById("regimeBox");
        const isBull = REGIME.ok;
        box.innerHTML = `
            <div class="regime ${isBull ? 'bull' : 'bear'}">
                <div class="regime-icon">${isBull ? '💹' : '⚠️'}</div>
                <div class="regime-info">
                    <b>Market Regime: ${isBull ? 'Bullish' : 'Bearish'}</b>
                    <span>KOSPI ${REGIME.kospi.toLocaleString()} (200MA Deviation: ${REGIME.diff_pct>=0?'+':''}${REGIME.diff_pct.toFixed(1)}%) — ${isBull ? 'Favorable for entries' : 'Caution advised'}</span>
                </div>
            </div>`;
    })();

    const num = (v, d=0) => v == null ? "-" : Number(v).toLocaleString("ko-KR", {minimumFractionDigits: d, maximumFractionDigits: d});
    const pct = (v) => {
        if (v == null) return "-";
        return `<span class="${v >= 0 ? 'pos' : 'neg'}">${v >= 0 ? '+' : ''}${v.toFixed(1)}%</span>`;
    };
    const scoreCell = (v) => {
        const cls = v >= 80 ? "score-hi" : v >= 60 ? "score-md" : "score-lo";
        return `<span class="pill ${cls}">${v}</span>`;
    };
    const sentimentCell = (v, n) => {
        if (v == null || n === 0) return `<span class="sent-na">no news</span>`;
        const cls = v >= 20 ? "sent-pos" : v <= -20 ? "sent-neg" : "sent-neu";
        const sign = v > 0 ? "+" : "";
        return `<span class="sentiment-pill ${cls}">${sign}${v.toFixed(0)}</span> <span class="news-count">(${n})</span>`;
    };

    let grid = null;

    function render(mode) {
        let rows = [...DATA];
        if (mode === "swing") {
            rows = rows.filter(r => r.swing > 0).sort((a,b) => b.swing - a.swing);
        } else if (mode === "trend") {
            // 종합 점수(기술+감성)로 정렬, 동점이면 추세 점수로
            rows = rows.filter(r => r.trend > 0).sort((a,b) => (b.combined - a.combined) || (b.trend - a.trend));
        } else {
            rows.sort((a,b) => (b.combined - a.combined) || (b.trend - a.trend));
        }

        const SCORE_TH = 80;
        const candCount = rows.filter(r => (mode === "swing" ? r.swing : r.trend) >= SCORE_TH).length;
        const analyzed = rows.filter(r => r.news_count > 0).length;
        document.getElementById("candidatesBar").innerHTML = `
            ${candCount > 0 ? `<div class="candidates-bar">✨ Priority Signals: ${candCount} stocks (Score 80+)</div>` : ""}
            <div class="info-pill">📰 News-analyzed: ${analyzed} stocks · Combined = 70% Technical + 30% Narrative</div>
        `;

        const cols = [
            { name: "종목", width: "180px", formatter: cell => gridjs.html(cell), attributes: { 'title': '종목명/코드' } },
            { name: "코드", hidden: true },
            { name: "이름", hidden: true },
            { name: "종합", width: "75px", formatter: v => gridjs.html(scoreCell(Math.round(v))),
              attributes: { 'title': '기술 70% + 뉴스 감성 30%' } },
            { name: "추세", width: "70px", formatter: v => gridjs.html(scoreCell(v)), attributes: { 'title': '기술적 추세 점수' } },
            { name: "감성", width: "100px", formatter: (_, r) => gridjs.html(sentimentCell(r.cells[15].data, r.cells[16].data)),
              attributes: { 'title': '야후파이낸스 헤드라인 감성 (-100~+100)' }, sort: { compare: (a, b) => a - b } },
            { name: "시장", width: "95px" },
            { name: "종가", width: "95px", formatter: v => num(v) },
            { name: "1일", width: "80px", formatter: v => gridjs.html(pct(v)) },
            { name: "20일", width: "80px", formatter: v => gridjs.html(pct(v)) },
            { name: "60일", width: "80px", formatter: v => gridjs.html(pct(v)) },
            { name: "RSI", width: "65px", formatter: v => v==null ? "-" : v.toFixed(0) },
            { name: "거래량x", width: "85px", formatter: v => v==null ? "-" : v.toFixed(2) },
            { name: "사유", width: "260px", formatter: v => gridjs.html(`<div class="reason-cell">${v || ""}</div>`) },
            { name: "손절가", width: "95px", formatter: v => num(v) },
            { name: "목표가", width: "95px", formatter: v => num(v) },
            { name: "_sentiment", hidden: true },
            { name: "_news_count", hidden: true },
            { name: "스윙", width: "70px", hidden: true, formatter: v => gridjs.html(scoreCell(v)) },
        ];

        const data = rows.map(r => [
            `<a class="stock-link" href="https://finance.naver.com/item/main.naver?code=${r.code}" target="_blank">${r.name}</a><span class="stock-code">${r.code}</span>`,
            r.code, r.name,
            r.combined != null ? r.combined : r.trend,
            r.trend,
            r.sentiment,  // formatter용 placeholder
            r.market, r.close, r.chg1d, r.chg20d, r.chg60d,
            r.rsi, r.vol_ratio,
            mode === "trend" ? r.trend_why : (r.swing_why || r.trend_why),
            r.stop, r.target,
            r.sentiment, r.news_count, r.swing,
        ]);

        if (grid) grid.destroy();
        grid = new gridjs.Grid({
            columns: cols, data, sort: true, pagination: { limit: 25 }, search: true, resizable: true,
            language: { search: { placeholder: "Search stocks (한글/영어/숫자)..." } },
            rowAttributes: (row) => {
                const score = mode === "swing" ? row.cells[18].data : row.cells[4].data;
                return (score >= SCORE_TH) ? { class: "row-priority" } : {};
            },
        }).render(document.getElementById("grid"));

        // 헤드라인 상세 패널 — 상위 후보 5개
        renderNewsDetail(rows, mode);
    }

    function renderNewsDetail(rows, mode) {
        const box = document.getElementById("newsDetail");
        const tops = rows.filter(r => r.news_count > 0 && r.trend >= 60).slice(0, 5);
        if (!tops.length) { box.innerHTML = ""; return; }

        let html = `<details class="news-detail" open><summary>📰 상위 후보 뉴스 헤드라인 (${tops.length}개 종목)</summary>`;
        for (const r of tops) {
            html += `<div class="news-card"><h4>${r.name} (${r.code}) · 감성 ${r.sentiment>=0?'+':''}${r.sentiment} / ${r.news_count}건</h4><ul>`;
            for (const h of (r.headlines || [])) {
                const titleEsc = (h.title || "").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                const link = h.url ? `<a href="${h.url}" target="_blank">${titleEsc}</a>` : titleEsc;
                html += `<li>${h.tag || ""} ${link}</li>`;
            }
            html += `</ul></div>`;
        }
        html += `</details>`;
        box.innerHTML = html;
    }

    document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", (e) => {
        document.querySelectorAll(".tab").forEach(x => x.classList.remove("on"));
        e.target.classList.add("on");
        render(e.target.dataset.mode);
    }));

    render("trend");
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 6) 메인
# ---------------------------------------------------------------------------

def main():
    print("📈 시장 레짐 확인 중 (KOSPI vs 200MA)...")
    regime = check_regime()
    if regime["kospi"] is not None:
        sign = "위 ✅" if regime["ok"] else "아래 ⚠️"
        print(f"  KOSPI {regime['kospi']:,.1f} / 200MA {regime['ma']:,.1f} "
              f"({regime['diff_pct']:+.1f}%) — 200MA {sign}\n")
    if regime["warn"] and REGIME_FILTER:
        print("  ⚠️ 시장 약세 신호. 신규 진입 자제 권고.\n")

    print("📋 종목 리스트 조회 중...")
    universe = get_universe()
    print(f"  코스피200 + 코스닥150 = {len(universe)}개 종목\n")

    print("📊 시세 분석 중 (5~10분 소요)...")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(analyze_one, code, name, market): code
                for code, name, market in universe[["code", "name", "market"]]
                                          .itertuples(index=False, name=None)}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(universe)} 처리됨")

    print(f"\n✓ 기술적 분석 완료: {len(results)}개\n")

    # 뉴스 감성 분석 — 추세 점수 상위 N개만
    if ENABLE_NEWS_SENTIMENT and results:
        candidates = sorted(results, key=lambda x: -x["trend"])[:NEWS_TOP_N]
        print(f"📰 야후파이낸스 뉴스 감성 분석 중 (상위 {len(candidates)}개 종목)...")
        with ThreadPoolExecutor(max_workers=NEWS_WORKERS) as ex:
            futs = {ex.submit(fetch_news_sentiment, r["code"], r["market"]): r for r in candidates}
            done = 0
            for f in as_completed(futs):
                r = futs[f]
                try:
                    s = f.result()
                    r["sentiment"] = s["score"]
                    r["news_count"] = s["n"]
                    r["headlines"] = s["headlines"]
                except Exception:
                    pass
                done += 1
                if done % 5 == 0:
                    print(f"  {done}/{len(candidates)}")
        analyzed = sum(1 for r in results if r.get("news_count", 0) > 0)
        print(f"✓ 뉴스 분석 완료: {analyzed}개 종목에서 헤드라인 발견\n")

    # 종합 점수 계산 (기술 0.7 + 뉴스 0.3)
    for r in results:
        tech = r["trend"]
        sent = r.get("sentiment")
        if sent is not None and r.get("news_count", 0) > 0:
            sent_norm = (sent + 100) / 2  # -100~+100 → 0~100
            r["combined"] = round((1 - NEWS_WEIGHT) * tech + NEWS_WEIGHT * sent_norm, 1)
        else:
            r["combined"] = float(tech)

    # 종합 점수 기준 TOP 10
    top10 = sorted([r for r in results if r["trend"] >= SCORE_THRESHOLD],
                   key=lambda x: (-x["combined"], -x["trend"]))[:10]
    swing_top = sorted([r for r in results if r["swing"] >= SCORE_THRESHOLD],
                       key=lambda x: -x["swing"])[:10]

    today = dt.date.today().strftime("%Y-%m-%d")
    html = (HTML
            .replace("__DATE__", today)
            .replace("__DATA__", json.dumps(results, ensure_ascii=False))
            .replace("__REGIME__", json.dumps(regime, ensure_ascii=False)))

    primary_path = Path.cwd() / "dashboard.html"
    primary_path.write_text(html, encoding="utf-8")
    print(f"✅ 대시보드 생성 완료 → {primary_path.resolve()}")
    if OUTPUT_DIR.resolve() != Path.cwd().resolve():
        (OUTPUT_DIR / "dashboard.html").write_text(html, encoding="utf-8")
    sys.stdout.flush()

    # GitHub Issue 용 마크다운
    pages_url = os.environ.get("PAGES_URL", "")
    md = [f"# 📊 한국주식 매매 신호 — {today}\n"]

    if regime["kospi"] is not None:
        if regime["ok"]:
            md.append(f"### 🟢 시장 레짐: 강세 (KOSPI {regime['kospi']:,.1f} / "
                      f"200MA {regime['ma']:,.1f}, {regime['diff_pct']:+.1f}%)\n")
            md.append("→ 추세 신호 진입 가능 구간\n")
        else:
            md.append(f"### 🔴 시장 레짐: **약세** (KOSPI {regime['kospi']:,.1f} / "
                      f"200MA {regime['ma']:,.1f}, {regime['diff_pct']:+.1f}%)\n")
            md.append("> ⚠️ **신규 진입 자제 권고.**\n")

    md.append(f"분석 종목 수: **{len(results)}개** · 종합 점수 = 70% 기술 + 30% 뉴스 감성\n")

    md.append("\n## 🏆 종합 매수 우선순위 TOP 10")
    md.append("*기술적 추세 + 야후파이낸스 뉴스 감성 결합*\n")
    if top10:
        md.append("| 순위 | 종합 | 추세 | 감성 | 종목 | 현재가 | 손절 | 목표 | 사유 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(top10, 1):
            link = f"[{r['name']}](https://finance.naver.com/item/main.naver?code={r['code']})"
            sent = r.get("sentiment")
            sent_str = f"{sent:+.0f}" if sent is not None and r.get("news_count", 0) > 0 else "-"
            md.append(f"| {i} | **{r['combined']:.0f}** | {r['trend']} | {sent_str} | "
                      f"{link} ({r['code']}) | ₩{r['close']:,.0f} | "
                      f"₩{r['stop']:,.0f} | ₩{r['target']:,.0f} | {r['trend_why']} |")
    else:
        md.append(f"_점수 {SCORE_THRESHOLD} 이상 후보 없음. 오늘은 진입할 종목이 없습니다._\n")

    # 상위 후보 뉴스 헤드라인
    with_news = [r for r in top10 if r.get("news_count", 0) > 0][:5]
    if with_news:
        md.append("\n## 📰 상위 후보 뉴스 헤드라인")
        for r in with_news:
            sent = r.get("sentiment", 0)
            md.append(f"\n**{r['name']} ({r['code']})** · 감성 {sent:+.0f} / {r['news_count']}건")
            for h in r.get("headlines", []):
                title_clean = (h.get("title", "") or "").replace("|", "\\|")
                tag = h.get("tag", "·")
                url = h.get("url", "")
                if url:
                    md.append(f"- {tag} [{title_clean}]({url})")
                else:
                    md.append(f"- {tag} {title_clean}")

    if ENABLE_SWING and swing_top:
        md.append("\n## 🎯 단기 스윙 TOP 10 (참고용)\n")
        md.append("| # | 점수 | 종목 | 현재가 | 손절 | 목표 | 사유 |")
        md.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(swing_top, 1):
            link = f"[{r['name']}](https://finance.naver.com/item/main.naver?code={r['code']})"
            md.append(f"| {i} | **{r['swing']}** | {link} ({r['code']}) | "
                      f"₩{r['close']:,.0f} | ₩{r['stop']:,.0f} | ₩{r['target']:,.0f} | {r['swing_why']} |")

    md.append("\n---\n")
    md.append("⚠️ 알고리즘 신호일 뿐, 매매 결정과 책임은 본인에게 있습니다.")
    md.append("\n💡 운용: 동시 포지션 10종목 균등 분할, 같은 종목 30일 재진입 쿨다운.")
    if pages_url:
        md.append(f"\n📊 [전체 대시보드 보기]({pages_url})")
    md_path = Path.cwd() / "top.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"📝 마크다운 요약 → {md_path.resolve()}")
    sys.stdout.flush()

    print("\n🏆 종합 매수 우선순위 TOP 10 (점수 80+)")
    if top10:
        for i, r in enumerate(top10, 1):
            sent = r.get("sentiment")
            sent_str = f"감성{sent:+.0f}" if sent is not None and r.get("news_count", 0) > 0 else "감성-"
            print(f"  {i:2d}. [종합 {r['combined']:5.1f} | 추세 {r['trend']:3d} | {sent_str:>7s}] "
                  f"{r['name']:<10s} ({r['code']}) ₩{r['close']:>9,.0f} | {r['trend_why']}")
    else:
        print(f"  (점수 {SCORE_THRESHOLD} 이상 후보 없음 — 오늘은 관망)")

    if regime["warn"] and REGIME_FILTER:
        print("\n⚠️ 시장 약세 — 신규 매수보다는 기존 포지션 손절선 관리 우선!")
    print("\n👉 dashboard.html 을 브라우저로 열어 검색·정렬·전체 종목 확인.\n"
          "   매매는 본인이 직접 HTS/MTS에서 진행해 주세요.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(1)
