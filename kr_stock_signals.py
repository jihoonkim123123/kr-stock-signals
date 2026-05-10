"""
한국주식 매매 신호 대시보드 (미니멀 다크 모드 + 원본 데이터 풀 복구)
=========================================================
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

warnings.filterwarnings("ignore")

# [설정] 원본 로직 유지
LOOKBACK_DAYS = 260
WORKERS = 8
SCORE_THRESHOLD = 80
ENABLE_SWING = False
REGIME_FILTER = True
REGIME_INDEX = "KS11"
REGIME_MA = 200

# ---------------------------------------------------------------------------
# 0) 시장 레짐 필터 (원본 유지)
# ---------------------------------------------------------------------------
def check_regime() -> dict:
    import FinanceDataReader as fdr
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=REGIME_MA * 2 + 60)
        df = fdr.DataReader(REGIME_INDEX, start, end)
        if len(df) < REGIME_MA:
            return {"ok": True, "kospi": None, "ma": None, "diff_pct": 0.0}
        ma = float(df["Close"].rolling(REGIME_MA).mean().iloc[-1])
        last = float(df["Close"].iloc[-1])
        return {"ok": bool(last > ma), "kospi": last, "ma": ma, "diff_pct": float((last/ma-1)*100)}
    except Exception:
        return {"ok": True, "kospi": None, "ma": None, "diff_pct": 0.0}

# ---------------------------------------------------------------------------
# 1) 분석 로직 (원본 데이터 구조 100% 유지)
# ---------------------------------------------------------------------------
def get_universe() -> pd.DataFrame:
    import FinanceDataReader as fdr
    listing = fdr.StockListing("KRX")
    cap_col = next((c for c in ("Marcap", "MarketCap", "marcap") if c in listing.columns), None)
    listing = listing.dropna(subset=[cap_col, "Market", "Name"])
    listing = listing[~listing["Name"].str.contains("스팩|우$|우B|우C", regex=True, na=False)]
    kospi = listing[listing["Market"] == "KOSPI"].sort_values(cap_col, ascending=False).head(200)
    kosdaq = listing[listing["Market"] == "KOSDAQ"].sort_values(cap_col, ascending=False).head(150)
    rows = [{"code": r["Code"], "name": r["Name"], "market": "KOSPI200"} for _, r in kospi.iterrows()]
    rows += [{"code": r["Code"], "name": r["Name"], "market": "KOSDAQ150"} for _, r in kosdaq.iterrows()]
    return pd.DataFrame(rows)

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in (5, 20, 60, 120): df[f"MA{n}"] = df["Close"].rolling(n).mean()
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
    df["Pct20"] = (df["Close"] / df["Close"].shift(20) - 1) * 100
    df["Pct60"] = (df["Close"] / df["Close"].shift(60) - 1) * 100
    df["RSI_prev"] = df["RSI"].shift(1)
    return df

def score_trend(r: pd.Series):
    if any(pd.isna(r[c]) for c in ["MA5", "MA20", "MA60", "MA120", "RSI"]): return 0, []
    score, reasons = 0, []
    if r["MA5"] > r["MA20"] > r["MA60"] > r["MA120"]: score += 35; reasons.append("정배열")
    if r["Close"] > r["MA20"]: score += 12; reasons.append("MA20위")
    if r["MACD"] > r["MACD_sig"]: score += 18; reasons.append("MACD매수")
    if 50 <= r["RSI"] <= 70: score += 15; reasons.append(f"RSI {r['RSI']:.0f}")
    if not pd.isna(r["Pct60"]) and r["Pct60"] > 0: score += min(15, int(r["Pct60"] / 2)); reasons.append(f"상승추세")
    return max(0, min(100, score)), reasons

def analyze_one(code: str, name: str, market: str):
    import FinanceDataReader as fdr
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=LOOKBACK_DAYS)
        df = fdr.DataReader(code, start, end)
        if len(df) < 130: return None
        df = calc_indicators(df)
        last = df.iloc[-1]
        t_score, t_reasons = score_trend(last)
        atr = last["ATR"] if not pd.isna(last["ATR"]) else 0
        close = float(last["Close"])
        return {
            "code": code, "name": name, "market": market, "close": close,
            "chg1d": float((close / df["Close"].iloc[-2] - 1) * 100),
            "chg60d": float(last["Pct60"]) if not pd.isna(last["Pct60"]) else 0.0,
            "rsi": float(last["RSI"]) if not pd.isna(last["RSI"]) else 0.0,
            "trend": t_score, "trend_why": ", ".join(t_reasons),
            "stop": float(close - 2 * atr) if atr else None,
            "target": float(close + 3 * atr) if atr else None,
        }
    except Exception: return None

# ---------------------------------------------------------------------------
# 3) UI 대시보드 (최신 미니멀 다크 모드)
# ---------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>KR Stock Signal — __DATE__</title>
    <link href="https://cdn.jsdelivr.net/npm/gridjs/dist/theme/mermaid.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/gridjs/dist/gridjs.umd.js"></script>
    <style>
        :root {
            --bg: #0d1117; --card: #161b22; --text: #c9d1d9; --sub: #8b949e;
            --accent: #58a6ff; --border: #30363d; --pos: #39d353; --neg: #f85149;
            --hi-bg: rgba(56, 139, 253, 0.15);
        }
        body { background: var(--bg); color: var(--text); font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 2rem; line-height: 1.5; }
        .container { max-width: 1400px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }
        h1 { font-size: 1.5rem; margin: 0; font-weight: 600; letter-spacing: -0.5px; }
        .meta { font-size: 0.85rem; color: var(--sub); }
        .regime { display: inline-flex; align-items: center; padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.9rem; font-weight: 500; margin-bottom: 2rem; border: 1px solid var(--border); }
        .regime.bull { background: rgba(57, 211, 83, 0.1); color: var(--pos); border-color: rgba(57, 211, 83, 0.2); }
        .regime.bear { background: rgba(248, 81, 73, 0.1); color: var(--neg); border-color: rgba(248, 81, 73, 0.2); }
        
        /* GridJS Custom Dark Theme */
        .gridjs-container { padding: 0; }
        .gridjs-wrapper { border: 1px solid var(--border) !important; border-radius: 8px; background: var(--card) !important; }
        .gridjs-table { background: var(--card) !important; width: 100%; }
        .gridjs-thead th { background: #161b22 !important; color: var(--sub) !important; font-weight: 600; text-transform: uppercase; font-size: 11px; border-bottom: 1px solid var(--border) !important; }
        .gridjs-tbody td { background: transparent !important; color: var(--text) !important; border-bottom: 1px solid var(--border) !important; font-size: 13px; padding: 12px 16px !important; }
        .gridjs-tr:hover td { background: rgba(255,255,255,0.02) !important; }
        .gridjs-search-input { background: var(--bg) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 6px; }
        .gridjs-pagination .gridjs-pages button { background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
        
        .row-hi { background: var(--hi-bg) !important; }
        .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; background: var(--border); color: var(--sub); }
        .badge.hi { background: var(--accent); color: white; }
        .code { color: var(--sub); font-size: 11px; font-family: monospace; }
        a { color: inherit; text-decoration: none; }
        a:hover { color: var(--accent); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>KR Stock Quantitative Signals</h1>
                <div class="meta">Generated on __DATE__ • __COUNT__ assets scanned</div>
            </div>
            <div id="regimeBox"></div>
        </header>

        <div id="grid"></div>
    </div>

    <script>
        const DATA = __DATA__;
        const REGIME = __REGIME__;
        
        if (REGIME.kospi) {
            const el = document.getElementById("regimeBox");
            el.className = `regime ${REGIME.ok ? 'bull' : 'bear'}`;
            el.innerHTML = `${REGIME.ok ? '● Market Bullish' : '○ Market Bearish'} (KOSPI ${Math.round(REGIME.kospi)})`;
        }

        const fmt = (v) => v ? Math.round(v).toLocaleString() : '-';
        const pct = (v) => `<span style="color:${v>=0?'var(--pos)':'var(--neg)'}">${v>=0?'+':''}${v.toFixed(1)}%</span>`;

        new gridjs.Grid({
            columns: [
                { name: "Ticker", width: "200px", formatter: (cell, row) => gridjs.html(`<div><a href="https://finance.naver.com/item/main.naver?code=${row.cells[1].data}" target="_blank"><b>${cell}</b></a><br><span class="code">${row.cells[1].data}</span></div>`) },
                { name: "Code", hidden: true },
                { name: "Market", width: "110px", formatter: v => gridjs.html(`<span class="badge">${v}</span>`) },
                { name: "Price", width: "100px", formatter: v => fmt(v) },
                { name: "1D", width: "80px", formatter: v => gridjs.html(pct(v)) },
                { name: "Trend", width: "80px", formatter: v => gridjs.html(`<span class="badge ${v>=80?'hi':''}">${v}</span>`) },
                { name: "Signals", width: "250px" },
                { name: "Stop Loss", width: "110px", formatter: v => fmt(v) },
                { name: "Target", width: "110px", formatter: v => fmt(v) }
            ],
            data: DATA.sort((a,b) => b.trend - a.trend).map(r => [r.name, r.code, r.market, r.close, r.chg1d, r.trend, r.trend_why, r.stop, r.target]),
            search: true, pagination: { limit: 20 }, sort: true,
            style: { table: { 'white-space': 'nowrap' } }
        }).render(document.getElementById("grid"));
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# 4) 메인 실행
# ---------------------------------------------------------------------------
def main():
    print("📈 Analyzing market signals...")
    regime = check_regime()
    universe = get_universe()
    
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(analyze_one, r.code, r.name, r.market): r.code for _, r in universe.iterrows()}
        for f in as_completed(futs):
            res = f.result()
            if res: results.append(res)

    today = dt.date.today().strftime("%Y-%m-%d")
    final_html = (HTML
                  .replace("__DATE__", today)
                  .replace("__COUNT__", str(len(results)))
                  .replace("__DATA__", json.dumps(results, ensure_ascii=False))
                  .replace("__REGIME__", json.dumps(regime, ensure_ascii=False)))

    # 파일 저장 (GitHub Actions 경로 대응)
    Path("dashboard.html").write_text(final_html, encoding="utf-8")
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(final_html, encoding="utf-8")

    # 마크다운 요약 (원본 데이터 필드 모두 유지)
    trend_top = sorted([r for r in results if r["trend"] >= SCORE_THRESHOLD], key=lambda x: -x["trend"])[:10]
    md = [f"# 📊 Signal Summary ({today})\n", "| 종목(코드) | 점수 | 종가 | 손절 | 목표 | 신호 |\n|---|---|---|---|---|---|"]
    for r in trend_top:
        md.append(f"| {r['name']}({r['code']}) | **{r['trend']}** | {r['close']:,.0f} | {r['stop']:,.0f} | {r['target']:,.0f} | {r['trend_why']} |")
    Path("top.md").write_text("\n".join(md), encoding="utf-8")
    print(f"✅ Success: dashboard.html, docs/index.html, top.md generated.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
