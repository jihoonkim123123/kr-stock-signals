"""
한국주식 매매 신호 대시보드 (원본 로직 + UI 최신화 버전)
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

# 원본 설정 유지
LOOKBACK_DAYS = 260
WORKERS = 8
SCORE_THRESHOLD = 80          # 점수 80 이상만 진입 후보
ENABLE_SWING = False          # 단기 스윙은 알파 약해서 끔
REGIME_FILTER = True          # KOSPI 가 200MA 위일 때만 신규 진입 권장
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
            return {"ok": True, "kospi": None, "ma": None, "diff_pct": 0.0, "warn": False}
        ma = float(df["Close"].rolling(REGIME_MA).mean().iloc[-1])
        last = float(df["Close"].iloc[-1])
        diff = float((last / ma - 1) * 100)
        return {
            "ok": bool(last > ma),
            "kospi": last,
            "ma": ma,
            "diff_pct": diff,
            "warn": bool(last <= ma),
        }
    except Exception as e:
        print(f"  ⚠️ 레짐 데이터 조회 실패 ({e}). 필터 비활성 처리.")
        return {"ok": True, "kospi": None, "ma": None, "diff_pct": 0.0, "warn": False}

# ---------------------------------------------------------------------------
# 1) 종목 유니버스 (원본 유지)
# ---------------------------------------------------------------------------
def get_universe() -> pd.DataFrame:
    import FinanceDataReader as fdr
    listing = fdr.StockListing("KRX")
    cap_col = next((c for c in ("Marcap", "MarketCap", "marcap") if c in listing.columns), None)
    if cap_col is None:
        raise RuntimeError("시가총액 컬럼을 찾지 못했습니다.")
    listing = listing.dropna(subset=[cap_col, "Market", "Name"])
    listing = listing[~listing["Name"].str.contains("스팩|우$|우B|우C", regex=True, na=False)]
    kospi = listing[listing["Market"] == "KOSPI"].sort_values(cap_col, ascending=False).head(200)
    kosdaq = listing[listing["Market"] == "KOSDAQ"].sort_values(cap_col, ascending=False).head(150)
    rows = [{"code": r["Code"], "name": r["Name"], "market": "KOSPI200"} for _, r in kospi.iterrows()]
    rows += [{"code": r["Code"], "name": r["Name"], "market": "KOSDAQ150"} for _, r in kosdaq.iterrows()]
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 2) 기술적 지표 및 점수 (원본 로직 유지)
# ---------------------------------------------------------------------------
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
    df["Hist_prev"] = df["MACD_hist"].shift(1)
    return df

def score_swing(r: pd.Series):
    if pd.isna(r["RSI"]) or pd.isna(r["BB_pct"]): return 0, []
    score, reasons = 0, []
    if 30 <= r["RSI"] <= 45 and r["RSI"] > r["RSI_prev"]: score += 30; reasons.append(f"RSI반등")
    elif r["RSI"] < 30: score += 22; reasons.append(f"RSI과매도")
    if r["BB_pct"] < 0.2: score += 22; reasons.append("BB하단")
    if not pd.isna(r["Vol_ratio"]) and r["Vol_ratio"] >= 2.0: score += 18; reasons.append(f"거래량x{r['Vol_ratio']:.1f}")
    if r["Close"] > r["Open"] and not pd.isna(r["MA5"]) and r["Close"] >= r["MA5"]: score += 15; reasons.append("양봉+MA5↑")
    return max(0, min(100, score)), reasons

def score_trend(r: pd.Series):
    needed = ["MA5", "MA20", "MA60", "MA120", "RSI", "MACD", "MACD_sig"]
    if any(pd.isna(r[c]) for c in needed): return 0, []
    score, reasons = 0, []
    if r["MA5"] > r["MA20"] > r["MA60"] > r["MA120"]: score += 35; reasons.append("완전정배열")
    elif r["MA20"] > r["MA60"] > r["MA120"]: score += 22; reasons.append("중장기정배열")
    if r["Close"] > r["MA20"]: score += 12; reasons.append("MA20위")
    if r["MACD"] > 0 and r["MACD"] > r["MACD_sig"]: score += 18; reasons.append("MACD매수")
    if 50 <= r["RSI"] <= 70: score += 15; reasons.append(f"RSI {r['RSI']:.0f}")
    if not pd.isna(r["Pct60"]) and r["Pct60"] > 0: score += min(15, int(r["Pct60"] / 2)); reasons.append(f"60일+{r['Pct60']:.0f}%")
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
        s_score, s_reasons = score_swing(last)
        t_score, t_reasons = score_trend(last)
        atr = last["ATR"] if not pd.isna(last["ATR"]) else 0
        close = float(last["Close"])
        return {
            "code": code, "name": name, "market": market, "close": close,
            "chg1d": float((close / df["Close"].iloc[-2] - 1) * 100),
            "chg20d": float(last["Pct20"]) if not pd.isna(last["Pct20"]) else 0.0,
            "chg60d": float(last["Pct60"]) if not pd.isna(last["Pct60"]) else 0.0,
            "rsi": float(last["RSI"]) if not pd.isna(last["RSI"]) else None,
            "vol_ratio": float(last["Vol_ratio"]) if not pd.isna(last["Vol_ratio"]) else None,
            "atr": float(atr),
            "stop": float(close - 2 * atr) if atr else None,
            "target": float(close + 3 * atr) if atr else None,
            "swing": s_score, "swing_why": ", ".join(s_reasons),
            "trend": t_score, "trend_why": ", ".join(t_reasons),
        }
    except Exception: return None

# ---------------------------------------------------------------------------
# 3) UI 대시보드 (최신화된 HTML/JS)
# ---------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="ko"><meta charset="utf-8">
<title>한국주식 매매 신호 대시보드 — __DATE__</title>
<link href="https://cdn.jsdelivr.net/npm/gridjs/dist/theme/mermaid.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/gridjs/dist/gridjs.umd.js"></script>
<style>
 :root { color-scheme: light dark; }
 body { font-family: -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif; margin: 24px; max-width: 1500px; }
 h1 { margin: 0 0 4px; font-size: 22px; }
 .sub { color: #777; font-size: 13px; margin-bottom: 18px; }
 .tabs { display: flex; gap: 8px; margin: 14px 0; }
 .tab { padding: 8px 16px; background: #eee; border-radius: 6px; cursor: pointer; font-weight: 600; border: 1px solid transparent; }
 .tab.on { background: #1f6feb; color: white; }
 .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:600; }
 .pos { color:#d0342c; } .neg { color:#1862ce; }
 .score-hi { background:#fde0dc; color:#9d2914; }
 .regime { padding: 12px 16px; border-radius: 8px; margin: 12px 0; font-size: 14px; font-weight: 600; }
 .regime.bull { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
 .regime.bear { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
 .gridjs-table { font-size: 13px; width: 100% !important; }
 .candidates-bar { display: inline-block; padding: 6px 12px; background: #1f6feb; color: white; border-radius: 6px; font-size: 13px; margin: 8px 0; }
 .row-priority td { background: rgba(31, 111, 235, 0.08) !important; }
 @media (prefers-color-scheme: dark) { .tab { background:#222; color:#ddd; } .regime.bull { background:#1f3a26; color:#7ed99a; } .regime.bear { background:#3d1f23; color:#f97583; } }
</style>
<body>
<h1>📊 한국주식 매매 신호 대시보드</h1>
<div class="sub">기준일 <b>__DATE__</b> · 종목 수 <b id="cnt"></b> · 원본 알고리즘(80점 기준) 적용</div>
<div id="regimeBox"></div>
<div class="tabs">
 <div class="tab on" data-mode="trend">📈 장기 추세 (메인)</div>
 <div class="tab" data-mode="swing">🎯 단기 스윙 (참고)</div>
 <div class="tab" data-mode="all">전체 종목</div>
</div>
<div id="candidatesBar"></div>
<div id="grid"></div>
<script>
 const DATA = __DATA__; const REGIME = __REGIME__;
 document.getElementById("cnt").textContent = DATA.length;
 (function() {
   if (!REGIME.kospi) return;
   const box = document.getElementById("regimeBox");
   const cls = REGIME.ok ? "bull" : "bear";
   box.innerHTML = `<div class="regime ${cls}">${REGIME.ok?'🟢':'🔴'} 시장 레짐: ${REGIME.ok?'강세 (진입가능)':'약세 (관망권고)'} 
   <small> (KOSPI ${REGIME.kospi.toFixed(1)} / 200MA ${REGIME.ma.toFixed(1)})</small></div>`;
 })();
 const num = (v) => v == null ? "-" : Number(v).toLocaleString();
 const pct = (v) => `<span class="${v>=0?'pos':'neg'}">${v>=0?'+':''}${v.toFixed(1)}%</span>`;
 const scoreCell = (v) => `<span class="pill ${v>=80?'score-hi':''}">${v}</span>`;
 
 let grid;
 function render(mode) {
   let filtered = [...DATA];
   if(mode==='trend') filtered = filtered.filter(r => r.trend > 0).sort((a,b)=>b.trend-a.trend);
   else if(mode==='swing') filtered = filtered.filter(r => r.swing > 0).sort((a,b)=>b.swing-a.swing);
   
   const candCount = filtered.filter(r => (mode==='swing'?r.swing:r.trend) >= 80).length;
   document.getElementById("candidatesBar").innerHTML = candCount > 0 ? `<div class="candidates-bar">⭐ 80점 이상 진입 후보: ${candCount}개</div>` : "";

   if (grid) grid.destroy();
   grid = new gridjs.Grid({
     columns: [
       { name: "종목", formatter: (cell, row) => gridjs.html(`<a href="https://finance.naver.com/item/main.naver?code=${row.cells[1].data}" target="_blank" style="text-decoration:none;font-weight:bold;color:inherit;">${cell}</a>`) },
       { name: "코드", hidden: true },
       "시장", "종가", 
       { name: "1일", formatter: v => gridjs.html(pct(v)) },
       { name: "60일", formatter: v => gridjs.html(pct(v)) },
       "RSI", 
       { name: "스윙", formatter: v => gridjs.html(scoreCell(v)) },
       { name: "추세", formatter: v => gridjs.html(scoreCell(v)) },
       { name: "사유", width: "200px" },
       "손절가"
     ],
     data: filtered.map(r => [r.name, r.code, r.market, num(r.close), r.chg1d, r.chg60d, r.rsi?r.rsi.toFixed(0):"-", r.swing, r.trend, mode==='trend'?r.trend_why:r.swing_why, num(r.stop)]),
     sort: true, pagination: { limit: 25 }, search: true, resizable: true,
     rowAttributes: (row) => {
       const s = mode==='swing' ? row.cells[7].data : row.cells[8].data;
       return s >= 80 ? { class: 'row-priority' } : {};
     }
   }).render(document.getElementById("grid"));
 }
 document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", e => {
   document.querySelectorAll(".tab").forEach(x => x.classList.remove("on"));
   e.target.classList.add("on");
   render(e.target.dataset.mode);
 }));
 render("trend");
</script>
</body></html>
"""

# ---------------------------------------------------------------------------
# 4) 메인 실행부 (경로 에러 완벽 해결 버전)
# ---------------------------------------------------------------------------
def main():
    print("📈 분석 시작 (원본 로직 적용)...")
    regime = check_regime()
    universe = get_universe()
    
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(analyze_one, r.code, r.name, r.market): r.code for _, r in universe.iterrows()}
        for f in as_completed(futs):
            res = f.result()
            if res: results.append(res)

    today = dt.date.today().strftime("%Y-%m-%d")
    html_output = (HTML
                   .replace("__DATE__", today)
                   .replace("__DATA__", json.dumps(results, ensure_ascii=False))
                   .replace("__REGIME__", json.dumps(regime, ensure_ascii=False)))

    # GitHub Actions 환경 대응: 현재 폴더 및 docs 폴더 양쪽 저장
    Path("dashboard.html").write_text(html_output, encoding="utf-8")
    docs_path = Path("docs")
    docs_path.mkdir(exist_ok=True)
    (docs_path / "index.html").write_text(html_output, encoding="utf-8")

    # 마크다운 요약 생성 (원본 유지)
    trend_top = sorted([r for r in results if r["trend"] >= SCORE_THRESHOLD], key=lambda x: -x["trend"])[:10]
    md = [f"# 📊 주식 신호 요약 ({today})\n", "| 종목 | 점수 | 사유 |\n|---|---|---|"]
    for r in trend_top: md.append(f"| {r['name']} | {r['trend']} | {r['trend_why']} |")
    Path("top.md").write_text("\n".join(md), encoding="utf-8")

    print(f"✅ 분석 완료! (파일생성: dashboard.html, docs/index.html, top.md)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
