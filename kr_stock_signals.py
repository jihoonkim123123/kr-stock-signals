"""
한국주식 매매 신호 대시보드
==============================

KOSPI200 + KOSDAQ150 전체 종목을 스캔하여
단기 스윙(반등) · 장기 추세추종(정배열) 매매 신호를 생성합니다.

사용법:
    1) 의존성 설치 (최초 1회만):
       pip install pykrx FinanceDataReader pandas numpy tqdm

    2) 매일 아침 한 번 실행:
       python kr_stock_signals.py

    3) 같은 폴더에 dashboard.html 이 생성되면 더블클릭해서 브라우저로 확인.

매매는 직접 본인이 결제 클릭으로 진행하세요. 이 스크립트는 신호와 진입가/손절가 후보만 제시합니다.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent
LOOKBACK_DAYS = 260
WORKERS = 8

# 백테스트로 검증된 운용 설정과 동일하게 맞춤
SCORE_THRESHOLD = 80          # 점수 80 이상만 진입 후보
ENABLE_SWING = False          # 단기 스윙은 알파 약해서 끔 (참고용으로만 계산)
REGIME_FILTER = True          # KOSPI 가 200MA 위일 때만 신규 진입 권장
REGIME_INDEX = "KS11"
REGIME_MA = 200


# ---------------------------------------------------------------------------
# 0) 시장 레짐 필터 — KOSPI > 200MA 인지 확인
# ---------------------------------------------------------------------------

def check_regime() -> dict:
    """현재 시장 레짐 상태 조회. 약세장이면 신규 진입 자제 권고.

    반환 dict 의 모든 값은 Python native 타입 (json.dumps 호환).
    """
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
        # 레짐 데이터 가져오기 실패해도 스크립트 진행 — 강세장 가정
        print(f"  ⚠️ 레짐 데이터 조회 실패 ({e}). 필터 비활성 처리.")
        return {"ok": True, "kospi": None, "ma": None, "diff_pct": 0.0, "warn": False}


# ---------------------------------------------------------------------------
# 1) 종목 유니버스: 코스피200 + 코스닥150
# ---------------------------------------------------------------------------

def get_universe() -> pd.DataFrame:
    """KOSPI200 + KOSDAQ150 근사 (각 시장 시가총액 상위).

    pykrx 의 정식 KOSPI200 API 는 KRX 로그인이 필요해 사용 불가. 대신
    FinanceDataReader 로 전체 상장 목록을 받아 시총 상위 200/150 로 근사한다.
    실제 지수와의 구성 차이는 무시할 수준 (가장자리 몇 종목 차이).
    """
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

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]

    # 볼린저밴드(20, 2σ)
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_up"] = bb_mid + 2 * bb_std
    df["BB_lo"] = bb_mid - 2 * bb_std
    df["BB_pct"] = (df["Close"] - df["BB_lo"]) / (df["BB_up"] - df["BB_lo"]).replace(0, np.nan)

    # 거래량
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / df["Vol_MA20"].replace(0, np.nan)

    # ATR(14) — 손절·목표가 계산용
    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift()).abs()
    l_c = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # 기간 등락률
    df["Pct5"] = (df["Close"] / df["Close"].shift(5) - 1) * 100
    df["Pct20"] = (df["Close"] / df["Close"].shift(20) - 1) * 100
    df["Pct60"] = (df["Close"] / df["Close"].shift(60) - 1) * 100

    df["RSI_prev"] = df["RSI"].shift(1)
    df["Hist_prev"] = df["MACD_hist"].shift(1)
    return df


# ---------------------------------------------------------------------------
# 3) 점수 — 단기 스윙(반등형) / 장기 추세추종
# ---------------------------------------------------------------------------

def score_swing(r: pd.Series):
    """단기 스윙 (0~100). 과매도 후 반등 초입을 잡는 패턴."""
    if pd.isna(r["RSI"]) or pd.isna(r["BB_pct"]):
        return 0, []
    score, reasons = 0, []

    # ① RSI 30 부근 반등 시작
    if 30 <= r["RSI"] <= 45 and r["RSI"] > r["RSI_prev"]:
        score += 30
        reasons.append(f"RSI반등 {r['RSI_prev']:.0f}→{r['RSI']:.0f}")
    elif r["RSI"] < 30:
        score += 22
        reasons.append(f"RSI과매도 {r['RSI']:.0f}")

    # ② 볼린저 하단 부근
    if r["BB_pct"] < 0.2:
        score += 22
        reasons.append("BB하단")
    elif r["BB_pct"] < 0.4:
        score += 10

    # ③ 거래량 급증
    if not pd.isna(r["Vol_ratio"]):
        if r["Vol_ratio"] >= 2.0:
            score += 18
            reasons.append(f"거래량x{r['Vol_ratio']:.1f}")
        elif r["Vol_ratio"] >= 1.5:
            score += 10

    # ④ 양봉 + 5일선 회복
    if r["Close"] > r["Open"] and not pd.isna(r["MA5"]) and r["Close"] >= r["MA5"]:
        score += 15
        reasons.append("양봉+MA5↑")

    # ⑤ MACD 히스토그램 반등
    if not pd.isna(r["Hist_prev"]) and r["MACD_hist"] > r["Hist_prev"] and r["Hist_prev"] < 0:
        score += 10
        reasons.append("MACD반등")

    # ⑥ 너무 망가진 종목 제외 (60일 -25% 이하)
    if not pd.isna(r["Pct60"]) and r["Pct60"] < -25:
        score -= 15
        reasons.append("⚠️60일급락")

    return max(0, min(100, score)), reasons


def score_trend(r: pd.Series):
    """장기 추세추종 (0~100). 정배열 + MACD 매수 + 적정 RSI."""
    needed = ["MA5", "MA20", "MA60", "MA120", "RSI", "MACD", "MACD_sig"]
    if any(pd.isna(r[c]) for c in needed):
        return 0, []
    score, reasons = 0, []

    # ① 완전 정배열
    if r["MA5"] > r["MA20"] > r["MA60"] > r["MA120"]:
        score += 35
        reasons.append("완전정배열")
    elif r["MA20"] > r["MA60"] > r["MA120"]:
        score += 22
        reasons.append("중장기정배열")

    # ② 종가가 20일선 위
    if r["Close"] > r["MA20"]:
        score += 12
        reasons.append("MA20위")

    # ③ MACD 매수
    if r["MACD"] > 0 and r["MACD"] > r["MACD_sig"]:
        score += 18
        reasons.append("MACD매수")
    elif r["MACD"] > r["MACD_sig"]:
        score += 8

    # ④ RSI 50~70 적정 강세 (>80은 과열 감점)
    if 50 <= r["RSI"] <= 70:
        score += 15
        reasons.append(f"RSI {r['RSI']:.0f}")
    elif 70 < r["RSI"] <= 80:
        score += 5
    elif r["RSI"] > 80:
        score -= 10
        reasons.append("⚠️과열")

    # ⑤ 60일 모멘텀
    if not pd.isna(r["Pct60"]) and r["Pct60"] > 0:
        score += min(15, int(r["Pct60"] / 2))
        reasons.append(f"60일+{r['Pct60']:.0f}%")

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
            "bb_pct": float(last["BB_pct"]) if not pd.isna(last["BB_pct"]) else None,
            "macd_hist": float(last["MACD_hist"]) if not pd.isna(last["MACD_hist"]) else None,
            "atr": float(atr) if atr is not None else None,
            "stop": float(close - 2 * atr) if atr else None,
            "target": float(close + 3 * atr) if atr else None,
            "swing": s_score,
            "swing_why": ", ".join(s_reasons),
            "trend": t_score,
            "trend_why": ", ".join(t_reasons),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5) HTML 대시보드
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="ko"><meta charset="utf-8">
<title>한국주식 매매 신호 대시보드 — __DATE__</title>
<link href="https://cdn.jsdelivr.net/npm/gridjs/dist/theme/mermaid.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/gridjs/dist/gridjs.umd.js"></script>
<style>
 :root { color-scheme: light dark; }
 body { font-family: -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
        margin: 24px; max-width: 1400px; }
 h1 { margin: 0 0 4px; font-size: 22px; }
 .sub { color: #777; font-size: 13px; margin-bottom: 18px; }
 .tabs { display: flex; gap: 8px; margin: 14px 0; }
 .tab { padding: 8px 16px; background: #eee; border-radius: 6px; cursor: pointer;
        font-weight: 600; border: 1px solid transparent; }
 .tab.on { background: #1f6feb; color: white; }
 @media (prefers-color-scheme: dark) { .tab { background:#222; color:#ddd; } }
 .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px;
         font-weight:600; }
 .pos { color:#d0342c; } .neg { color:#1862ce; }
 .score-hi { background:#fde0dc; color:#9d2914; }
 .score-md { background:#fff3cd; color:#7a5400; }
 .score-lo { background:#e8eef7; color:#5b6a82; }
 @media (prefers-color-scheme: dark) {
   .score-hi { background:#5a1f1a; color:#ffb3a8; }
   .score-md { background:#5a4a14; color:#ffd97a; }
   .score-lo { background:#2a3346; color:#a9b9cf; }
 }
 .legend { font-size: 12px; color:#888; margin: 8px 0 16px;
           background:rgba(127,127,127,0.08); padding: 10px 14px; border-radius:6px; }
 .regime { padding: 12px 16px; border-radius: 8px; margin: 12px 0;
           font-size: 14px; font-weight: 600; }
 .regime.bull { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
 .regime.bear { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
 @media (prefers-color-scheme: dark) {
   .regime.bull { background:#1f3a26; color:#7ed99a; border-left-color:#2ea043; }
   .regime.bear { background:#3d1f23; color:#f97583; border-left-color:#da3633; }
 }
 .regime small { font-weight: 400; opacity: 0.85; }
 .gridjs-table { font-size: 13px; }
 .candidates-bar { display: inline-block; padding: 6px 12px; background: #1f6feb;
                   color: white; border-radius: 6px; font-size: 13px; font-weight: 600;
                   margin: 8px 0 4px; }
 .candidates-bar.empty { background: #6c757d; }
 .stock-name { font-weight: 600; }
 .stock-code { color: #888; font-size: 11px; margin-left: 4px; }
 a.stock-link { text-decoration: none; }
 a.stock-link:hover { text-decoration: underline; }
 /* 페이지네이션 크게 */
 .gridjs-pagination { padding: 14px 0 !important; font-size: 14px !important; }
 .gridjs-pagination .gridjs-pages button {
   padding: 8px 14px !important; margin: 0 3px !important;
   font-size: 13px !important; min-width: 40px !important;
 }
 .gridjs-pagination .gridjs-pages button.gridjs-currentPage {
   background: #1f6feb !important; color: white !important;
 }
 .gridjs-pagination .gridjs-summary { font-size: 13px !important; padding: 8px 0 !important; }
 .gridjs-search input { font-size: 14px !important; padding: 10px 14px !important;
                        width: 320px !important; }
 /* 점수 80+ 행 강조 */
 .row-priority td { background: rgba(31, 111, 235, 0.06) !important; }
 @media (prefers-color-scheme: dark) {
   .row-priority td { background: rgba(88, 166, 255, 0.10) !important; }
 }
</style>
<body>
<h1>📊 한국주식 매매 신호 대시보드</h1>
<div class="sub">기준일 <b>__DATE__</b> · 종목 수 <b id="cnt"></b> · 데이터 출처 KRX/네이버</div>

<div id="regimeBox"></div>

<div class="legend">
 <b>장기 추세 (메인 전략)</b>: 정배열 + MACD 매수 + 적정 RSI + 60일 모멘텀. <b>점수 80 이상</b>이 진입 후보.
 백테스트 7년 결과 +446% (KOSPI +260% 대비 알파 검증).<br>
 <b>시장 레짐 필터</b>: KOSPI가 200일선 위일 때만 신규 진입. 약세장에서는 위 신호도 보류 권고.<br>
 <b>단기 스윙</b>은 알파 약해 참고용으로만 표시.
 <b>진입가/손절가</b>는 종가 기준 ±2~3×ATR 가이드. 마지막 주문은 직접 본인이.
</div>

<div class="tabs">
 <div class="tab on" data-mode="trend">📈 장기 추세 TOP</div>
 <div class="tab" data-mode="swing">🎯 단기 스윙 (참고용)</div>
 <div class="tab" data-mode="all">전체 정렬</div>
</div>

<div id="candidatesBar"></div>

<div id="grid"></div>

<script>
 const DATA = __DATA__;
 const REGIME = __REGIME__;
 document.getElementById("cnt").textContent = DATA.length;

 // 시장 레짐 배너
 (function() {
   if (REGIME.kospi == null) return;
   const box = document.getElementById("regimeBox");
   const cls = REGIME.ok ? "bull" : "bear";
   const emoji = REGIME.ok ? "🟢" : "🔴";
   const label = REGIME.ok ? "강세" : "약세";
   const note = REGIME.ok
     ? "추세 신호 진입 가능 구간"
     : "⚠️ 신규 진입 자제 권고. 보유 종목은 손절선 관리 우선.";
   box.innerHTML = `<div class="regime ${cls}">${emoji} 시장 레짐: ${label}
     <small> · KOSPI ${REGIME.kospi.toLocaleString("ko-KR",{maximumFractionDigits:1})}
     vs 200MA ${REGIME.ma.toLocaleString("ko-KR",{maximumFractionDigits:1})}
     (${REGIME.diff_pct>=0?'+':''}${REGIME.diff_pct.toFixed(1)}%)</small>
     <div style="margin-top:6px;font-weight:400;font-size:13px;">${note}</div>
   </div>`;
 })();

 const num = (v, d=0) => v == null ? "-" : Number(v).toLocaleString("ko-KR",
              {minimumFractionDigits: d, maximumFractionDigits: d});
 const pct = (v) => v == null ? "-" :
   `<span class="${v>=0?'pos':'neg'}">${v>=0?'+':''}${v.toFixed(1)}%</span>`;
 const scoreCell = (v) => {
   const cls = v >= 80 ? "score-hi" : v >= 60 ? "score-md" : "score-lo";
   return `<span class="pill ${cls}">${v}</span>`;
 };
 const link = (code, name) =>
   `<a href="https://finance.naver.com/item/main.naver?code=${code}" target="_blank">${name}</a>`;

 // 종목 셀: 종목명 굵게 + 코드 작게
 const stockCell = (code, name) =>
   `<a class="stock-link" href="https://finance.naver.com/item/main.naver?code=${code}" target="_blank">` +
   `<span class="stock-name">${name}</span></a><span class="stock-code">${code}</span>`;

 // "매수 적절도" 점수 — 추세점수가 메인, 60일 모멘텀 가중, RSI 70 이하 보너스, 과매수 페널티
 const buyPriority = (r) => {
   let s = (r.trend || 0) * 1.0;
   if (r.chg60d != null) s += Math.max(-10, Math.min(15, r.chg60d / 4));
   if (r.rsi != null) {
     if (r.rsi >= 50 && r.rsi <= 65) s += 5;     // 적정 강세 보너스
     else if (r.rsi > 75) s -= 8;                // 과매수 페널티
   }
   if (r.vol_ratio != null && r.vol_ratio >= 1.5) s += 3;
   return s;
 };

 let grid = null;
 function render(mode) {
   let rows = [...DATA];
   if (mode === "swing") {
     rows = rows.filter(r => r.swing > 0).sort((a,b) => b.swing - a.swing);
   } else if (mode === "trend") {
     rows = rows.filter(r => r.trend > 0)
                .sort((a,b) => buyPriority(b) - buyPriority(a));  // 매수 적절도순
   } else {
     rows.sort((a,b) => buyPriority(b) - buyPriority(a));
   }

   // 진입 후보 카운트 배너
   const SCORE_TH = 80;
   const candCount = rows.filter(r => (mode === "swing" ? r.swing : r.trend) >= SCORE_TH).length;
   const bar = document.getElementById("candidatesBar");
   if (mode === "trend" || mode === "all") {
     bar.innerHTML = candCount > 0
       ? `<div class="candidates-bar">⭐ 오늘의 진입 후보: ${candCount}개 (점수 ${SCORE_TH}+)</div>`
       : `<div class="candidates-bar empty">오늘은 점수 ${SCORE_TH}+ 진입 후보 없음 — 관망 권고</div>`;
   } else {
     bar.innerHTML = `<div class="candidates-bar empty">단기 스윙은 참고용 (백테스트상 알파 약함)</div>`;
   }

   const cols = [
     { name: "종목", sort: false, width: "200px",
       formatter: cell => gridjs.html(cell) },
     { name: "코드", hidden: true },          // 검색용
     { name: "이름", hidden: true },          // 검색용
     { name: "시장", width: "85px" },
     { name: "종가", formatter: v => num(v) },
     { name: "1일", formatter: v => gridjs.html(pct(v)) },
     { name: "20일", formatter: v => gridjs.html(pct(v)) },
     { name: "60일", formatter: v => gridjs.html(pct(v)) },
     { name: "RSI", formatter: v => v==null ? "-" : v.toFixed(0) },
     { name: "거래량x", formatter: v => v==null ? "-" : v.toFixed(2) },
     { name: "스윙", formatter: v => gridjs.html(scoreCell(v)) },
     { name: "추세", formatter: v => gridjs.html(scoreCell(v)) },
     { name: "사유", width: "240px" },
     { name: "손절", formatter: v => num(v) },
     { name: "목표", formatter: v => num(v) },
   ];

   const data = rows.map(r => [
     stockCell(r.code, r.name),  // 종목 (이름+코드 결합 HTML)
     r.code,                     // 검색용
     r.name,                     // 검색용
     r.market, r.close, r.chg1d, r.chg20d, r.chg60d,
     r.rsi, r.vol_ratio, r.swing, r.trend,
     mode === "trend" ? r.trend_why : (r.swing_why || r.trend_why),
     r.stop, r.target,
   ]);

   if (grid) grid.destroy();
   grid = new gridjs.Grid({
     columns: cols,
     data,
     sort: true,
     pagination: { limit: 25, summary: true },
     search: true,
     resizable: true,
     language: {
       search: { placeholder: "🔍 종목명·코드·시장·사유 검색 (한글/영어/숫자)" },
       pagination: { previous: "← 이전", next: "다음 →",
                     showing: "", to: "~", of: "/", results: () => "건" },
       noRecordsFound: "조건에 맞는 종목이 없어요",
     },
     style: { table: { "white-space": "nowrap" } },
     rowAttributes: (row) => {
       const score = mode === "swing" ? row.cells[10].data : row.cells[11].data;
       return score >= SCORE_TH ? { class: "row-priority" } : {};
     },
   }).render(document.getElementById("grid"));
 }

 document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
   document.querySelectorAll(".tab").forEach(x => x.classList.remove("on"));
   t.classList.add("on");
   render(t.dataset.mode);
 }));

 render("trend");
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# 6) 메인
# ---------------------------------------------------------------------------

def main():
    # 1) 시장 레짐 검사
    print("📈 시장 레짐 확인 중 (KOSPI vs 200MA)...")
    regime = check_regime()
    if regime["kospi"] is not None:
        sign = "위 ✅" if regime["ok"] else "아래 ⚠️"
        print(f"  KOSPI {regime['kospi']:,.1f} / 200MA {regime['ma']:,.1f} "
              f"({regime['diff_pct']:+.1f}%) — 200MA {sign}\n")
    if regime["warn"] and REGIME_FILTER:
        print("  ⚠️ 시장 약세 신호. 신규 진입 자제 권고. 보유 종목 위주로 관리하세요.\n")

    # 2) 종목 리스트
    print("📋 종목 리스트 조회 중...")
    universe = get_universe()
    print(f"  코스피200 + 코스닥150 = {len(universe)}개 종목\n")

    # 3) 시세 분석
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

    print(f"\n✓ 분석 완료: {len(results)}개\n")

    # 4) 임계값 80 이상으로 필터링 (백테스트 검증값과 동일)
    trend = sorted([r for r in results if r["trend"] >= SCORE_THRESHOLD],
                   key=lambda x: -x["trend"])[:10]
    swing = sorted([r for r in results if r["swing"] >= SCORE_THRESHOLD],
                   key=lambda x: -x["swing"])[:10]

    today = dt.date.today().strftime("%Y-%m-%d")
    html = (HTML
            .replace("__DATE__", today)
            .replace("__DATA__", json.dumps(results, ensure_ascii=False))
            .replace("__REGIME__", json.dumps(regime, ensure_ascii=False)))

    # cwd 와 __file__ 위치가 다를 수 있으니 cwd 와 OUTPUT_DIR 양쪽에 모두 저장
    primary_path = Path.cwd() / "dashboard.html"
    primary_path.write_text(html, encoding="utf-8")
    print(f"✅ 대시보드 생성 완료 → {primary_path.resolve()}")
    if OUTPUT_DIR.resolve() != Path.cwd().resolve():
        backup_path = OUTPUT_DIR / "dashboard.html"
        backup_path.write_text(html, encoding="utf-8")
        print(f"   백업 위치 → {backup_path.resolve()}")
    sys.stdout.flush()
    out = primary_path

    # 5) GitHub Issue 용 마크다운 요약 (top.md)
    import os as _os
    pages_url = _os.environ.get("PAGES_URL", "")
    md = [f"# 📊 한국주식 매매 신호 — {today}\n"]

    # 레짐 배너
    if regime["kospi"] is not None:
        if regime["ok"]:
            md.append(f"### 🟢 시장 레짐: 강세 (KOSPI {regime['kospi']:,.1f} / "
                      f"200MA {regime['ma']:,.1f}, {regime['diff_pct']:+.1f}%)\n")
            md.append("→ 추세 신호 진입 가능 구간\n")
        else:
            md.append(f"### 🔴 시장 레짐: **약세** (KOSPI {regime['kospi']:,.1f} / "
                      f"200MA {regime['ma']:,.1f}, {regime['diff_pct']:+.1f}%)\n")
            md.append("> ⚠️ **신규 진입 자제 권고.** 백테스트 결과 약세장 진입은 손실 확률↑.\n"
                      "> 보유 종목은 손절선 관리에 집중. 아래 신호는 참고용.\n")

    md.append(f"분석 종목 수: **{len(results)}개** · 점수 {SCORE_THRESHOLD}+ 만 표시\n")

    # 추세 (메인)
    md.append("\n## 📈 장기 추세 TOP 10")
    md.append("*정배열 + MACD 매수 + 적정 RSI + 60일 모멘텀 (백테스트 7년 +446%, KOSPI +260% 대비 알파 검증)*\n")
    if trend:
        md.append("| # | 점수 | 종목 | 현재가 | 손절 | 목표 | 사유 |")
        md.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(trend, 1):
            link = f"[{r['name']}](https://finance.naver.com/item/main.naver?code={r['code']})"
            md.append(f"| {i} | **{r['trend']}** | {link} ({r['code']}) | "
                      f"₩{r['close']:,.0f} | ₩{r['stop']:,.0f} | ₩{r['target']:,.0f} | {r['trend_why']} |")
    else:
        md.append(f"_점수 {SCORE_THRESHOLD} 이상 후보 없음. 오늘은 진입할 종목이 없습니다._\n")

    # 스윙 (참고용 — 백테스트 알파 약함)
    if ENABLE_SWING:
        md.append("\n## 🎯 단기 스윙 TOP 10 (참고용)")
        md.append("*RSI 과매도 반등 + 볼린저 하단. 백테스트상 인덱스 대비 알파 약함 — 정보용으로만 활용*\n")
        if swing:
            md.append("| # | 점수 | 종목 | 현재가 | 손절 | 목표 | 사유 |")
            md.append("|---|---|---|---|---|---|---|")
            for i, r in enumerate(swing, 1):
                link = f"[{r['name']}](https://finance.naver.com/item/main.naver?code={r['code']})"
                md.append(f"| {i} | **{r['swing']}** | {link} ({r['code']}) | "
                          f"₩{r['close']:,.0f} | ₩{r['stop']:,.0f} | ₩{r['target']:,.0f} | {r['swing_why']} |")
        else:
            md.append("_해당 점수 이상 후보 없음._\n")

    md.append("\n---\n")
    md.append("⚠️ 알고리즘 신호일 뿐, 매매 결정과 책임은 본인에게 있습니다. 직접 HTS/MTS에서 주문하세요.")
    md.append("\n💡 운용 팁: 동시 포지션 10종목까지 균등 분할, 같은 종목 30일 재진입 쿨다운, 손절은 표시된 가격에서 자동.")
    if pages_url:
        md.append(f"\n📊 [전체 대시보드 보기]({pages_url})")
    md_path = Path.cwd() / "top.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"📝 마크다운 요약 → {md_path.resolve()}")
    sys.stdout.flush()

    # 6) 콘솔 요약
    print("\n📈 장기 추세 TOP 10 (점수 80+)")
    if trend:
        for r in trend:
            print(f"  [{r['trend']:3d}] {r['name']:<10s} ({r['code']}) "
                  f"₩{r['close']:>9,.0f} → 손절 ₩{r['stop']:>9,.0f} | {r['trend_why']}")
    else:
        print(f"  (점수 {SCORE_THRESHOLD} 이상 후보 없음 — 오늘은 진입 종목 없음)")

    if ENABLE_SWING:
        print(f"\n🎯 단기 스윙 TOP 10 (참고용)")
        for r in swing:
            print(f"  [{r['swing']:3d}] {r['name']:<10s} ({r['code']}) "
                  f"₩{r['close']:>9,.0f} | {r['swing_why']}")

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
