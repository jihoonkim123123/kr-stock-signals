"""
한국주식 매매 신호 대시보드
==============================

KOSPI200 + KOSDAQ150 전체 종목을 스캔하여 장기 추세추종(정배열) 매매 신호를 생성합니다.
백테스트 검증된 운용 룰: 점수 80+ 진입 / KOSPI 200MA 레짐 필터 / 동시 10포지션 분할.

사용법:
    pip install -r requirements.txt
    python kr_stock_signals.py
    → dashboard.html 더블클릭하면 브라우저로 검색·정렬되는 대시보드 열림

매매는 직접 본인이 HTS/MTS에서 진행. 이 스크립트는 신호와 진입가/손절가 후보만 제시.
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

OUTPUT_DIR = Path(__file__).parent
LOOKBACK_DAYS = 260
WORKERS = 8

# 백테스트로 검증된 운용 설정과 동일하게 맞춤
SCORE_THRESHOLD = 80          # 점수 80 이상만 진입 후보
ENABLE_SWING = False          # 단기 스윙은 알파 약해서 끔
REGIME_FILTER = True          # KOSPI 가 200MA 위일 때만 신규 진입 권장
REGIME_INDEX = "KS11"
REGIME_MA = 200


# ---------------------------------------------------------------------------
# 0) 시장 레짐 필터 — KOSPI > 200MA
# ---------------------------------------------------------------------------

def check_regime() -> dict:
    """현재 시장 레짐 상태 조회. 모든 값은 Python native (json 호환)."""
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
# 1) 종목 유니버스: 코스피200 + 코스닥150 (시총 상위로 근사)
# ---------------------------------------------------------------------------

def get_universe() -> pd.DataFrame:
    """KOSPI200 + KOSDAQ150 근사 (각 시장 시가총액 상위)."""
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

    # 강세 RSI 다이버전스 — 가격은 저점 더 낮춤(LL), RSI는 저점 더 높임(HL)
    df["price_min_recent"] = df["Close"].rolling(10).min()
    df["price_min_prev"] = df["Close"].rolling(20).min().shift(10)
    df["rsi_min_recent"] = df["RSI"].rolling(10).min()
    df["rsi_min_prev"] = df["RSI"].rolling(20).min().shift(10)
    df["bullish_div"] = (
        (df["price_min_recent"] < df["price_min_prev"]) &
        (df["rsi_min_recent"] > df["rsi_min_prev"]) &
        (df["RSI"] < 60)  # 과매수 영역에서는 의미 없음
    )

    # 거래량 동반 횡보 → 돌파 임박 (축적 패턴)
    hi20 = df["High"].rolling(20).max()
    lo20 = df["Low"].rolling(20).min()
    df["range_20"] = (hi20 - lo20) / df["Close"]
    df["price_pos_20"] = (df["Close"] - lo20) / (hi20 - lo20).replace(0, np.nan)
    df["vol_5"] = df["Volume"].rolling(5).mean()
    df["vol_25"] = df["Volume"].rolling(25).mean()
    df["consolidation_breakout"] = (
        (df["range_20"] < 0.18) &                    # 박스 좁음
        (df["price_pos_20"] > 0.65) &                # 박스 상단 부근
        (df["vol_5"] > df["vol_25"].replace(0, np.nan) * 1.2)  # 거래량 1.2배+
    )

    return df


# ---------------------------------------------------------------------------
# 3) 점수 — 단기 스윙 / 장기 추세
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

    # ⑥ 강세 RSI 다이버전스 — 반전 직전 매수 신호 (+15)
    bd = r.get("bullish_div", False)
    if bd is True or (bd is not None and not pd.isna(bd) and bool(bd)):
        score += 15
        reasons.append("RSI다이버전스↑")

    # ⑦ 거래량 동반 횡보 돌파 임박 — 축적 패턴 (+12)
    cb = r.get("consolidation_breakout", False)
    if cb is True or (cb is not None and not pd.isna(cb) and bool(cb)):
        score += 12
        reasons.append("횡보+거래량↑")

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
<html lang="ko">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>한국주식 매매 신호 대시보드 — __DATE__</title>
    <link href="https://cdn.jsdelivr.net/npm/gridjs/dist/theme/mermaid.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/gridjs/dist/gridjs.umd.js"></script>
    <style>
        :root {
            --bg-body: #0f172a;
            --bg-card: #1e293b;
            --primary: #38bdf8;
            --up-color: #fb7185;
            --down-color: #38bdf8;
            --text-main: #f1f5f9;
            --text-sub: #94a3b8;
            --border: #334155;
        }

        body { 
            font-family: -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
            background-color: var(--bg-body); color: var(--text-main);
            margin: 0; padding: 16px; line-height: 1.5;
        }

        .header-section { max-width: 1500px; margin: 0 auto 20px; }
        h1 { font-size: 22px; font-weight: 800; margin: 0 0 8px; color: var(--text-main); }
        .sub { color: var(--text-sub); font-size: 13px; }

        /* 레짐 박스 */
        .regime { padding: 16px; border-radius: 12px; margin: 15px 0; font-size: 14px; border-left: 5px solid #444; background: var(--bg-card); }
        .regime.bull { border-left-color: #10b981; background: rgba(16, 185, 129, 0.1); }
        .regime.bear { border-left-color: #f43f5e; background: rgba(244, 63, 94, 0.1); }

        /* 탭 디자인 */
        .tabs { display: flex; gap: 8px; margin: 20px 0; overflow-x: auto; padding-bottom: 4px; }
        .tab {
            white-space: nowrap; padding: 10px 20px; background: var(--bg-card); 
            border-radius: 10px; cursor: pointer; font-weight: 600; font-size: 13px;
            border: 1px solid var(--border); color: var(--text-sub);
        }
        .tab.on { background: var(--primary); color: #0f172a; border-color: var(--primary); }

        /* 테이블 컨테이너 (가로 스크롤의 핵심) */
        .table-wrap { 
            background: var(--bg-card); border-radius: 12px; padding: 12px;
            border: 1px solid var(--border); overflow-x: auto; 
        }

        /* GridJS 제목 잘림 방지 및 레이아웃 최적화 */
        .gridjs-table { table-layout: auto !important; width: 100% !important; min-width: 1300px; } 
        .gridjs-th { 
            background-color: rgba(15, 23, 42, 0.8) !important; color: var(--text-sub) !important; 
            padding: 12px 8px !important; font-size: 12px !important; white-space: nowrap !important;
        }
        .gridjs-td { padding: 12px 8px !important; border-bottom: 1px solid var(--border) !important; font-size: 13px; color: var(--text-main); }
        
        /* 유틸리티 컬러 */
        .pos { color: var(--up-color); font-weight: 600; }
        .neg { color: var(--down-color); font-weight: 600; }
        .pill { display:inline-block; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:700; }
        .score-hi { background:#fb7185; color:#fff; }
        .score-md { background:#fbbf24; color:#000; }
        .score-lo { background:#475569; color:#fff; }

        .stock-link { text-decoration: none; color: var(--primary); font-weight: 700; }
        .stock-code { color: var(--text-sub); font-size: 11px; margin-left: 4px; }
        .reason-cell { white-space: normal !important; min-width: 200px; font-size: 12px; color: var(--text-sub); }
        
        .candidates-bar { display: inline-block; padding: 8px 16px; background: var(--primary); color: #0f172a; border-radius: 8px; font-size: 13px; font-weight: 700; margin-bottom: 12px; }
        .row-priority td { background: rgba(56, 189, 248, 0.08) !important; }

        /* 검색창 모바일 대응 */
        .gridjs-search-input { background: var(--bg-body) !important; color: white !important; border-color: var(--border) !important; }
    </style>
</head>
<body>

<div class="header-section">
    <h1>📊 한국주식 매매 신호 대시보드</h1>
    <div class="sub">기준일 <b>__DATE__</b> · 종목 수 <b id="cnt"></b> · 데이터 출처 KRX/네이버</div>
    <div id="regimeBox"></div>
</div>

<div class="tabs">
    <div class="tab on" data-mode="trend">📈 장기 추세 TOP</div>
    <div class="tab" data-mode="swing">🎯 단기 스윙 (참고용)</div>
    <div class="tab" data-mode="all">🔍 전체 정렬</div>
</div>

<div id="candidatesBar"></div>

<div class="table-wrap">
    <div id="grid"></div>
</div>

<script>
    const DATA = __DATA__;
    const REGIME = __REGIME__;
    document.getElementById("cnt").textContent = DATA.length;

    (function() {
        if (REGIME.kospi == null) return;
        const box = document.getElementById("regimeBox");
        const cls = REGIME.ok ? "bull" : "bear";
        const emoji = REGIME.ok ? "🟢" : "🔴";
        box.innerHTML = `<div class="regime ${cls}">
            <b>${emoji} 시장 레짐: ${REGIME.ok ? '강세' : '약세'}</b>
            <div style="font-size:12px; margin-top:4px; opacity:0.8;">
                KOSPI ${REGIME.kospi.toLocaleString()} (200MA 대비 ${REGIME.diff_pct>=0?'+':''}${REGIME.diff_pct.toFixed(1)}%) | 
                ${REGIME.ok ? '추세 신호 진입 가능 구간' : '⚠️ 신규 진입 자제 권고'}
            </div>
        </div>`;
    })();

    const num = (v, d=0) => v == null ? "-" : Number(v).toLocaleString("ko-KR", {minimumFractionDigits: d, maximumFractionDigits: d});
    const pct = (v) => {
        if (v == null) return "-";
        return `<span class="${v>=0?'pos':'neg'}">${v>=0?'+':''}${v.toFixed(1)}%</span>`;
    }
    const scoreCell = (v) => {
        const cls = v >= 80 ? "score-hi" : v >= 60 ? "score-md" : "score-lo";
        return `<span class="pill ${cls}">${v}</span>`;
    };

    const buyPriority = (r) => {
        let s = (r.trend || 0) * 1.0;
        if (r.chg60d != null) s += Math.max(-10, Math.min(15, r.chg60d / 4));
        return s;
    };

    let grid = null;
    function render(mode) {
        let rows = [...DATA];
        if (mode === "swing") rows = rows.filter(r => r.swing > 0).sort((a,b) => b.swing - a.swing);
        else if (mode === "trend") rows = rows.filter(r => r.trend > 0).sort((a,b) => buyPriority(b) - buyPriority(a));

        const SCORE_TH = 80;
        const candCount = rows.filter(r => (mode === "swing" ? r.swing : r.trend) >= SCORE_TH).length;
        const bar = document.getElementById("candidatesBar");
        bar.innerHTML = candCount > 0 ? `<div class="candidates-bar">⭐ 진입 후보: ${candCount}개 (점수 ${SCORE_TH}+)</div>` : "";

        // 원본 컬럼 15개 모두 유지
        const cols = [
            { name: "종목", width: "180px", formatter: cell => gridjs.html(cell) },
            { name: "코드", hidden: true },
            { name: "이름", hidden: true },
            { name: "시장", width: "90px" },
            { name: "종가", width: "90px", formatter: v => num(v) },
            { name: "1일", width: "75px", formatter: v => gridjs.html(pct(v)) },
            { name: "20일", width: "75px", formatter: v => gridjs.html(pct(v)) },
            { name: "60일", width: "75px", formatter: v => gridjs.html(pct(v)) },
            { name: "RSI", width: "65px", formatter: v => v==null ? "-" : v.toFixed(0) },
            { name: "거래량x", width: "75px", formatter: v => v==null ? "-" : v.toFixed(2) },
            { name: "스윙", width: "65px", formatter: v => gridjs.html(scoreCell(v)) },
            { name: "추세", width: "65px", formatter: v => gridjs.html(scoreCell(v)) },
            { name: "사유", width: "250px", formatter: v => gridjs.html(`<div class="reason-cell">${v || ""}</div>`) },
            { name: "손절", width: "90px", formatter: v => num(v) },
            { name: "목표", width: "90px", formatter: v => num(v) },
        ];

        const data = rows.map(r => [
            `<a class="stock-link" href="https://finance.naver.com/item/main.naver?code=${r.code}" target="_blank">${r.name}</a><span class="stock-code">${r.code}</span>`,
            r.code, r.name, r.market, r.close, r.chg1d, r.chg20d, r.chg60d,
            r.rsi, r.vol_ratio, r.swing, r.trend,
            mode === "trend" ? r.trend_why : (r.swing_why || r.trend_why),
            r.stop, r.target,
        ]);

        if (grid) grid.destroy();
        grid = new gridjs.Grid({
            columns: cols,
            data,
            sort: true,
            pagination: { limit: 20 },
            search: true,
            resizable: true,
            language: { search: { placeholder: "🔍 검색..." } },
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
        print("  ⚠️ 시장 약세 신호. 신규 진입 자제 권고. 보유 종목 위주로 관리하세요.\n")

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

    print(f"\n✓ 분석 완료: {len(results)}개\n")

    trend = sorted([r for r in results if r["trend"] >= SCORE_THRESHOLD],
                   key=lambda x: -x["trend"])[:10]
    swing = sorted([r for r in results if r["swing"] >= SCORE_THRESHOLD],
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
        backup_path = OUTPUT_DIR / "dashboard.html"
        backup_path.write_text(html, encoding="utf-8")
        print(f"   백업 위치 → {backup_path.resolve()}")
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
            md.append("> ⚠️ **신규 진입 자제 권고.** 백테스트상 약세장 진입은 손실 확률↑.\n"
                      "> 보유 종목은 손절선 관리에 집중. 아래 신호는 참고용.\n")

    md.append(f"분석 종목 수: **{len(results)}개** · 점수 {SCORE_THRESHOLD}+ 만 표시\n")

    md.append("\n## 📈 장기 추세 TOP 10")
    md.append("*정배열 + MACD 매수 + 적정 RSI + 60일 모멘텀 (백테스트 알파 검증)*\n")
    if trend:
        md.append("| # | 점수 | 종목 | 현재가 | 손절 | 목표 | 사유 |")
        md.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(trend, 1):
            link = f"[{r['name']}](https://finance.naver.com/item/main.naver?code={r['code']})"
            md.append(f"| {i} | **{r['trend']}** | {link} ({r['code']}) | "
                      f"₩{r['close']:,.0f} | ₩{r['stop']:,.0f} | ₩{r['target']:,.0f} | {r['trend_why']} |")
    else:
        md.append(f"_점수 {SCORE_THRESHOLD} 이상 후보 없음. 오늘은 진입할 종목이 없습니다._\n")

    if ENABLE_SWING:
        md.append("\n## 🎯 단기 스윙 TOP 10 (참고용)")
        md.append("*RSI 과매도 반등 + 볼린저 하단. 백테스트상 알파 약함 — 정보용으로만 활용*\n")
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

    print("\n📈 장기 추세 TOP 10 (점수 80+)")
    if trend:
        for r in trend:
            print(f"  [{r['trend']:3d}] {r['name']:<10s} ({r['code']}) "
                  f"₩{r['close']:>9,.0f} → 손절 ₩{r['stop']:>9,.0f} | {r['trend_why']}")
    else:
        print(f"  (점수 {SCORE_THRESHOLD} 이상 후보 없음 — 오늘은 진입 종목 없음)")

    if ENABLE_SWING:
        print("\n🎯 단기 스윙 TOP 10 (참고용)")
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
