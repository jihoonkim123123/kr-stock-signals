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


# ---------------------------------------------------------------------------
# 1) 종목 유니버스: 코스피200 + 코스닥150
# ---------------------------------------------------------------------------

def get_universe() -> pd.DataFrame:
    """KOSPI200 + KOSDAQ150 구성 종목."""
    from pykrx import stock

    today = dt.date.today().strftime("%Y%m%d")
    # 휴장일 대비: 최근 영업일 찾기 (최대 10일 전까지)
    for back in range(0, 10):
        d = (dt.date.today() - dt.timedelta(days=back)).strftime("%Y%m%d")
        try:
            kospi200 = stock.get_index_portfolio_deposit_file("1028", d)
            kosdaq150 = stock.get_index_portfolio_deposit_file("2203", d)
            if kospi200 and kosdaq150:
                break
        except Exception:
            continue
    else:
        raise RuntimeError("종목 리스트를 가져오지 못했어요. 인터넷 연결을 확인하세요.")

    rows = []
    for code in kospi200:
        rows.append({"code": code, "name": stock.get_market_ticker_name(code), "market": "KOSPI200"})
    for code in kosdaq150:
        if code in kospi200:
            continue
        rows.append({"code": code, "name": stock.get_market_ticker_name(code), "market": "KOSDAQ150"})
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
 .gridjs-table { font-size: 13px; }
</style>
<body>
<h1>📊 한국주식 매매 신호 대시보드</h1>
<div class="sub">기준일 <b>__DATE__</b> · 종목 수 <b id="cnt"></b> · 데이터 출처 KRX/네이버</div>

<div class="legend">
 <b>단기 스윙</b>: RSI 과매도 반등 + 볼린저 하단 + 거래량 급증 + 양봉을 종합 점수화. 70점 이상이면 진입 검토 구간.<br>
 <b>장기 추세</b>: 정배열 + MACD 매수 + 적정 RSI + 60일 모멘텀. 70점 이상이면 추세추종 매수 후보.<br>
 <b>진입가/손절가</b>는 종가 기준이며 손절은 -2×ATR, 목표는 +3×ATR (대략적인 가이드).
 마지막 주문은 직접 HTS/MTS에서 본인이 넣으세요.
</div>

<div class="tabs">
 <div class="tab on" data-mode="swing">🎯 단기 스윙 TOP</div>
 <div class="tab" data-mode="trend">📈 장기 추세 TOP</div>
 <div class="tab" data-mode="all">전체 정렬</div>
</div>

<div id="grid"></div>

<script>
 const DATA = __DATA__;
 document.getElementById("cnt").textContent = DATA.length;

 const num = (v, d=0) => v == null ? "-" : Number(v).toLocaleString("ko-KR",
              {minimumFractionDigits: d, maximumFractionDigits: d});
 const pct = (v) => v == null ? "-" :
   `<span class="${v>=0?'pos':'neg'}">${v>=0?'+':''}${v.toFixed(1)}%</span>`;
 const scoreCell = (v) => {
   const cls = v >= 70 ? "score-hi" : v >= 50 ? "score-md" : "score-lo";
   return `<span class="pill ${cls}">${v}</span>`;
 };
 const link = (code, name) =>
   `<a href="https://finance.naver.com/item/main.naver?code=${code}" target="_blank">${name}</a>`;

 let grid = null;
 function render(mode) {
   let rows = [...DATA];
   if (mode === "swing") {
     rows = rows.filter(r => r.swing > 0).sort((a,b) => b.swing - a.swing);
   } else if (mode === "trend") {
     rows = rows.filter(r => r.trend > 0).sort((a,b) => b.trend - a.trend);
   } else {
     rows.sort((a,b) => Math.max(b.swing,b.trend) - Math.max(a.swing,a.trend));
   }

   const cols = [
     { name: "종목", formatter: (_, r) =>
        gridjs.html(`${link(r.cells[0].data, r.cells[1].data)} <span style="color:#888">${r.cells[0].data}</span>`),
        sort: false },
     { name: "code", hidden: true },
     { name: "이름", hidden: true },
     { name: "시장" },
     { name: "종가", formatter: v => num(v) },
     { name: "1일", formatter: v => gridjs.html(pct(v)) },
     { name: "20일", formatter: v => gridjs.html(pct(v)) },
     { name: "60일", formatter: v => gridjs.html(pct(v)) },
     { name: "RSI", formatter: v => v==null ? "-" : v.toFixed(0) },
     { name: "거래량x", formatter: v => v==null ? "-" : v.toFixed(2) },
     { name: "스윙", formatter: v => gridjs.html(scoreCell(v)) },
     { name: "추세", formatter: v => gridjs.html(scoreCell(v)) },
     { name: "사유", width: "260px" },
     { name: "손절", formatter: v => num(v) },
     { name: "목표", formatter: v => num(v) },
   ];

   const data = rows.map(r => [
     r.code, r.code, r.name, r.market, r.close, r.chg1d, r.chg20d, r.chg60d,
     r.rsi, r.vol_ratio, r.swing, r.trend,
     mode === "trend" ? r.trend_why : (r.swing_why || r.trend_why),
     r.stop, r.target,
   ]);

   if (grid) grid.destroy();
   grid = new gridjs.Grid({
     columns: cols,
     data,
     sort: true,
     pagination: { limit: 30 },
     search: true,
     resizable: true,
     style: { table: { "white-space": "nowrap" } },
   }).render(document.getElementById("grid"));
 }

 document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
   document.querySelectorAll(".tab").forEach(x => x.classList.remove("on"));
   t.classList.add("on");
   render(t.dataset.mode);
 }));

 render("swing");
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# 6) 메인
# ---------------------------------------------------------------------------

def main():
    print("📋 종목 리스트 조회 중...")
    universe = get_universe()
    print(f"  코스피200 + 코스닥150 = {len(universe)}개 종목\n")

    print("📊 시세 분석 중 (5~10분 소요)...")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(analyze_one, r.code, r["name"], r.market): r.code
                for r in universe.itertuples(index=False)}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(universe)} 처리됨")

    print(f"\n✓ 분석 완료: {len(results)}개\n")

    today = dt.date.today().strftime("%Y-%m-%d")
    html = (HTML
            .replace("__DATE__", today)
            .replace("__DATA__", json.dumps(results, ensure_ascii=False)))

    out = OUTPUT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ 대시보드 생성 완료 → {out}")

    # 콘솔에도 TOP 10 요약
    swing = sorted([r for r in results if r["swing"] > 0],
                   key=lambda x: -x["swing"])[:10]
    trend = sorted([r for r in results if r["trend"] > 0],
                   key=lambda x: -x["trend"])[:10]

    # GitHub Issue 등에서 쓸 마크다운 요약 (top.md)
    import os as _os
    pages_url = _os.environ.get("PAGES_URL", "")
    md = [f"# 📊 한국주식 매매 신호 — {today}\n",
          f"분석 종목 수: **{len(results)}개**\n"]
    md.append("\n## 🎯 단기 스윙 TOP 10\n")
    md.append("RSI 과매도 반등 + 볼린저 하단 + 거래량 급증 종합\n")
    md.append("| # | 점수 | 종목 | 현재가 | 손절 | 목표 | 사유 |")
    md.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(swing, 1):
        link = f"[{r['name']}](https://finance.naver.com/item/main.naver?code={r['code']})"
        md.append(f"| {i} | **{r['swing']}** | {link} ({r['code']}) | "
                  f"₩{r['close']:,.0f} | ₩{r['stop']:,.0f} | ₩{r['target']:,.0f} | {r['swing_why']} |")
    md.append("\n## 📈 장기 추세 TOP 10\n")
    md.append("정배열 + MACD 매수 + 적정 RSI + 60일 모멘텀\n")
    md.append("| # | 점수 | 종목 | 현재가 | 손절 | 목표 | 사유 |")
    md.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(trend, 1):
        link = f"[{r['name']}](https://finance.naver.com/item/main.naver?code={r['code']})"
        md.append(f"| {i} | **{r['trend']}** | {link} ({r['code']}) | "
                  f"₩{r['close']:,.0f} | ₩{r['stop']:,.0f} | ₩{r['target']:,.0f} | {r['trend_why']} |")
    md.append("\n---\n")
    md.append("⚠️ 알고리즘 신호일 뿐, 매매 결정과 책임은 본인에게 있습니다. 직접 HTS/MTS에서 주문하세요.")
    if pages_url:
        md.append(f"\n📊 [전체 대시보드 보기]({pages_url})")
    md_path = OUTPUT_DIR / "top.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"📝 마크다운 요약 → {md_path}")

    print("\n🎯 단기 스윙 TOP 10")
    for r in swing:
        print(f"  [{r['swing']:3d}] {r['name']:<10s} ({r['code']}) "
              f"₩{r['close']:>9,.0f} | {r['swing_why']}")

    print("\n📈 장기 추세 TOP 10")
    for r in trend:
        print(f"  [{r['trend']:3d}] {r['name']:<10s} ({r['code']}) "
              f"₩{r['close']:>9,.0f} | {r['trend_why']}")

    print("\n👉 dashboard.html 을 브라우저로 열어 확인하세요.\n"
          "   매매는 본인이 직접 HTS/MTS에서 진행해 주세요.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(1)
