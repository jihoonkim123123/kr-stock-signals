"""
한국주식 매매 신호 백테스터
============================

`kr_stock_signals.py` 와 동일한 점수 체계(단기 스윙 / 장기 추세추종)를 과거 데이터에
적용해서 실제로 돈을 벌었을지 검증합니다.

기본 설정 (CONFIG 섹션에서 수정 가능):
- 테스트 기간: 최근 3년
- 유니버스: KOSPI200
- 진입 기준: 점수 70점 이상
- 단기 스윙: 진입 후 +3×ATR 익절 / -2×ATR 손절 / 최대 10거래일 보유
- 장기 추세: 진입 후 MA20 < MA60 추세파괴 시 청산 / -8% 트레일링 스톱
- 거래 비용: 매수·매도 0.015% + 매도 시 거래세 0.18%, 슬리피지 0.3%

실행:
    pip install pykrx FinanceDataReader pandas numpy
    python kr_stock_backtest.py

출력:
- 콘솔에 단기/장기 전략별 통계 (승률, 평균수익률, 프로핏팩터, MDD, 샤프 등)
- backtest_report.html — 자산곡선·월별 수익률·거래 리스트가 담긴 리포트
- backtest_trades.csv — 모든 시뮬레이션 거래 내역
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "test_years": 3,
    "universe": "KOSPI200",          # "KOSPI200" / "KOSDAQ150" / "BOTH"
    "score_threshold": 70,
    "swing_target_atr": 3.0,
    "swing_stop_atr": 2.0,
    "swing_max_hold": 10,
    "trend_trail_pct": 8.0,
    "commission_buy": 0.00015,       # 0.015%
    "commission_sell": 0.00015,
    "tax_sell": 0.0018,              # 거래세 0.18%
    "slippage": 0.003,               # 0.3% 양방향
    "workers": 8,
    "max_concurrent_per_stock": 1,   # 종목당 동시 1포지션
}


# =============================================================================
# 유니버스 + 시세
# =============================================================================

def get_universe(which: str) -> list[tuple[str, str]]:
    """KOSPI200 / KOSDAQ150 근사 (각 시장 시가총액 상위).

    pykrx 의 정식 KOSPI200 API 가 KRX 로그인을 요구해 사용 불가. 대신
    FinanceDataReader 의 전체 상장 목록을 받아 시총 상위 200/150 으로 근사한다.
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

    if which == "KOSPI200":
        df = kospi
    elif which == "KOSDAQ150":
        df = kosdaq
    else:  # BOTH
        df = pd.concat([kospi, kosdaq]).drop_duplicates("Code")

    return [(r["Code"], r["Name"]) for _, r in df.iterrows()]


# =============================================================================
# 지표 (signals 스크립트와 동일)
# =============================================================================

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in (5, 20, 60, 120):
        df[f"MA{n}"] = df["Close"].rolling(n).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    e12 = df["Close"].ewm(span=12, adjust=False).mean()
    e26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_h"] = df["MACD"] - df["MACD_sig"]

    bm = df["Close"].rolling(20).mean()
    bs = df["Close"].rolling(20).std()
    df["BB_pct"] = (df["Close"] - (bm - 2 * bs)) / ((bm + 2 * bs) - (bm - 2 * bs)).replace(0, np.nan)
    df["VR"] = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)

    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift()).abs()
    l_c = (df["Low"] - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()

    df["P60"] = (df["Close"] / df["Close"].shift(60) - 1) * 100
    df["RSIp"] = df["RSI"].shift(1)
    df["Hp"] = df["MACD_h"].shift(1)
    return df


def score_swing(r: pd.Series) -> int:
    if pd.isna(r["RSI"]) or pd.isna(r["BB_pct"]) or pd.isna(r["RSIp"]):
        return 0
    s = 0
    if 30 <= r["RSI"] <= 45 and r["RSI"] > r["RSIp"]: s += 30
    elif r["RSI"] < 30: s += 22
    if r["BB_pct"] < 0.2: s += 22
    elif r["BB_pct"] < 0.4: s += 10
    if not pd.isna(r["VR"]):
        if r["VR"] >= 2: s += 18
        elif r["VR"] >= 1.5: s += 10
    if r["Close"] > r["Open"] and not pd.isna(r["MA5"]) and r["Close"] >= r["MA5"]:
        s += 15
    if not pd.isna(r["Hp"]) and r["MACD_h"] > r["Hp"] and r["Hp"] < 0:
        s += 10
    if not pd.isna(r["P60"]) and r["P60"] < -25:
        s -= 15
    return max(0, min(100, s))


def score_trend(r: pd.Series) -> int:
    needed = ["MA5", "MA20", "MA60", "MA120", "RSI", "MACD", "MACD_sig"]
    if any(pd.isna(r[c]) for c in needed):
        return 0
    s = 0
    if r["MA5"] > r["MA20"] > r["MA60"] > r["MA120"]: s += 35
    elif r["MA20"] > r["MA60"] > r["MA120"]: s += 22
    if r["Close"] > r["MA20"]: s += 12
    if r["MACD"] > 0 and r["MACD"] > r["MACD_sig"]: s += 18
    elif r["MACD"] > r["MACD_sig"]: s += 8
    if 50 <= r["RSI"] <= 70: s += 15
    elif 70 < r["RSI"] <= 80: s += 5
    elif r["RSI"] > 80: s -= 10
    if not pd.isna(r["P60"]) and r["P60"] > 0:
        s += min(15, int(r["P60"] / 2))
    return max(0, min(100, s))


# =============================================================================
# 트레이드 시뮬레이션
# =============================================================================

@dataclass
class Trade:
    code: str
    name: str
    strategy: str            # "swing" / "trend"
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    days_held: int
    return_pct: float        # 거래비용 반영 후
    exit_reason: str
    score_at_entry: int


def apply_costs(entry: float, exit_: float) -> float:
    """거래비용 반영 후 수익률(%)."""
    eff_entry = entry * (1 + CONFIG["slippage"] / 2) * (1 + CONFIG["commission_buy"])
    eff_exit = exit_ * (1 - CONFIG["slippage"] / 2) * (1 - CONFIG["commission_sell"] - CONFIG["tax_sell"])
    return (eff_exit / eff_entry - 1) * 100


def simulate_swing(code: str, name: str, df: pd.DataFrame, start_idx: int) -> list[Trade]:
    trades, in_pos = [], None
    th = CONFIG["score_threshold"]

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        if in_pos is not None:
            entry_price, entry_date, stop, target, score, days = in_pos
            days += 1
            exit_price, reason = None, None

            # 익절 — 다음날 고가가 목표가 이상이면 목표가에 체결 가정
            if nxt["High"] >= target:
                exit_price, reason = target, "target"
            elif nxt["Low"] <= stop:
                exit_price, reason = stop, "stop"
            elif days >= CONFIG["swing_max_hold"]:
                exit_price, reason = float(nxt["Close"]), "max_hold"

            if exit_price is not None:
                trades.append(Trade(
                    code=code, name=name, strategy="swing",
                    entry_date=entry_date, entry_price=entry_price,
                    exit_date=nxt.name, exit_price=exit_price,
                    days_held=days,
                    return_pct=apply_costs(entry_price, exit_price),
                    exit_reason=reason, score_at_entry=score,
                ))
                in_pos = None
            else:
                in_pos = (entry_price, entry_date, stop, target, score, days)

        if in_pos is None:
            score = score_swing(row)
            if score >= th and not pd.isna(row["ATR"]):
                entry_price = float(nxt["Open"])
                atr = float(row["ATR"])
                stop = entry_price - CONFIG["swing_stop_atr"] * atr
                target = entry_price + CONFIG["swing_target_atr"] * atr
                in_pos = (entry_price, nxt.name, stop, target, score, 0)

    return trades


def simulate_trend(code: str, name: str, df: pd.DataFrame, start_idx: int) -> list[Trade]:
    trades, in_pos = [], None
    th = CONFIG["score_threshold"]
    trail = CONFIG["trend_trail_pct"] / 100

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        if in_pos is not None:
            entry_price, entry_date, score, days, peak = in_pos
            days += 1
            peak = max(peak, float(row["Close"]))
            exit_price, reason = None, None

            # 추세 파괴: MA20 < MA60
            if not pd.isna(row["MA20"]) and not pd.isna(row["MA60"]) and row["MA20"] < row["MA60"]:
                exit_price, reason = float(nxt["Open"]), "trend_break"
            # 트레일링 스톱: 고점 대비 -8%
            elif nxt["Low"] <= peak * (1 - trail):
                exit_price, reason = peak * (1 - trail), "trail_stop"

            if exit_price is not None:
                trades.append(Trade(
                    code=code, name=name, strategy="trend",
                    entry_date=entry_date, entry_price=entry_price,
                    exit_date=nxt.name, exit_price=exit_price,
                    days_held=days,
                    return_pct=apply_costs(entry_price, exit_price),
                    exit_reason=reason, score_at_entry=score,
                ))
                in_pos = None
            else:
                in_pos = (entry_price, entry_date, score, days, peak)

        if in_pos is None:
            score = score_trend(row)
            if score >= th:
                entry_price = float(nxt["Open"])
                in_pos = (entry_price, nxt.name, score, 0, entry_price)

    return trades


def backtest_one(code: str, name: str, start: dt.date) -> list[Trade]:
    import FinanceDataReader as fdr
    try:
        # 워밍업 위해 추가 1년치 받기
        df = fdr.DataReader(code, start - dt.timedelta(days=400), dt.date.today())
        if len(df) < 200:
            return []
        df = calc_indicators(df)
        # 워밍업 끝나는 인덱스 (start_date 이후 첫 행)
        start_idx = df.index.get_indexer([pd.Timestamp(start)], method="bfill")[0]
        if start_idx < 130:
            start_idx = 130
        return simulate_swing(code, name, df, start_idx) + simulate_trend(code, name, df, start_idx)
    except Exception:
        return []


# =============================================================================
# 통계 + 자산곡선
# =============================================================================

def stats(trades: list[Trade], strategy: str) -> dict:
    rs = [t for t in trades if t.strategy == strategy]
    if not rs:
        return {"strategy": strategy, "n": 0}
    rets = np.array([t.return_pct for t in rs])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    days = np.array([t.days_held for t in rs])

    n_win, n_loss = len(wins), len(losses)
    win_rate = n_win / len(rs) * 100
    avg_w = wins.mean() if n_win else 0
    avg_l = losses.mean() if n_loss else 0
    profit_factor = (wins.sum() / -losses.sum()) if (n_loss and losses.sum() < 0) else np.inf
    expectancy = rets.mean()

    # 자산곡선 (시간순)
    sorted_trades = sorted(rs, key=lambda t: t.exit_date)
    equity = [10000.0]
    for t in sorted_trades:
        equity.append(equity[-1] * (1 + t.return_pct / 100))
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    mdd = dd.min() * 100 if len(dd) else 0

    # 거래당 표준편차 → 거래 단위 샤프
    sharpe = (rets.mean() / rets.std() * np.sqrt(252 / max(days.mean(), 1))) if rets.std() > 0 else 0

    return {
        "strategy": strategy,
        "n": len(rs),
        "win_rate": win_rate,
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "total_return": (equity[-1] / equity[0] - 1) * 100,
        "mdd": mdd,
        "sharpe": sharpe,
        "avg_days": days.mean(),
        "best": rets.max(),
        "worst": rets.min(),
        "equity": equity.tolist(),
        "exit_dates": [str(t.exit_date.date()) for t in sorted_trades],
    }


def benchmark_kospi(start: dt.date) -> dict:
    """KOSPI 지수 매수보유 — 같은 기간."""
    import FinanceDataReader as fdr
    df = fdr.DataReader("KS11", start, dt.date.today())
    ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    return {"label": "KOSPI 매수보유", "return_pct": float(ret),
            "equity": (10000 * df["Close"] / df["Close"].iloc[0]).tolist(),
            "dates": [str(d.date()) for d in df.index]}


# =============================================================================
# HTML 리포트
# =============================================================================

REPORT = r"""<!doctype html>
<html lang="ko"><meta charset="utf-8">
<title>백테스트 리포트 — __DATE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://cdn.jsdelivr.net/npm/gridjs/dist/theme/mermaid.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/gridjs/dist/gridjs.umd.js"></script>
<style>
 :root { color-scheme: light dark; }
 body { font-family: -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
        margin: 24px; max-width: 1300px; }
 h1 { margin: 0 0 6px; font-size: 22px; }
 h2 { margin: 28px 0 10px; font-size: 18px; }
 .sub { color:#888; font-size:13px; margin-bottom: 18px; }
 .grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
 .card { padding:16px; border:1px solid rgba(127,127,127,.25); border-radius:10px; }
 table.kpi { width:100%; font-size: 13px; border-collapse: collapse; }
 table.kpi td { padding:6px 0; }
 table.kpi td:first-child { color:#888; }
 table.kpi td:last-child { text-align:right; font-weight:600; }
 .pos { color:#d0342c; } .neg { color:#1862ce; }
 canvas { max-height: 300px; }
</style>
<body>
<h1>📊 한국주식 매매 신호 백테스트</h1>
<div class="sub">기간 <b>__START__ ~ __END__</b> · 유니버스 <b>__UNIV__</b> · 진입 점수 <b>__TH__점</b> 이상</div>

<div class="grid">
 <div class="card">
   <h2>🎯 단기 스윙</h2>
   <table class="kpi" id="swingKpi"></table>
 </div>
 <div class="card">
   <h2>📈 장기 추세추종</h2>
   <table class="kpi" id="trendKpi"></table>
 </div>
</div>

<h2>자산곡선 (₩10,000 시작 기준)</h2>
<div class="card"><canvas id="eq"></canvas></div>

<h2>전체 거래 내역</h2>
<div id="grid"></div>

<script>
 const SWING = __SWING__;
 const TREND = __TREND__;
 const BENCH = __BENCH__;
 const TRADES = __TRADES__;

 const fmtPct = v => (v == null ? "-" :
   `<span class="${v >= 0 ? 'pos' : 'neg'}">${v >= 0 ? '+' : ''}${v.toFixed(2)}%</span>`);
 const fmtNum = (v, d=2) => v == null ? "-" : Number(v).toFixed(d);

 function fillKpi(elId, s) {
   const el = document.getElementById(elId);
   if (!s.n) { el.innerHTML = "<tr><td colspan=2>거래 없음</td></tr>"; return; }
   const rows = [
     ["거래 횟수", s.n],
     ["승률", s.win_rate.toFixed(1) + "%"],
     ["기대값/거래", fmtPct(s.expectancy)],
     ["평균 익절", fmtPct(s.avg_win)],
     ["평균 손절", fmtPct(s.avg_loss)],
     ["프로핏 팩터", isFinite(s.profit_factor) ? fmtNum(s.profit_factor) : "∞"],
     ["누적 수익률", fmtPct(s.total_return)],
     ["최대 낙폭(MDD)", fmtPct(s.mdd)],
     ["샤프 (대략)", fmtNum(s.sharpe)],
     ["평균 보유일", fmtNum(s.avg_days, 1)],
     ["최고/최악 거래", fmtPct(s.best) + " / " + fmtPct(s.worst)],
   ];
   el.innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("");
 }
 fillKpi("swingKpi", SWING);
 fillKpi("trendKpi", TREND);

 // 자산곡선 — 거래 기준
 const buildLine = (s) => s.exit_dates.length ?
   s.exit_dates.map((d, i) => ({ x: d, y: s.equity[i+1] })) : [];
 new Chart(document.getElementById("eq"), {
   type: "line",
   data: {
     datasets: [
       { label: "단기 스윙", data: buildLine(SWING), borderColor: "#d0342c",
         backgroundColor: "transparent", tension: 0.2, pointRadius: 0 },
       { label: "장기 추세", data: buildLine(TREND), borderColor: "#1862ce",
         backgroundColor: "transparent", tension: 0.2, pointRadius: 0 },
       { label: BENCH.label,
         data: BENCH.dates.map((d, i) => ({ x: d, y: BENCH.equity[i] })),
         borderColor: "#888", borderDash: [4, 4], backgroundColor: "transparent",
         tension: 0.1, pointRadius: 0 },
     ],
   },
   options: {
     responsive: true,
     scales: { x: { type: "category" } },
   }
 });

 new gridjs.Grid({
   columns: ["전략", "종목", "코드", "진입일", "진입가", "청산일", "청산가",
             {name:"수익률", formatter: v => gridjs.html(fmtPct(v))},
             "보유일", "사유", "점수"],
   data: TRADES,
   sort: true, search: true, pagination: { limit: 25 },
   resizable: true,
   style: { table: { "white-space": "nowrap", "font-size": "12px" } },
 }).render(document.getElementById("grid"));
</script>
</body></html>
"""


# =============================================================================
# MAIN
# =============================================================================

def main():
    start = dt.date.today() - dt.timedelta(days=365 * CONFIG["test_years"])
    print(f"\n🔍 백테스트 시작")
    print(f"  기간      : {start} ~ {dt.date.today()}")
    print(f"  유니버스  : {CONFIG['universe']}")
    print(f"  진입 기준 : 점수 ≥ {CONFIG['score_threshold']}\n")

    universe = get_universe(CONFIG["universe"])
    print(f"📋 {len(universe)}개 종목 시뮬레이션 중...\n")

    all_trades: list[Trade] = []
    with ThreadPoolExecutor(max_workers=CONFIG["workers"]) as ex:
        futs = {ex.submit(backtest_one, c, n, start): c for c, n in universe}
        done = 0
        for f in as_completed(futs):
            all_trades.extend(f.result())
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(universe)} 완료 (누적 거래 {len(all_trades)})")

    print(f"\n✓ 시뮬레이션 완료: 총 {len(all_trades)}개 거래\n")

    swing = stats(all_trades, "swing")
    trend = stats(all_trades, "trend")
    bench = benchmark_kospi(start)

    def show(s, label):
        if not s.get("n"):
            print(f"  {label}: 거래 없음")
            return
        print(f"\n📊 {label}")
        print(f"  거래 수      : {s['n']}")
        print(f"  승률         : {s['win_rate']:.1f}%")
        print(f"  기대값/거래  : {s['expectancy']:+.2f}%")
        print(f"  평균 익절/손절: {s['avg_win']:+.2f}% / {s['avg_loss']:+.2f}%")
        pf = "∞" if not np.isfinite(s["profit_factor"]) else f"{s['profit_factor']:.2f}"
        print(f"  프로핏 팩터  : {pf}")
        print(f"  누적 수익률  : {s['total_return']:+.1f}%")
        print(f"  최대 낙폭    : {s['mdd']:.1f}%")
        print(f"  샤프 (대략)  : {s['sharpe']:.2f}")
        print(f"  평균 보유일  : {s['avg_days']:.1f}일")

    show(swing, "단기 스윙 결과")
    show(trend, "장기 추세 결과")
    print(f"\n📌 같은 기간 KOSPI 매수보유: {bench['return_pct']:+.1f}%")

    # CSV
    csv_path = OUTPUT_DIR / "backtest_trades.csv"
    pd.DataFrame([t.__dict__ for t in all_trades]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 거래 내역 → {csv_path}")

    # HTML
    trades_for_html = [
        [t.strategy, t.name, t.code,
         str(t.entry_date.date()), round(t.entry_price),
         str(t.exit_date.date()), round(t.exit_price),
         t.return_pct, t.days_held, t.exit_reason, t.score_at_entry]
        for t in sorted(all_trades, key=lambda x: x.exit_date, reverse=True)
    ]
    html = (REPORT
            .replace("__DATE__", str(dt.date.today()))
            .replace("__START__", str(start))
            .replace("__END__", str(dt.date.today()))
            .replace("__UNIV__", CONFIG["universe"])
            .replace("__TH__", str(CONFIG["score_threshold"]))
            .replace("__SWING__", json.dumps(swing, ensure_ascii=False, default=str))
            .replace("__TREND__", json.dumps(trend, ensure_ascii=False, default=str))
            .replace("__BENCH__", json.dumps(bench, ensure_ascii=False))
            .replace("__TRADES__", json.dumps(trades_for_html, ensure_ascii=False)))
    html_path = OUTPUT_DIR / "backtest_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"📄 리포트 → {html_path}")
    print(f"\n👉 브라우저로 backtest_report.html 을 열어 자산곡선과 거래 분석을 확인하세요.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(1)
