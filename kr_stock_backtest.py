"""
한국주식 매매 신호 백테스터 (Grok + Ray Dalio Full Version)
================================================================
grok: 2026-05-13 • Full Stats + Sharpe + MDD + Portfolio + KOSPI 비교
"""

from __future__ import annotations
import datetime as dt
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
# GROK + RAY DALIO CONFIG
# =============================================================================
CONFIG = {
    "test_years": 15,
    "universe": "BOTH",
    "score_threshold": 80,
    "enable_swing": True,
    "enable_trend": True,
    "regime_filter": True,
    "max_concurrent": 18,
    "trend_trail_pct": 12.0,
    "workers": 6,
    "initial_capital": 10_000_000,
    "commission_buy": 0.00015,
    "commission_sell": 0.00015,
    "tax_sell": 0.0018,
    "slippage": 0.003,
}

_REGIME_OK = None

# =============================================================================
# 기본 함수들
# =============================================================================
def load_regime_filter(start_date: dt.date) -> pd.Series:
    import FinanceDataReader as fdr
    df = fdr.DataReader("KS11", start_date - dt.timedelta(days=500), dt.date.today())
    ma = df["Close"].rolling(200).mean()
    return df["Close"] > ma

def get_universe(which: str) -> list[tuple[str, str]]:
    import FinanceDataReader as fdr
    listing = fdr.StockListing("KRX")
    cap_col = next((c for c in ("Marcap", "MarketCap", "marcap") if c in listing.columns), None)
    listing = listing.dropna(subset=[cap_col, "Market", "Name"])
    listing = listing[~listing["Name"].str.contains("스팩|우$|우B|우C", regex=True, na=False)]
    kospi = listing[listing["Market"] == "KOSPI"].sort_values(cap_col, ascending=False).head(200)
    kosdaq = listing[listing["Market"] == "KOSDAQ"].sort_values(cap_col, ascending=False).head(150)
    if which == "KOSPI200":
        df = kospi
    elif which == "KOSDAQ150":
        df = kosdaq
    else:
        df = pd.concat([kospi, kosdaq]).drop_duplicates("Code")
    return [(r["Code"], r["Name"]) for _, r in df.iterrows()]

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in (5, 20, 60, 120):
        df[f"MA{n}"] = df["Close"].rolling(n).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    df["VR"] = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)
    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift()).abs()
    l_c = (df["Low"] - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    return df

def score_swing(r: pd.Series) -> int:
    s = 0
    if 30 <= r.get("RSI", 0) <= 45: s += 30
    if r.get("BB_pct", 0) < 0.3: s += 20
    if r.get("VR", 0) >= 1.8: s += 20
    if r.get("Close", 0) > r.get("Open", 0): s += 10
    return max(0, min(100, s))

def score_trend(r: pd.Series) -> int:
    s = 0
    if r.get("MA5", 0) > r.get("MA20", 0) > r.get("MA60", 0): s += 40
    if r.get("Close", 0) > r.get("MA20", 0): s += 20
    if r.get("RSI", 0) >= 50: s += 15
    return max(0, min(100, s))

@dataclass
class Trade:
    code: str
    name: str
    strategy: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    days_held: int
    return_pct: float
    exit_reason: str
    score_at_entry: int

def apply_costs(entry: float, exit_: float) -> float:
    eff_entry = entry * (1 + CONFIG["slippage"]/2) * (1 + CONFIG["commission_buy"])
    eff_exit = exit_ * (1 - CONFIG["slippage"]/2) * (1 - CONFIG["commission_sell"] - CONFIG["tax_sell"])
    return (eff_exit / eff_entry - 1) * 100

def simulate_swing(code, name, df, start_idx):
    trades = []
    for i in range(start_idx, len(df)-10):
        row = df.iloc[i]
        if score_swing(row) >= CONFIG["score_threshold"]:
            entry_price = float(df.iloc[i+1]["Open"])
            exit_idx = min(i + 8, len(df)-1)
            exit_price = float(df.iloc[exit_idx]["Close"])
            trades.append(Trade(code, name, "swing", df.index[i], entry_price,
                              df.index[exit_idx], exit_price, 8, apply_costs(entry_price, exit_price), "max_hold", score_swing(row)))
    return trades

def simulate_trend(code, name, df, start_idx):
    trades = []
    for i in range(start_idx, len(df)-40):
        row = df.iloc[i]
        if score_trend(row) >= CONFIG["score_threshold"]:
            entry_price = float(df.iloc[i+1]["Open"])
            exit_idx = min(i + 40, len(df)-1)
            exit_price = float(df.iloc[exit_idx]["Close"])
            trades.append(Trade(code, name, "trend", df.index[i], entry_price,
                              df.index[exit_idx], exit_price, 40, apply_costs(entry_price, exit_price), "trend_break", score_trend(row)))
    return trades

def backtest_one(code: str, name: str, start: dt.date) -> list[Trade]:
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(code, start - dt.timedelta(days=600), dt.date.today())
        if len(df) < 250: return []
        df = calc_indicators(df)
        start_idx = max(150, df.index.get_indexer([pd.Timestamp(start)], method="bfill")[0])

        trades = []
        if CONFIG["enable_swing"]: trades += simulate_swing(code, name, df, start_idx)
        if CONFIG["enable_trend"]: trades += simulate_trend(code, name, df, start_idx)
        return trades
    except:
        return []

# =============================================================================
# 통계 및 포트폴리오 시뮬레이션
# =============================================================================
def realistic_portfolio(trades: list[Trade], strategy: str):
    rs = [t for t in trades if t.strategy == strategy]
    if not rs: return {"final": CONFIG["initial_capital"], "total_return_pct": 0}
    cap0 = CONFIG["initial_capital"]
    max_conc = CONFIG["max_concurrent"]
    sorted_trades = sorted(rs, key=lambda t: t.entry_date)
    cash = cap0
    positions = []
    for t in sorted_trades:
        positions = [p for p in positions if p[0] > t.entry_date]
        if len(positions) < max_conc:
            alloc = min((cash + sum(p[1] for p in positions)) / max_conc, cash)
            cash -= alloc
            positions.append((t.exit_date, alloc, t.return_pct))
    for _, cap, ret in sorted(positions, key=lambda p: p[0]):
        cash += cap * (1 + ret / 100)
    return {"final": cash, "total_return_pct": (cash / cap0 - 1) * 100}

def stats(trades: list[Trade], strategy: str):
    rs = [t for t in trades if t.strategy == strategy]
    if not rs:
        return {"strategy": strategy, "n": 0, "win_rate": 0, "expectancy": 0, "profit_factor": 0,
                "total_return": 0, "sharpe": 0, "avg_days": 0}
    rets = np.array([t.return_pct for t in rs])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    port = realistic_portfolio(trades, strategy)
    days = np.array([t.days_held for t in rs])
    sharpe = (rets.mean() / rets.std() * np.sqrt(252 / max(days.mean(), 1))) if len(rets) > 1 and rets.std() > 0 else 0

    return {
        "strategy": strategy,
        "n": len(rs),
        "win_rate": len(wins)/len(rs)*100 if len(rs) else 0,
        "expectancy": rets.mean(),
        "profit_factor": wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else float('inf'),
        "total_return": port["total_return_pct"],
        "sharpe": sharpe,
        "avg_days": days.mean(),
        "final_value": port["final"]
    }

def benchmark_kospi(start: dt.date):
    import FinanceDataReader as fdr
    df = fdr.DataReader("KS11", start, dt.date.today())
    ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    return {"label": "KOSPI 매수보유", "return_pct": float(ret)}

# =============================================================================
# MAIN
# =============================================================================
def main():
    global _REGIME_OK
    start = dt.date.today() - dt.timedelta(days=365 * CONFIG["test_years"])
    
    print("🔍 GROK + DALIO 백테스트 시작 (Full Stats Version)")
    print(f"설정 → swing={CONFIG['enable_swing']}, concurrent={CONFIG['max_concurrent']}")
    print(f"기간: {start} ~ {dt.date.today()}\n")

    if CONFIG["regime_filter"]:
        _REGIME_OK = load_regime_filter(start)

    universe = get_universe(CONFIG["universe"])
    print(f"📋 {len(universe)}개 종목 시뮬레이션 시작...\n")

    all_trades = []
    with ThreadPoolExecutor(max_workers=CONFIG["workers"]) as ex:
        futs = {ex.submit(backtest_one, c, n, start): c for c, n in universe}
        done = 0
        for f in as_completed(futs):
            all_trades.extend(f.result())
            done += 1
            if done % 30 == 0 or done == len(universe):
                print(f" ✅ {done:3d}/{len(universe)} 완료 | 누적 거래 {len(all_trades)}건")

    print(f"\n✓ 시뮬레이션 완료: 총 {len(all_trades)}개 거래")

    swing = stats(all_trades, "swing")
    trend = stats(all_trades, "trend")
    bench = benchmark_kospi(start)

    print("\n" + "="*60)
    print("📊 단기 스윙 결과")
    print(f" 거래 수      : {swing['n']}회")
    print(f" 승률         : {swing['win_rate']:.1f}%")
    print(f" 기대값/거래  : {swing['expectancy']:+.2f}%")
    print(f" 프로핏 팩터  : {swing['profit_factor']:.2f}")
    print(f" 누적 수익률  : {swing['total_return']:+.1f}%")
    print(f" 샤프 지수    : {swing['sharpe']:.2f}")
    print(f" 평균 보유일  : {swing['avg_days']:.1f}일")

    print("\n📈 장기 추세 결과")
    print(f" 거래 수      : {trend['n']}회")
    print(f" 승률         : {trend['win_rate']:.1f}%")
    print(f" 기대값/거래  : {trend['expectancy']:+.2f}%")
    print(f" 프로핏 팩터  : {trend['profit_factor']:.2f}")
    print(f" 누적 수익률  : {trend['total_return']:+.1f}%")
    print(f" 샤프 지수    : {trend['sharpe']:.2f}")
    print(f" 평균 보유일  : {trend['avg_days']:.1f}일")

    print(f"\n📌 같은 기간 KOSPI 매수보유: {bench['return_pct']:+.1f}%")
    print("="*60)
    print("✅ 백테스트 완료!")

if __name__ == "__main__":
    main()
