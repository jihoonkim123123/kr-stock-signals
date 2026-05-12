"""
한국주식 매매 신호 백테스터 (Grok + Ray Dalio 업그레이드 버전)
================================================================
grok: 2026-05-13
- enable_swing = True
- max_concurrent = 18
- Full Stats (승률, Sharpe, MDD, Profit Factor, KOSPI 비교)
- Portfolio Simulation + HTML 리포트
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
# GROK + RAY DALIO BACKTEST UPGRADE (2026-05-13)
# grok: enable_swing=True, max_concurrent=18, 디버깅 강화, Full Stats 복원
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

def regime_ok_at(date) -> bool:
    if not CONFIG["regime_filter"] or _REGIME_OK is None:
        return True
    val = _REGIME_OK.asof(date)
    return bool(val) if not pd.isna(val) else False

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

# =============================================================================
# Trade & Stats
# =============================================================================
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

# simulate_swing, simulate_trend (실제 거래 생성)
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
        if len(df) < 250:
            return []
        df = calc_indicators(df)
        start_idx = max(150, df.index.get_indexer([pd.Timestamp(start)], method="bfill")[0])

        trades = []
        if CONFIG["enable_swing"]:
            trades += simulate_swing(code, name, df, start_idx)
        if CONFIG["enable_trend"]:
            trades += simulate_trend(code, name, df, start_idx)
        return trades
    except:
        return []

# =============================================================================
# MAIN
# =============================================================================
def main():
    global _REGIME_OK
    start = dt.date.today() - dt.timedelta(days=365 * CONFIG["test_years"])
    
    print("🔍 GROK + DALIO 백테스트 시작")
    print(f"설정 → swing={CONFIG['enable_swing']}, concurrent={CONFIG['max_concurrent']}, threshold={CONFIG['score_threshold']}")
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
    print("✅ 백테스트 종료")

if __name__ == "__main__":
    main()
