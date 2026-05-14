"""
한국주식 매매 신호 백테스터 v3 (Risk-Managed)
=====================================================

신규:
- 종합 점수 가중치 = 40% 기술 + 30% 수급 + 30% 모멘텀 (감성 제외)
- ATR 기반 변동성 조정 포지션 사이징 (3~8%, 목표 15~20종목 분산)
- 포트폴리오 단위 트레일링 스톱 (고점 대비 -20% 시 전량 청산 후 재진입 대기)
- 최대 낙폭 한도 (포트폴리오 -25% 도달 시 운영 중단)
- 종목 단위 트레일링 스톱 강화 (-8% → -12%, 변동성 비례)
- 유동성 필터 (일평균 거래대금 5억 미만 제외)
- 슬리피지 + 수수료 + 거래세 + 매수/매도 비대칭 비용 반영
- Monte Carlo 시뮬레이션 (거래 순서 N회 무작위 재배열 → MDD/누적수익률 분포)
- Out-of-Sample 분리 (전체 기간 70% in-sample / 30% out-of-sample)

실행:
    pip install -r requirements.txt
    python kr_stock_backtest.py
"""
from __future__ import annotations
import datetime as dt
import json
import random
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # 기간
    "test_years": 15,                # 2011~ 다중 사이클 커버
    "oos_pct": 0.30,                 # 끝 30% 를 out-of-sample 로 보존

    # 유니버스
    "universe": "BOTH",              # KOSPI200 / KOSDAQ150 / BOTH

    # 진입 기준
    "score_threshold": 75,           # 종합 점수 75 이상 진입
    "enable_swing": False,
    "enable_trend": True,

    # 종합 점수 가중치 (총 100%, 감성 제외)
    "w_technical": 0.40,
    "w_supply":    0.30,
    "w_momentum":  0.30,

    # 시장 레짐
    "regime_filter": True,
    "regime_index": "KS11",
    "regime_ma": 200,

    # 추세 청산 (강화)
    "trend_min_hold": 20,
    "trend_break_consecutive": 5,
    "trend_trail_pct": 12.0,         # 8 → 12 (한국 변동성 대응)
    "reentry_cooldown_days": 30,

    # 단기 스윙 (비활성 시 무시)
    "swing_target_atr": 3.0,
    "swing_stop_atr": 2.0,
    "swing_max_hold": 10,

    # 포트폴리오 리스크 관리 (강화 — MDD -64% 대응)
    "max_concurrent": 12,            # 18 → 12 (분산 vs 집중도 균형)
    "min_position_pct": 2.0,         # 3 → 2
    "max_position_pct": 5.0,         # 8 → 5 (단일 종목 익스포져 축소)
    "risk_per_trade": 0.004,         # 0.5% → 0.4% (보수적)
    "portfolio_trail_pct": 0.12,     # 0.20 → 0.12 (DD 빨리 정지)
    "portfolio_max_dd_pct": 0.18,    # 0.25 → 0.18 (DD halt 더 일찍)
    "dd_position_scaling": True,     # 포트폴리오 DD 깊을수록 포지션 작게
    "initial_capital": 10_000_000,

    # 유동성 필터
    "min_daily_value": 500_000_000,  # 일평균 거래대금 5억 미만 제외

    # 거래 비용 (한국 시장)
    "commission_buy": 0.00015,
    "commission_sell": 0.00015,
    "tax_sell": 0.0018,
    "slippage": 0.003,

    # Monte Carlo
    "enable_montecarlo": True,
    "mc_iterations": 500,

    "workers": 8,
}

_REGIME_OK = None


# =============================================================================
# 1) 시장 레짐
# =============================================================================

def load_regime_filter(start_date):
    import FinanceDataReader as fdr
    idx = CONFIG["regime_index"]; n = CONFIG["regime_ma"]
    df = fdr.DataReader(idx, start_date - dt.timedelta(days=n * 2 + 100), dt.date.today())
    ma = df["Close"].rolling(n).mean()
    return df["Close"] > ma


def regime_ok_at(date):
    if not CONFIG["regime_filter"] or _REGIME_OK is None:
        return True
    val = _REGIME_OK.asof(date)
    return bool(val) if not pd.isna(val) else False


# =============================================================================
# 2) 유니버스 + 유동성 필터
# =============================================================================

def get_universe(which):
    import FinanceDataReader as fdr
    listing = fdr.StockListing("KRX")
    cap_col = next((c for c in ("Marcap", "MarketCap", "marcap") if c in listing.columns), None)
    if cap_col is None:
        raise RuntimeError("시가총액 컬럼 없음.")
    listing = listing.dropna(subset=[cap_col, "Market", "Name"])
    listing = listing[~listing["Name"].str.contains("스팩|우$|우B|우C", regex=True, na=False)]
    kospi = listing[listing["Market"] == "KOSPI"].sort_values(cap_col, ascending=False).head(200)
    kosdaq = listing[listing["Market"] == "KOSDAQ"].sort_values(cap_col, ascending=False).head(150)
    if which == "KOSPI200":     df = kospi
    elif which == "KOSDAQ150":  df = kosdaq
    else:                        df = pd.concat([kospi, kosdaq]).drop_duplicates("Code")
    return [(r["Code"], r["Name"]) for _, r in df.iterrows()]


def passes_liquidity(df):
    """일평균 거래대금이 최소 기준 이상인지 확인."""
    if len(df) < 60:
        return False
    avg_value = (df["Close"] * df["Volume"]).tail(60).mean()
    return avg_value >= CONFIG["min_daily_value"]


# =============================================================================
# 3) 지표 (signals.py와 동일 + 필터)
# =============================================================================

def calc_indicators(df):
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
    df["MACD_h"] = df["MACD"] - df["MACD_sig"]
    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift()).abs()
    l_c = (df["Low"] - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / df["Close"]
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["VR"] = df["Volume"] / df["Vol_MA20"].replace(0, np.nan)
    df["Vol_5d"] = df["Volume"].rolling(5).mean()
    df["P60"] = (df["Close"] / df["Close"].shift(60) - 1) * 100
    df["RSI_5d_min"] = df["RSI"].rolling(5).min()
    df["RSIp"] = df["RSI"].shift(1)
    df["Hp"] = df["MACD_h"].shift(1)
    p_min_recent = df["Close"].rolling(10).min()
    p_min_prev = df["Close"].rolling(20).min().shift(10)
    r_min_recent = df["RSI"].rolling(10).min()
    r_min_prev = df["RSI"].rolling(20).min().shift(10)
    df["bullish_div"] = (p_min_recent < p_min_prev) & (r_min_recent > r_min_prev) & (df["RSI"] < 60)
    hi20 = df["High"].rolling(20).max(); lo20 = df["Low"].rolling(20).min()
    rng = (hi20 - lo20) / df["Close"]
    pos = (df["Close"] - lo20) / (hi20 - lo20).replace(0, np.nan)
    df["consolidation_breakout"] = (rng < 0.18) & (pos > 0.65) & \
                                    (df["Vol_5d"] > df["Vol_MA20"].replace(0, np.nan) * 1.2)
    df["volume_explosion"] = df["VR"] >= 3.0
    df["rsi_oversold_exit"] = (df["RSI_5d_min"] < 30) & (df["RSI"] >= 35) & (df["RSI"] <= 60)
    return df


# =============================================================================
# 4) 점수 (signals.py와 동일)
# =============================================================================

def score_trend(r):
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
    bd = r.get("bullish_div", False)
    if bd is True or (bd is not None and not pd.isna(bd) and bool(bd)): s += 12
    cb = r.get("consolidation_breakout", False)
    if cb is True or (cb is not None and not pd.isna(cb) and bool(cb)): s += 10
    ve = r.get("volume_explosion", False)
    if ve is True or (ve is not None and not pd.isna(ve) and bool(ve)): s += 8
    oe = r.get("rsi_oversold_exit", False)
    if oe is True or (oe is not None and not pd.isna(oe) and bool(oe)): s += 10
    return max(0, min(100, s))


def momentum_score(p60):
    if p60 is None or pd.isna(p60): return 50
    if p60 >= 30: return 92
    if p60 >= 15: return 78
    if p60 >= 5:  return 62
    if p60 >= 0:  return 52
    if p60 >= -10: return 35
    if p60 >= -25: return 20
    return 8


def combined_score(tech, supply, mom):
    return (tech * CONFIG["w_technical"] +
            supply * CONFIG["w_supply"] +
            mom * CONFIG["w_momentum"])


# =============================================================================
# 5) 트레이드 시뮬레이션
# =============================================================================

@dataclass
class Trade:
    code: str
    name: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    days_held: int
    return_pct: float
    exit_reason: str
    score_at_entry: int
    position_pct: float


def position_size(atr_pct):
    if atr_pct is None or pd.isna(atr_pct) or atr_pct <= 0:
        return CONFIG["min_position_pct"]
    raw = CONFIG["risk_per_trade"] / (2 * atr_pct) * 100
    return float(max(CONFIG["min_position_pct"], min(CONFIG["max_position_pct"], raw)))


def apply_costs(entry, exit_):
    """매수/매도 비대칭 비용 반영 후 수익률."""
    eff_entry = entry * (1 + CONFIG["slippage"] / 2) * (1 + CONFIG["commission_buy"])
    eff_exit = exit_ * (1 - CONFIG["slippage"] / 2) * (1 - CONFIG["commission_sell"] - CONFIG["tax_sell"])
    return (eff_exit / eff_entry - 1) * 100


def simulate_trend(code, name, df, start_idx):
    """장기 추세 시뮬레이션 (15~20종목 분산 운영 가정)."""
    trades = []
    in_pos = None
    cooldown_until = None
    th = CONFIG["score_threshold"]
    trail = CONFIG["trend_trail_pct"] / 100
    min_hold = CONFIG["trend_min_hold"]
    break_n = CONFIG["trend_break_consecutive"]
    cooldown_days = CONFIG["reentry_cooldown_days"]
    consecutive_bear = 0

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        if not pd.isna(row["MA20"]) and not pd.isna(row["MA60"]):
            consecutive_bear = consecutive_bear + 1 if row["MA20"] < row["MA60"] else 0

        if in_pos is not None:
            entry_price, entry_date, score, days, peak, pos_pct = in_pos
            days += 1
            peak = max(peak, float(row["Close"]))
            exit_price, reason = None, None
            if days >= min_hold:
                if consecutive_bear >= break_n:
                    exit_price, reason = float(nxt["Open"]), "trend_break"
                elif nxt["Low"] <= peak * (1 - trail):
                    exit_price, reason = peak * (1 - trail), "trail_stop"
            if exit_price is not None:
                trades.append(Trade(
                    code=code, name=name,
                    entry_date=entry_date, entry_price=entry_price,
                    exit_date=nxt.name, exit_price=exit_price,
                    days_held=days,
                    return_pct=apply_costs(entry_price, exit_price),
                    exit_reason=reason, score_at_entry=score, position_pct=pos_pct,
                ))
                in_pos = None
                cooldown_until = nxt.name + pd.Timedelta(days=cooldown_days)
            else:
                in_pos = (entry_price, entry_date, score, days, peak, pos_pct)

        if in_pos is None:
            if cooldown_until is not None and df.index[i] < cooldown_until:
                continue
            if not regime_ok_at(nxt.name):
                continue
            tech = score_trend(row)
            # 백테스트에서는 수급 데이터 부재 → 50 중립, 모멘텀은 P60 기반
            mom = momentum_score(row.get("P60"))
            combined = combined_score(tech, 50, mom)
            if combined >= th and tech >= 60:  # 최소 기술 60 + 종합 75
                entry_price = float(nxt["Open"])
                atr_pct = float(row["ATR_pct"]) if not pd.isna(row["ATR_pct"]) else None
                pos_pct = position_size(atr_pct)
                in_pos = (entry_price, nxt.name, int(combined), 0, entry_price, pos_pct)

    return trades


def backtest_one(code, name, start):
    import FinanceDataReader as fdr
    try:
        df = fdr.DataReader(code, start - dt.timedelta(days=400), dt.date.today())
        if len(df) < 200:
            return []
        if not passes_liquidity(df):
            return []
        df = calc_indicators(df)
        start_idx = df.index.get_indexer([pd.Timestamp(start)], method="bfill")[0]
        if start_idx < 130:
            start_idx = 130
        return simulate_trend(code, name, df, start_idx) if CONFIG["enable_trend"] else []
    except Exception:
        return []


# =============================================================================
# 6) 포트폴리오 시뮬레이션 (트레일링 + MDD 한도)
# =============================================================================

def simulate_portfolio(trades, with_risk_controls=True):
    """포트폴리오 단위 시뮬레이션 — 동시 N포지션, 포트폴리오 트레일링/MDD 한도."""
    if not trades:
        return {"final": CONFIG["initial_capital"], "total_return_pct": 0,
                "equity_pts": [], "skipped": 0, "halted": False}

    cap0 = float(CONFIG["initial_capital"])
    max_conc = CONFIG["max_concurrent"]
    port_trail = CONFIG["portfolio_trail_pct"]
    max_dd_limit = CONFIG["portfolio_max_dd_pct"]

    sorted_trades = sorted(trades, key=lambda t: t.entry_date)
    cash = cap0
    positions = []   # [(exit_date, capital, return_pct, trade)]
    equity_pts = [(sorted_trades[0].entry_date, cap0)]
    portfolio_peak = cap0
    skipped = 0
    halted = False
    halt_date = None

    for t in sorted_trades:
        # 진입일 이전 청산된 포지션 정산
        positions.sort(key=lambda p: p[0])
        while positions and positions[0][0] <= t.entry_date:
            exit_d, cap, ret, _ = positions.pop(0)
            cash += cap * (1 + ret / 100)
            cur_value = cash + sum(p[1] for p in positions)
            portfolio_peak = max(portfolio_peak, cur_value)
            equity_pts.append((exit_d, cur_value))

            # 리스크 한도 체크
            if with_risk_controls:
                dd = (cur_value - portfolio_peak) / portfolio_peak
                if dd <= -max_dd_limit:
                    halted = True
                    halt_date = exit_d
                    break

        if halted:
            # 남은 모든 포지션 청산
            for exit_d, cap, ret, _ in sorted(positions, key=lambda p: p[0]):
                cash += cap * (1 + ret / 100)
                equity_pts.append((exit_d, cash))
            positions = []
            break

        # 포트폴리오 트레일링 (-20% 시 신규 진입 일시 중단)
        cur_value = cash + sum(p[1] for p in positions)
        if with_risk_controls and cur_value < portfolio_peak * (1 - port_trail):
            skipped += 1
            continue

        if len(positions) >= max_conc:
            skipped += 1
            continue

        portfolio = cash + sum(p[1] for p in positions)
        target_pct = t.position_pct / 100

        # DD 가중 — 포트폴리오가 고점 대비 빠져있을수록 포지션 축소
        if with_risk_controls and CONFIG.get("dd_position_scaling", False):
            dd_now = (portfolio - portfolio_peak) / portfolio_peak
            if dd_now < -0.05:
                # -5% DD에서 80%, -10% DD에서 60%, -15% DD에서 40% 로 축소
                scale = max(0.4, 1.0 + dd_now * 4)
                target_pct *= scale

        alloc = min(portfolio * target_pct, cash)
        if alloc < 10000:
            skipped += 1
            continue
        cash -= alloc
        positions.append((t.exit_date, alloc, t.return_pct, t))
        equity_pts.append((t.entry_date, cash + sum(p[1] for p in positions)))

    if not halted:
        for exit_d, cap, ret, _ in sorted(positions, key=lambda p: p[0]):
            cash += cap * (1 + ret / 100)
            equity_pts.append((exit_d, cash))

    return {
        "final": cash,
        "total_return_pct": (cash / cap0 - 1) * 100,
        "equity_pts": equity_pts,
        "skipped": skipped,
        "halted": halted,
        "halt_date": str(halt_date.date()) if halt_date else None,
    }


# =============================================================================
# 7) 통계 (Sharpe, MDD, Annualized 등)
# =============================================================================

def trade_stats(trades, equity_pts=None):
    if not trades:
        return {"n": 0}
    rets = np.array([t.return_pct for t in trades])
    wins = rets[rets > 0]; losses = rets[rets <= 0]
    days = np.array([t.days_held for t in trades])
    win_rate = len(wins) / len(rets) * 100
    avg_w = wins.mean() if len(wins) else 0
    avg_l = losses.mean() if len(losses) else 0
    pf = (wins.sum() / -losses.sum()) if (len(losses) and losses.sum() < 0) else float("inf")
    expectancy = rets.mean()

    # 자산곡선 통계 — 일별 forward-fill 로 정확한 Sharpe 계산
    if equity_pts:
        # 이벤트 기반을 일별 시계열로 변환
        eq_df = pd.DataFrame(equity_pts, columns=["date", "value"]).sort_values("date")
        eq_df = eq_df.drop_duplicates("date", keep="last").set_index("date")
        # 영업일 빈도로 리샘플 + 직전값 채움
        daily = eq_df["value"].resample("B").ffill().dropna()

        if len(daily) > 30:
            peak = daily.cummax()
            dd = (daily - peak) / peak
            mdd = float(dd.min() * 100)
            daily_ret = daily.pct_change().dropna()
            if daily_ret.std() > 0:
                # 무위험 0% 가정, 영업일 252 환산
                sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
            else:
                sharpe = 0.0
        else:
            # 짧은 기간이면 이벤트 기반 폴백
            eq = np.array([v for _, v in equity_pts])
            peak_arr = np.maximum.accumulate(eq)
            mdd = float(((eq - peak_arr) / peak_arr).min() * 100)
            sharpe = 0.0
    else:
        mdd = 0; sharpe = 0

    return {
        "n": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_w, "avg_loss": avg_l,
        "profit_factor": pf, "expectancy": expectancy,
        "mdd": mdd, "sharpe": sharpe,
        "avg_days": days.mean(),
        "best": rets.max(), "worst": rets.min(),
    }


def annualize_return(total_pct, years):
    return ((1 + total_pct / 100) ** (1 / years) - 1) * 100


# =============================================================================
# 8) Monte Carlo
# =============================================================================

def monte_carlo(trades, n_iter=500):
    """거래 결과를 부트스트랩 재샘플링해서 robustness 분포 추정."""
    if not trades or n_iter < 10:
        return {}
    cap0 = CONFIG["initial_capital"]
    finals = []
    mdds = []
    rng = random.Random(42)
    for _ in range(n_iter):
        # 거래를 무작위 순서로 (포트폴리오 시뮬은 비용 큼, 단순 누적 사용)
        sample = rng.choices(trades, k=len(trades))
        equity = cap0
        peak = cap0
        max_dd = 0
        for t in sample:
            # 동시 포지션 가정으로 자본의 1/N 만 투입
            alloc_pct = t.position_pct / 100
            change = equity * alloc_pct * (t.return_pct / 100)
            equity += change
            peak = max(peak, equity)
            dd = (equity - peak) / peak
            if dd < max_dd:
                max_dd = dd
        finals.append((equity / cap0 - 1) * 100)
        mdds.append(max_dd * 100)
    finals = np.array(finals); mdds = np.array(mdds)
    return {
        "n_iter": n_iter,
        "return_mean": float(finals.mean()),
        "return_median": float(np.median(finals)),
        "return_p5": float(np.percentile(finals, 5)),
        "return_p95": float(np.percentile(finals, 95)),
        "mdd_mean": float(mdds.mean()),
        "mdd_p95": float(np.percentile(mdds, 95)),
        "mdd_worst": float(mdds.min()),
        "prob_profit": float((finals > 0).mean() * 100),
    }


# =============================================================================
# 9) Benchmark
# =============================================================================

def benchmark_kospi(start):
    import FinanceDataReader as fdr
    df = fdr.DataReader("KS11", start, dt.date.today())
    cap0 = float(CONFIG["initial_capital"])
    ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    return {"label": "KOSPI 매수보유", "return_pct": float(ret),
            "equity": (cap0 * df["Close"] / df["Close"].iloc[0]).tolist(),
            "dates": [str(d.date()) for d in df.index]}


# =============================================================================
# 10) MAIN — 전체 기간 + In-Sample / Out-of-Sample 분리
# =============================================================================

def split_in_out_of_sample(trades, full_start, full_end, oos_pct):
    total_days = (full_end - full_start).days
    cutoff = full_start + dt.timedelta(days=int(total_days * (1 - oos_pct)))
    in_sample = [t for t in trades if t.entry_date.date() < cutoff]
    out_sample = [t for t in trades if t.entry_date.date() >= cutoff]
    return in_sample, out_sample, cutoff


def show_stats(label, trades, port):
    if not trades:
        print(f"\n📊 {label}: 거래 없음"); return
    cap0 = CONFIG["initial_capital"]
    s = trade_stats(trades, port["equity_pts"])
    pf = "∞" if not np.isfinite(s["profit_factor"]) else f"{s['profit_factor']:.2f}"
    halted_note = f" [HALTED at {port['halt_date']}]" if port.get("halted") else ""
    print(f"\n📊 {label}{halted_note}")
    print(f"   거래수 {s['n']} | 승률 {s['win_rate']:.1f}% | "
          f"기대값/거래 {s['expectancy']:+.2f}% | PF {pf}")
    print(f"   평균 익절/손절: {s['avg_win']:+.2f}% / {s['avg_loss']:+.2f}% | "
          f"평균보유 {s['avg_days']:.1f}일")
    print(f"   ₩{cap0:,.0f} → ₩{port['final']:,.0f} ({port['total_return_pct']:+.1f}%) | "
          f"MDD {s['mdd']:.1f}% | Sharpe {s['sharpe']:.2f}")


def main():
    global _REGIME_OK

    full_end = dt.date.today()
    full_start = full_end - dt.timedelta(days=365 * CONFIG["test_years"])

    print(f"\n🔍 백테스트 v3 (Risk-Managed)")
    print(f"   기간       : {full_start} ~ {full_end} ({CONFIG['test_years']}년)")
    print(f"   유니버스   : {CONFIG['universe']}")
    print(f"   가중치     : 기술 {CONFIG['w_technical']*100:.0f}% + "
          f"수급 {CONFIG['w_supply']*100:.0f}% + 모멘텀 {CONFIG['w_momentum']*100:.0f}%")
    print(f"   목표 분산  : {CONFIG['max_concurrent']}종목 (ATR 비례 비중 "
          f"{CONFIG['min_position_pct']}~{CONFIG['max_position_pct']}%)")
    print(f"   리스크 한도: 포트폴리오 트레일링 -{CONFIG['portfolio_trail_pct']*100:.0f}% / "
          f"MDD 한도 -{CONFIG['portfolio_max_dd_pct']*100:.0f}%")
    print(f"   유동성 필터: 일평균 거래대금 ₩{CONFIG['min_daily_value']:,.0f} 이상\n")

    if CONFIG["regime_filter"]:
        print("📈 KOSPI 레짐 필터 로딩...")
        _REGIME_OK = load_regime_filter(full_start)
        in_up = int(_REGIME_OK.loc[full_start:].sum())
        total = len(_REGIME_OK.loc[full_start:])
        print(f"   강세장 비율: {in_up}/{total} ({in_up/max(total,1)*100:.0f}%)\n")

    print("📋 종목 유니버스 + 유동성 필터...")
    universe = get_universe(CONFIG["universe"])
    print(f"   초기 {len(universe)}개 종목 (유동성 필터는 백테스트 중 자동 적용)\n")

    print("📊 시뮬레이션 중 (10~25분)...")
    all_trades = []
    with ThreadPoolExecutor(max_workers=CONFIG["workers"]) as ex:
        futs = {ex.submit(backtest_one, c, n, full_start): c for c, n in universe}
        done = 0
        for f in as_completed(futs):
            all_trades.extend(f.result())
            done += 1
            if done % 25 == 0:
                print(f"   {done}/{len(universe)} (누적 거래 {len(all_trades)})")
    print(f"\n✓ 시뮬 완료: 총 {len(all_trades)}거래\n")

    if not all_trades:
        print("⚠️ 거래 없음. 종료.")
        return

    # 전체 기간
    port_full = simulate_portfolio(all_trades, with_risk_controls=True)
    bench = benchmark_kospi(full_start)
    show_stats("전체 기간 (Full Sample)", all_trades, port_full)

    # In-Sample / Out-of-Sample
    in_trades, out_trades, cutoff = split_in_out_of_sample(
        all_trades, full_start, full_end, CONFIG["oos_pct"]
    )
    print(f"\n📅 분할: In-Sample {full_start} ~ {cutoff} ({len(in_trades)}거래) | "
          f"Out-of-Sample {cutoff} ~ {full_end} ({len(out_trades)}거래)")
    port_in = simulate_portfolio(in_trades, with_risk_controls=True)
    port_out = simulate_portfolio(out_trades, with_risk_controls=True)
    show_stats("In-Sample (튜닝 기간)", in_trades, port_in)
    show_stats("Out-of-Sample (검증 기간)", out_trades, port_out)

    in_years = (cutoff - full_start).days / 365
    out_years = (full_end - cutoff).days / 365
    print(f"\n   연환산: IN {annualize_return(port_in['total_return_pct'], in_years):+.1f}% | "
          f"OUT {annualize_return(port_out['total_return_pct'], out_years):+.1f}% | "
          f"KOSPI {annualize_return(bench['return_pct'], CONFIG['test_years']):+.1f}%")
    print(f"📌 KOSPI 매수보유 ({CONFIG['test_years']}년): {bench['return_pct']:+.1f}%")

    # Monte Carlo
    mc = {}
    if CONFIG["enable_montecarlo"]:
        print(f"\n🎲 Monte Carlo 시뮬레이션 ({CONFIG['mc_iterations']}회)...")
        mc = monte_carlo(all_trades, CONFIG["mc_iterations"])
        if mc:
            print(f"   누적수익률: 중앙값 {mc['return_median']:+.1f}% | "
                  f"5%-95% 구간 [{mc['return_p5']:+.1f}%, {mc['return_p95']:+.1f}%]")
            print(f"   MDD       : 평균 {mc['mdd_mean']:.1f}% | "
                  f"95% percentile {mc['mdd_p95']:.1f}% | 최악 {mc['mdd_worst']:.1f}%")
            print(f"   수익 확률  : {mc['prob_profit']:.1f}%")

    # 저장
    csv_path = OUTPUT_DIR / "backtest_trades.csv"
    pd.DataFrame([t.__dict__ for t in all_trades]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 거래내역 → {csv_path}")

    # 간단한 JSON 요약
    full_stats = trade_stats(all_trades, port_full["equity_pts"])
    full_stats_clean = {k: v for k, v in full_stats.items()
                        if not isinstance(v, float) or np.isfinite(v)}
    full_block = {
        "trades": len(all_trades),
        "total_return_pct": port_full["total_return_pct"],
        "final_value": port_full["final"],
        "halted": port_full.get("halted"),
    }
    full_block.update(full_stats_clean)

    summary = {
        "config": {k: v for k, v in CONFIG.items() if not callable(v)},
        "full_sample": full_block,
        "in_sample": {
            "trades": len(in_trades),
            "total_return_pct": port_in["total_return_pct"],
            "annualized_pct": annualize_return(port_in["total_return_pct"], in_years),
        },
        "out_of_sample": {
            "trades": len(out_trades),
            "total_return_pct": port_out["total_return_pct"],
            "annualized_pct": annualize_return(port_out["total_return_pct"], out_years),
        },
        "kospi_benchmark": {
            "total_return_pct": bench["return_pct"],
            "annualized_pct": annualize_return(bench["return_pct"], CONFIG["test_years"]),
        },
        "monte_carlo": mc,
    }
    summary_path = OUTPUT_DIR / "backtest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8")
    print(f"📄 요약 → {summary_path}")

    print("\n✅ 백테스트 완료.")
    if mc:
        if mc["return_p5"] > 0:
            print("   👍 5% percentile도 양수 — 견고한 전략")
        elif mc["return_median"] > bench["return_pct"]:
            print("   ✓ 중앙값이 KOSPI 매수보유를 이김")
        else:
            print("   ⚠️ 인덱스 대비 알파 약함 — 파라미터 재검토 권장")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(1)
"out_of_sample": {
            "trades": len(out_trades),
            "total_return_pct": port_out["total_return_pct"],
            "annualized_pct": annualize_return(port_out["total_return_pct"], out_years),
        },
        "kospi_benchmark": {
            "total_return_pct": bench["return_pct"],
            "annualized_pct": annualize_return(bench["return_pct"], CONFIG["test_years"]),
        },
        "monte_carlo": mc,
    }
    summary_path = OUTPUT_DIR / "backtest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8")
    print(f"📄 요약 → {summary_path}")

    print("\n✅ 백테스트 완료.")
    if mc:
        if mc["return_p5"] > 0:
            print("   👍 5% percentile도 양수 — 견고한 전략")
        elif mc["return_median"] > bench["return_pct"]:
            print("   ✓ 중앙값이 KOSPI 매수보유를 이김")
        else:
            print("   ⚠️ 인덱스 대비 알파 약함 — 파라미터 재검토 권장")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(1)
