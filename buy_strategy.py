"""
매수 전략 자동 생성 모듈
====================
종목 클릭 시 표시할 구체적인 매수 전략을 자동 생성.

전략 구성:
1. 진입 타이밍 (3단계: Tier 1/2/3)
2. 가격대별 매수 비중
3. Stop-Loss 시스템
4. 차익실현 사다리
5. 8/13 목표 도달 시나리오
"""
from __future__ import annotations
from typing import Optional
import numpy as np


def generate_buy_strategy(
    stock_data: dict,
    target_date: str = "2026-08-13",
    portfolio_pct: float = 5.0,
    total_budget: float = 1_000_000,
) -> dict:
    """
    종목별 구체적 매수 전략 생성.
    
    Args:
        stock_data: analyze_one()의 결과 + 수급/감성 데이터
        target_date: 목표 매도일
        portfolio_pct: 전체 포트폴리오 내 비중 (%)
        total_budget: 전체 투자 가능 자금 (원)
    
    Returns:
        {
            "summary": "...",
            "entry_tiers": [...],  # Tier 1/2/3 진입 계획
            "stop_loss": {...},
            "take_profit": [...],
            "expected_return": {...},
            "risk_management": [...],
        }
    """
    close = stock_data.get("close", 0)
    atr = stock_data.get("atr", close * 0.03)
    rsi = stock_data.get("rsi", 50)
    ma20 = stock_data.get("ma20", close)
    ma60 = stock_data.get("ma60", close)
    trend_score = stock_data.get("trend", 50)
    combined_score = stock_data.get("combined", trend_score)
    
    # 수급 데이터
    sd = stock_data.get("supply_demand", {})
    smart_score = sd.get("smart_money_score", 50)
    
    # 감성 데이터
    sentiment = stock_data.get("sentiment", 0)
    
    # 종목별 투자 자금
    allocation = total_budget * (portfolio_pct / 100)
    
    # ---------- Tier 진입 가격 계산 ----------
    # Tier 1: 즉시 매수 (-1% ~ +2%)
    tier1_low = round(close * 0.99, 0)
    tier1_high = round(close * 1.02, 0)
    
    # Tier 2: 1차 조정 (-5% ~ -7%) - MA20 근처
    tier2_low = round(min(close * 0.93, ma20 * 0.98), 0)
    tier2_high = round(min(close * 0.95, ma20 * 1.00), 0)
    
    # Tier 3: 2차 조정 (-10% ~ -15%) - MA60 근처
    tier3_low = round(min(close * 0.85, ma60 * 0.95), 0)
    tier3_high = round(min(close * 0.90, ma60 * 1.00), 0)
    
    # ---------- 매수 비중 결정 ----------
    # 신호 강도에 따라 Tier 1 비중 조정
    if combined_score >= 80 and smart_score >= 75:
        # 강한 신호: Tier 1에 적극 진입
        tier1_pct = 50
        tier2_pct = 30
        tier3_pct = 20
        strategy_type = "AGGRESSIVE"
    elif combined_score >= 70:
        # 중간 신호: 균형 분할
        tier1_pct = 35
        tier2_pct = 40
        tier3_pct = 25
        strategy_type = "BALANCED"
    else:
        # 약한 신호: 보수적 진입
        tier1_pct = 25
        tier2_pct = 35
        tier3_pct = 40
        strategy_type = "CONSERVATIVE"
    
    # ---------- 주식 수 계산 ----------
    tier1_amt = allocation * tier1_pct / 100
    tier2_amt = allocation * tier2_pct / 100
    tier3_amt = allocation * tier3_pct / 100
    
    tier1_avg = (tier1_low + tier1_high) / 2
    tier2_avg = (tier2_low + tier2_high) / 2
    tier3_avg = (tier3_low + tier3_high) / 2
    
    tier1_shares = int(tier1_amt / tier1_avg) if tier1_avg else 0
    tier2_shares = int(tier2_amt / tier2_avg) if tier2_avg else 0
    tier3_shares = int(tier3_amt / tier3_avg) if tier3_avg else 0
    
    # 가중평균 매수가
    total_shares = tier1_shares + tier2_shares + tier3_shares
    total_cost = (
        tier1_shares * tier1_avg
        + tier2_shares * tier2_avg
        + tier3_shares * tier3_avg
    )
    avg_buy_price = total_cost / total_shares if total_shares > 0 else close
    
    # ---------- Stop-Loss ----------
    stop_atr = round(close - 2.5 * atr, 0)  # 보수적
    stop_ma60 = round(ma60 * 0.95, 0)  # 60일선 -5%
    stop_pct = round(avg_buy_price * 0.88, 0)  # 평단 -12%
    
    # 가장 가까운 손절가 선택 (가장 높은 값)
    final_stop = max(stop_atr, stop_ma60, stop_pct)
    
    # ---------- Take Profit (차익실현 사다리) ----------
    # 8/13까지 목표 수익률 계산
    target_return_low = 15  # 보수적
    target_return_mid = 25  # 기본
    target_return_high = 40  # 공격적
    
    if combined_score >= 80 and smart_score >= 75:
        target_return_mid = 30
        target_return_high = 50
    
    tp1 = round(avg_buy_price * (1 + target_return_low / 100), 0)
    tp2 = round(avg_buy_price * (1 + target_return_mid / 100), 0)
    tp3 = round(avg_buy_price * (1 + target_return_high / 100), 0)
    
    # ---------- 기대 수익 ----------
    expected = {
        "bull": {
            "price": tp3,
            "return_pct": target_return_high,
            "profit": int(total_shares * (tp3 - avg_buy_price)),
            "probability": 30 if combined_score >= 75 else 20,
        },
        "base": {
            "price": tp2,
            "return_pct": target_return_mid,
            "profit": int(total_shares * (tp2 - avg_buy_price)),
            "probability": 50,
        },
        "bear": {
            "price": final_stop,
            "return_pct": round((final_stop - avg_buy_price) / avg_buy_price * 100, 1),
            "profit": int(total_shares * (final_stop - avg_buy_price)),
            "probability": 20 if combined_score >= 75 else 30,
        },
    }
    
    # 확률 가중 평균 기대 수익
    weighted_return = (
        expected["bull"]["return_pct"] * expected["bull"]["probability"]
        + expected["base"]["return_pct"] * expected["base"]["probability"]
        + expected["bear"]["return_pct"] * expected["bear"]["probability"]
    ) / 100
    
    # 리스크 관리 메시지 구성
    risk_mgmt = [
        f"Stop-Loss ₩{final_stop:,.0f} 이탈 시 즉시 손절 (-{round((avg_buy_price - final_stop) / avg_buy_price * 100, 1)}%)",
        f"신용/레버리지 사용 금지",
        f"단일 종목 최대 비중: 포트폴리오의 {portfolio_pct * 1.5:.0f}% 초과 금지",
    ]
    
    if rsi > 70:
        risk_mgmt.append(f"⚠️ RSI {rsi:.0f} = 과매수 영역, Tier 1 비중 축소 권장")
    elif rsi < 30:
        risk_mgmt.append(f"⚠️ RSI {rsi:.0f} = 과매도 영역, 추가 하락 가능성")
    
    if smart_score < 40:
        risk_mgmt.append(f"⚠️ 스마트머니 점수 {smart_score} = 외국인/기관 매도 우위")
    
    # ✅ [수정 포인트] sentiment가 딕셔너리인 경우를 완벽 방어
    actual_sentiment = sentiment.get('score', 0) if isinstance(sentiment, dict) else sentiment
    
    if actual_sentiment < -20:
        risk_mgmt.append(f"⚠️ 뉴스 감성 {actual_sentiment:+.0f} = 부정적 뉴스 우세")
    
    # ---------- 한줄 요약 ----------
    smart_money_signal = (
        "🚀 스마트머니 진입"
        if smart_score >= 75
        else "⭐ 외국인/기관 관심"
        if smart_score >= 60
        else "📊 수급 중립"
    )
    
    summary = (
        f"{strategy_type} 전략 · {smart_money_signal} · "
        f"평단 ₩{avg_buy_price:,.0f} 형성 → 8/13까지 {weighted_return:+.1f}% 기대"
    )
    
    return {
        "summary": summary,
        "strategy_type": strategy_type,
        "weighted_return": round(weighted_return, 1),
        "entry_tiers": [
            {
                "tier": 1,
                "label": "🥇 즉시 매수",
                "price_low": tier1_low,
                "price_high": tier1_high,
                "pct": tier1_pct,
                "shares": tier1_shares,
                "amount": int(tier1_shares * tier1_avg),
                "trigger": "현재가 매수 가능",
                "condition": (
                    "강한 신호 (점수 80+)"
                    if strategy_type == "AGGRESSIVE"
                    else "안정적 진입"
                ),
            },
            {
                "tier": 2,
                "label": "🥈 1차 조정 매수",
                "price_low": tier2_low,
                "price_high": tier2_high,
                "pct": tier2_pct,
                "shares": tier2_shares,
                "amount": int(tier2_shares * tier2_avg),
                "trigger": f"MA20 ₩{ma20:,.0f} 터치 시",
                "condition": "5-7% 조정 시 분할",
            },
            {
                "tier": 3,
                "label": "🥉 2차 조정 매수",
                "price_low": tier3_low,
                "price_high": tier3_high,
                "pct": tier3_pct,
                "shares": tier3_shares,
                "amount": int(tier3_shares * tier3_avg),
                "trigger": f"MA60 ₩{ma60:,.0f} 터치 시",
                "condition": "10-15% 조정 시 (최고 기회)",
            },
        ],
        "average_buy_price": round(avg_buy_price, 0),
        "total_shares": total_shares,
        "total_investment": int(total_cost),
        "stop_loss": {
            "price": final_stop,
            "pct_from_avg": round((final_stop - avg_buy_price) / avg_buy_price * 100, 1),
            "reason": "ATR/MA60/평단 -12% 중 가장 보수적",
        },
        "take_profit": [
            {
                "level": 1,
                "price": tp1,
                "return_pct": target_return_low,
                "sell_pct": 25,
                "label": "1차 차익실현 (안전 확보)",
            },
            {
                "level": 2,
                "price": tp2,
                "return_pct": target_return_mid,
                "sell_pct": 50,
                "label": "2차 차익실현 (절반 회수)",
            },
            {
                "level": 3,
                "price": tp3,
                "return_pct": target_return_high,
                "sell_pct": 25,
                "label": "3차 차익실현 (Bull 시나리오)",
            },
        ],
        "expected_returns": expected,
        "risk_management": risk_mgmt,
        "target_date": target_date,
    }


def format_strategy_for_display(strategy: dict) -> str:
    """전략을 마크다운 형식으로 포맷팅."""
    md = []
    md.append(f"### 💡 {strategy['summary']}\n")
    
    # 진입 계획
    md.append("#### 🎯 단계별 매수 전략\n")
    for tier in strategy["entry_tiers"]:
        md.append(
            f"**{tier['label']} ({tier['pct']}%)**  \n"
            f"₩{tier['price_low']:,.0f} ~ ₩{tier['price_high']:,.0f} · "
            f"{tier['shares']}주 · ₩{tier['amount']:,.0f}  \n"
            f"_{tier['condition']}_\n"
        )
    
    md.append(
        f"\n→ **가중평균 매수가**: ₩{strategy['average_buy_price']:,.0f}  \n"
        f"→ **총 매수 수량**: {strategy['total_shares']}주  \n"
        f"→ **총 투자금**: ₩{strategy['total_investment']:,.0f}\n"
    )
    
    # Stop Loss
    md.append(
        f"\n#### 🛡️ Stop-Loss\n"
        f"**₩{strategy['stop_loss']['price']:,.0f}** "
        f"({strategy['stop_loss']['pct_from_avg']:+.1f}%) 이탈 시 손절  \n"
        f"_{strategy['stop_loss']['reason']}_\n"
    )
    
    # Take Profit
    md.append("\n#### 💰 차익실현 사다리\n")
    for tp in strategy["take_profit"]:
        md.append(
            f"- **₩{tp['price']:,.0f}** (+{tp['return_pct']}%) → "
            f"{tp['sell_pct']}% 매도 _{tp['label']}_"
        )
    
    # Expected
    md.append("\n\n#### 📊 시나리오별 기대수익\n")
    md.append(f"확률 가중 기대수익률: **{strategy['weighted_return']:+.1f}%**\n")
    
    return "\n".join(md)


if __name__ == "__main__":
    # 테스트
    test_data = {
        "close": 131100,
        "atr": 3500,
        "rsi": 58,
        "ma20": 128000,
        "ma60": 115000,
        "trend": 85,
        "combined": 82,
        "supply_demand": {"smart_money_score": 78},
        "sentiment": 25,
    }
    
    strategy = generate_buy_strategy(
        test_data,
        portfolio_pct=10.0,
        total_budget=60_000_000,
    )
    
    print(format_strategy_for_display(strategy))
