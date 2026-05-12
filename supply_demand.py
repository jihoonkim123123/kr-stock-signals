"""
1개월 수급 분석 모듈
====================
개인/기관/외국인 수급 데이터로 매수 우선순위 산출.
pykrx 활용 (별도 API 키 불필요).

매수 우선순위 로직:
- 외국인 + 기관 동반 매수 = 가장 강력한 신호
- 개인은 매도, 외국인/기관 매수 = 스마트머니 진입
- 거래량 폭증 + 수급 일치 = 우선 진입

설치: pip install pykrx
"""
from __future__ import annotations

import datetime as dt
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np


def fetch_supply_demand_1m(stock_code: str, days: int = 22) -> Optional[dict]:
    """
    1개월(22거래일) 개인/기관/외국인 순매수 데이터.
    
    Returns:
        {
            "stock_code": "005930",
            "individual_net": -1500000,  # 개인 순매수 (주식수)
            "foreign_net": 2300000,       # 외국인 순매수
            "institution_net": 800000,    # 기관 순매수
            "individual_value": -150e8,   # 개인 순매수 금액
            "foreign_value": 230e8,
            "institution_value": 80e8,
            "smart_money_score": 75,      # 0~100점
            "daily_data": [...],          # 일별 데이터 (차트용)
        }
    """
    try:
        from pykrx import stock
    except ImportError:
        print("⚠️ pykrx 미설치: pip install pykrx")
        return None
    
    today = dt.date.today()
    end = today.strftime("%Y%m%d")
    start = (today - dt.timedelta(days=days * 2)).strftime("%Y%m%d")
    
    try:
        # 거래주체별 순매수 (주식수 기준)
        df_vol = stock.get_market_trading_volume_by_date(start, end, stock_code)
        # 거래주체별 순매수 (금액 기준)
        df_val = stock.get_market_trading_value_by_date(start, end, stock_code)
        
        if df_vol.empty or df_val.empty:
            return None
        
        # 최근 22거래일
        df_vol = df_vol.tail(22)
        df_val = df_val.tail(22)
        
        # 순매수 합계
        ind_net = int(df_vol.get("개인", pd.Series([0])).sum())
        for_net = int(df_vol.get("외국인합계", df_vol.get("외국인", pd.Series([0]))).sum())
        inst_net = int(df_vol.get("기관합계", df_vol.get("기관", pd.Series([0]))).sum())
        
        ind_val = float(df_val.get("개인", pd.Series([0])).sum())
        for_val = float(df_val.get("외국인합계", df_val.get("외국인", pd.Series([0]))).sum())
        inst_val = float(df_val.get("기관합계", df_val.get("기관", pd.Series([0]))).sum())
        
        # 스마트 머니 스코어 (외국인+기관 매수 vs 개인 매도)
        smart_score = calculate_smart_money_score(for_val, inst_val, ind_val)
        
        # 일별 데이터 (차트용)
        daily_data = []
        for idx in df_val.index:
            daily_data.append({
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                "individual": float(df_val.loc[idx].get("개인", 0)),
                "foreign": float(df_val.loc[idx].get("외국인합계", df_val.loc[idx].get("외국인", 0))),
                "institution": float(df_val.loc[idx].get("기관합계", df_val.loc[idx].get("기관", 0))),
            })
        
        return {
            "stock_code": stock_code,
            "individual_net": ind_net,
            "foreign_net": for_net,
            "institution_net": inst_net,
            "individual_value": ind_val,
            "foreign_value": for_val,
            "institution_value": inst_val,
            "smart_money_score": smart_score,
            "daily_data": daily_data,
        }
    except Exception as e:
        return None


def calculate_smart_money_score(for_val: float, inst_val: float, ind_val: float) -> int:
    """
    스마트머니 점수 (0~100).
    
    로직:
    - 외국인 매수 + 기관 매수 + 개인 매도 = 100점 (이상적)
    - 외국인 매수만 = 60점
    - 기관 매수만 = 50점
    - 개인만 매수 (외국인/기관 매도) = 0~20점
    """
    score = 50  # 중립
    
    # 외국인 매수 (가중치 40)
    if for_val > 0:
        score += min(20, int(for_val / 1e10))  # 100억 단위
    else:
        score -= min(15, int(-for_val / 1e10))
    
    # 기관 매수 (가중치 30)
    if inst_val > 0:
        score += min(15, int(inst_val / 1e10))
    else:
        score -= min(10, int(-inst_val / 1e10))
    
    # 개인 매도 = 외국인/기관 매수에 추가 가산
    if ind_val < 0 and (for_val > 0 or inst_val > 0):
        score += min(15, int(-ind_val / 1e10))
    
    return max(0, min(100, score))


def analyze_supply_demand_batch(stock_codes: list, workers: int = 4) -> dict:
    """여러 종목 1개월 수급 일괄 분석."""
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_supply_demand_1m, code): code for code in stock_codes}
        for f in as_completed(futs):
            code = futs[f]
            try:
                data = f.result()
                if data:
                    results[code] = data
            except Exception:
                pass
    return results


def get_priority_signal(sd_data: dict) -> dict:
    """
    수급 데이터로 매수 우선순위 신호 생성.
    
    Returns:
        {
            "priority": "TIER1" | "TIER2" | "TIER3" | "AVOID",
            "reason": "외국인+기관 동반 매수 (스마트머니 ↑)",
            "icon": "🚀",
        }
    """
    score = sd_data["smart_money_score"]
    for_val = sd_data["foreign_value"]
    inst_val = sd_data["institution_value"]
    ind_val = sd_data["individual_value"]
    
    # TIER 1: 외국인 + 기관 동반 매수 (스마트머니 강세)
    if for_val > 0 and inst_val > 0 and score >= 75:
        return {
            "priority": "TIER1",
            "icon": "🚀",
            "reason": "외국인+기관 동반 매수 (스마트머니 진입)",
            "score": score,
        }
    
    # TIER 2: 외국인 OR 기관 강한 매수
    if (for_val > 1e10 or inst_val > 1e10) and score >= 60:
        which = "외국인" if for_val > inst_val else "기관"
        return {
            "priority": "TIER2",
            "icon": "⭐",
            "reason": f"{which} 매수 우위",
            "score": score,
        }
    
    # AVOID: 외국인 + 기관 동반 매도
    if for_val < -1e10 and inst_val < -1e10:
        return {
            "priority": "AVOID",
            "icon": "🚫",
            "reason": "외국인+기관 동반 매도",
            "score": score,
        }
    
    # TIER 3: 그 외
    return {
        "priority": "TIER3",
        "icon": "📊",
        "reason": "수급 중립",
        "score": score,
    }


if __name__ == "__main__":
    # 테스트
    test_codes = ["005930", "000660", "035720"]  # 삼성전자, SK하이닉스, 카카오
    
    print("📊 1개월 수급 분석 테스트")
    for code in test_codes:
        data = fetch_supply_demand_1m(code)
        if data:
            signal = get_priority_signal(data)
            print(f"\n{code}:")
            print(f"  외국인: {data['foreign_value']/1e8:+,.0f}억")
            print(f"  기관:   {data['institution_value']/1e8:+,.0f}억")
            print(f"  개인:   {data['individual_value']/1e8:+,.0f}억")
            print(f"  점수:   {data['smart_money_score']}/100")
            print(f"  신호:   {signal['icon']} {signal['priority']} - {signal['reason']}")
