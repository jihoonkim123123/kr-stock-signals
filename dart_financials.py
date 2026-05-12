"""
DART 재무제표 모듈
====================
OPEN DART API로 코스피/코스닥 종목의 재무 데이터를 조회.
koreantickers.com 스타일 9개 섹션 데이터를 제공.

준비:
1. https://opendart.fss.or.kr/ 회원가입 → API 키 발급
2. 환경변수 DART_API_KEY 설정 또는 .env 파일

사용법:
    from dart_financials import get_full_financials
    data = get_full_financials("005930")  # 삼성전자
"""
from __future__ import annotations

import os
import re
import time
import zipfile
import io
import datetime as dt
from pathlib import Path
from typing import Optional
from functools import lru_cache

import requests
import pandas as pd

DART_API_KEY = os.environ.get("DART_API_KEY", "")
DART_BASE = "https://opendart.fss.or.kr/api"
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 0) 종목코드 → corp_code 매핑 (DART 고유번호)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_corp_code_map() -> dict:
    """KRX 종목코드(6자리) → DART 고유번호(8자리) 매핑."""
    cache_file = CACHE_DIR / "corpcode.csv"
    
    # 캐시가 7일 이내면 사용
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 7 * 86400:
            df = pd.read_csv(cache_file, dtype={"corp_code": str, "stock_code": str})
            return dict(zip(df["stock_code"], df["corp_code"]))
    
    # DART에서 다운로드 (ZIP)
    url = f"{DART_BASE}/corpCode.xml?crtfc_key={DART_API_KEY}"
    r = requests.get(url, timeout=30)
    
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("CORPCODE.xml") as f:
            xml = f.read().decode("utf-8")
    
    # 간단 파싱 (BeautifulSoup 의존 회피)
    rows = []
    for m in re.finditer(
        r"<corp_code>(\d+)</corp_code>\s*<corp_name>([^<]+)</corp_name>"
        r"\s*<corp_eng_name>[^<]*</corp_eng_name>\s*<stock_code>([^<]*)</stock_code>",
        xml,
    ):
        corp_code, corp_name, stock_code = m.groups()
        if stock_code.strip():
            rows.append({
                "corp_code": corp_code,
                "corp_name": corp_name,
                "stock_code": stock_code.strip(),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(cache_file, index=False)
    return dict(zip(df["stock_code"], df["corp_code"]))


# ---------------------------------------------------------------------------
# 1) 재무제표 API 호출
# ---------------------------------------------------------------------------
def fetch_fnltt(corp_code: str, year: int, reprt_code: str = "11011") -> Optional[pd.DataFrame]:
    """
    단일회사 주요계정 조회.
    reprt_code: 11011=사업보고서(연간), 11012=반기, 11013=1Q, 11014=3Q
    """
    url = f"{DART_BASE}/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "fs_div": "CFS",  # 연결재무제표 (없으면 OFS 시도)
    }
    
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("status") != "000":
            # 연결 없으면 별도 시도
            params["fs_div"] = "OFS"
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if data.get("status") != "000":
                return None
        return pd.DataFrame(data.get("list", []))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2) 9개 섹션 데이터 정리
# ---------------------------------------------------------------------------
def parse_section_data(df: pd.DataFrame) -> dict:
    """DART 응답을 koreantickers 9개 섹션 구조로 변환."""
    
    def find(account_keywords, default=0):
        """계정명 키워드로 값 찾기 (당기금액)."""
        if df is None or df.empty:
            return default
        for kw in account_keywords:
            mask = df["account_nm"].str.contains(kw, regex=False, na=False)
            if mask.any():
                val = df[mask].iloc[0].get("thstrm_amount", "0")
                try:
                    return int(val.replace(",", "")) if val else default
                except (ValueError, AttributeError):
                    return default
        return default
    
    # 손익계산서
    revenue = find(["수익(매출액)", "매출액", "영업수익"])
    op_income = find(["영업이익"])
    net_income = find(["당기순이익", "분기순이익", "반기순이익"])
    
    # 재무상태표
    total_assets = find(["자산총계"])
    total_liab = find(["부채총계"])
    total_equity = find(["자본총계"])
    current_assets = find(["유동자산"])
    current_liab = find(["유동부채"])
    cash = find(["현금및현금성자산"])
    receivables = find(["매출채권"])
    inventory = find(["재고자산"])
    
    # 현금흐름표
    op_cf = find(["영업활동"])
    inv_cf = find(["투자활동"])
    fin_cf = find(["재무활동"])
    
    return {
        # Section 1: Growth
        "revenue": revenue,
        "op_income": op_income,
        "net_income": net_income,
        
        # Section 2: Profitability
        "op_margin": (op_income / revenue * 100) if revenue else 0,
        "net_margin": (net_income / revenue * 100) if revenue else 0,
        "roe": (net_income / total_equity * 100) if total_equity else 0,
        
        # Section 3: Balance Sheet
        "total_assets": total_assets,
        "total_liab": total_liab,
        "total_equity": total_equity,
        
        # Section 4: Cash Flow
        "op_cf": op_cf,
        "inv_cf": inv_cf,
        "fin_cf": fin_cf,
        
        # Section 5: Liquidity
        "cash": cash,
        "current_liab": current_liab,
        "current_ratio": (current_assets / current_liab) if current_liab else 0,
        
        # Section 6: Working Capital
        "receivables": receivables,
        "inventory": inventory,
        "current_assets": current_assets,
    }


# ---------------------------------------------------------------------------
# 3) 메인 함수 — 5분기 시계열
# ---------------------------------------------------------------------------
def get_full_financials(stock_code: str) -> Optional[dict]:
    """
    종목코드 (6자리) → 9개 섹션 5분기 시계열 데이터.
    
    Returns:
        {
            "stock_code": "005930",
            "corp_code": "00126380",
            "periods": ["24 3Q", "24 4Q", "25 1Q", "25 2Q", "25 3Q"],
            "growth": {...},  # 5분기 시계열
            "profitability": {...},
            "balance_sheet": {...},
            "cash_flow": {...},
            "liquidity": {...},
            "working_capital": {...},
            "valuation": {...},  # 최신 분기만
            "filing_timeline": [...],
        }
    """
    if not DART_API_KEY:
        return None
    
    corp_map = get_corp_code_map()
    corp_code = corp_map.get(stock_code)
    if not corp_code:
        return None
    
    # 최근 5분기 조회 (간단화: 최근 2년)
    today = dt.date.today()
    year = today.year
    
    quarters = []
    section_data = []
    
    # 분기 코드: 1Q=11013, 반기=11012, 3Q=11014, 사업=11011
    quarter_codes = [
        (year, "11013", "Q1"),
        (year - 1, "11014", "3Q"),
        (year - 1, "11012", "H1"),
        (year - 1, "11013", "Q1"),
        (year - 2, "11011", "FY"),
    ]
    
    for y, rcode, qlabel in quarter_codes:
        df = fetch_fnltt(corp_code, y, rcode)
        if df is not None and not df.empty:
            quarters.append(f"{str(y)[-2:]} {qlabel}")
            section_data.append(parse_section_data(df))
        time.sleep(0.1)  # API rate limit
    
    if not section_data:
        return None
    
    # 시계열로 재구성
    def to_series(key):
        return [d[key] for d in section_data]
    
    return {
        "stock_code": stock_code,
        "corp_code": corp_code,
        "periods": quarters,
        "growth": {
            "revenue": to_series("revenue"),
            "op_income": to_series("op_income"),
            "net_income": to_series("net_income"),
        },
        "profitability": {
            "op_margin": to_series("op_margin"),
            "net_margin": to_series("net_margin"),
            "roe": to_series("roe"),
        },
        "balance_sheet": {
            "total_assets": to_series("total_assets"),
            "total_liab": to_series("total_liab"),
            "total_equity": to_series("total_equity"),
        },
        "cash_flow": {
            "op_cf": to_series("op_cf"),
            "inv_cf": to_series("inv_cf"),
            "fin_cf": to_series("fin_cf"),
        },
        "liquidity": {
            "cash": to_series("cash"),
            "current_liab": to_series("current_liab"),
            "total_liab": to_series("total_liab"),
            "current_ratio": section_data[-1]["current_ratio"] if section_data else 0,
        },
        "working_capital": {
            "receivables": to_series("receivables"),
            "inventory": to_series("inventory"),
            "current_assets": to_series("current_assets"),
        },
    }


def calculate_valuation(stock_code: str, market_cap: float, latest_fin: dict) -> dict:
    """밸류에이션 멀티플 계산 (Section 7)."""
    annual_revenue = latest_fin.get("growth", {}).get("revenue", [0])[-1]
    annual_ni = latest_fin.get("growth", {}).get("net_income", [0])[-1]
    total_equity = latest_fin.get("balance_sheet", {}).get("total_equity", [0])[-1]
    cash = latest_fin.get("liquidity", {}).get("cash", [0])[-1]
    total_liab = latest_fin.get("balance_sheet", {}).get("total_liab", [0])[-1]
    
    ev = market_cap - cash + total_liab
    
    return {
        "per": (market_cap / annual_ni) if annual_ni else None,
        "pbr": (market_cap / total_equity) if total_equity else None,
        "psr": (market_cap / annual_revenue) if annual_revenue else None,
        "ev_sales": (ev / annual_revenue) if annual_revenue else None,
        "ev_op": (ev / latest_fin.get("growth", {}).get("op_income", [0])[-1])
                 if latest_fin.get("growth", {}).get("op_income", [0])[-1] else None,
        "earnings_yield": (annual_ni / market_cap * 100) if market_cap else None,
        "debt_mkt_cap": (total_liab / market_cap * 100) if market_cap else None,
        "market_cap": market_cap,
        "enterprise_value": ev,
    }


# ---------------------------------------------------------------------------
# 4) 최신 공시 조회 (Section 9)
# ---------------------------------------------------------------------------
def get_recent_filings(corp_code: str, n: int = 10) -> list:
    """최근 공시 N건."""
    url = f"{DART_BASE}/list.json"
    end = dt.date.today().strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=180)).strftime("%Y%m%d")
    
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": start,
        "end_de": end,
        "pblntf_ty": "A",  # 정기공시
        "page_count": n,
    }
    
    try:
        r = requests.get(url, params=params, timeout=15)
        return r.json().get("list", [])[:n]
    except Exception:
        return []


if __name__ == "__main__":
    # 테스트: 삼성전자
    if not DART_API_KEY:
        print("⚠️ DART_API_KEY 환경변수 설정 필요!")
        print("   https://opendart.fss.or.kr/ → 회원가입 → API 키 발급")
    else:
        data = get_full_financials("005930")
        if data:
            print(f"✅ {data['stock_code']} 재무 데이터:")
            print(f"   기간: {data['periods']}")
            print(f"   매출: {data['growth']['revenue']}")
            print(f"   영업이익률: {[f'{x:.1f}%' for x in data['profitability']['op_margin']]}")
