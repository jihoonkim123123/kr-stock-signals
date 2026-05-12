"""
네이버 뉴스 감성 분석 모듈
====================
네이버 금융 종목별 뉴스를 크롤링하여 감성 점수 산출.
야후파이낸스보다 한국 주식에 훨씬 정확.

URL 패턴:
- https://finance.naver.com/item/news_news.naver?code=005930

추출 데이터:
- 헤드라인
- 발행일/시간
- 언론사
- 본문 일부 (선택)
"""
from __future__ import annotations

import re
import time
import datetime as dt
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup


# 한국어 긍정/부정 키워드 (확장)
POSITIVE_KW = [
    # 실적 관련
    "어닝서프라이즈", "호실적", "사상최고", "신고가", "최대실적", "사상최대",
    "분기최대", "역대최고", "흑자전환", "이익급증", "실적개선", "수익성개선",
    "영업이익증가", "매출증가", "전년대비증가", "yoy증가", "qoq증가",
    
    # 주가 관련
    "급등", "강세", "상승", "반등", "회복", "상한가", "신고가갱신",
    "목표가상향", "투자의견상향", "매수추천", "buy", "outperform",
    "overweight", "strong buy", "best pick", "탑픽",
    
    # 사업 관련
    "수주", "계약체결", "독점공급", "특허", "신제품", "확대", "진출",
    "성공", "돌파", "최초", "글로벌", "1위", "선두", "주도",
    "투자유치", "MOU", "파트너십", "협력", "공급계약", "수출확대",
    
    # AI/기술
    "AI", "엔비디아", "HBM", "슈퍼사이클", "수혜", "공급", "양산",
    "FAANG", "M7", "데이터센터", "반도체", "메모리", "DRAM",
    
    # 외국인/기관
    "외국인매수", "기관매수", "패시브유입", "ETF편입", "MSCI편입",
]

NEGATIVE_KW = [
    # 실적 관련
    "어닝쇼크", "적자전환", "적자확대", "실적부진", "이익감소",
    "매출감소", "전년대비감소", "yoy감소", "qoq감소", "예상하회",
    "컨센서스하회", "가이던스하향",
    
    # 주가 관련
    "급락", "약세", "하락", "폭락", "하한가", "신저가", "전저점",
    "목표가하향", "투자의견하향", "매도추천", "sell", "underperform",
    "underweight", "downgrade",
    
    # 사업 관련
    "리콜", "수사", "조사", "벌금", "소송", "분쟁", "철수",
    "감원", "구조조정", "결함", "리스크", "우려", "위기",
    "지연", "취소", "철회", "보류", "축소", "감소",
    
    # 거시
    "경기침체", "둔화", "위축", "약화", "부진", "불황",
    "외국인매도", "기관매도", "공매도", "shortselling",
]


def fetch_naver_news(stock_code: str, days: int = 14, max_items: int = 30) -> list:
    """
    네이버 금융 종목 뉴스 크롤링.
    
    Returns:
        [
            {
                "title": "삼성전자, HBM4 양산 본격화...",
                "date": "2026.05.12 09:30",
                "press": "이데일리",
                "url": "https://...",
            },
            ...
        ]
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://finance.naver.com/item/main.naver?code={stock_code}",
    }
    
    url = f"https://finance.naver.com/item/news_news.naver?code={stock_code}&page=1"
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "euc-kr"  # 네이버 금융 인코딩
        soup = BeautifulSoup(r.text, "html.parser")
        
        items = []
        cutoff = dt.datetime.now() - dt.timedelta(days=days)
        
        # 뉴스 테이블 행
        rows = soup.select("table.type5 tr")
        for row in rows:
            title_el = row.select_one("td.title a")
            date_el = row.select_one("td.date")
            press_el = row.select_one("td.info")
            
            if not title_el or not date_el:
                continue
            
            title = title_el.get_text(strip=True)
            date_str = date_el.get_text(strip=True)
            press = press_el.get_text(strip=True) if press_el else ""
            href = title_el.get("href", "")
            
            # 날짜 파싱
            try:
                article_date = dt.datetime.strptime(date_str, "%Y.%m.%d %H:%M")
                if article_date < cutoff:
                    continue
            except ValueError:
                article_date = None
            
            news_url = (
                f"https://finance.naver.com{href}" if href.startswith("/")
                else href
            )
            
            items.append({
                "title": title,
                "date": date_str,
                "press": press,
                "url": news_url,
                "datetime": article_date,
            })
            
            if len(items) >= max_items:
                break
        
        return items
    except Exception as e:
        return []


def analyze_sentiment(news_items: list) -> dict:
    """
    뉴스 헤드라인 감성 분석.
    
    Returns:
        {
            "score": 25.5,           # -100 ~ +100
            "positive_count": 8,
            "negative_count": 3,
            "neutral_count": 5,
            "total": 16,
            "headlines": [
                {"title": "...", "tag": "📈", "url": "..."},
                ...
            ],
        }
    """
    if not news_items:
        return {
            "score": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "total": 0,
            "headlines": [],
        }
    
    pos_total, neg_total = 0, 0
    headlines = []
    
    for item in news_items:
        title = item["title"].lower()
        
        # 키워드 카운트
        pos_kw = sum(1 for kw in POSITIVE_KW if kw.lower() in title)
        neg_kw = sum(1 for kw in NEGATIVE_KW if kw.lower() in title)
        
        pos_total += pos_kw
        neg_total += neg_kw
        
        # 태그 결정
        if pos_kw > neg_kw:
            tag = "📈"
            sentiment = "positive"
        elif neg_kw > pos_kw:
            tag = "📉"
            sentiment = "negative"
        else:
            tag = "·"
            sentiment = "neutral"
        
        headlines.append({
            "title": item["title"],
            "tag": tag,
            "url": item["url"],
            "date": item["date"],
            "press": item["press"],
            "sentiment": sentiment,
        })
    
    # 감성 점수 계산
    total = pos_total + neg_total
    if total > 0:
        score = round((pos_total - neg_total) / total * 100, 1)
    else:
        score = 0.0
    
    # 카운트
    pos_count = sum(1 for h in headlines if h["sentiment"] == "positive")
    neg_count = sum(1 for h in headlines if h["sentiment"] == "negative")
    neu_count = sum(1 for h in headlines if h["sentiment"] == "neutral")
    
    return {
        "score": score,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "total": len(headlines),
        "headlines": headlines[:10],  # 상위 10개만 저장
    }


def fetch_and_analyze(stock_code: str, days: int = 14) -> dict:
    """원스톱: 뉴스 크롤링 + 감성 분석."""
    items = fetch_naver_news(stock_code, days=days)
    result = analyze_sentiment(items)
    result["stock_code"] = stock_code
    return result


def batch_sentiment_analysis(stock_codes: list, workers: int = 4) -> dict:
    """여러 종목 일괄 뉴스 감성 분석."""
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_and_analyze, code): code for code in stock_codes}
        for f in as_completed(futs):
            code = futs[f]
            try:
                data = f.result()
                results[code] = data
            except Exception:
                pass
            time.sleep(0.1)  # 네이버 rate limit 고려
    return results


if __name__ == "__main__":
    # 테스트: 삼성전자
    print("📰 네이버 뉴스 감성 분석 테스트")
    print("=" * 60)
    
    result = fetch_and_analyze("005930", days=7)
    
    print(f"\n📊 삼성전자 (005930)")
    print(f"   감성 점수: {result['score']:+.1f} / 100")
    print(f"   긍정: {result['positive_count']}건")
    print(f"   부정: {result['negative_count']}건")
    print(f"   중립: {result['neutral_count']}건")
    print(f"\n📑 헤드라인 (상위 5개):")
    for h in result["headlines"][:5]:
        print(f"   {h['tag']} {h['title']}")
        print(f"      {h['date']} · {h['press']}")
