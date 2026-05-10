# 한국주식 매매 신호 자동화

KOSPI200 + KOSDAQ150 전체 종목을 매일 평일 아침 자동으로 스캔해서
**단기 스윙(반등형)** · **장기 추세추종(정배열)** 매매 신호 TOP 10을 GitHub 이슈로 알림으로 받고,
GitHub Pages에 호스팅된 대시보드에서 검색·정렬해 볼 수 있는 시스템입니다.

> ⚠️ 이 도구는 **신호와 후보 종목**만 자동으로 알려줍니다. 자동 매매는 하지 않습니다.
> 모든 매매 결정과 주문 실행은 본인이 HTS/MTS에서 직접 합니다.

## 구성

```
.
├── kr_stock_signals.py     # 매일 신호 스캔 (대시보드 + 마크다운 생성)
├── kr_stock_backtest.py    # 같은 점수 체계 과거 검증
├── requirements.txt
└── .github/workflows/
    ├── daily-signals.yml   # 평일 8:30 KST 자동 실행 → GitHub Pages + Issue
    └── backtest.yml        # 토요일 7:00 KST 자동 + 수동 실행 가능
```

## 1. GitHub repo 만들기

이 폴더 전체가 그대로 repo가 되도록 구성되어 있어요. 두 가지 방법:

### 방법 A — 웹에서 만들고 업로드 (추천, 가장 쉬움)

1. https://github.com/new 에서 새 repo 생성
   - 이름: `kr-stock-signals` (자유)
   - **Public** 으로 만드세요 (Pages 무료 호스팅을 위해; 사적인 정보 없음)
2. 생성 직후 repo 페이지의 "uploading an existing file" 링크 클릭
3. 이 폴더의 파일들 **전체 드래그&드롭**으로 업로드
   - `kr_stock_signals.py`, `kr_stock_backtest.py`, `requirements.txt`, `.gitignore`, `README.md`
   - `.github` 폴더 통째로
4. "Commit changes" 클릭

### 방법 B — git CLI 로 push

```bash
cd "<이 폴더>"
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```

## 2. GitHub Pages 활성화 (대시보드 호스팅)

repo 페이지에서:

1. **Settings** 탭 → 좌측 메뉴 **Pages**
2. "Source" 드롭다운에서 **GitHub Actions** 선택 (Branch 가 아닌!)
3. 저장

이게 끝나면 첫 번째 자동 실행 후 `https://USERNAME.github.io/REPO/` 에서 대시보드를 볼 수 있어요.

## 3. 첫 실행 (테스트)

자동 실행을 기다릴 필요 없이 지금 바로 한번 돌려보세요:

1. **Actions** 탭 → 좌측에서 **📊 Daily KR Stock Signals** 선택
2. 오른쪽 **Run workflow** 버튼 → 다시 **Run workflow** 클릭
3. 약 5~10분 후 초록색 체크 ✓ 가 뜨면 완료
4. 동시에:
   - **Issues** 탭에 "📊 매매 신호 YYYY-MM-DD" 라는 이슈가 생성됨 (등록한 이메일로 알림 옴)
   - `https://USERNAME.github.io/REPO/` 에 대시보드 업데이트

## 4. 자동 스케줄

### 매일 신호 (`daily-signals.yml`)

- **평일 오전 8:30 KST** (= 23:30 UTC 일~목)
- GitHub Actions cron 은 분산 처리로 5~30분 지연될 수 있어 실제로는 8:30~9:00 사이 실행
- 시장 개장(9:00) 전에 결과를 받을 가능성이 높지만, 늦으면 개장 직후 도착할 수도 있음

### 주간 백테스트 (`backtest.yml`)

- **토요일 오전 7:00 KST** 자동 실행 (한 주 마감 후 검증)
- Actions 탭에서 **Run workflow** 로 언제든 수동 실행 가능 (유니버스·기간·점수 임계값 조정 옵션 제공)
- 결과는 **Actions → 해당 run → Artifacts** 에서 `backtest_report.html` 다운로드해서 브라우저로 열기

## 5. 알림 받는 법

GitHub 이슈가 생성되면 자동으로 등록된 이메일로 알림이 와요. 모바일에서 GitHub 앱을 깔아두면 푸시 알림도 받을 수 있고요.

이메일 알림 설정 확인: https://github.com/settings/notifications

## 6. 점수 체계 요약

### 단기 스윙 (반등형)

- RSI 30~45 + 어제 대비 반등 시작 (+30점)
- 볼린저 밴드 하단 부근 (+22점)
- 거래량 평소 대비 2배 이상 (+18점)
- 양봉 + 5일선 회복 (+15점)
- MACD 히스토그램 반등 (+10점)
- 60일 -25% 이상 급락 종목 감점 (-15점)

진입가는 다음날 시가, **손절 -2×ATR / 익절 +3×ATR / 최대 10거래일 보유**.

### 장기 추세추종 (정배열)

- MA5 > MA20 > MA60 > MA120 완전 정배열 (+35점)
- 종가가 MA20 위 (+12점)
- MACD 매수 시그널 (+18점)
- RSI 50~70 적정 강세 (+15점, 80 이상은 -10점 과열 감점)
- 60일 모멘텀 (최대 +15점)

진입 후 **MA20 < MA60 추세파괴 청산 또는 -8% 트레일링 스톱**.

70점 이상이 진입 검토 구간이에요.

## 7. 파라미터 튜닝

- `kr_stock_backtest.py` 상단 `CONFIG` 딕셔너리 수정 후 push
- 또는 **Actions → 🧪 Strategy Backtest → Run workflow** 에서 즉석 변경 가능
  (유니버스, 기간, 점수 임계값 셋만 바꿀 수 있음)

수정 가능한 항목:
- `score_threshold` (70 → 80 으로 올리면 거래 적지만 신호 품질 ↑)
- `swing_target_atr` / `swing_stop_atr` (수익 vs 손절 비율)
- `swing_max_hold` (스윙 최대 보유일)
- `trend_trail_pct` (추세 트레일링 스톱 폭)
- 거래 비용 (`commission_*`, `tax_sell`, `slippage`) — 실제 사용 증권사 수치로 조정

## 8. 비용

- **GitHub 무료 플랜으로 충분** — Public repo 의 Actions·Pages 모두 무제한 무료
- Private repo 는 월 2000분 Actions 무료, 신호 10분 × 22일 = 220분 + 백테스트 = 충분

## 9. 로컬에서도 돌려보고 싶다면

```bash
pip install -r requirements.txt
python kr_stock_signals.py    # dashboard.html, top.md 생성
python kr_stock_backtest.py   # backtest_report.html, backtest_trades.csv 생성
```

## 10. 주의사항

- **이 시스템은 신호만 만들고, 실제 매매는 하지 않습니다.**
- 모든 매매 결정과 책임은 본인에게 있어요.
- 백테스트 성과가 좋아도 미래 수익을 보장하지 않습니다 — 시장은 변합니다.
- 큰 자본 투입 전에 1~2주 가벼운 금액으로 신호 적중률을 직접 검증하세요.
- 시장 변동성이 큰 날 (실적 발표, 거시 이슈, 옵션 만기) 에는 평소보다 보수적으로 접근.
