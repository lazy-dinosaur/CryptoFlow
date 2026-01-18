# Channel Bounce Strategy - 연구 결과 및 계획

> 마지막 업데이트: 2026-01-18

---

## 목차

1. [전략 개요](#1-전략-개요)
2. [백테스트 결과](#2-백테스트-결과)
3. [주요 연구 결과](#3-주요-연구-결과)
4. [시스템 아키텍처](#4-시스템-아키텍처)
5. [향후 계획](#5-향후-계획)

---

## 1. 전략 개요

### 1.1 기본 컨셉

**채널 바운스 (Channel Bounce)** 전략은 가격이 수평 채널의 지지선/저항선에서 반등하는 패턴을 활용합니다.

```
저항선 ─────────────────────  ← SHORT 진입
         ╱╲      ╱╲
        ╱  ╲    ╱  ╲
       ╱    ╲  ╱    ╲
      ╱      ╲╱      ╲
지지선 ─────────────────────  ← LONG 진입
```

### 1.2 타임프레임

| 용도            | 타임프레임 | 설명                     |
| --------------- | ---------- | ------------------------ |
| 채널 감지 (HTF) | 1H         | 스윙 포인트로 채널 형성  |
| 진입 (LTF)      | 15M        | 터치 감지 및 진입 타이밍 |

### 1.3 채널 감지 조건

```python
confirm_candles = 3      # 스윙 확정에 필요한 캔들 수
min_touches = 2          # 최소 터치 횟수
channel_tolerance = 0.4% # 같은 레벨로 인정하는 오차
channel_width = 0.5~3.0% # 유효 채널 폭
```

### 1.4 진입/청산 조건

**LONG 진입:**

- 가격이 지지선 근처 터치 (0.3% 이내)
- 캔들 종가가 지지선 위에서 마감

**SHORT 진입:**

- 가격이 저항선 근처 터치 (0.3% 이내)
- 캔들 종가가 저항선 아래에서 마감

**청산 전략 (현재 사용):**

```
TP1: 채널 중간 (50% 청산)
TP2: 반대편 경계 (50% 청산)
SL:  진입 경계 + 0.08% 버퍼
BE:  TP1 도달 후 SL을 진입가로 이동
```

### 1.5 리스크 관리

```python
risk_per_trade = 1.5%    # 매매당 리스크
max_leverage = 15x       # 최대 레버리지
fee = 0.04%              # 거래 수수료 (편도)
```

---

## 2. 백테스트 결과

### 2.1 전체 성과 (2022-2025)

| 지표               | 값               |
| ------------------ | ---------------- |
| 초기 자본          | $10,000          |
| 최종 자본          | ~$450,000+       |
| 총 수익률          | +4,502%          |
| 총 매매 수         | 984회            |
| 승률 (Win Rate)    | 67.1%            |
| 평균 수익          | +4.58% per trade |
| 최대 낙폭 (Max DD) | -10.4%           |

### 2.2 연도별 성과

| 연도 | 매매 수 | 수익 |
| ---- | ------- | ---- |
| 2022 | ~250    | 양호 |
| 2023 | ~280    | 양호 |
| 2024 | ~300    | 양호 |
| 2025 | ~150    | 양호 |

### 2.3 방향별 성과

| 방향  | 매매 수 | 승률 | 비고      |
| ----- | ------- | ---- | --------- |
| LONG  | ~500    | ~68% | 약간 우수 |
| SHORT | ~480    | ~66% | 양호      |

---

## 3. 주요 연구 결과

### 3.1 TP1 100% vs 현재 전략 비교

| 전략                   | 총 수익 | 승률 | 결론          |
| ---------------------- | ------- | ---- | ------------- |
| TP1 100% 청산          | +4,214% | 높음 | 안정적        |
| TP1 50% + TP2 50% + BE | +4,445% | 67%  | **약간 우수** |

**결론:** 현재 전략이 5.5% 더 좋음. TP2까지 가는 경우의 추가 수익이 BE로 인한 손실을 상쇄.

### 3.2 ML 필터링 효과

| 설정     | 매매 수 | 승률  | 총 수익 |
| -------- | ------- | ----- | ------- |
| ML 없음  | 608     | 70.6% | +3,377% |
| ML p≥0.6 | 204     | 75.0% | +1,296% |

**결론:** ML 필터링은 승률을 높이지만 매매 수 감소로 **총 수익 감소**. 사용하지 않기로 결정.

### 3.3 동시 포지션 분석

| 동시 포지션 수 | 비율 |
| -------------- | ---- |
| 1개            | 67%  |
| 2개            | 24%  |
| 3개+           | 9%   |
| 최대           | 6개  |

**평균 매매 기간:** 18.1시간 (72.5개 15분 캔들)

### 3.4 포지션 제한별 성과

| 설정        | 매매    | 총수익      | 최대 리스크 |
| ----------- | ------- | ----------- | ----------- |
| 무제한      | 984     | +4,502%     | 무제한      |
| L1 + S1     | 729     | +3,043%     | 3%          |
| **L2 + S2** | **933** | **+4,130%** | **6%**      |
| L3 + S3     | 976     | +4,440%     | 9%          |

**결론:** L2 + S2 (LONG 2개 + SHORT 2개) 채택

- 무제한 대비 92% 수익 유지
- 최대 리스크 6%로 제한
- 합리적인 균형점

---

## 4. 시스템 아키텍처

### 4.1 현재 구조

```
┌─────────────────────────────────────────────────────────────┐
│  Data Source (Binance API)                                  │
│  - 15분봉 실시간 수신                                         │
│  - 1시간봉 (채널 감지용)                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Signal Service (channel_signal_service.py)                 │
│  - 채널 감지 (HTF)                                           │
│  - BOUNCE 시그널 생성                                        │
│  - WebSocket으로 프론트엔드 전송                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Paper Trading (ml_paper_trading.py)                        │
│  - 시그널 수신                                               │
│  - 가상 매매 실행                                            │
│  - TP1/TP2/SL 모니터링                                       │
│  - 자본 추적                                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Frontend (React)                                           │
│  - FootprintChart: 차트 + 시그널 표시                        │
│  - MLDashboard: 통계 대시보드                                │
│  - 실시간 WebSocket 연결                                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 데이터베이스 (DuckDB)

```sql
-- 활성 매매
trades: id, symbol, direction, entry_price, sl_price, tp1_price, tp2_price,
        status, entry_time, exit_time, pnl, ...

-- 통계
strategy_stats: total_trades, wins, losses, total_pnl, ...

-- 시그널 히스토리
signals: timestamp, type, direction, price, sl, tp1, tp2, ...
```

### 4.3 프론트엔드 컴포넌트

| 컴포넌트       | 역할                               |
| -------------- | ---------------------------------- |
| FootprintChart | 메인 차트, 시그널 마커, TP/SL 라인 |
| AnalysisLayer  | 차트 오버레이 (채널, 시그널)       |
| MLDashboard    | 통계, 승률, 자본 추적              |

---

## 5. 향후 계획

### 5.1 Phase 1: 실전 트레이딩 준비

#### 거래소 선택

| 거래소    | 수수료 (Taker) | API      | 유동성 | 추천 |
| --------- | -------------- | -------- | ------ | ---- |
| Binance   | 0.04%          | 우수     | 최고   | -    |
| **Bybit** | 0.055%         | **최고** | 우수   | ⭐   |
| Bitget    | 0.06%          | 양호     | 양호   | -    |

**Bybit 선택 이유:**

- API가 알고리즘 트레이딩에 최적화
- Testnet 제공 (무위험 테스트)
- 부분 청산 지원 (TP1 50% + TP2 50%)
- CCXT 라이브러리 호환

#### 구현 항목

- [ ] Bybit API 연동
- [ ] 실제 주문 실행 모듈
- [ ] Testnet 테스트
- [ ] 포지션 제한 로직 (L2 + S2)

### 5.2 Phase 2: 라이브 트레이딩

```python
# Trading Bot 구조
class TradingBot:
    max_long_positions = 2
    max_short_positions = 2
    risk_per_trade = 0.015  # 1.5%

    def on_signal(self, signal):
        # 포지션 제한 체크
        if signal.direction == 'LONG' and self.active_longs >= 2:
            return  # SKIP
        if signal.direction == 'SHORT' and self.active_shorts >= 2:
            return  # SKIP

        # 포지션 크기 계산
        size = self.calculate_position_size(signal)

        # 주문 실행
        self.execute_order(signal, size)
```

### 5.3 Phase 3: 모니터링 및 개선

- [ ] 실시간 대시보드 (실제 수익/손실)
- [ ] 알림 시스템 (Telegram/Discord)
- [ ] 성과 리포트 자동화
- [ ] 파라미터 최적화 (주기적)

### 5.4 리스크 관리 체크리스트

- [ ] 일일 최대 손실 한도 (-5%)
- [ ] 연속 손실 시 일시 중지 (3회)
- [ ] 자본 분리 (트레이딩용 vs 예비)
- [ ] 거래소 API 키 권한 최소화

---

## 6. 기준 스크립트 (Baseline)

### 6.1 실매매 기준 백테스트

**파일:** `backtest/ml_channel_tiebreaker_proper.py`

이 스크립트가 **실제 페이퍼트레이딩 및 라이브 트레이딩에 복사해야 하는 기준 로직**입니다.

```bash
# 실행 방법
cd backtest
python ml_channel_tiebreaker_proper.py narrow
```

### 6.2 Baseline 핵심 설정

| 설정 | 값 | 설명 |
|------|-----|------|
| Tiebreaker | **NARROW** | 동일 점수 채널 중 가장 좁은 채널 선택 |
| HTF | 1H | 채널 감지 |
| LTF | 15M | 진입 실행 |
| Touch Threshold | 0.003 (0.3%) | 터치 인식 범위 |
| SL Buffer | 0.0008 (0.08%) | 손절 버퍼 |
| Cooldown | 20 캔들 (5시간) | 같은 채널 재진입 대기 |

### 6.3 Baseline 결과 (2024년)

```
Trades: 351
Win Rate: 55.0% (outcome >= 0.5 기준)
Return: +5,527,993,669.7%
Max DD: 7.1%
Final Capital: $552,799,376,965.76
```

### 6.4 채널 선택 로직 (NARROW Tiebreaker)

```python
# 1. 모든 유효한 확정 채널 수집
candidates = []
for channel in confirmed_channels:
    if price_inside_channel(channel):
        score = channel.support_touches + channel.resistance_touches
        width_pct = (channel.resistance - channel.support) / channel.support
        candidates.append((score, width_pct, channel))

# 2. 최고 점수 채널들 필터링
max_score = max(c[0] for c in candidates)
top_candidates = [c for c in candidates if c[0] == max_score]

# 3. NARROW tiebreaker: 가장 좁은 채널 선택
if len(top_candidates) == 1:
    best_channel = top_candidates[0][2]
else:
    best_channel = min(top_candidates, key=lambda c: c[1])[2]  # 최소 width
```

### 6.5 관련 파일

| 파일 | 역할 | 비고 |
|------|------|------|
| `backtest/ml_channel_tiebreaker_proper.py` | **Baseline 백테스트** | 기준 로직 |
| `server/ml_paper_trading.py` | 페이퍼 트레이딩 | Baseline 복사됨 |
| `backtest/ml_mtf_bounce.py` | ML 연구용 백테스트 | Baseline 기반 |

### 6.6 주의사항

- **항상 `ml_channel_tiebreaker_proper.py`를 기준으로 로직 검증**
- 페이퍼/라이브 트레이딩 수정 시 반드시 백테스트와 결과 비교
- ML 필터링은 효과 없음 → 사용하지 않음

---

## 부록

### A. 주요 파일 목록

| 파일                                     | 역할               |
| ---------------------------------------- | ------------------ |
| `server/channel_signal_service.py`       | 시그널 서비스      |
| `server/ml_paper_trading.py`             | 페이퍼 트레이딩    |
| `backtest/ml_channel_proper_mtf.py`      | 채널 감지 로직     |
| `backtest/compare_tp1_vs_current.py`     | 전략 비교 백테스트 |
| `backtest/visualize_equity_curve.py`     | 수익 곡선 시각화   |
| `backtest/test_direction_limit.py`       | 포지션 제한 테스트 |
| `pinescripts/channel_bounce_tp1.pine`    | TradingView 전략   |
| `pinescripts/channel_bounce_alerts.pine` | TradingView 알림   |

### B. 핵심 파라미터

```python
# 채널 감지
CONFIRM_CANDLES = 3
MIN_TOUCHES = 2
CHANNEL_TOLERANCE = 0.004  # 0.4%
CHANNEL_WIDTH_MIN = 0.005  # 0.5%
CHANNEL_WIDTH_MAX = 0.03   # 3.0%

# 진입
TOUCH_THRESHOLD = 0.003    # 0.3%
SL_BUFFER = 0.0008         # 0.08%

# 리스크
RISK_PER_TRADE = 0.015     # 1.5%
MAX_LEVERAGE = 15
MAX_LONG_POSITIONS = 2
MAX_SHORT_POSITIONS = 2

# 수수료
TRADING_FEE = 0.0004       # 0.04% (편도)
```

### C. 성과 요약

```
┌────────────────────────────────────────┐
│  Channel Bounce Strategy Summary       │
├────────────────────────────────────────┤
│  기간: 2022-2025 (4년)                  │
│  총 수익: +4,130% (L2+S2 제한)          │
│  승률: 66.2%                           │
│  Max DD: -10.4%                        │
│  매매 수: 933회                         │
│  평균 매매당 수익: +4.43%               │
│  리스크/리워드: 우수                    │
└────────────────────────────────────────┘
```

---

_이 문서는 CryptoFlow 프로젝트의 Channel Bounce 전략 연구 결과를 정리한 것입니다._
