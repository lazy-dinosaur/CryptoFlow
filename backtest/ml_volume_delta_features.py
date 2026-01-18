#!/usr/bin/env python3
"""
Volume/Delta Pattern Feature Extraction for 1-Minute Candles

1분봉 윈도우에서 볼륨/델타 패턴 피처를 추출하여
채널 터치 시 반전 확률을 예측하는 ML 모델에 사용

피처 카테고리:
1. 볼륨 패턴 (5개)
2. 델타 패턴 (5개)
3. 가격 패턴 (3개)
"""

import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats


def extract_1m_features(
    df_1m: pd.DataFrame,
    direction: str,
    channel_support: float,
    channel_resistance: float
) -> dict:
    """
    1분봉 윈도우에서 볼륨/델타 패턴 피처 추출

    Args:
        df_1m: 최근 20-30개 1분봉 데이터 (time, open, high, low, close, volume, delta)
        direction: 'LONG' or 'SHORT'
        channel_support: 채널 지지선
        channel_resistance: 채널 저항선

    Returns:
        dict: 13-15개 피처
    """
    if len(df_1m) < 5:
        return _get_default_features()

    # 기본 데이터 추출
    opens = df_1m['open'].values
    highs = df_1m['high'].values
    lows = df_1m['low'].values
    closes = df_1m['close'].values
    volumes = df_1m['volume'].values
    deltas = df_1m['delta'].values if 'delta' in df_1m.columns else np.zeros(len(df_1m))

    # 윈도우 분할: 최근 5분 vs 이전 15분
    recent_n = min(5, len(df_1m))
    lookback_n = min(15, len(df_1m) - recent_n)

    # 최근 5분
    recent_volumes = volumes[-recent_n:]
    recent_deltas = deltas[-recent_n:]
    recent_closes = closes[-recent_n:]
    recent_opens = opens[-recent_n:]
    recent_highs = highs[-recent_n:]
    recent_lows = lows[-recent_n:]

    # 이전 15분 (lookback)
    if lookback_n > 0:
        lookback_volumes = volumes[-(recent_n + lookback_n):-recent_n]
        lookback_deltas = deltas[-(recent_n + lookback_n):-recent_n]
    else:
        lookback_volumes = recent_volumes
        lookback_deltas = recent_deltas

    features = {}

    # ========== 1. 볼륨 패턴 피처 (5개) ==========

    # vol_spike_ratio: 최근 5분 평균 볼륨 / 이전 15분 평균 볼륨
    avg_recent_vol = np.mean(recent_volumes) if len(recent_volumes) > 0 else 1
    avg_lookback_vol = np.mean(lookback_volumes) if len(lookback_volumes) > 0 else 1
    features['vol_spike_ratio'] = avg_recent_vol / (avg_lookback_vol + 1e-10)

    # vol_trend: 볼륨 선형 회귀 기울기 (증가/감소 추세)
    if len(volumes) >= 5:
        x = np.arange(len(volumes[-10:]))
        slope, _, _, _, _ = stats.linregress(x, volumes[-10:])
        features['vol_trend'] = slope / (np.mean(volumes[-10:]) + 1e-10)
    else:
        features['vol_trend'] = 0.0

    # vol_climax: 마지막 캔들이 윈도우 내 최대 볼륨인지 (0 or 1)
    if len(volumes) >= 3:
        features['vol_climax'] = 1.0 if volumes[-1] == np.max(volumes[-10:]) else 0.0
    else:
        features['vol_climax'] = 0.0

    # vol_exhaustion: 볼륨이 점점 줄어드는지 (최근 5개 중 감소 비율)
    if len(recent_volumes) >= 3:
        decreasing = sum(1 for i in range(1, len(recent_volumes))
                        if recent_volumes[i] < recent_volumes[i-1])
        features['vol_exhaustion'] = decreasing / (len(recent_volumes) - 1)
    else:
        features['vol_exhaustion'] = 0.0

    # vol_distribution: 상승캔들 볼륨 비율 vs 하락캔들
    up_candle_mask = recent_closes > recent_opens
    down_candle_mask = ~up_candle_mask
    up_vol = np.sum(recent_volumes[up_candle_mask]) if np.any(up_candle_mask) else 0
    down_vol = np.sum(recent_volumes[down_candle_mask]) if np.any(down_candle_mask) else 0
    total_vol = up_vol + down_vol + 1e-10
    features['vol_distribution'] = up_vol / total_vol  # 높을수록 매수 우세

    # ========== 2. 델타 패턴 피처 (5개) ==========

    # delta_reversal: 델타 방향 전환 감지
    # LONG 진입 시: 음→양 전환 = 좋음, SHORT 진입 시: 양→음 전환 = 좋음
    if len(recent_deltas) >= 3:
        prev_delta_sum = np.sum(recent_deltas[:-2])
        last_delta_sum = np.sum(recent_deltas[-2:])

        if direction == 'LONG':
            # 음→양 전환이면 1, 아니면 0
            features['delta_reversal'] = 1.0 if (prev_delta_sum < 0 and last_delta_sum > 0) else 0.0
        else:
            # 양→음 전환이면 1
            features['delta_reversal'] = 1.0 if (prev_delta_sum > 0 and last_delta_sum < 0) else 0.0
    else:
        features['delta_reversal'] = 0.0

    # delta_momentum: 최근 5분 델타 합 / 이전 15분 합
    recent_delta_sum = np.sum(recent_deltas)
    lookback_delta_sum = np.sum(lookback_deltas) if len(lookback_deltas) > 0 else 1
    features['delta_momentum'] = recent_delta_sum / (np.abs(lookback_delta_sum) + 1e-10)

    # delta_divergence: 가격 vs 델타 다이버전스
    # LONG: 가격 하락 + 델타 양전환 = 불 다이버전스
    # SHORT: 가격 상승 + 델타 음전환 = 베어 다이버전스
    if len(recent_closes) >= 3 and len(recent_deltas) >= 3:
        price_change = recent_closes[-1] - recent_closes[0]
        delta_trend = recent_deltas[-1] - recent_deltas[0]

        if direction == 'LONG':
            # 가격 하락 + 델타 상승 = 불 다이버전스
            features['delta_divergence'] = 1.0 if (price_change < 0 and delta_trend > 0) else 0.0
        else:
            # 가격 상승 + 델타 하락 = 베어 다이버전스
            features['delta_divergence'] = 1.0 if (price_change > 0 and delta_trend < 0) else 0.0
    else:
        features['delta_divergence'] = 0.0

    # delta_absorption: 매도 볼륨 나오는데 가격 유지 (LONG의 경우)
    # 매수 볼륨 나오는데 가격 유지 (SHORT의 경우)
    if len(recent_deltas) >= 3:
        price_range = np.max(recent_highs) - np.min(recent_lows)
        price_change_pct = abs(recent_closes[-1] - recent_closes[0]) / (recent_closes[0] + 1e-10)
        delta_magnitude = np.abs(np.sum(recent_deltas))

        if direction == 'LONG':
            # 큰 매도(음의 델타)가 있지만 가격이 안 떨어짐
            if recent_delta_sum < 0 and price_change_pct < 0.001:
                features['delta_absorption'] = min(delta_magnitude / 100, 1.0)
            else:
                features['delta_absorption'] = 0.0
        else:
            # 큰 매수(양의 델타)가 있지만 가격이 안 오름
            if recent_delta_sum > 0 and price_change_pct < 0.001:
                features['delta_absorption'] = min(delta_magnitude / 100, 1.0)
            else:
                features['delta_absorption'] = 0.0
    else:
        features['delta_absorption'] = 0.0

    # cvd_slope: CVD(Cumulative Volume Delta) 선형 회귀 기울기
    if len(deltas) >= 5:
        cvd = np.cumsum(deltas)
        x = np.arange(len(cvd[-10:]))
        slope, _, _, _, _ = stats.linregress(x, cvd[-10:])
        # 정규화: LONG은 양의 기울기가 좋고, SHORT은 음의 기울기가 좋음
        features['cvd_slope'] = slope / (np.abs(np.mean(deltas[-10:])) + 1e-10)
    else:
        features['cvd_slope'] = 0.0

    # ========== 3. 가격 패턴 피처 (3개) ==========

    # wick_rejection: 서포트/레지스턴스에서 긴 꼬리 형성
    last_candle_idx = -1
    body = abs(closes[last_candle_idx] - opens[last_candle_idx])
    full_range = highs[last_candle_idx] - lows[last_candle_idx]

    if full_range > 0:
        if direction == 'LONG':
            # 아래꼬리 비율 (서포트에서 반등)
            lower_wick = min(closes[last_candle_idx], opens[last_candle_idx]) - lows[last_candle_idx]
            features['wick_rejection'] = lower_wick / full_range
        else:
            # 위꼬리 비율 (레지스턴스에서 밀림)
            upper_wick = highs[last_candle_idx] - max(closes[last_candle_idx], opens[last_candle_idx])
            features['wick_rejection'] = upper_wick / full_range
    else:
        features['wick_rejection'] = 0.0

    # price_momentum_1m: 1분봉 기준 모멘텀 (최근 5분)
    if len(recent_closes) >= 2:
        momentum = (recent_closes[-1] - recent_closes[0]) / (recent_closes[0] + 1e-10)
        # LONG은 양의 모멘텀이, SHORT은 음의 모멘텀이 유리
        if direction == 'LONG':
            features['price_momentum_1m'] = momentum
        else:
            features['price_momentum_1m'] = -momentum
    else:
        features['price_momentum_1m'] = 0.0

    # touch_precision: 정확히 채널에 닿았는지 vs 오버슈트
    current_low = lows[-1]
    current_high = highs[-1]

    if direction == 'LONG':
        # 서포트 터치 정밀도: 서포트에 얼마나 가까이 닿았는지
        distance_to_support = (current_low - channel_support) / channel_support
        features['touch_precision'] = 1.0 - min(abs(distance_to_support) * 100, 1.0)  # 가까울수록 1
    else:
        # 레지스턴스 터치 정밀도
        distance_to_resistance = (channel_resistance - current_high) / channel_resistance
        features['touch_precision'] = 1.0 - min(abs(distance_to_resistance) * 100, 1.0)

    # ========== 4. 추가 컨텍스트 피처 (2개) ==========

    # direction_aligned_delta: 방향에 맞는 델타인지
    if direction == 'LONG':
        features['direction_aligned_delta'] = 1.0 if recent_delta_sum > 0 else 0.0
    else:
        features['direction_aligned_delta'] = 1.0 if recent_delta_sum < 0 else 0.0

    # recent_delta_strength: 최근 델타의 강도
    features['recent_delta_strength'] = np.abs(recent_delta_sum)

    return features


def extract_1m_features_for_df(
    df_1m: pd.DataFrame,
    idx: int,
    direction: str,
    channel_support: float,
    channel_resistance: float,
    window_size: int = 20
) -> dict:
    """
    DataFrame의 특정 인덱스에서 1분봉 피처 추출 (백테스트용)

    Args:
        df_1m: 전체 1분봉 데이터
        idx: 현재 인덱스
        direction: 'LONG' or 'SHORT'
        channel_support: 채널 지지선
        channel_resistance: 채널 저항선
        window_size: 피처 추출에 사용할 윈도우 크기

    Returns:
        dict: 피처 딕셔너리
    """
    start_idx = max(0, idx - window_size + 1)
    window_df = df_1m.iloc[start_idx:idx + 1].reset_index(drop=True)

    return extract_1m_features(
        window_df,
        direction,
        channel_support,
        channel_resistance
    )


def _get_default_features() -> dict:
    """데이터가 부족할 때 기본값 반환"""
    return {
        # 볼륨 패턴
        'vol_spike_ratio': 1.0,
        'vol_trend': 0.0,
        'vol_climax': 0.0,
        'vol_exhaustion': 0.0,
        'vol_distribution': 0.5,
        # 델타 패턴
        'delta_reversal': 0.0,
        'delta_momentum': 0.0,
        'delta_divergence': 0.0,
        'delta_absorption': 0.0,
        'cvd_slope': 0.0,
        # 가격 패턴
        'wick_rejection': 0.0,
        'price_momentum_1m': 0.0,
        'touch_precision': 0.5,
        # 추가 컨텍스트
        'direction_aligned_delta': 0.0,
        'recent_delta_strength': 0.0,
    }


def get_feature_names() -> list:
    """피처 이름 목록 반환"""
    return list(_get_default_features().keys())


def calculate_dynamic_sl(
    df_1m: pd.DataFrame,
    direction: str,
    channel_support: float,
    channel_resistance: float,
    buffer_pct: float = 0.0003
) -> float:
    """
    1분봉 진입 캔들 기반 동적 SL 계산

    Args:
        df_1m: 1분봉 데이터 (최근 데이터 포함)
        direction: 'LONG' or 'SHORT'
        channel_support: 채널 지지선 (폴백용)
        channel_resistance: 채널 저항선 (폴백용)
        buffer_pct: SL 버퍼 (기본 0.03%)

    Returns:
        float: 동적 SL 가격
    """
    if len(df_1m) < 1:
        # 폴백: 채널 기반 SL
        if direction == 'LONG':
            return channel_support * (1 - 0.0008)
        else:
            return channel_resistance * (1 + 0.0008)

    # 진입 캔들 (마지막 캔들)
    entry_candle_low = df_1m['low'].iloc[-1]
    entry_candle_high = df_1m['high'].iloc[-1]

    if direction == 'LONG':
        # 진입 캔들 저점 - 버퍼
        sl = entry_candle_low * (1 - buffer_pct)
        # 채널 지지선보다 위에 있으면 채널 기반으로 (더 보수적)
        channel_sl = channel_support * (1 - 0.0008)
        return min(sl, channel_sl)  # 더 가까운 SL 사용 (보수적)
    else:
        # 진입 캔들 고점 + 버퍼
        sl = entry_candle_high * (1 + buffer_pct)
        # 채널 저항선보다 아래에 있으면 채널 기반으로
        channel_sl = channel_resistance * (1 + 0.0008)
        return max(sl, channel_sl)  # 더 가까운 SL 사용


if __name__ == '__main__':
    # 테스트
    import pandas as pd

    # 테스트 데이터 생성
    np.random.seed(42)
    n = 30
    test_df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='1min'),
        'open': 100000 + np.random.randn(n).cumsum() * 10,
        'high': 100100 + np.random.randn(n).cumsum() * 10,
        'low': 99900 + np.random.randn(n).cumsum() * 10,
        'close': 100050 + np.random.randn(n).cumsum() * 10,
        'volume': np.random.uniform(100, 500, n),
        'delta': np.random.uniform(-100, 100, n)
    })

    # 피처 추출 테스트
    features = extract_1m_features(
        test_df,
        direction='LONG',
        channel_support=99000,
        channel_resistance=101000
    )

    print("=" * 60)
    print("1M Volume/Delta Features")
    print("=" * 60)
    for name, value in features.items():
        print(f"  {name:25s}: {value:+.4f}")

    print(f"\nTotal features: {len(features)}")
    print(f"Feature names: {get_feature_names()}")

    # 동적 SL 테스트
    sl_long = calculate_dynamic_sl(test_df, 'LONG', 99000, 101000)
    sl_short = calculate_dynamic_sl(test_df, 'SHORT', 99000, 101000)
    print(f"\nDynamic SL (LONG): {sl_long:.2f}")
    print(f"Dynamic SL (SHORT): {sl_short:.2f}")
