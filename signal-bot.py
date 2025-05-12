# ===============================================
# signal_bot.py – OKX USDT-Perp 전용
#   • 개선된 4-Step 필터 구현
#   • 추세 분석(HH/HL, LH/LL), 채널, RSI 다이버전스, 손익비
#   • 다중 시간대 분석(MTF) 추가 (상위 → 중간 → 하위)
#   • 매물대 근처 시그널 우선순위 반영
#   • OKX = Long·Short 포지션 분석
#   • 결과를 Telegram으로 전송 (GitHub Actions 등에서 스케줄링)
# -----------------------------------------------
# requirements.txt:
# pandas
# numpy
# ta
# requests
# ccxt
# ===============================================
import os
import sys
import time
import datetime as dt
import pandas as pd
import numpy as np
import ta
import requests
import ccxt

# ─────────── 환경변수 (깃허브 Secrets) ───────────
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT = os.getenv("TG_CHAT")
LOGS_DIR = "logs"  # 로그 저장 디렉토리

# ─────────── 매개변수 ───────────
# 다중 시간대 분석을 위한 설정
INTERVAL_DAILY = '1d'        # 일봉 (상위 추세)
INTERVAL_4H = '4h'           # 4시간봉 (중기 추세)
INTERVAL_1H = '1h'           # 1시간봉 (단기 타이밍)

# 각 시간대별 가져올 캔들 수
CANDLES_DAILY = 90           # 약 3개월
CANDLES_4H = 200             # 약 33일 (LEN_CHAN + 여유분)
CANDLES_1H = 180             # 약 7.5일

# 채널 및 필터 설정
LEN_CHAN = 120               # 채널 EMA 길이 (4시간봉 기준 약 20일)
RSI_PERIOD = 14              # RSI 주기
TREND_CHECK_CANDLES = 3      # 추세 확인 시 사용할 최근 캔들 수 (예: 3개 캔들이 연속 HH/HL)
DIVERGENCE_LOOKBACK = 10     # 다이버전스 확인 기간 (캔들 수)
VOLUME_PROFILE_PERIODS = 30  # 매물대 분석 기간 (캔들 수)

# 조건별 가중치 설정 (0~1) - 현재 signal_strength 계산에는 직접 반영되지 않음. 개념적 중요도.
WEIGHT_TREND = 0.3           # 추세 가중치
WEIGHT_CHANNEL = 0.25        # 채널 가중치
WEIGHT_RSI = 0.25            # RSI 가중치
WEIGHT_VOLUME_PROFILE = 0.2  # 매물대 가중치

# 매매 조건 설정
RSI_OVERSOLD = 30            # RSI 과매도
RSI_OVERBOUGHT = 70          # RSI 과매수
MARGIN = 0.02                # 채널 마진 (2%)
SMA_DEVIATION = 0.02         # SMA 편차 허용 범위
MIN_RISK_REWARD = 1.5        # 최소 손익비 (1:1.5)

# 거래량 필터
VOL_MIN_USDT = 1_000_000     # OKX 24h 거래대금

# ─────────── 거래소 인스턴스 ───────────
okx = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

# ─────────── 유틸리티 함수 ───────────

def ensure_dir(directory):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_signal(exchange, symbol, side, signal_data):
    """신호 로깅"""
    ensure_dir(LOGS_DIR)
    timestamp_file = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d") # 파일명 날짜 기준 UTC
    filename = f"{LOGS_DIR}/{exchange}_{timestamp_file}.csv"
    
    log_entry = {
        'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(), # 로그 시간 UTC
        'symbol': symbol,
        'side': side,
        **signal_data
    }
    df = pd.DataFrame([log_entry])
    
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(filename, index=False, encoding='utf-8-sig')

# ─────────── OHLCV 데이터 가져오기 ───────────

def fetch_ohlcv_with_retry(exchange_func, symbol, timeframe, limit, max_retries=3, delay=5):
    """지정된 횟수만큼 재시도하며 OHLCV 데이터 가져오기"""
    for attempt in range(max_retries):
        try:
            ohlcv = exchange_func(symbol, timeframe=timeframe, limit=limit)
            if len(ohlcv) < limit * 0.8: # 데이터가 충분히 오지 않은 경우 (네트워크 등 이슈)
                raise ccxt.NetworkError(f"Insufficient data for {symbol} {timeframe}: got {len(ohlcv)}, expected {limit}")
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # 필수 컬럼 존재 및 NA 값 확인 (최근 데이터에 NA가 있으면 안됨)
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols) or df.iloc[-5:][required_cols].isnull().any().any():
                raise ValueError(f"OHLCV data for {symbol} {timeframe} contains NaNs or missing columns in recent data.")
            return df
        except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as e:
            print(f"[Fetch Retry {attempt+1}/{max_retries}] Failed to fetch {symbol} {timeframe}: {e}")
            if attempt + 1 == max_retries:
                raise  # 마지막 재시도 실패 시 예외 발생
            time.sleep(delay)

def fetch_mtf_data(exchange, symbol):
    """거래소에서 다중 시간대 데이터 가져오기"""
    fetch_func = okx.fetch_ohlcv
    
    daily = fetch_ohlcv_with_retry(fetch_func, symbol, INTERVAL_DAILY, CANDLES_DAILY)
    time.sleep(0.1) # API rate limit
    h4 = fetch_ohlcv_with_retry(fetch_func, symbol, INTERVAL_4H, CANDLES_4H)
    time.sleep(0.1) # API rate limit
    h1 = fetch_ohlcv_with_retry(fetch_func, symbol, INTERVAL_1H, CANDLES_1H)
    
    # 데이터 길이 검증 (각 분석 함수에서 추가 검증 필요)
    if daily is None or h4 is None or h1 is None or \
       len(daily) < TREND_CHECK_CANDLES or len(h4) < LEN_CHAN or len(h1) < RSI_PERIOD:
        print(f"[OKX MTF Data Warning] Insufficient data length for {symbol} after fetch.")
        return None

    return {'daily': daily, '4h': h4, '1h': h1}


# ─────────── 기술적 분석 함수 ───────────

def check_trend(df, side):
    """
    추세 분석 - 최근 TREND_CHECK_CANDLES 동안 연속적인 HH/HL 또는 LH/LL 확인
    """
    if len(df) < TREND_CHECK_CANDLES:
        return False

    highs = df['high'].values[-TREND_CHECK_CANDLES:]
    lows = df['low'].values[-TREND_CHECK_CANDLES:]

    if side == 'long':
        # 모든 고점이 이전 고점보다 높거나 같고 (최소 하나는 확실히 높아야 함), 모든 저점이 이전 저점보다 높음
        all_higher_highs = all(highs[i] >= highs[i-1] for i in range(1, len(highs))) and any(highs[i] > highs[i-1] for i in range(1, len(highs)))
        all_higher_lows = all(lows[i] > lows[i-1] for i in range(1, len(lows)))
        return all_higher_highs and all_higher_lows
    elif side == 'short':
        # 모든 고점이 이전 고점보다 낮고, 모든 저점이 이전 저점보다 낮거나 같음 (최소 하나는 확실히 낮아야 함)
        all_lower_highs = all(highs[i] < highs[i-1] for i in range(1, len(highs)))
        all_lower_lows = all(lows[i] <= lows[i-1] for i in range(1, len(lows))) and any(lows[i] < lows[i-1] for i in range(1, len(lows)))
        return all_lower_highs and all_lower_lows
    return False

def analyze_channel(df):
    """채널 분석 - EMA 기반 채널 계산 및 위치 확인"""
    if len(df) < LEN_CHAN:
        print(f"[Channel Analysis Warning] Insufficient data length for channel: {len(df)} < {LEN_CHAN}")
        return None
    close = df['close']
    
    basis = ta.trend.ema_indicator(close, window=LEN_CHAN)
    if basis is None or basis.isnull().all(): return None # EMA 계산 실패
        
    dev = (close - basis).abs().rolling(window=LEN_CHAN, min_periods=max(1, LEN_CHAN//2)).max() # min_periods 추가
    if dev is None or dev.isnull().all(): return None

    # 마지막 값들이 유효한지 확인
    if pd.isna(basis.iloc[-1]) or pd.isna(dev.iloc[-1]) or pd.isna(close.iloc[-1]):
        return None

    lower = basis - dev
    upper = basis + dev
    mid = (upper + lower) / 2

    if pd.isna(lower.iloc[-1]) or pd.isna(upper.iloc[-1]) or (upper.iloc[-1] - lower.iloc[-1]) == 0:
        return None # 채널 폭이 0이거나 계산 불가

    channel_position = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
    channel_slope = (basis.iloc[-1] - basis.iloc[-5]) / basis.iloc[-5] if len(basis) >= 5 and not pd.isna(basis.iloc[-5]) and basis.iloc[-5] != 0 else 0
    
    return {
        'lower': lower.iloc[-1], 'upper': upper.iloc[-1], 'mid': mid.iloc[-1],
        'position': channel_position, 'slope': channel_slope,
        'width': (upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1] if mid.iloc[-1] !=0 else 0
    }

def analyze_rsi(df, side):
    """RSI 분석 - 과매수/과매도 및 다이버전스 확인"""
    if len(df) < RSI_PERIOD + DIVERGENCE_LOOKBACK: # 다이버전스 계산 위해 충분한 데이터 필요
        print(f"[RSI Analysis Warning] Insufficient data length: {len(df)}")
        return None
        
    close = df['close']
    rsi_series = ta.momentum.rsi(close, window=RSI_PERIOD)
    if rsi_series is None or rsi_series.isnull().all() or len(rsi_series) < DIVERGENCE_LOOKBACK:
        return None
    current_rsi = rsi_series.iloc[-1]
    if pd.isna(current_rsi): return None

    oversold = current_rsi < RSI_OVERSOLD
    overbought = current_rsi > RSI_OVERBOUGHT
    
    bull_div = False
    bear_div = False
    
    # 다이버전스 확인 (최근 DIVERGENCE_LOOKBACK 봉 기준)
    # 가격: 저점은 낮아지는데, RSI 저점은 높아지는 경우 (상승 다이버전스)
    # 가격: 고점은 높아지는데, RSI 고점은 낮아지는 경우 (하락 다이버전스)
    
    # 다이버전스 검출을 위해 가격 데이터와 RSI 데이터의 최근 DIVERGENCE_LOOKBACK 기간을 사용
    recent_prices_low = df['low'].iloc[-DIVERGENCE_LOOKBACK:]
    recent_prices_high = df['high'].iloc[-DIVERGENCE_LOOKBACK:]
    recent_rsi = rsi_series.iloc[-DIVERGENCE_LOOKBACK:]

    if len(recent_prices_low) < 2 or len(recent_rsi) < 2: # 최소 2점 비교
        return {'value': current_rsi, 'oversold': oversold, 'overbought': overbought, 'bull_div': False, 'bear_div': False}

    # 상승 다이버전스: 가격 저점 하락, RSI 저점 상승
    min_price_idx = recent_prices_low.idxmin()
    min_rsi_idx = recent_rsi.idxmin()
    if recent_prices_low.iloc[-1] < recent_prices_low.iloc[0] and recent_rsi.iloc[-1] > recent_rsi.iloc[0]: # 단순화된 시작/끝점 비교
         # 좀 더 정교한 로직: 최근 N개 봉에서 최저점과 그 이전 최저점을 비교
        if len(recent_prices_low) >= 3: # 최소 3개 봉 이상일 때
            idx_last_low = len(recent_prices_low) - 1 - np.argmin(np.array(recent_prices_low)[::-1]) # 마지막 저점
            idx_prev_lows = recent_prices_low.iloc[:idx_last_low]
            if not idx_prev_lows.empty:
                idx_prev_low = idx_prev_lows.idxmin()
                if recent_prices_low.loc[recent_prices_low.index[idx_last_low]] < recent_prices_low.loc[idx_prev_low] and \
                   recent_rsi.loc[recent_rsi.index[idx_last_low]] > recent_rsi.loc[idx_prev_low]:
                    bull_div = True
    
    # 하락 다이버전스: 가격 고점 상승, RSI 고점 하락
    max_price_idx = recent_prices_high.idxmax()
    max_rsi_idx = recent_rsi.idxmax()
    if recent_prices_high.iloc[-1] > recent_prices_high.iloc[0] and recent_rsi.iloc[-1] < recent_rsi.iloc[0]: # 단순화된 시작/끝점 비교
        if len(recent_prices_high) >= 3:
            idx_last_high = len(recent_prices_high) - 1 - np.argmax(np.array(recent_prices_high)[::-1]) # 마지막 고점
            idx_prev_highs = recent_prices_high.iloc[:idx_last_high]
            if not idx_prev_highs.empty:
                idx_prev_high = idx_prev_highs.idxmax()
                if recent_prices_high.loc[recent_prices_high.index[idx_last_high]] > recent_prices_high.loc[idx_prev_high] and \
                   recent_rsi.loc[recent_rsi.index[idx_last_high]] < recent_rsi.loc[idx_prev_high]:
                    bear_div = True
                    
    return {'value': current_rsi, 'oversold': oversold, 'overbought': overbought, 'bull_div': bull_div, 'bear_div': bear_div}

def analyze_volume_profile(df):
    """매물대 분석 - 최근 VOLUME_PROFILE_PERIODS 기간 거래량 집중 구간 확인"""
    if len(df) < VOLUME_PROFILE_PERIODS:
        print(f"[Volume Profile Warning] Insufficient data: {len(df)} < {VOLUME_PROFILE_PERIODS}")
        return None
        
    profile_df = df.iloc[-VOLUME_PROFILE_PERIODS:]
    price_min = profile_df['low'].min()
    price_max = profile_df['high'].max()
    if pd.isna(price_min) or pd.isna(price_max) or price_min == price_max: return None

    bins = np.linspace(price_min, price_max, 11) # 10개 구간
    volume_at_price = np.zeros(10)
    
    for _, row in profile_df.iterrows():
        candle_low, candle_high, candle_vol = row['low'], row['high'], row['volume']
        if pd.isna(candle_low) or pd.isna(candle_high) or pd.isna(candle_vol) or candle_vol == 0: continue
        
        candle_price_range = candle_high - candle_low
        
        for j in range(10):
            bin_low, bin_high = bins[j], bins[j+1]
            overlap_low = max(candle_low, bin_low)
            overlap_high = min(candle_high, bin_high)
            
            if overlap_high > overlap_low:
                if candle_price_range > 0:
                    overlap_ratio = (overlap_high - overlap_low) / candle_price_range
                    volume_at_price[j] += candle_vol * overlap_ratio
                elif candle_low >= bin_low and candle_high <= bin_high: # 가격 변동 없는 봉이 구간 내 포함
                    volume_at_price[j] += candle_vol


    if np.sum(volume_at_price) == 0: return None # 거래량이 없는 경우
        
    max_vol_bin_idx = np.argmax(volume_at_price)
    poc_min, poc_max = bins[max_vol_bin_idx], bins[max_vol_bin_idx+1] # Point of Control range
    
    current_price = df['close'].iloc[-1]
    if pd.isna(current_price): return None
    
    # 이 부분 수정 - distance_to_poc 변수 선언 추가
    distance_to_poc = 0  # 초기값 설정
    if current_price >= poc_min and current_price <= poc_max:
        distance_to_poc = 0
    else:
        distance_to_poc = min(abs(current_price - poc_min), abs(current_price - poc_max)) / (price_max - price_min)

    return {'poc_min': poc_min, 'poc_max': poc_max, 'distance': distance_to_poc}


def calculate_risk_reward(df, side, channel_data):
    """손익비 계산"""
    if channel_data is None or len(df) < 2: return None
    close = df['close'].iloc[-1]
    if pd.isna(close): return None

    if side == 'long':
        stop_loss = min(df['low'].iloc[-TREND_CHECK_CANDLES:].min(), channel_data['lower'] * 0.985) # 최근 N개 저점 또는 채널하단-1.5%
        take_profit_mid = channel_data['mid']
        take_profit_upper = channel_data['upper']
        if pd.isna(stop_loss) or pd.isna(take_profit_mid): return None

        risk = close - stop_loss
        reward_mid = take_profit_mid - close
        
        rr_mid = reward_mid / risk if risk > 0.000001 else 0 # 0으로 나누기 방지
        return {'rr_mid': rr_mid, 'meets_min_rr': rr_mid >= MIN_RISK_REWARD, 'stop_loss': stop_loss, 'take_profit': take_profit_mid}
        
    elif side == 'short':
        stop_loss = max(df['high'].iloc[-TREND_CHECK_CANDLES:].max(), channel_data['upper'] * 1.015) # 최근 N개 고점 또는 채널상단+1.5%
        take_profit_mid = channel_data['mid']
        take_profit_lower = channel_data['lower']
        if pd.isna(stop_loss) or pd.isna(take_profit_mid): return None
            
        risk = stop_loss - close
        reward_mid = close - take_profit_mid
        
        rr_mid = reward_mid / risk if risk > 0.000001 else 0
        return {'rr_mid': rr_mid, 'meets_min_rr': rr_mid >= MIN_RISK_REWARD, 'stop_loss': stop_loss, 'take_profit': take_profit_mid}
    return None

def check_sma_margin(df):
    """SMA20 근처 여부 확인"""
    if len(df) < 20: return False
    close_price = df['close'].iloc[-1]
    sma20 = ta.trend.sma_indicator(df['close'], window=20).iloc[-1]
    if pd.isna(close_price) or pd.isna(sma20) or sma20 == 0 : return False
    
    return abs(close_price - sma20) / sma20 <= SMA_DEVIATION

# ─────────── 4-Step 분석 함수 (개선됨) ───────────

def four_step_analysis(mtf_data, side):
    """개선된 4-Step 분석: 다중 시간대 분석을 통한 신호 평가"""
    if mtf_data is None: return {'valid': False, 'strength': 0, 'details': {}, 'risk_reward': None}

    results = {}
    
    # 각 분석 결과가 None일 경우 처리
    daily_trend = check_trend(mtf_data['daily'], side)
    
    h4_trend = check_trend(mtf_data['4h'], side)
    h4_channel = analyze_channel(mtf_data['4h'])
    h4_rsi = analyze_rsi(mtf_data['4h'], side) # side 전달
    h4_volume = analyze_volume_profile(mtf_data['4h'])
    h4_risk_reward = calculate_risk_reward(mtf_data['4h'], side, h4_channel)
    
    h1_channel = analyze_channel(mtf_data['1h']) # 1H 채널도 참고용으로 분석
    h1_rsi = analyze_rsi(mtf_data['1h'], side) # side 전달
    h1_sma_margin = check_sma_margin(mtf_data['1h'])

    results = {
        'daily_trend': daily_trend, 'h4_trend': h4_trend, 'h4_channel': h4_channel,
        'h4_rsi': h4_rsi, 'h4_volume': h4_volume, 'h4_risk_reward': h4_risk_reward,
        'h1_channel': h1_channel, 'h1_rsi': h1_rsi, 'h1_sma_margin': h1_sma_margin
    }
    
    # 필수 분석 결과 누락 시 유효하지 않은 신호로 처리
    if not all([
        daily_trend is not None, 
        h4_trend is not None, 
        h4_channel is not None, 
        h4_rsi is not None, 
        h4_volume is not None, 
        h4_risk_reward is not None, 
        h1_channel is not None, 
        h1_rsi is not None, 
        h1_sma_margin is not None
    ]):
        print(f"[4-Step Validation Fail] Missing critical analysis data for {side}.")
        return {'valid': False, 'strength': 0, 'details': results, 'risk_reward': None}

    # 신호 평가 및 종합 점수 계산
    if side == 'long':
        trend_condition = daily_trend or (h4_trend) # 일봉 추세 또는 4시간봉 추세 (둘 중 하나 만족)
        channel_condition = h4_channel['position'] <= 0.25 or \
                            (h1_channel['position'] <= 0.2 and h4_channel['position'] <= 0.35) # 4시간봉 채널 하단 또는 1시간봉 채널 하단 근접
        rsi_condition = h4_rsi['oversold'] or h4_rsi['bull_div'] or \
                        (h1_rsi['oversold'] or h1_rsi['bull_div']) # 4시간 또는 1시간 RSI 조건
        additional_condition = h1_sma_margin and h4_risk_reward['meets_min_rr'] and h4_volume['distance'] <= 0.3 # SMA, 손익비, 매물대 근접
    elif side == 'short':
        trend_condition = daily_trend or (h4_trend)
        channel_condition = h4_channel['position'] >= 0.75 or \
                            (h1_channel['position'] >= 0.8 and h4_channel['position'] >= 0.65)
        rsi_condition = h4_rsi['overbought'] or h4_rsi['bear_div'] or \
                        (h1_rsi['overbought'] or h1_rsi['bear_div'])
        additional_condition = h1_sma_margin and h4_risk_reward['meets_min_rr'] and h4_volume['distance'] <= 0.3
    else: return {'valid': False, 'strength': 0, 'details': results, 'risk_reward': h4_risk_reward}

    signal_valid = trend_condition and channel_condition and rsi_condition and additional_condition
    
    signal_strength = 0
    if signal_valid:
        # 점수 체계: 추세(30), 채널(25), RSI(25), 매물대(20) = 총 100점
        trend_score = 30 if daily_trend else (15 if h4_trend else 0) # 일봉 추세에 더 큰 가중치
        
        channel_score_h4 = (1 - h4_channel['position']) if side == 'long' else h4_channel['position']
        channel_score_h1 = (1 - h1_channel['position']) if side == 'long' else h1_channel['position']
        channel_score = 25 * max(channel_score_h4 * 0.7, channel_score_h1 * 0.3) # 4시간봉 채널에 더 비중

        rsi_val_h4, rsi_val_h1 = h4_rsi['value'], h1_rsi['value']
        if side == 'long':
            rsi_score_h4 = (1 - min(rsi_val_h4 / RSI_OVERSOLD, 1)) if rsi_val_h4 <= RSI_OVERSOLD * 1.5 else 0 # 과매도 근처일수록 높은 점수
            rsi_score_h1 = (1 - min(rsi_val_h1 / RSI_OVERSOLD, 1)) if rsi_val_h1 <= RSI_OVERSOLD * 1.5 else 0
            rsi_score = 15 * rsi_score_h4 + 10 * rsi_score_h1
            if h4_rsi['bull_div']: rsi_score = max(rsi_score, 20) # 다이버전스 발생 시 추가 점수
            if h1_rsi['bull_div']: rsi_score = max(rsi_score, 22)
        else: # short
            rsi_score_h4 = min((rsi_val_h4 - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT), 1) if rsi_val_h4 >= RSI_OVERBOUGHT * 0.9 else 0
            rsi_score_h1 = min((rsi_val_h1 - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT), 1) if rsi_val_h1 >= RSI_OVERBOUGHT * 0.9 else 0
            rsi_score = 15 * rsi_score_h4 + 10 * rsi_score_h1
            if h4_rsi['bear_div']: rsi_score = max(rsi_score, 20)
            if h1_rsi['bear_div']: rsi_score = max(rsi_score, 22)
        rsi_score = min(rsi_score, 25) # 최대 25점

        volume_score = 20 * (1 - h4_volume['distance']) # 매물대 가까울수록 높은 점수
        signal_strength = trend_score + channel_score + rsi_score + volume_score
        signal_strength = max(0, min(signal_strength, 100)) # 0~100점 범위

    return {'valid': signal_valid, 'strength': signal_strength, 'details': results, 'risk_reward': h4_risk_reward}

# ─────────── OKX 스캔 (선물) ───────────
def scan_okx():
    """OKX 선물 스캔"""
    longs, shorts = [], []
    
    try:
        print("[OKX Scan] Loading markets...")
        markets = okx.load_markets()
        print(f"[OKX Scan] Loaded {len(markets)} markets")
        
        # 필터링된 심볼 리스트 생성
        filtered_symbols = []
        for symbol_key in markets:
            m = markets[symbol_key]
            if not (m['swap'] and m.get('settleId') == 'USDT' and m.get('active', True)): # 스왑, USDT 정산, 활성 시장
                continue
            filtered_symbols.append(m['symbol'])
        
        print(f"[OKX Scan] Filtered down to {len(filtered_symbols)} USDT swap markets")
        
        # 옵션: 테스트를 위해 처음 몇 개 심볼만 분석
        # filtered_symbols = filtered_symbols[:10]  # 처음 10개만
    
        for sym in filtered_symbols:
            try:
                base_symbol = markets[sym]['baseId']  # 예: BTC
                
                print(f"[OKX Scan] Fetching ticker for {sym}")
                tick = okx.fetch_ticker(sym)
                
                # 볼륨 정보 가져오기 - 다양한 필드 시도
                vol_24h_usdt = None
                
                # 1. 기본 quoteVolume 확인
                if tick.get('quoteVolume') is not None:
                    vol_24h_usdt = float(tick.get('quoteVolume', 0))
                
                # 2. 대체 필드 확인 - baseVolume(USD로 곱해야 할 수 있음)
                elif tick.get('baseVolume') is not None:
                    base_volume = float(tick.get('baseVolume', 0))
                    # 베이스 볼륨을 USDT 가치로 변환 (대략적인 계산)
                    if tick.get('last') is not None:
                        vol_24h_usdt = base_volume * float(tick.get('last', 0))
                
                # 3. info 객체 내의 필드 확인
                elif 'info' in tick and isinstance(tick['info'], dict):
                    # 볼륨 관련 키 찾기
                    vol_keys = [k for k in tick['info'].keys() if 'vol' in k.lower() or 'amount' in k.lower()]
                    for key in vol_keys:
                        try:
                            value = tick['info'][key]
                            if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                                vol_24h_usdt = float(value)
                                break
                        except (ValueError, TypeError):
                            continue
                
                # 볼륨 정보가 없거나 최소 볼륨 미만인 경우 건너뛰기
                if vol_24h_usdt is None or vol_24h_usdt < VOL_MIN_USDT:
                    print(f"[OKX Scan] Skipping {sym} - {'No volume data' if vol_24h_usdt is None else f'Low volume: {vol_24h_usdt:.0f} USDT'}")
                    continue
                
                print(f"[OKX Scan] Analyzing {sym} (Vol: {vol_24h_usdt:.0f} USDT)")
                mtf_data = fetch_mtf_data(okx, sym)
                if mtf_data is None: 
                    print(f"[OKX Scan] Failed to fetch complete MTF data for {sym}")
                    continue
                time.sleep(0.25) # API Rate Limit
    
                # 롱 포지션 분석
                print(f"[OKX Scan] Running long analysis for {sym}")
                long_analysis = four_step_analysis(mtf_data, 'long')
                if long_analysis['valid']:
                    rr = long_analysis['risk_reward']['rr_mid'] if long_analysis['risk_reward'] else 0
                    rsi_val = long_analysis['details']['h1_rsi']['value'] if long_analysis['details'].get('h1_rsi') else -1
                    longs.append({'symbol': base_symbol, 'strength': long_analysis['strength'], 'rr': rr, 'rsi': rsi_val})
                    log_signal('okx', base_symbol, 'long', {'strength': long_analysis['strength'], 'rr': rr, 'rsi': rsi_val})
                    print(f"[OKX Scan] ✅ Valid LONG signal for {sym}: Strength={long_analysis['strength']:.1f}, RR={rr:.1f}")
                
                # 숏 포지션 분석
                print(f"[OKX Scan] Running short analysis for {sym}")
                short_analysis = four_step_analysis(mtf_data, 'short')
                if short_analysis['valid']:
                    rr = short_analysis['risk_reward']['rr_mid'] if short_analysis['risk_reward'] else 0
                    rsi_val = short_analysis['details']['h1_rsi']['value'] if short_analysis['details'].get('h1_rsi') else -1
                    shorts.append({'symbol': base_symbol, 'strength': short_analysis['strength'], 'rr': rr, 'rsi': rsi_val})
                    log_signal('okx', base_symbol, 'short', {'strength': short_analysis['strength'], 'rr': rr, 'rsi': rsi_val})
                    print(f"[OKX Scan] ✅ Valid SHORT signal for {sym}: Strength={short_analysis['strength']:.1f}, RR={rr:.1f}")
    
            except ccxt.RateLimitExceeded as e:
                print(f"[OKX RateLimit] {sym}: {e}. Sleeping for 60s...")
                time.sleep(60)
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                print(f"[OKX Error] Skipping {sym}: {type(e).__name__} - {e}")
            except Exception as e:
                print(f"[OKX Unexpected Error] {sym}: {type(e).__name__} - {e}")
        
    except Exception as e:
        print(f"[OKX Scan Critical Error]: {type(e).__name__} - {e}")
        longs, shorts = [], []
    
    longs.sort(key=lambda x: x['strength'], reverse=True)
    shorts.sort(key=lambda x: x['strength'], reverse=True)
    return [f"{item['symbol']} ({item['strength']:.0f}|{item['rr']:.1f})" for item in longs], \
           [f"{item['symbol']} ({item['strength']:.0f}|{item['rr']:.1f})" for item in shorts]

# ─────────── 텔레그램 ───────────
def send_telegram(msg):
    """텔레그램으로 메시지 전송"""
    if not TG_TOKEN or not TG_CHAT:
        print("Telegram TOKEN or CHAT_ID missing. Skipping notification.")
        return
    max_length = 4096
    for i in range(0, len(msg), max_length):
        chunk = msg[i:i+max_length]
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
            response = requests.post(url, json={"chat_id": TG_CHAT, "text": chunk, "parse_mode": "Markdown"})
            response.raise_for_status() # 오류 발생 시 예외 처리
            print("Telegram message chunk sent successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send Telegram message: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while sending Telegram message: {e}")


# ─────────── 메인 ───────────
def main():
    """메인 함수"""
    start_time = time.time()
    print(f"===== Signal Scan Started at {dt.datetime.now(dt.timezone(dt.timedelta(hours=9))):%Y-%m-%d %H:%M:%S KST} =====")
    
    try:
        long_results, short_results = scan_okx()
        
        now_korea = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))) # 한국 시간
        
        fmt = lambda x_list: ", ".join(x_list) if x_list else "―"
        msg_body = (f"📊 *4-Step Signals* — `{now_korea:%Y-%m-%d %H:%M} KST`\n\n"
                    f"🎯 *Long (OKX USDT-Perp)*\n{fmt(long_results)}\n\n"
                    f"📉 *Short (OKX USDT-Perp)*\n{fmt(short_results)}")
        
        send_telegram(msg_body)
        
        elapsed_time = time.time() - start_time
        print(f"OKX Long: {len(long_results)}, OKX Short: {len(short_results)}")
        print(f"===== Signal Scan Completed in {elapsed_time:.2f} seconds =====")

    except Exception as e:
        error_msg = f"❌ Critical Error in signal bot main process: {type(e).__name__} - {str(e)}"
        print(error_msg)
        send_telegram(f"*CRITICAL ERROR*: {error_msg}")

if __name__ == "__main__":
    if not TG_TOKEN or not TG_CHAT:
        print("⚠️ TG_TOKEN / TG_CHAT environment variables missing. Telegram notifications will be disabled.")
    
    main()
