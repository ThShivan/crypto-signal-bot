# ===============================================
# signal_bot.py – OKX USDT-Perp + Upbit KRW Spot
#   • 개선된 4-Step 필터 구현
#   • 추세 분석(HH/HL, LH/LL), 채널, RSI 다이버전스, 손익비
#   • 다중 시간대 분석(MTF) 추가 (상위 → 중간 → 하위)
#   • 매물대 근처 시그널 우선순위 반영
#   • OKX = Long·Short / Upbit = Spot(Long)
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
INTERVAL_UPBIT_DAILY = '1d'  # 업비트 일봉
INTERVAL_UPBIT_4H = '240m'   # 업비트 4시간봉
INTERVAL_UPBIT_1H = '60m'    # 업비트 1시간봉

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
MARGIN = 0.02                # 채널 마진 (2%) - 현재 코드에서는 직접 사용되지 않음 (analyze_channel 에서 dev로 동적 계산)
SMA_DEVIATION = 0.02         # SMA 편차 허용 범위
MIN_RISK_REWARD = 1.5        # 최소 손익비 (1:1.5) - 상향 조정 고려

# 거래량 필터
VOL_MIN_USDT = 1_000_000       # OKX 24h 거래대금
VOL_MIN_KRW = 1_000_000_000    # Upbit 24h 거래대금(원)

# ─────────── 거래소 인스턴스 ───────────
okx = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
upbit = ccxt.upbit({'enableRateLimit': True})

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
            if not ohlcv or len(ohlcv) < limit * 0.8: # 데이터가 없거나 충분히 오지 않은 경우
                raise ccxt.NetworkError(f"Insufficient data for {symbol} {timeframe}: got {len(ohlcv) if ohlcv else 0}, expected near {limit}")
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols) or df.iloc[-min(5, len(df)):][required_cols].isnull().any().any(): # 최근 데이터에 NA 확인
                raise ValueError(f"OHLCV data for {symbol} {timeframe} contains NaNs or missing columns in recent data.")
            return df
        except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as e:
            print(f"[Fetch Retry {attempt+1}/{max_retries}] Failed to fetch {symbol} {timeframe}: {type(e).__name__} - {e}")
            if attempt + 1 == max_retries:
                raise
            time.sleep(delay)

def fetch_mtf_data(exchange_instance, symbol, is_okx=True):
    """거래소에서 다중 시간대 데이터 가져오기"""
    fetch_func = exchange_instance.fetch_ohlcv
    tf_daily = INTERVAL_DAILY if is_okx else INTERVAL_UPBIT_DAILY
    tf_4h = INTERVAL_4H if is_okx else INTERVAL_UPBIT_4H
    tf_1h = INTERVAL_1H if is_okx else INTERVAL_UPBIT_1H

    try:
        daily = fetch_ohlcv_with_retry(fetch_func, symbol, tf_daily, CANDLES_DAILY)
        time.sleep(0.1) 
        h4 = fetch_ohlcv_with_retry(fetch_func, symbol, tf_4h, CANDLES_4H)
        time.sleep(0.1) 
        h1 = fetch_ohlcv_with_retry(fetch_func, symbol, tf_1h, CANDLES_1H)
    except Exception as e:
        print(f"[{'OKX' if is_okx else 'Upbit'} MTF Data Error] Failed to fetch all MTF data for {symbol}: {e}")
        return None
    
    if daily is None or h4 is None or h1 is None or \
       len(daily) < TREND_CHECK_CANDLES or len(h4) < LEN_CHAN or len(h1) < RSI_PERIOD:
        print(f"[{'OKX' if is_okx else 'Upbit'} MTF Data Warning] Insufficient data length for {symbol} after fetch.")
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
    
    if len(highs) < 2 or len(lows) < 2: # 비교할 데이터가 최소 2개는 있어야 함
        return False

    if side == 'long':
        all_higher_highs = all(highs[i] >= highs[i-1] for i in range(1, len(highs))) and any(highs[i] > highs[i-1] for i in range(1, len(highs)))
        all_higher_lows = all(lows[i] > lows[i-1] for i in range(1, len(lows)))
        return all_higher_highs and all_higher_lows
    elif side == 'short':
        all_lower_highs = all(highs[i] < highs[i-1] for i in range(1, len(highs)))
        all_lower_lows = all(lows[i] <= lows[i-1] for i in range(1, len(lows))) and any(lows[i] < lows[i-1] for i in range(1, len(lows)))
        return all_lower_highs and all_lower_lows
    return False

def analyze_channel(df):
    """채널 분석 - EMA 기반 채널 계산 및 위치 확인"""
    if df is None or len(df) < LEN_CHAN:
        # print(f"[Channel Analysis Warning] Insufficient data length for channel: got {len(df) if df is not None else 'None'}")
        return None
    close = df['close']
    
    basis = ta.trend.ema_indicator(close, window=LEN_CHAN)
    if basis is None or basis.isnull().all() or len(basis) < 1: return None
        
    dev = (close - basis).abs().rolling(window=LEN_CHAN, min_periods=max(1, LEN_CHAN//2)).max()
    if dev is None or dev.isnull().all() or len(dev) < 1: return None

    if pd.isna(basis.iloc[-1]) or pd.isna(dev.iloc[-1]) or pd.isna(close.iloc[-1]):
        return None

    lower_val, upper_val = basis.iloc[-1] - dev.iloc[-1], basis.iloc[-1] + dev.iloc[-1]
    mid_val = (upper_val + lower_val) / 2

    if pd.isna(lower_val) or pd.isna(upper_val) or (upper_val - lower_val) == 0:
        return None

    channel_position = (close.iloc[-1] - lower_val) / (upper_val - lower_val)
    channel_slope = 0
    if len(basis) >= 5 and not pd.isna(basis.iloc[-5]) and basis.iloc[-5] != 0:
        channel_slope = (basis.iloc[-1] - basis.iloc[-5]) / basis.iloc[-5] 
    
    channel_width = 0
    if mid_val != 0:
        channel_width = (upper_val - lower_val) / mid_val
        
    return {
        'lower': lower_val, 'upper': upper_val, 'mid': mid_val,
        'position': channel_position, 'slope': channel_slope,
        'width': channel_width
    }

def analyze_rsi(df, side):
    """RSI 분석 - 과매수/과매도 및 다이버전스 확인"""
    if df is None or len(df) < RSI_PERIOD + DIVERGENCE_LOOKBACK:
        # print(f"[RSI Analysis Warning] Insufficient data length: got {len(df) if df is not None else 'None'}")
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
    
    recent_prices_low = df['low'].iloc[-DIVERGENCE_LOOKBACK:]
    recent_prices_high = df['high'].iloc[-DIVERGENCE_LOOKBACK:]
    recent_rsi = rsi_series.iloc[-DIVERGENCE_LOOKBACK:]

    if len(recent_prices_low) >= 3 and len(recent_rsi) >=3: # 최소 3개 지점 비교
        # 상승 다이버전스 (최근 저점 < 이전 저점 AND 최근 RSI 저점 > 이전 RSI 저점)
        price_low_idx1 = len(recent_prices_low) - 1 - np.argmin(np.array(recent_prices_low)[::-1][:DIVERGENCE_LOOKBACK//2]) # 최근 절반에서 저점
        price_low_idx0 = np.argmin(np.array(recent_prices_low)[:DIVERGENCE_LOOKBACK//2]) # 이전 절반에서 저점
        
        if recent_prices_low.iloc[price_low_idx1] < recent_prices_low.iloc[price_low_idx0] and \
           recent_rsi.iloc[price_low_idx1] > recent_rsi.iloc[price_low_idx0]:
            bull_div = True

        # 하락 다이버전스 (최근 고점 > 이전 고점 AND 최근 RSI 고점 < 이전 RSI 고점)
        price_high_idx1 = len(recent_prices_high) - 1 - np.argmax(np.array(recent_prices_high)[::-1][:DIVERGENCE_LOOKBACK//2]) # 최근 절반에서 고점
        price_high_idx0 = np.argmax(np.array(recent_prices_high)[:DIVERGENCE_LOOKBACK//2]) # 이전 절반에서 고점

        if recent_prices_high.iloc[price_high_idx1] > recent_prices_high.iloc[price_high_idx0] and \
           recent_rsi.iloc[price_high_idx1] < recent_rsi.iloc[price_high_idx0]:
            bear_div = True
            
    return {'value': current_rsi, 'oversold': oversold, 'overbought': overbought, 'bull_div': bull_div, 'bear_div': bear_div}

def analyze_volume_profile(df):
    """매물대 분석 - 최근 VOLUME_PROFILE_PERIODS 기간 거래량 집중 구간 확인"""
    if df is None or len(df) < VOLUME_PROFILE_PERIODS:
        # print(f"[Volume Profile Warning] Insufficient data: got {len(df) if df is not None else 'None'}")
        return None
        
    profile_df = df.iloc[-VOLUME_PROFILE_PERIODS:]
    price_min = profile_df['low'].min()
    price_max = profile_df['high'].max()
    if pd.isna(price_min) or pd.isna(price_max) or price_min == price_max: return None

    bins = np.linspace(price_min, price_max, 11) 
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
                if candle_price_range > 1e-9: # Check for non-zero range
                    overlap_ratio = (overlap_high - overlap_low) / candle_price_range
                    volume_at_price[j] += candle_vol * overlap_ratio
                elif candle_low >= bin_low and candle_high <= bin_high:
                    volume_at_price[j] += candle_vol

    if np.sum(volume_at_price) < 1e-9 : return None 
        
    max_vol_bin_idx = np.argmax(volume_at_price)
    poc_min, poc_max = bins[max_vol_bin_idx], bins[max_vol_bin_idx+1]
    
    current_price = df['close'].iloc[-1]
    if pd.isna(current_price): return None
        
    distance_to_poc = 0.0
    price_range_for_norm = price_max - price_min
    if price_range_for_norm < 1e-9 : price_range_for_norm = 1.0 # Prevent division by zero if all prices are same

    if not (current_price >= poc_min and current_price <= poc_max):
        distance_to_poc = min(abs(current_price - poc_min), abs(current_price - poc_max)) / price_range_for_norm

    return {'poc_min': poc_min, 'poc_max': poc_max, 'distance': distance_to_poc}


def calculate_risk_reward(df, side, channel_data):
    """손익비 계산"""
    if df is None or channel_data is None or len(df) < TREND_CHECK_CANDLES : return None # TREND_CHECK_CANDLES 사용
    close = df['close'].iloc[-1]
    if pd.isna(close): return None

    if side == 'long':
        stop_loss = min(df['low'].iloc[-TREND_CHECK_CANDLES:].min(), channel_data['lower'] * 0.985) 
        take_profit_mid = channel_data['mid']
        if pd.isna(stop_loss) or pd.isna(take_profit_mid): return None

        risk = close - stop_loss
        reward_mid = take_profit_mid - close
        
        rr_mid = reward_mid / risk if risk > 1e-9 else 0
        return {'rr_mid': rr_mid, 'meets_min_rr': rr_mid >= MIN_RISK_REWARD, 'stop_loss': stop_loss, 'take_profit': take_profit_mid}
        
    elif side == 'short':
        stop_loss = max(df['high'].iloc[-TREND_CHECK_CANDLES:].max(), channel_data['upper'] * 1.015)
        take_profit_mid = channel_data['mid']
        if pd.isna(stop_loss) or pd.isna(take_profit_mid): return None
            
        risk = stop_loss - close
        reward_mid = close - take_profit_mid
        
        rr_mid = reward_mid / risk if risk > 1e-9 else 0
        return {'rr_mid': rr_mid, 'meets_min_rr': rr_mid >= MIN_RISK_REWARD, 'stop_loss': stop_loss, 'take_profit': take_profit_mid}
    return None

def check_sma_margin(df):
    """SMA20 근처 여부 확인"""
    if df is None or len(df) < 20: return False
    close_price = df['close'].iloc[-1]
    sma20_series = ta.trend.sma_indicator(df['close'], window=20)
    if sma20_series is None or sma20_series.empty: return False
    sma20 = sma20_series.iloc[-1]
    
    if pd.isna(close_price) or pd.isna(sma20) or sma20 == 0 : return False
    
    return abs(close_price - sma20) / sma20 <= SMA_DEVIATION

# ─────────── 4-Step 분석 함수 (개선됨) ───────────

def four_step_analysis(mtf_data, side):
    """개선된 4-Step 분석: 다중 시간대 분석을 통한 신호 평가"""
    if mtf_data is None: return {'valid': False, 'strength': 0, 'details': {}, 'risk_reward': None}

    results = {}
    
    daily_trend = check_trend(mtf_data['daily'], side)
    h4_trend = check_trend(mtf_data['4h'], side)
    h4_channel = analyze_channel(mtf_data['4h'])
    h4_rsi = analyze_rsi(mtf_data['4h'], side) 
    h4_volume = analyze_volume_profile(mtf_data['4h'])
    h4_risk_reward = calculate_risk_reward(mtf_data['4h'], side, h4_channel)
    
    h1_channel = analyze_channel(mtf_data['1h']) 
    h1_rsi = analyze_rsi(mtf_data['1h'], side) 
    h1_sma_margin = check_sma_margin(mtf_data['1h'])

    results = {
        'daily_trend': daily_trend, 'h4_trend': h4_trend, 'h4_channel': h4_channel,
        'h4_rsi': h4_rsi, 'h4_volume': h4_volume, 'h4_risk_reward': h4_risk_reward,
        'h1_channel': h1_channel, 'h1_rsi': h1_rsi, 'h1_sma_margin': h1_sma_margin
    }
    
    if not all([res is not None for res_key, res in results.items() if res_key not in ['h4_channel', 'h1_channel'] and not isinstance(res, bool)]) or \
       results['h4_channel'] is None or results['h1_channel'] is None: # check if key dicts are None
        # print(f"[4-Step Validation Fail] Missing critical analysis data for {side} due to None results.")
        # for k,v in results.items(): 
        #     if v is None or (isinstance(v, dict) and any(sub_v is None for sub_v in v.values())): print(f"Missing: {k}")
        return {'valid': False, 'strength': 0, 'details': results, 'risk_reward': None}


    if side == 'long':
        trend_condition = daily_trend or h4_trend 
        channel_condition = results['h4_channel']['position'] <= 0.25 or \
                            (results['h1_channel']['position'] <= 0.2 and results['h4_channel']['position'] <= 0.35) 
        rsi_condition = results['h4_rsi']['oversold'] or results['h4_rsi']['bull_div'] or \
                        results['h1_rsi']['oversold'] or results['h1_rsi']['bull_div'] 
        additional_condition = results['h1_sma_margin'] and results['h4_risk_reward']['meets_min_rr'] and results['h4_volume']['distance'] <= 0.3 
    elif side == 'short':
        trend_condition = daily_trend or h4_trend 
        channel_condition = results['h4_channel']['position'] >= 0.75 or \
                            (results['h1_channel']['position'] >= 0.8 and results['h4_channel']['position'] >= 0.65)
        rsi_condition = results['h4_rsi']['overbought'] or results['h4_rsi']['bear_div'] or \
                        results['h1_rsi']['overbought'] or results['h1_rsi']['bear_div']
        additional_condition = results['h1_sma_margin'] and results['h4_risk_reward']['meets_min_rr'] and results['h4_volume']['distance'] <= 0.3
    else: return {'valid': False, 'strength': 0, 'details': results, 'risk_reward': results.get('h4_risk_reward')}

    signal_valid = trend_condition and channel_condition and rsi_condition and additional_condition
    
    signal_strength = 0
    if signal_valid:
        trend_score = 30 if results['daily_trend'] else (15 if results['h4_trend'] else 0) 
        
        channel_score_h4 = (1 - results['h4_channel']['position']) if side == 'long' else results['h4_channel']['position']
        channel_score_h1 = (1 - results['h1_channel']['position']) if side == 'long' else results['h1_channel']['position']
        channel_score = 25 * max(channel_score_h4 * 0.7, channel_score_h1 * 0.3) 

        rsi_val_h4, rsi_val_h1 = results['h4_rsi']['value'], results['h1_rsi']['value']
        rsi_score = 0
        if side == 'long':
            rsi_score_h4_val = (1 - min(rsi_val_h4 / RSI_OVERSOLD, 1)) if rsi_val_h4 <= RSI_OVERSOLD * 1.5 else 0 
            rsi_score_h1_val = (1 - min(rsi_val_h1 / RSI_OVERSOLD, 1)) if rsi_val_h1 <= RSI_OVERSOLD * 1.5 else 0
            rsi_score = 15 * rsi_score_h4_val + 10 * rsi_score_h1_val
            if results['h4_rsi']['bull_div']: rsi_score = max(rsi_score, 20) 
            if results['h1_rsi']['bull_div']: rsi_score = max(rsi_score, 22)
        else: 
            rsi_score_h4_val = min((rsi_val_h4 - RSI_OVERBOUGHT*0.9) / (100 - RSI_OVERBOUGHT*0.9), 1) if rsi_val_h4 >= RSI_OVERBOUGHT * 0.9 else 0
            rsi_score_h1_val = min((rsi_val_h1 - RSI_OVERBOUGHT*0.9) / (100 - RSI_OVERBOUGHT*0.9), 1) if rsi_val_h1 >= RSI_OVERBOUGHT * 0.9 else 0
            rsi_score = 15 * rsi_score_h4_val + 10 * rsi_score_h1_val
            if results['h4_rsi']['bear_div']: rsi_score = max(rsi_score, 20)
            if results['h1_rsi']['bear_div']: rsi_score = max(rsi_score, 22)
        rsi_score = min(rsi_score, 25) 

        volume_score = 20 * (1 - results['h4_volume']['distance']) 
        signal_strength = trend_score + channel_score + rsi_score + volume_score
        signal_strength = max(0, min(signal_strength, 100)) 

    return {'valid': signal_valid, 'strength': signal_strength, 'details': results, 'risk_reward': results.get('h4_risk_reward')}

# ─────────── OKX 스캔 (선물) ───────────
def scan_okx():
    """OKX 선물 스캔"""
    longs, shorts = [], []
    try:
        markets = okx.load_markets()
    except Exception as e:
        print(f"[OKX Error] Failed to load markets: {e}")
        return [], []
    
    for symbol_key in markets:
        m = markets[symbol_key]
        if not (m.get('swap', False) and m.get('settleId', '').upper() == 'USDT' and m.get('active', True)):
            continue
        
        sym = m['symbol']
        base_symbol = m.get('baseId', sym.split('/')[0].split(':')[0])
        
        try:
            tick = okx.fetch_ticker(sym)
            vol_24h_usdt = tick.get('quoteVolume') or 0
            if vol_24h_usdt < VOL_MIN_USDT:
                continue
            
            print(f"[OKX Scan] Analyzing {sym} (Vol: {vol_24h_usdt:.0f} USDT)")
            mtf_data = fetch_mtf_data(okx, sym, is_okx=True)
            if mtf_data is None: continue
            time.sleep(okx.rateLimit / 1000 * 1.1) # Use exchange's rateLimit property

            long_analysis = four_step_analysis(mtf_data, 'long')
            if long_analysis['valid'] and long_analysis.get('risk_reward'):
                rr = long_analysis['risk_reward']['rr_mid']
                rsi_val = long_analysis['details']['h1_rsi']['value'] if long_analysis['details'].get('h1_rsi') else -1
                longs.append({'symbol': base_symbol, 'strength': long_analysis['strength'], 'rr': rr, 'rsi': rsi_val})
                log_signal('okx', base_symbol, 'long', {'strength': long_analysis['strength'], 'rr': rr, 'rsi': rsi_val})
            
            short_analysis = four_step_analysis(mtf_data, 'short')
            if short_analysis['valid'] and short_analysis.get('risk_reward'):
                rr = short_analysis['risk_reward']['rr_mid']
                rsi_val = short_analysis['details']['h1_rsi']['value'] if short_analysis['details'].get('h1_rsi') else -1
                shorts.append({'symbol': base_symbol, 'strength': short_analysis['strength'], 'rr': rr, 'rsi': rsi_val})
                log_signal('okx', base_symbol, 'short', {'strength': short_analysis['strength'], 'rr': rr, 'rsi': rsi_val})

        except ccxt.RateLimitExceeded as e:
            print(f"[OKX RateLimit] {sym}: {e}. Sleeping for 60s...")
            time.sleep(60)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e: # More specific ccxt errors
            print(f"[OKX CCXT Error] Skipping {sym}: {type(e).__name__} - {e}")
        except Exception as e: # Generic errors
            print(f"[OKX Generic Error] Skipping {sym}: {type(e).__name__} - {e}")
    
    longs.sort(key=lambda x: x['strength'], reverse=True)
    shorts.sort(key=lambda x: x['strength'], reverse=True)
    return [f"{item['symbol']} ({item['strength']:.0f}|{item['rr']:.1f})" for item in longs], \
           [f"{item['symbol']} ({item['strength']:.0f}|{item['rr']:.1f})" for item in shorts]


# ─────────── Upbit 스캔 (현물) ───────────
def scan_upbit():
    """Upbit 현물 스캔"""
    spot = []
    try:
        markets = upbit.load_markets()
    except Exception as e:
        print(f"[Upbit Error] Failed to load markets: {e}")
        return []

    for symbol_key in markets:
        m = markets[symbol_key]
        if not (m.get('quoteId', '').upper() == 'KRW' and m.get('active', True)):
            continue
        
        sym = m['symbol'] 
        base_symbol = m.get('baseId', sym.split('/')[0])

        try:
            tick = upbit.fetch_ticker(sym)
            vol_24h_krw = float(tick['info'].get('acc_trade_price_24h', 0)) # Upbit specific field
            if vol_24h_krw < VOL_MIN_KRW:
                continue
            
            print(f"[Upbit Scan] Analyzing {sym} (Vol: {vol_24h_krw:,.0f} KRW)")
            mtf_data = fetch_mtf_data(upbit, sym, is_okx=False)
            if mtf_data is None: continue
            time.sleep(upbit.rateLimit / 1000 * 1.1) 

            analysis = four_step_analysis(mtf_data, 'long') 
            if analysis['valid'] and analysis.get('risk_reward'):
                rr = analysis['risk_reward']['rr_mid']
                rsi_val = analysis['details']['h1_rsi']['value'] if analysis['details'].get('h1_rsi') else -1
                spot.append({'symbol': base_symbol, 'strength': analysis['strength'], 'rr': rr, 'rsi': rsi_val})
                log_signal('upbit', base_symbol, 'long', {'strength': analysis['strength'], 'rr': rr, 'rsi': rsi_val})
        
        except ccxt.RateLimitExceeded as e:
            print(f"[Upbit RateLimit] {sym}: {e}. Sleeping for 60s...")
            time.sleep(60)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
             print(f"[Upbit CCXT Error] Skipping {sym}: {type(e).__name__} - {e}")
        except Exception as e:
            print(f"[Upbit Generic Error] Skipping {sym}: {type(e).__name__} - {e}")
            # If "invalid literal for int() with base 10: 'da'" error persists with 'days',
            # try INTERVAL_UPBIT_DAILY = '1440m' as a next step.
            # Also ensure your ccxt library is up to date: pip install -U ccxt

    spot.sort(key=lambda x: x['strength'], reverse=True)
    return [f"{item['symbol']} ({item['strength']:.0f}|{item['rr']:.1f})" for item in spot]

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
            response.raise_for_status() 
            print("Telegram message chunk sent successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send Telegram message chunk: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while sending Telegram message chunk: {e}")


# ─────────── 메인 ───────────
def main():
    """메인 함수"""
    start_time = time.time()
    kst = dt.timezone(dt.timedelta(hours=9))
    print(f"===== Signal Scan Started at {dt.datetime.now(kst):%Y-%m-%d %H:%M:%S KST} =====")
    
    okx_long_results, okx_short_results = [], []
    upbit_spot_results = []

    try:
        okx_long_results, okx_short_results = scan_okx()
        upbit_spot_results = scan_upbit()
        
        now_korea = dt.datetime.now(kst) 
        
        fmt = lambda x_list: ", ".join(x_list) if x_list else "―"
        msg_body = (f"📊 *4-Step Signals* — `{now_korea:%Y-%m-%d %H:%M} KST`\n\n"
                    f"🎯 *Long (OKX USDT-Perp)*\n{fmt(okx_long_results)}\n\n"
                    f"📉 *Short (OKX USDT-Perp)*\n{fmt(okx_short_results)}\n\n"
                    f"💰 *Spot (Upbit KRW)*\n{fmt(upbit_spot_results)}")
        
        send_telegram(msg_body)
        
        elapsed_time = time.time() - start_time
        print(f"OKX Long: {len(okx_long_results)}, OKX Short: {len(okx_short_results)}, Upbit Spot: {len(upbit_spot_results)}")
        print(f"===== Signal Scan Completed in {elapsed_time:.2f} seconds =====")

    except Exception as e:
        error_msg = f"❌ Critical Error in signal bot main process: {type(e).__name__} - {str(e)}"
        import traceback
        print(traceback.format_exc()) # 스택 트레이스 출력
        send_telegram(f"*CRITICAL ERROR*: ```\n{error_msg}\n{traceback.format_exc_info()[2]}```") # 간략한 스택 정보 포함

if __name__ == "__main__":
    if not TG_TOKEN or not TG_CHAT:
        print("❌ TG_TOKEN / TG_CHAT environment variables missing. Telegram notifications will be disabled.")
    
    main()
