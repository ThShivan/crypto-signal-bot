# ===============================================
# signal_bot.py – OKX USDT-Perp + Upbit KRW Spot
#   • 개선된 4-Step 필터 구현
#   • 추세 분석(HH/HL, LH/LL), 채널, RSI 다이버전스, 손익비
#   • 다중 시간대 분석(MTF) 추가 (상위 → 중간 → 하위)
#   • 매물대 근처 시그널 우선순위 반영
#   • OKX = Long·Short / Upbit = Spot(Long)
#   • 결과를 Telegram으로 전송 (09:00 KST, GitHub Actions)
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
INTERVAL_DAILY = '1d'       # 일봉 (상위 추세)
INTERVAL_4H = '4h'          # 4시간봉 (중기 추세)
INTERVAL_1H = '1h'          # 1시간봉 (단기 타이밍)
INTERVAL_UPBIT_DAILY = '1440m'
INTERVAL_UPBIT_4H = '240m'
INTERVAL_UPBIT_1H = '60m'

# 각 시간대별 가져올 캔들 수
CANDLES_DAILY = 90          # 약 3개월
CANDLES_4H = 200            # 약 33일
CANDLES_1H = 180            # 약 7.5일

# 채널 및 필터 설정
LEN_CHAN = 120              # 채널 길이
RSI_PERIOD = 14             # RSI 주기
TREND_CHECK_DAYS = 5        # 추세 확인 캔들 수
DIVERGENCE_LOOKBACK = 5     # 다이버전스 확인 기간
VOLUME_PROFILE_PERIODS = 20 # 매물대 분석 기간

# 조건별 가중치 설정 (0~1)
WEIGHT_TREND = 0.3          # 추세 가중치
WEIGHT_CHANNEL = 0.25       # 채널 가중치
WEIGHT_RSI = 0.25           # RSI 가중치
WEIGHT_VOLUME_PROFILE = 0.2 # 매물대 가중치

# 매매 조건 설정
RSI_OVERSOLD = 30           # RSI 과매도
RSI_OVERBOUGHT = 70         # RSI 과매수
MARGIN = 0.02               # 채널 마진 (2%)
SMA_DEVIATION = 0.02        # SMA 편차 허용 범위
MIN_RISK_REWARD = 2         # 최소 손익비 (1:2)

# 거래량 필터
VOL_MIN_USDT = 1_000_000         # OKX 24h 거래대금
VOL_MIN_KRW = 1_000_000_000      # Upbit 24h 거래대금(원)

# ─────────── 거래소 인스턴스 ───────────
okx = ccxt.okx({'enableRateLimit': True})
upbit = ccxt.upbit({'enableRateLimit': True})

# ─────────── 유틸리티 함수 ───────────

def ensure_dir(directory):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_signal(exchange, symbol, side, signal_data):
    """신호 로깅"""
    ensure_dir(LOGS_DIR)
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"{LOGS_DIR}/{exchange}_{timestamp}.csv"
    
    # 신호 데이터를 DataFrame으로 변환
    df = pd.DataFrame([{
        'timestamp': dt.datetime.utcnow().isoformat(),
        'symbol': symbol,
        'side': side,
        **signal_data
    }])
    
    # 파일이 있으면 추가, 없으면 생성
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# ─────────── OHLCV 데이터 가져오기 ───────────

def fetch_ohlcv_okx(symbol, timeframe, limit):
    """OKX에서 OHLCV 데이터 가져오기"""
    ohlcv = okx.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def fetch_ohlcv_upbit(symbol, timeframe, limit):
    """Upbit에서 OHLCV 데이터 가져오기"""
    ohlcv = upbit.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def fetch_mtf_data_okx(symbol):
    """OKX에서 다중 시간대 데이터 가져오기"""
    daily = fetch_ohlcv_okx(symbol, INTERVAL_DAILY, CANDLES_DAILY)
    h4 = fetch_ohlcv_okx(symbol, INTERVAL_4H, CANDLES_4H)
    h1 = fetch_ohlcv_okx(symbol, INTERVAL_1H, CANDLES_1H)
    return {
        'daily': daily,
        '4h': h4,
        '1h': h1
    }

def fetch_mtf_data_upbit(symbol):
    """Upbit에서 다중 시간대 데이터 가져오기"""
    daily = fetch_ohlcv_upbit(symbol, INTERVAL_UPBIT_DAILY, CANDLES_DAILY)
    h4 = fetch_ohlcv_upbit(symbol, INTERVAL_UPBIT_4H, CANDLES_4H)
    h1 = fetch_ohlcv_upbit(symbol, INTERVAL_UPBIT_1H, CANDLES_1H)
    return {
        'daily': daily,
        '4h': h4,
        '1h': h1
    }

# ─────────── 기술적 분석 함수 ───────────

def check_trend(df, side):
    """
    추세 분석 - Higher Highs & Higher Lows 또는 Lower Highs & Lower Lows 확인
    """
    highs = df['high'].values
    lows = df['low'].values
    
    if side == 'long':
        # 상승 추세 확인 (Higher Highs & Higher Lows)
        higher_highs = True
        higher_lows = True
        
        for i in range(TREND_CHECK_DAYS, 1, -1):
            if highs[-i] >= highs[-i+2]:  # 고점이 이전 고점보다 낮으면
                higher_highs = False
            if lows[-i] >= lows[-i+2]:    # 저점이 이전 저점보다 낮으면
                higher_lows = False
        
        # Higher Highs와 Higher Lows 둘 다 충족해야 함
        return higher_highs and higher_lows
    
    elif side == 'short':
        # 하락 추세 확인 (Lower Highs & Lower Lows)
        lower_highs = True
        lower_lows = True
        
        for i in range(TREND_CHECK_DAYS, 1, -1):
            if highs[-i] <= highs[-i+2]:  # 고점이 이전 고점보다 높으면
                lower_highs = False
            if lows[-i] <= lows[-i+2]:    # 저점이 이전 저점보다 높으면
                lower_lows = False
        
        # Lower Highs와 Lower Lows 둘 다 충족해야 함
        return lower_highs and lower_lows
    
    return False

def analyze_channel(df):
    """
    채널 분석 - EMA 기반 채널 계산 및 위치 확인
    """
    close = df['close']
    
    # EMA 채널 계산
    basis = ta.trend.ema_indicator(close, window=LEN_CHAN)
    dev = (close - basis).abs().rolling(LEN_CHAN).max()
    lower = basis - dev
    upper = basis + dev
    mid = (upper + lower) / 2  # 0.5 중간선
    
    # 현재 가격이 채널 내 어디에 위치하는지 계산 (0~1, 0=하단, 0.5=중간, 1=상단)
    channel_position = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
    
    # 채널 기울기 (상승/하락 강도)
    channel_slope = (basis.iloc[-1] - basis.iloc[-5]) / basis.iloc[-5]
    
    return {
        'lower': lower.iloc[-1],
        'upper': upper.iloc[-1],
        'mid': mid.iloc[-1],
        'position': channel_position,
        'slope': channel_slope,
        'width': (upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1]  # 채널 폭 (%)
    }

def analyze_rsi(df, side):
    """
    RSI 분석 - 과매수/과매도 및 다이버전스 확인
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # RSI 계산
    rsi = ta.momentum.rsi(close, window=RSI_PERIOD)
    current_rsi = rsi.iloc[-1]
    
    # 과매수/과매도 상태 확인
    oversold = current_rsi < RSI_OVERSOLD
    overbought = current_rsi > RSI_OVERBOUGHT
    
    # 다이버전스 확인
    bull_div = False  # 상승 다이버전스 (가격↓ RSI↑)
    bear_div = False  # 하락 다이버전스 (가격↑ RSI↓)
    
    # 지난 N봉 사이에서 다이버전스 찾기
    for i in range(3, DIVERGENCE_LOOKBACK + 1):
        # 상승 다이버전스: 가격은 낮은 저점, RSI는 높은 저점
        if low.iloc[-1] < low.iloc[-i] and rsi.iloc[-1] > rsi.iloc[-i]:
            bull_div = True
            break
            
        # 하락 다이버전스: 가격은 높은 고점, RSI는 낮은 고점
        if high.iloc[-1] > high.iloc[-i] and rsi.iloc[-1] < rsi.iloc[-i]:
            bear_div = True
            break
    
    return {
        'value': current_rsi,
        'oversold': oversold,
        'overbought': overbought,
        'bull_div': bull_div,
        'bear_div': bear_div
    }

def analyze_volume_profile(df):
    """
    매물대 분석 - 과거 거래량 집중 구간 확인
    """
    # 가격 범위 설정
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    
    # 가격 구간 나누기 (10개 구간)
    bins = np.linspace(price_min, price_max, 11)
    
    # 각 봉마다 해당 가격 구간에 거래량 할당
    volume_profile = np.zeros(10)
    
    for i in range(len(df) - VOLUME_PROFILE_PERIODS, len(df)):
        # 각 봉의 거래 범위
        candle_min = df['low'].iloc[i]
        candle_max = df['high'].iloc[i]
        candle_volume = df['volume'].iloc[i]
        
        # 봉이 걸쳐있는 구간들에 거래량 비례 배분
        for j in range(10):
            bin_min = bins[j]
            bin_max = bins[j+1]
            
            # 해당 구간과 봉의 겹치는 부분 계산
            overlap_min = max(candle_min, bin_min)
            overlap_max = min(candle_max, bin_max)
            
            if overlap_max > overlap_min:
                # 겹치는 비율만큼 거래량 할당
                overlap_ratio = (overlap_max - overlap_min) / (candle_max - candle_min)
                volume_profile[j] += candle_volume * overlap_ratio
    
    # 최대 거래량 구간 찾기
    max_volume_bin = np.argmax(volume_profile)
    
    # 현재 가격과 최대 거래량 구간의 관계
    current_price = df['close'].iloc[-1]
    max_vol_price_min = bins[max_volume_bin]
    max_vol_price_max = bins[max_volume_bin + 1]
    
    # 현재 가격이 최대 거래량 구간에 얼마나 가까운지 계산 (0~1, 0=매우 가까움)
    if current_price >= max_vol_price_min and current_price <= max_vol_price_max:
        distance = 0  # 구간 내에 있음
    else:
        # 구간까지의 거리를 전체 가격 범위로 정규화
        distance = min(
            abs(current_price - max_vol_price_min),
            abs(current_price - max_vol_price_max)
        ) / price_range
    
    return {
        'max_vol_price_min': max_vol_price_min,
        'max_vol_price_max': max_vol_price_max,
        'distance': distance,
        'volume_profile': volume_profile.tolist()
    }

def calculate_risk_reward(df, side, channel_data):
    """
    손익비 계산 - 리스크 대비 리워드 비율 확인
    """
    close = df['close'].iloc[-1]
    
    if side == 'long':
        # 롱 포지션의 경우
        # 손절: 직전 저점 또는 채널 하단 -2%
        stop_loss = min(df['low'].iloc[-2], channel_data['lower'] * 0.98)
        
        # 익절: 채널 중단선 또는 채널 상단
        take_profit_mid = channel_data['mid']
        take_profit_upper = channel_data['upper']
        
        # 손익비 계산 (리워드/리스크)
        risk = close - stop_loss
        reward_mid = take_profit_mid - close
        reward_upper = take_profit_upper - close
        
        # 중간선까지의 손익비
        rr_mid = reward_mid / risk if risk > 0 else 0
        # 상단까지의 손익비
        rr_upper = reward_upper / risk if risk > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'take_profit_mid': take_profit_mid,
            'take_profit_upper': take_profit_upper,
            'risk': risk,
            'reward_mid': reward_mid,
            'reward_upper': reward_upper,
            'rr_mid': rr_mid,
            'rr_upper': rr_upper,
            'meets_min_rr': rr_mid >= MIN_RISK_REWARD
        }
    
    elif side == 'short':
        # 숏 포지션의 경우
        # 손절: 직전 고점 또는 채널 상단 +2%
        stop_loss = max(df['high'].iloc[-2], channel_data['upper'] * 1.02)
        
        # 익절: 채널 중단선 또는 채널 하단
        take_profit_mid = channel_data['mid']
        take_profit_lower = channel_data['lower']
        
        # 손익비 계산 (리워드/리스크)
        risk = stop_loss - close
        reward_mid = close - take_profit_mid
        reward_lower = close - take_profit_lower
        
        # 중간선까지의 손익비
        rr_mid = reward_mid / risk if risk > 0 else 0
        # 하단까지의 손익비
        rr_lower = reward_lower / risk if risk > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'take_profit_mid': take_profit_mid,
            'take_profit_lower': take_profit_lower,
            'risk': risk,
            'reward_mid': reward_mid,
            'reward_lower': reward_lower,
            'rr_mid': rr_mid,
            'rr_lower': rr_lower,
            'meets_min_rr': rr_mid >= MIN_RISK_REWARD
        }
    
    return None

def check_sma_margin(df):
    """SMA20 근처 여부 확인"""
    close = df['close']
    sma20 = close.rolling(20).mean()
    current_price = close.iloc[-1]
    current_sma = sma20.iloc[-1]
    
    # 현재 가격이 SMA20에서 ±2% 이내인지 확인
    deviation = abs(current_price - current_sma) / current_sma
    return deviation <= SMA_DEVIATION

# ─────────── 4-Step 분석 함수 (개선됨) ───────────

def four_step_analysis(mtf_data, side):
    """
    개선된 4-Step 분석: 다중 시간대 분석을 통한 신호 평가
    """
    results = {}
    weights = {}
    
    # 1. 상위 시간대 추세 분석 (일봉)
    daily_trend = check_trend(mtf_data['daily'], side)
    results['daily_trend'] = daily_trend
    weights['daily_trend'] = WEIGHT_TREND * 0.4  # 40% 가중치
    
    # 2. 중간 시간대 분석 (4시간봉)
    h4_trend = check_trend(mtf_data['4h'], side)
    h4_channel = analyze_channel(mtf_data['4h'])
    h4_rsi = analyze_rsi(mtf_data['4h'], side)
    h4_volume = analyze_volume_profile(mtf_data['4h'])
    h4_risk_reward = calculate_risk_reward(mtf_data['4h'], side, h4_channel)
    
    results['h4_trend'] = h4_trend
    results['h4_channel'] = h4_channel
    results['h4_rsi'] = h4_rsi
    results['h4_volume'] = h4_volume
    results['h4_risk_reward'] = h4_risk_reward
    
    weights['h4_trend'] = WEIGHT_TREND * 0.4  # 40% 가중치
    weights['h4_channel'] = WEIGHT_CHANNEL * 0.6  # 60% 가중치
    weights['h4_rsi'] = WEIGHT_RSI * 0.6  # 60% 가중치
    weights['h4_volume'] = WEIGHT_VOLUME_PROFILE * 0.7  # 70% 가중치
    
    # 3. 하위 시간대 분석 (1시간봉)
    h1_trend = check_trend(mtf_data['1h'], side)
    h1_channel = analyze_channel(mtf_data['1h'])
    h1_rsi = analyze_rsi(mtf_data['1h'], side)
    h1_sma_margin = check_sma_margin(mtf_data['1h'])
    
    results['h1_trend'] = h1_trend
    results['h1_channel'] = h1_channel
    results['h1_rsi'] = h1_rsi
    results['h1_sma_margin'] = h1_sma_margin
    
    weights['h1_trend'] = WEIGHT_TREND * 0.2  # 20% 가중치
    weights['h1_channel'] = WEIGHT_CHANNEL * 0.4  # 40% 가중치
    weights['h1_rsi'] = WEIGHT_RSI * 0.4  # 40% 가중치
    
    # 4. 신호 평가 및 종합 점수 계산
    # 롱 포지션 조건
    if side == 'long':
        # 추세 조건
        trend_condition = (
            (daily_trend) or  # 일봉 상승 추세
            (h4_trend and h1_trend)  # 4시간 및 1시간 모두 상승 추세
        )
        
        # 채널 조건
        channel_condition = (
            h4_channel['position'] <= 0.2 or  # 4시간 채널 하단 20% 이내
            h1_channel['position'] <= 0.15  # 1시간 채널 하단 15% 이내
        )
        
        # RSI 조건
        rsi_condition = (
            h4_rsi['oversold'] or  # 4시간 RSI 과매도
            h4_rsi['bull_div'] or  # 4시간 상승 다이버전스
            h1_rsi['oversold'] or  # 1시간 RSI 과매도
            h1_rsi['bull_div']  # 1시간 상승 다이버전스
        )
        
        # 추가 조건
        additional_condition = (
            h1_sma_margin and  # 1시간 SMA20 근처
            h4_risk_reward['meets_min_rr']  # 최소 손익비 충족
        )
        
    # 숏 포지션 조건
    elif side == 'short':
        # 추세 조건
        trend_condition = (
            (not daily_trend) or  # 일봉 하락 추세
            (not h4_trend and not h1_trend)  # 4시간 및 1시간 모두 하락 추세
        )
        
        # 채널 조건
        channel_condition = (
            h4_channel['position'] >= 0.8 or  # 4시간 채널 상단 20% 이내
            h1_channel['position'] >= 0.85  # 1시간 채널 상단 15% 이내
        )
        
        # RSI 조건
        rsi_condition = (
            h4_rsi['overbought'] or  # 4시간 RSI 과매수
            h4_rsi['bear_div'] or  # 4시간 하락 다이버전스
            h1_rsi['overbought'] or  # 1시간 RSI 과매수
            h1_rsi['bear_div']  # 1시간 하락 다이버전스
        )
        
        # 추가 조건
        additional_condition = (
            h1_sma_margin and  # 1시간 SMA20 근처
            h4_risk_reward['meets_min_rr']  # 최소 손익비 충족
        )
    
    # 종합 판단
    signal_valid = trend_condition and channel_condition and rsi_condition and additional_condition
    
    # 신호 강도 계산 (0~100)
    signal_strength = 0
    if signal_valid:
        # 추세 점수 (0~30)
        trend_score = (
            (30 if daily_trend else 0) if side == 'long' else 
            (30 if not daily_trend else 0)
        )
        
        # 채널 점수 (0~25)
        if side == 'long':
            channel_score = 25 * (1 - h1_channel['position'])
        else:
            channel_score = 25 * h1_channel['position']
        
        # RSI 점수 (0~25)
        if side == 'long':
            rsi_score = 25 * (1 - min(h1_rsi['value'] / 50, 1))
        else:
            rsi_score = 25 * min(h1_rsi['value'] / 100, 1)
        
        # 매물대 점수 (0~20)
        volume_score = 20 * (1 - h4_volume['distance'])
        
        signal_strength = trend_score + channel_score + rsi_score + volume_score
    
    return {
        'valid': signal_valid,
        'strength': signal_strength,
        'details': results,
        'risk_reward': h4_risk_reward
    }

# ─────────── OKX 스캔 (선물) ───────────

def scan_okx():
    """OKX 선물 스캔"""
    longs = []
    shorts = []
    
    for m in okx.load_markets().values():
        if m['type'] != 'swap' or m['settle'] != 'USDT':
            continue
        
        sym = m['symbol']                           # BTC/USDT:USDT
        
        try:
            # 거래량 필터링
            tick = okx.fetch_ticker(sym)
            vol = tick.get('quoteVolume') or 0
            
            if vol < VOL_MIN_USDT:
                continue
                
            # 베이스 심볼 추출
            base = sym.split(':')[0].replace('/USDT', '')
            
            # 다중 시간대 데이터 가져오기
            mtf_data = fetch_mtf_data_okx(sym)
            time.sleep(0.2)  # API 속도 제한 준수
            
            # 롱 포지션 분석
            long_analysis = four_step_analysis(mtf_data, 'long')
            if long_analysis['valid']:
                longs.append({
                    'symbol': base,
                    'strength': long_analysis['strength'],
                    'risk_reward': long_analysis['risk_reward']['rr_mid'],
                    'rsi': long_analysis['details']['h1_rsi']['value']
                })
                # 신호 로깅
                log_signal('okx', base, 'long', {
                    'strength': long_analysis['strength'],
                    'risk_reward': long_analysis['risk_reward']['rr_mid'],
                    'rsi': long_analysis['details']['h1_rsi']['value']
                })
            
            # 숏 포지션 분석
            short_analysis = four_step_analysis(mtf_data, 'short')
            if short_analysis['valid']:
                shorts.append({
                    'symbol': base,
                    'strength': short_analysis['strength'],
                    'risk_reward': short_analysis['risk_reward']['rr_mid'],
                    'rsi': short_analysis['details']['h1_rsi']['value']
                })
                # 신호 로깅
                log_signal('okx', base, 'short', {
                    'strength': short_analysis['strength'],
                    'risk_reward': short_analysis['risk_reward']['rr_mid'],
                    'rsi': short_analysis['details']['h1_rsi']['value']
                })
                
        except Exception as e:
            print(f"[OKX skip] {sym}: {e}")
    
    # 신호 강도로 정렬
    longs.sort(key=lambda x: x['strength'], reverse=True)
    shorts.sort(key=lambda x: x['strength'], reverse=True)
    
    # 심볼만 추출
    long_symbols = [item['symbol'] for item in longs]
    short_symbols = [item['symbol'] for item in shorts]
    
    return long_symbols, short_symbols

# ─────────── Upbit 스캔 (현물) ───────────

def scan_upbit():
    """Upbit 현물 스캔"""
    spot = []
    
    for m in upbit.load_markets().values():
        if m['quote'] != 'KRW':
            continue
        
        sym = m['symbol']  # BTC/KRW
        
        try:
            # 거래량 필터링
            tick = upbit.fetch_ticker(sym)
            vol_krw = tick['info'].get('acc_trade_price_24h', 0)
            
            if float(vol_krw) < VOL_MIN_KRW:
                continue
            
            # 다중 시간대 데이터 가져오기
            mtf_data = fetch_mtf_data_upbit(sym)
            time.sleep(0.3)  # API 속도 제한 준수
            
            # 롱 포지션만 분석 (현물)
            analysis = four_step_analysis(mtf_data, 'long')
            if analysis['valid']:
                base = sym.replace('/KRW', '')
                spot.append({
                    'symbol': base,
                    'strength': analysis['strength'],
                    'risk_reward': analysis['risk_reward']['rr_mid'],
                    'rsi': analysis['details']['h1_rsi']['value']
                })
                # 신호 로깅
                log_signal('upbit', base, 'long', {
                    'strength': analysis['strength'],
                    'risk_reward': analysis['risk_reward']['rr_mid'],
                    'rsi': analysis['details']['h1_rsi']['value']
                })
                
        except Exception as e:
            print(f"[Upbit skip] {sym}: {e}")
    
    # 신호 강도로 정렬
    spot.sort(key=lambda x: x['strength'], reverse=True)
    
    # 심볼만 추출
    spot_symbols = [item['symbol'] for item in spot]
    
    return spot_symbols

# ─────────── 텔레그램 ───────────

def send_telegram(msg):
    """텔레그램으로 메시지 전송"""
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"})

# ─────────── 메인 ───────────

def main():
    """메인 함수"""
    try:
        # 스캔 실행
        long_, short_ = scan_okx()
        spot_ = scan_upbit()
        
        # 현재 시간 (한국 시간)
        now = dt.datetime.utcnow() + dt.timedelta(hours=9)
        
        # 메시지 포맷팅
        fmt = lambda x: ", ".join(x) if x else "―"
        msg = (f"*📊 4-Step Signals — {now:%Y-%m-%d %H:%M} KST*\n\n"
               f"*Long (OKX USDT-Perp)*\n{fmt(long_)}\n\n"
               f"*Short (OKX USDT-Perp)*\n{fmt(short_)}\n\n"
               f"*Spot (Upbit KRW)*\n{fmt(spot_)}")
        
        # 텔레그램 전송
        send_telegram(msg)
        
        print(f"✅ Signal scanning completed at {now:%Y-%m-%d %H:%M} KST")
        print(f"Long signals: {len(long_)}, Short signals: {len(short_)}, Spot signals: {len(spot_)}")
        
    except Exception as e:
        error_msg = f"❌ Error in signal bot: {str(e)}"
        print(error_msg)
        send_telegram(f"*ERROR*: {error_msg}")

if __name__ == "__main__":
    # 환경변수 확인
    if not TG_TOKEN or not TG_CHAT:
        sys.exit("❌ TG_TOKEN / TG_CHAT missing")
    
    # 메인 실행
    main()
