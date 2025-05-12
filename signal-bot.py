# ------------------------------------------------------------------
# requirements.txt
# pandas
# numpy
# ta
# ccxt
# requests
# ------------------------------------------------------------------
import os, sys, time, datetime as dt, pandas as pd, ta, requests, ccxt

TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT  = os.getenv("TG_CHAT")

# ───── 파라미터 ──────────────────────────────────────────────────
INTERVAL     = '4h'
CANDLES      = 300
LEN_CHAN     = 120
MARGIN       = 0.02
VOL_MIN_USD  = 1_000_000     # Binance 선물
VOL_MIN_KRW  = 1_000_000_000 # Upbit 현물 (KRW)

# ───── CCXT 인스턴스 ────────────────────────────────────────────
binance = ccxt.binanceusdm({'enableRateLimit': True})
upbit   = ccxt.upbit({'enableRateLimit': True})

# ───── 유틸 ────────────────────────────────────────────────────
def four_step(close: pd.Series, side: str) -> bool:
    trend_up = close.iloc[-1] > close.iloc[-2] > close.iloc[-3]
    trend_dn = close.iloc[-1] < close.iloc[-2] < close.iloc[-3]

    basis = ta.trend.ema_indicator(close, LEN_CHAN)
    dev   = (close - basis).abs().rolling(LEN_CHAN).max()
    lower, upper = basis - dev, basis + dev
    chan_long  = close.iloc[-1] <= lower.iloc[-1]*(1+MARGIN)
    chan_short = close.iloc[-1] >= upper.iloc[-1]*(1-MARGIN)

    rsi = ta.momentum.rsi(close, 14).iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    mv_ok = abs(close.iloc[-1]-sma20)/sma20 < MARGIN

    if side == 'long':
        return trend_up and chan_long  and rsi < 30 and mv_ok
    if side == 'short':
        return trend_dn and chan_short and rsi > 70 and mv_ok
    return False

def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str) -> pd.Series:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=INTERVAL, limit=CANDLES)
    closes = [c[4] for c in ohlcv]
    return pd.Series(closes)

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT, "text": text,
                             "parse_mode": "Markdown"})

# ───── 스캔 로직 ────────────────────────────────────────────────
def scan_binance():
    longs, shorts = [], []
    for m in binance.load_markets().values():
        if m['quote'] != 'USDT': continue
        symbol = m['symbol']                      # 예: BTC/USDT:USDT
        vol = binance.fetch_ticker(symbol)['quoteVolume']
        if vol is None or vol < VOL_MIN_USD: continue
        try:
            close = fetch_ohlcv(binance, symbol); time.sleep(0.1)
            if four_step(close,'long'):  longs.append(symbol.replace(':USDT',''))
            if four_step(close,'short'): shorts.append(symbol.replace(':USDT',''))
        except Exception as e:
            print("[Binance skip]", symbol, e)
    return longs, shorts

def scan_upbit():
    spots = []
    for m in upbit.load_markets().values():
        if m['quote'] != 'KRW': continue
        symbol = m['symbol']          # 예: BTC/KRW
        vol = upbit.fetch_ticker(symbol)['quoteVolume'] * upbit.fetch_ticker(symbol)['last']
        if vol < VOL_MIN_KRW: continue
        try:
            close = fetch_ohlcv(upbit, symbol); time.sleep(0.2)
            if four_step(close,'long'):
                spots.append(symbol.replace('/KRW',''))
        except Exception as e:
            print("[Upbit skip]", symbol, e)
    return spots

def main():
    long_, short_ = scan_binance()
    spot_         = scan_upbit()

    now = dt.datetime.utcnow()+dt.timedelta(hours=9)
    md = (f"*📊 CCXT 4-Step Signals – {now:%Y-%m-%d %H:%M} KST*\n\n"
          f"*Long (Binance USDT-Perp)*\n" + (", ".join(long_) or "―") + "\n\n"
          f"*Short (Binance USDT-Perp)*\n" + (", ".join(short_) or "―") + "\n\n"
          f"*Spot (Upbit KRW)*\n" + (", ".join(spot_) or "―"))
    send_telegram(md)

if __name__ == "__main__":
    if not TG_TOKEN or not TG_CHAT:
        sys.exit("❌ TG_TOKEN / TG_CHAT env missing")
    main()
