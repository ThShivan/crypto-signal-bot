# ===============================================
# signal_bot.py – OKX USDT-Perp + Upbit KRW Spot
#   • 4-Step 필터(추세·회귀채널·RSI·SMA20±2 %)
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
import os, sys, time, datetime as dt
import pandas as pd, ta, requests, ccxt

# ─────────── 환경변수 (깃허브 Secrets) ───────────
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT  = os.getenv("TG_CHAT")

# ─────────── 매개변수 ───────────
INTERVAL_OKX   = '4h'       # OKX는 4h 프리셋 지원
INTERVAL_UPBIT = '240m'     # Upbit 4h = 240분
CANDLES_OKX    = 300
CANDLES_UPBIT  = 200
LEN_CHAN       = 120
MARGIN         = 0.02
VOL_MIN_USDT   = 1_000_000        # OKX 24h 거래대금
VOL_MIN_KRW    = 1_000_000_000    # Upbit 24h 거래대금(원)

# ─────────── 거래소 인스턴스 ───────────
okx   = ccxt.okx   ({'enableRateLimit': True})
upbit = ccxt.upbit({'enableRateLimit': True})

# ─────────── 4-Step 필터 함수 ───────────
def four_step(close: pd.Series, side: str) -> bool:
    trend_up = close[-1] > close[-2] > close[-3]
    trend_dn = close[-1] < close[-2] < close[-3]

    basis = ta.trend.ema_indicator(close, LEN_CHAN)
    dev   = (close - basis).abs().rolling(LEN_CHAN).max()
    lower, upper = basis - dev, basis + dev
    chan_long  = close[-1] <= lower[-1] * (1 + MARGIN)
    chan_short = close[-1] >= upper[-1] * (1 - MARGIN)

    rsi   = ta.momentum.rsi(close, 14).iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    mv_ok = abs(close[-1] - sma20) / sma20 < MARGIN

    if side == 'long':
        return trend_up and chan_long  and rsi < 30 and mv_ok
    if side == 'short':
        return trend_dn and chan_short and rsi > 70 and mv_ok
    return False

# ─────────── OHLCV 래퍼 ───────────
def fetch_close_okx(symbol):
    ohlcv = okx.fetch_ohlcv(symbol, timeframe=INTERVAL_OKX, limit=CANDLES_OKX)
    return pd.Series([c[4] for c in ohlcv])

def fetch_close_upbit(symbol):
    ohlcv = upbit.fetch_ohlcv(symbol, timeframe=INTERVAL_UPBIT, limit=CANDLES_UPBIT)
    return pd.Series([c[4] for c in ohlcv])

# ─────────── OKX 스캔 (선물) ───────────
def scan_okx():
    longs, shorts = [], []
    for m in okx.load_markets().values():
        if m['type'] != 'swap' or m['settle'] != 'USDT':
            continue
        sym = m['symbol']                           # BTC/USDT:USDT
        tick = okx.fetch_ticker(sym)
        vol  = tick.get('quoteVolume') or 0
        if vol < VOL_MIN_USDT:
            continue
        try:
            close = fetch_close_okx(sym); time.sleep(0.12)
            base  = sym.split(':')[0].replace('/USDT', '')
            if four_step(close, 'long'):
                longs.append(base)
            if four_step(close, 'short'):
                shorts.append(base)
        except Exception as e:
            print("[OKX skip]", sym, e)
    return longs, shorts

# ─────────── Upbit 스캔 (현물) ───────────
def scan_upbit():
    spot = []
    for m in upbit.load_markets().values():
        if m['quote'] != 'KRW':
            continue
        sym = m['symbol']                           # BTC/KRW
        tick = upbit.fetch_ticker(sym)
        vol_krw = tick['info'].get('acc_trade_price_24h', 0)
        if vol_krw < VOL_MIN_KRW:
            continue
        try:
            close = fetch_close_upbit(sym); time.sleep(0.25)
            if four_step(close, 'long'):
                spot.append(sym.replace('/KRW', ''))
        except Exception as e:
            print("[Upbit skip]", sym, e)
    return spot

# ─────────── 텔레그램 ───────────
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT, "text": msg,
                             "parse_mode": "Markdown"})

# ─────────── 메인 ───────────
def main():
    long_, short_ = scan_okx()
    spot_ = scan_upbit()

    now = dt.datetime.utcnow() + dt.timedelta(hours=9)
    fmt = lambda x: ", ".join(x) if x else "―"
    msg = (f"*📊 4-Step Signals — {now:%Y-%m-%d %H:%M} KST*\n\n"
           f"*Long (OKX USDT-Perp)*\n{fmt(long_)}\n\n"
           f"*Short (OKX USDT-Perp)*\n{fmt(short_)}\n\n"
           f"*Spot (Upbit KRW)*\n{fmt(spot_)}")
    send_telegram(msg)

if __name__ == "__main__":
    if not TG_TOKEN or not TG_CHAT:
        sys.exit("❌  TG_TOKEN / TG_CHAT missing")
    main()
