# OKX USDT-Perp + Upbit KRW Spot 4-Step Signal Bot
# -------------------------------------------------
# 필요 패키지: pandas, numpy, ta, ccxt, requests
import os, sys, time, datetime as dt, pandas as pd, ta, ccxt, requests

TG_TOKEN = os.getenv("TG_TOKEN")    # → GitHub Secrets
TG_CHAT  = os.getenv("TG_CHAT")

# 공통 파라미터
INTERVAL, CANDLES = '4h', 300       # 약 50일
LEN_CHAN, MARGIN  = 120, 0.02
VOL_MIN_USDT      = 1_000_000       # OKX 24h 거래대금
VOL_MIN_KRW       = 1_000_000_000   # Upbit 24h 거래대금(원)

# CCXT 인스턴스
okx   = ccxt.okx   ({'enableRateLimit': True})
upbit = ccxt.upbit({'enableRateLimit': True})

# ─────────────────── 4-Step 필터 ───────────────────
def four_step(close: pd.Series, side: str) -> bool:
    trend_up = close[-1] > close[-2] > close[-3]
    trend_dn = close[-1] < close[-2] < close[-3]

    basis = ta.trend.ema_indicator(close, LEN_CHAN)
    dev   = (close - basis).abs().rolling(LEN_CHAN).max()
    lower, upper = basis - dev, basis + dev
    chan_long  = close[-1] <= lower[-1]*(1+MARGIN)
    chan_short = close[-1] >= upper[-1]*(1-MARGIN)

    rsi = ta.momentum.rsi(close, 14).iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    mv_ok = abs(close[-1]-sma20)/sma20 < MARGIN

    if side == 'long':
        return trend_up and chan_long  and rsi < 30 and mv_ok
    if side == 'short':
        return trend_dn and chan_short and rsi > 70 and mv_ok
    return False

# ─────────────────── 유틸 ──────────────────────────
def fetch_close(ex, symbol):
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=INTERVAL, limit=CANDLES)
    return pd.Series([c[4] for c in ohlcv])

def send_telegram(msg):
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                  json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"})

# ─────────────────── 스캔 ──────────────────────────
def scan_okx():
    longs, shorts = []
    for m in okx.load_markets().values():
        if m['type'] != 'swap' or m['settle'] != 'USDT':
            continue
        sym = m['symbol']                      # BTC/USDT:USDT
        tick = okx.fetch_ticker(sym)

        # ➜ 거래대금 값이 없으면 0 으로 간주해 필터에서 제외
        vol = tick.get('quoteVolume') or tick.get('quoteVolume24h') or 0
        if vol < VOL_MIN_USDT:
            continue

        try:
            close = fetch_close(okx, sym); time.sleep(0.12)
            stripped = sym.split(':')[0]       # BTC/USDT
            base = stripped.replace('/USDT', '')
            if four_step(close, 'long'):  longs.append(base)
            if four_step(close, 'short'): shorts.append(base)
        except Exception as e:
            print("[OKX skip]", sym, e)
    return longs, shorts


def scan_upbit():
    spot = []
    for m in upbit.load_markets().values():
        if m['quote']!='KRW': continue
        sym = m['symbol']                             # BTC/KRW
        tick = upbit.fetch_ticker(sym)
        vol_krw = tick.get('quoteVolume') * tick.get('last', 0) if tick.get('quoteVolume') else 0
        if vol_krw < VOL_MIN_KRW: continue
        try:
            close = fetch_close(upbit, sym); time.sleep(0.25)
            if four_step(close,'long'):
                spot.append(sym.replace('/KRW',''))
        except Exception as e:
            print("[Upbit skip]", sym, e)
    return spot

# ─────────────────── 메인 ──────────────────────────
def main():
    long_, short_ = scan_okx()
    spot_ = scan_upbit()

    now = dt.datetime.utcnow() + dt.timedelta(hours=9)
    body = (f"*📊 4-Step Signals — {now:%Y-%m-%d %H:%M} KST*\n\n"
            f"*Long (OKX USDT-Perp)*\n{', '.join(long_) or '―'}\n\n"
            f"*Short (OKX USDT-Perp)*\n{', '.join(short_) or '―'}\n\n"
            f"*Spot (Upbit KRW)*\n{', '.join(spot_) or '―'}")
    send_telegram(body)

if __name__ == "__main__":
    if not TG_TOKEN or not TG_CHAT:
        sys.exit("❌  TG_TOKEN / TG_CHAT environment variables missing")
    main()
