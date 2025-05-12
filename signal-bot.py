# signal_bot.py  ─ Binance 4-Step Signal Bot (robust ver.)
# ------------------------------------------------------------------
# requirements.txt
# pandas
# numpy
# ta
# requests
# ------------------------------------------------------------------
import os, sys, time, datetime as dt, requests, pandas as pd, ta

# ▶️ 환경 변수 (GitHub Secrets 로 세팅)
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT  = os.getenv("TG_CHAT")

# ▶️ 상수
BASE         = "https://fapi.binance.com"
INTERVAL     = "4h"
CANDLES      = 300
VOL_MIN_USD  = 1_000_000          # 24h 거래대금 최소
LEN_CHAN     = 120
MARGIN       = 0.02

# ------------------------------------------------------------------
# 🔹 공용: 안전한 GET (재시도)
def safe_get(url, params=None, tries=3, wait=2):
    for i in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[safe_get] {url} (try {i}/{tries}) → {e}")
            time.sleep(wait)
    return None

# 🔹 1) 티커 목록
def get_usdt_perp_list():
    j = safe_get(f"{BASE}/fapi/v1/exchangeInfo")
    if j and "symbols" in j:
        return [s["symbol"] for s in j["symbols"]
                if s["quoteAsset"] == "USDT" and s["contractType"] == "PERPETUAL"]
    # fallback 최소 리스트
    return ["BTCUSDT", "ETHUSDT", "FETUSDT", "PEPEUSDT"]

# 🔹 2) 24h 거래대금 필터
def filter_by_volume(tickers):
    j = safe_get(f"{BASE}/fapi/v1/ticker/24hr")
    if not j:
        print("⚠️  24hr endpoint error, skip volume filter")
        return tickers
    vol = {d["symbol"]: float(d["quoteVolume"]) for d in j}
    return [tk for tk in tickers if vol.get(tk, 0) > VOL_MIN_USD]

# 🔹 3) OHLCV (클로즈 시리즈)
def fetch_close_series(symbol):
    params = {"symbol": symbol, "interval": INTERVAL, "limit": CANDLES}
    j = safe_get(f"{BASE}/fapi/v1/klines", params=params)
    if not j:
        raise RuntimeError("klines fetch failed")
    return pd.Series([float(x[4]) for x in j])

# 🔹 4) 4-Step 필터
def four_step(close, direction):
    # ① 추세 (최근 3봉 HH-HL or LH-LL)
    trend_up = close.iloc[-1] > close.iloc[-2] > close.iloc[-3]
    trend_dn = close.iloc[-1] < close.iloc[-2] < close.iloc[-3]

    # ② 회귀채널
    basis = ta.trend.ema_indicator(close, LEN_CHAN)
    dev   = (close - basis).abs().rolling(LEN_CHAN).max()
    lower, upper = basis - dev, basis + dev
    chan_long  = close.iloc[-1] <= lower.iloc[-1] * (1 + MARGIN)
    chan_short = close.iloc[-1] >= upper.iloc[-1] * (1 - MARGIN)

    # ③ RSI
    rsi = ta.momentum.rsi(close, 14).iloc[-1]
    r_long, r_short = rsi < 30, rsi > 70

    # ④ SMA20 ±2 %
    sma20 = close.rolling(20).mean().iloc[-1]
    mv_ok = abs(close.iloc[-1] - sma20) / sma20 < MARGIN

    if direction == "long":
        return trend_up and chan_long  and r_long  and mv_ok
    if direction == "short":
        return trend_dn and chan_short and r_short and mv_ok
    return False

# 🔹 5) 전체 스캔
def scan_market():
    tickers = filter_by_volume(get_usdt_perp_list())
    long_ls, short_ls = [], []
    for tk in tickers:
        try:
            s = fetch_close_series(tk)
            time.sleep(0.1)  # API 부담↓
            if four_step(s, "long"):  long_ls.append(tk)
            if four_step(s, "short"): short_ls.append(tk)
        except Exception as e:
            print(f"[scan] {tk} skipped – {e}", file=sys.stderr)
    return long_ls, short_ls

# 🔹 6) Telegram
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    safe_get(url, params=None, tries=1)  # ping
    requests.post(url, json={"chat_id": TG_CHAT,
                             "text": text,
                             "parse_mode": "Markdown"})

# 🔹 메인
def main():
    long_, short_ = scan_market()
    today = dt.datetime.utcnow() + dt.timedelta(hours=9)
    header = f"*📊 Binance 4-Step Signals – {today:%Y-%m-%d %H:%M} KST*"
    fmt = lambda lst: ", ".join(lst) if lst else "―"
    msg = f"{header}\n\n*Long*\n{fmt(long_)}\n\n*Short*\n{fmt(short_)}"
    send_telegram(msg)

if __name__ == "__main__":
    if not TG_TOKEN or not TG_CHAT:
        sys.exit("❌ TG_TOKEN / TG_CHAT missing")
    main()
