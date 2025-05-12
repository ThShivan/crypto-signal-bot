# ------------------------------------------------------------------
# requirements.txt
# pandas
# numpy
# ta
# requests
# ------------------------------------------------------------------
import os, sys, time, datetime as dt, requests, pandas as pd, ta

# ───────── 환경 변수 (GitHub-Secrets 입력) ─────────
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT  = os.getenv("TG_CHAT")

# ───────── 파라미터 ─────────
BASE          = "https://api.bybit.com/v5/market"
INTERVAL      = "240"               # Bybit kline 240 = 4h
CANDLES       = 300                 # 50 일 가량
VOL_MIN_USD   = 1_000_000           # 24 h turnover ≥ 1 M USD
LEN_CHAN      = 120
MARGIN        = 0.02                # 2 %

# ────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.bybit.com/",
}

def safe_get(url, params=None, tries=3, wait=2):
    for i in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 403:
                raise Exception("403 Forbidden")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[safe_get] {url} (try {i}/{tries}) → {e}")
            time.sleep(wait)
    return None


# 1) USDT-Perp 심볼 전체
def get_bybit_symbols():
    j = safe_get(f"{BASE}/instruments", params={"category": "linear"})
    if j and "list" in j.get("result", {}):
        return [d["symbol"] for d in j["result"]["list"]]
    print("⚠️  instruments endpoint error — fallback list")
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]            # 폴백 최소

# 2) 24 h 거래대금 필터
def filter_by_volume(symbols):
    j = safe_get(f"{BASE}/tickers", params={"category": "linear"})
    if not j:
        print("⚠️  tickers endpoint error, skip volume filter")
        return symbols
    vol = {d["symbol"]: float(d["turnover24h"]) for d in j["result"]["list"]}
    return [s for s in symbols if vol.get(s, 0) > VOL_MIN_USD]

# 3) OHLCV → Close Series
def fetch_close_series(symbol):
    params = {"category": "linear", "symbol": symbol,
              "interval": INTERVAL, "limit": CANDLES}
    j = safe_get(f"{BASE}/kline", params=params)
    if not j or not j["result"]["list"]:
        raise RuntimeError("kline fetch failed")
    closes = [float(c[4]) for c in j["result"]["list"]]
    return pd.Series(closes)

# 4) 4-Step 필터
def four_step(close, side):
    trend_up = close.iloc[-1] > close.iloc[-2] > close.iloc[-3]
    trend_dn = close.iloc[-1] < close.iloc[-2] < close.iloc[-3]

    basis = ta.trend.ema_indicator(close, LEN_CHAN)
    dev   = (close - basis).abs().rolling(LEN_CHAN).max()
    lower, upper = basis - dev, basis + dev
    chan_long  = close.iloc[-1] <= lower.iloc[-1] * (1 + MARGIN)
    chan_short = close.iloc[-1] >= upper.iloc[-1] * (1 - MARGIN)

    rsi = ta.momentum.rsi(close, 14).iloc[-1]
    r_long, r_short = rsi < 30, rsi > 70

    sma20 = close.rolling(20).mean().iloc[-1]
    mv_ok = abs(close.iloc[-1] - sma20) / sma20 < MARGIN

    if side == "long":
        return trend_up and chan_long  and r_long  and mv_ok
    if side == "short":
        return trend_dn and chan_short and r_short and mv_ok
    return False

# 5) 시장 스캔
def scan():
    symbols = filter_by_volume(get_bybit_symbols())
    longs, shorts = [], []
    for s in symbols:
        try:
            closes = fetch_close_series(s); time.sleep(0.3)
            if four_step(closes, "long"):  longs.append(s)
            if four_step(closes, "short"): shorts.append(s)
        except Exception as e:
            print(f"[scan] {s} skipped – {e}", file=sys.stderr)
    return longs, shorts

# 6) Telegram
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT, "text": msg,
                             "parse_mode": "Markdown"})

# ─────────── main ───────────
def main():
    long_, short_ = scan()
    now = dt.datetime.utcnow() + dt.timedelta(hours=9)
    head = f"*📊 Bybit USDT-Perp 4-Step Signals – {now:%Y-%m-%d %H:%M} KST*"
    fmt  = lambda lst: ", ".join(lst) if lst else "―"
    msg  = f"{head}\n\n*Long*\n{fmt(long_)}\n\n*Short*\n{fmt(short_)}"
    send_telegram(msg)

if __name__ == "__main__":
    if not TG_TOKEN or not TG_CHAT:
        sys.exit("❌  TG_TOKEN / TG_CHAT env missing")
    main()
