# signal_bot.py  – Binance 전용 4-Step 시그널 → Telegram
# ------------------------------------------------------------------
# 필요 패키지 : pandas, numpy, ta, requests
# requirements.txt:
# pandas
# numpy
# ta
# requests
# ------------------------------------------------------------------
import requests, pandas as pd, numpy as np, ta, time, datetime as dt, os, sys

# ---------- 기본 설정 ----------
INTERVAL   = "4h"           # 4시간봉
CANDLES    = 300            # 가져올 캔들 수 (300개 ≈ 50일)
VOL_MIN_USD= 1_000_000      # 24h 거래대금 필터
LEN_CHAN   = 120            # 회귀채널 길이
MARGIN     = 0.02           # 채널·SMA 허용 오차 2 %
TG_TOKEN   = os.getenv("TG_TOKEN")
TG_CHAT    = os.getenv("TG_CHAT")

# ---------- Binance API ----------
BASE = "https://fapi.binance.com"

def get_usdt_perp_list():
    url = f"{BASE}/fapi/v1/exchangeInfo"
    j = requests.get(url, timeout=10).json()
    return [s["symbol"] for s in j["symbols"]
            if s["quoteAsset"]=="USDT" and s["contractType"]=="PERPETUAL"]

def filter_by_volume(tickers):
    url = f"{BASE}/fapi/v1/ticker/24hr"
    j   = requests.get(url, timeout=10).json()
    vol_dict = {d["symbol"]: float(d["quoteVolume"]) for d in j}
    return [tk for tk in tickers if vol_dict.get(tk,0) > VOL_MIN_USD]

def fetch_ohlcv(symbol):
    url = f"{BASE}/fapi/v1/klines"
    params = {"symbol":symbol, "interval":INTERVAL, "limit":CANDLES}
    j = requests.get(url, params=params, timeout=10).json()
    closes = pd.Series([float(x[4]) for x in j])
    return closes

# ---------- 4-Step 필터 ----------
def four_step(close, direction):
    # ① 추세 (최근 3봉 HH/HL, LH/LL 단순 판별)
    trend_up = close.iloc[-1] > close.iloc[-2] > close.iloc[-3]
    trend_dn = close.iloc[-1] < close.iloc[-2] < close.iloc[-3]

    # ② 회귀채널
    basis = ta.trend.ema_indicator(close, LEN_CHAN)  # 근사치
    dev   = (close - basis).abs().rolling(LEN_CHAN).max()
    lower, upper = basis - dev, basis + dev
    chan_long  = close.iloc[-1] <= lower.iloc[-1]*(1+MARGIN)
    chan_short = close.iloc[-1] >= upper.iloc[-1]*(1-MARGIN)

    # ③ RSI
    rsi = ta.momentum.rsi(close, 14).iloc[-1]
    r_long, r_short = rsi < 30, rsi > 70

    # ④ SMA20 ±2 %
    sma20 = close.rolling(20).mean().iloc[-1]
    mv = abs(close.iloc[-1]-sma20)/sma20 < MARGIN

    if direction=="long":
        return trend_up and chan_long and r_long and mv
    if direction=="short":
        return trend_dn and chan_short and r_short and mv
    return False

# ---------- 스캔 ----------
def scan_market():
    tickers = filter_by_volume(get_usdt_perp_list())
    long_ls, short_ls = [], []
    for tk in tickers:
        try:
            s = fetch_ohlcv(tk); time.sleep(0.1)   # rate limit 보호
            if four_step(s,"long"):  long_ls.append(tk)
            if four_step(s,"short"): short_ls.append(tk)
        except Exception as e:
            print(f"{tk} fetch error:", e, file=sys.stderr)
    return long_ls, short_ls

# ---------- Telegram ----------
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id":TG_CHAT,
                             "text":text,
                             "parse_mode":"Markdown"})

# ---------- 메인 ----------
def main():
    long_, short_ = scan_market()
    today = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)
    header = f"*📊 Binance 4-Step Signals – {today:%Y-%m-%d} 09:00 KST*"
    def fmt(lst): return ", ".join(lst) if lst else "―"
    msg = (f"{header}\n\n*Long*\n{fmt(long_)}\n\n*Short*\n{fmt(short_)}")
    send_telegram(msg)

if __name__ == "__main__":
    if not TG_TOKEN or not TG_CHAT:
        sys.exit("❌  TG_TOKEN or TG_CHAT env missing")
    main()
