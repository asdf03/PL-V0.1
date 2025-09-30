"""
spy_flash_detector.py
Educational script: detect large intraday drops (flash-crash-like) and optionally paper-trade via Alpaca.
NOT financial advice. Use this for research / backtesting only. Do NOT run on a funded live account
without understanding the risks, slippage, and regulatory concerns.

Requirements:
  pip install yfinance pandas numpy alpaca-trade-api pytz
"""

import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import pytz
import logging

# Optional Alpaca integration (paper trading)
try:
    from alpaca_trade_api.rest import REST, TimeFrame
    ALPACA_AVAILABLE = True
except Exception:
    ALPACA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

@dataclass
class Config:
    symbol: str = "SPY"            # S&P 500 ETF (use SPY for equities)
    lookback_minutes: int = 60     # how much intraday history to fetch for backtests / start
    drop_threshold_1m: float = 0.015   # 1.5% drop in one minute -> candidate flash crash
    drop_threshold_5m: float = 0.03    # 3% drop over 5 minutes -> candidate
    volume_spike_mult: float = 3.0     # volume >= this * average volume over lookback window
    position_risk: float = 0.01        # risk fraction of account equity per trade (1%)
    take_profit: float = 0.01          # 1% take profit target
    stop_loss: float = 0.02            # 2% stop loss
    poll_interval_seconds: int = 30    # how often to poll for new 1m candle (live mode)
    alpaca_paper: bool = True          # if True and credentials set, place paper trades
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    timezone: str = "US/Eastern"

cfg = Config()

def fetch_intraday(symbol: str, period_minutes: int) -> pd.DataFrame:
    """
    Fetch intraday 1m bars using yfinance.
    Note: yfinance provides up-to-date intraday but it's not true streaming; there can be delays.
    """
    period = "2d" if period_minutes > 1440 else "1d"
    interval = "1m"
    logging.info("Fetching intraday data via yfinance (may be delayed)...")
    df = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned no data. Check symbol / market hours / connectivity.")
    # Keep only recent rows for lookback
    df = df.iloc[-(period_minutes+10):].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    return df

def detect_flash(df: pd.DataFrame, cfg: Config) -> Optional[Dict[str,Any]]:
    """
    Detects candidate flash crash events based on magnitude + volume.
    Returns detection info or None.
    """
    if len(df) < 6:
        return None
    # use the last timestamp as current candle
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    # 1-minute pct drop
    pct_1m = (prev.close - latest.close) / prev.close
    # 5-minute drop: compare to 5 bars ago (if available)
    pct_5m = None
    if len(df) >= 6:
        five_min_ago = df.iloc[-6].close
        pct_5m = (five_min_ago - latest.close) / five_min_ago

    avg_volume = df.volume[-(cfg.lookback_minutes or 60):].mean()
    vol_spike = latest.volume >= (cfg.volume_spike_mult * (avg_volume if avg_volume>0 else 1))

    logging.debug(f"pct_1m={pct_1m:.4f} pct_5m={pct_5m:.4f if pct_5m is not None else 'N/A'} vol_spike={vol_spike}")

    if pct_1m >= cfg.drop_threshold_1m and vol_spike:
        return {"type":"1m_drop", "pct_1m":pct_1m, "pct_5m":pct_5m, "time":latest.name, "price":latest.close}
    if pct_5m is not None and pct_5m >= cfg.drop_threshold_5m and vol_spike:
        return {"type":"5m_drop", "pct_1m":pct_1m, "pct_5m":pct_5m, "time":latest.name, "price":latest.close}
    return None

def backtest_on_history(symbol: str, cfg: Config) -> pd.DataFrame:
    """
    Simple backtester that scans historical minute bars and simulates an entry at the close of the detection bar,
    with TP and SL applied as percent moves from entry. This is a naive simulator (no slippage, fees).
    """
    df = fetch_intraday(symbol, period_minutes=cfg.lookback_minutes*6)  # fetch more for backtest
    results = []
    for i in range(6, len(df)):
        window = df.iloc[:i+1]
        candidate = detect_flash(window, cfg)
        if candidate:
            entry_price = window.iloc[-1].close
            tp_price = entry_price * (1 - cfg.take_profit)  # short: profit when price falls
            sl_price = entry_price * (1 + cfg.stop_loss)
            # naive forward scan to see which is hit first within next N bars (e.g., 120 bars = 2 hours)
            future = df.iloc[i+1:i+121]
            outcome = "no_hit"
            hit_price = None
            hit_time = None
            for idx, row in future.iterrows():
                if row.low <= tp_price:
                    outcome = "tp"
                    hit_price = tp_price
                    hit_time = idx
                    break
                if row.high >= sl_price:
                    outcome = "sl"
                    hit_price = sl_price
                    hit_time = idx
                    break
            results.append({
                "detect_time": candidate["time"],
                "type": candidate["type"],
                "entry": entry_price,
                "outcome": outcome,
                "hit_price": hit_price,
                "hit_time": hit_time
            })
    return pd.DataFrame(results)

# Alpaca helper (paper trading). Only used if ALPACA_AVAILABLE and cfg.alpaca_paper True.
def init_alpaca():
    if not ALPACA_AVAILABLE:
        raise RuntimeError("Alpaca SDK not installed (pip install alpaca-trade-api)")
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set environment variables APCA_API_KEY_ID and APCA_API_SECRET_KEY for Alpaca")
    client = REST(key, secret, base_url=cfg.alpaca_base_url)
    return client

def place_short_order(alpaca_client, symbol: str, qty: int, cfg: Config):
    """
    Place a market short order for 'qty' shares using Alpaca REST client.
    This is paper trading if using paper endpoint.
    """
    logging.info(f"Placing market sell (short) order for {qty} shares of {symbol} (paper mode if paper API keys).")
    try:
        order = alpaca_client.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        return order
    except Exception as e:
        logging.exception("Alpaca order failed")
        return None

def main_loop(cfg: Config):
    logging.info("Starting monitoring/backtesting script (educational).")
    # Backtest on recent history first
    logging.info("Running quick backtest on historical data (naive simulator)...")
    bt = backtest_on_history(cfg.symbol, cfg)
    logging.info("Backtest summary:\n%s", bt['outcome'].value_counts().to_dict())

    alpaca_client = None
    if cfg.alpaca_paper and ALPACA_AVAILABLE:
        try:
            alpaca_client = init_alpaca()
            logging.info("Alpaca client initialized (paper mode).")
        except Exception as e:
            logging.warning("Alpaca not initialized: %s", e)
            alpaca_client = None

    # Live monitoring loop (polling). Note: yfinance is not a low-latency stream; for live trading you'd use exchange feed / broker websockets.
    while True:
        try:
            df = fetch_intraday(cfg.symbol, cfg.lookback_minutes)
            detection = detect_flash(df, cfg)
            if detection:
                logging.info("DETECTION: %s", detection)
                # compute position size (naive)
                if alpaca_client:
                    account = alpaca_client.get_account()
                    eq = float(account.equity)
                    risk_amount = eq * cfg.position_risk
                    entry_price = detection["price"]
                    # For shorts, number of shares = risk_amount / (stop_loss * entry_price)
                    shares = int(risk_amount / (cfg.stop_loss * entry_price))
                    shares = max(shares, 1)
                    logging.info("Simulated position sizing: account equity=%s risk_amount=%.2f shares=%s", eq, risk_amount, shares)
                    # Place paper short order
                    place_short_order(alpaca_client, cfg.symbol, shares, cfg)
                    # IMPORTANT: you must implement TP/SL order placement or monitors to close positions.
                else:
                    logging.info("Alpaca not configured. Detection only (no live or paper trade placed).")
            else:
                logging.debug("No detection at %s", dt.datetime.now())
        except Exception as e:
            logging.exception("Error in main loop")
        time.sleep(cfg.poll_interval_seconds)

if __name__ == "__main__":
    # Quick configuration hints printed on startup
    logging.info("Script configured for symbol=%s. Remember: this is educational and simulated/backtest-first.", cfg.symbol)
    # Run main_loop() if you want real-time polling; for unit/backtest only, you can call backtest_on_history()
    # To keep this example safe, comment out main_loop() unless you explicitly intend to run live/paper mode.
    # main_loop(cfg)

    # Provide a backtest snapshot instead of starting live loop by default
    df_results = backtest_on_history(cfg.symbol, cfg)
    if df_results.empty:
        logging.info("No detections found in backtest window.")
    else:
        logging.info("Backtest detections found: %d", len(df_results))
        print(df_results.head(20).to_string(index=False))
