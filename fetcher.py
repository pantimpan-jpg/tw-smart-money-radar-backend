from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd
import twstock

from .config import (
    EXCLUDE_ETF_ETN,
    EXCLUDE_FINANCIAL,
    EXCLUDE_INDEX,
    EXCLUDE_WARRANT,
    LOOKBACK_MONTHS,
    MAX_WORKERS,
    RETRY_TIMES,
)

warnings.filterwarnings("ignore")


def safe_round(value, digits: int = 2) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, float) and math.isnan(value):
            return 0.0
        return round(float(value), digits)
    except Exception:
        return 0.0


def to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def is_common_stock(code_info) -> bool:
    code = code_info.code
    name = code_info.name
    group = getattr(code_info, "group", "") or ""

    if not code.isdigit():
        return False
    if len(code) not in {4, 5}:
        return False
    if EXCLUDE_INDEX and (code.startswith("0") or "指數" in name):
        return False
    if EXCLUDE_ETF_ETN and ("ETF" in name or "ETN" in name or group in {"ETF", "ETN"}):
        return False
    if EXCLUDE_WARRANT and ("權證" in name or "牛證" in name or "熊證" in name):
        return False
    if EXCLUDE_FINANCIAL and ("金融" in group or "保險" in group or "銀行" in group):
        return False
    return True


def get_all_taiwan_stocks() -> list:
    codes = []
    for _, info in twstock.codes.items():
        if is_common_stock(info):
            codes.append(info)
    codes.sort(key=lambda x: x.code)
    return codes


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
    obv = (direction * volume).cumsum()
    return pd.Series(obv, index=close.index)


def calc_bollinger(series: pd.Series, period: int = 20):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower


def fetch_history_df(stock_id: str, months_back: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    stock = twstock.Stock(stock_id)

    today = date.today()
    start_year = today.year
    start_month = today.month - months_back

    while start_month <= 0:
        start_month += 12
        start_year -= 1

    rows = stock.fetch_from(start_year, start_month)
    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        data.append(
            {
                "date": pd.to_datetime(r.date),
                "capacity": to_float(r.capacity),
                "turnover": to_float(r.turnover),
                "open": to_float(r.open),
                "high": to_float(r.high),
                "low": to_float(r.low),
                "close": to_float(r.close),
                "change": to_float(r.change),
                "transaction": to_float(r.transaction),
            }
        )

    df = pd.DataFrame(data).sort_values("date").reset_index(drop=True)
    df["volume_lot"] = df["capacity"] / 1000
    df["turnover_100m"] = df["turnover"] / 100_000_000
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["volume_avg20"] = df["volume_lot"].rolling(20).mean()
    df["pct_5d"] = df["close"].pct_change(5) * 100
    df["pct_20d"] = df["close"].pct_change(20) * 100
    df["platform_high_20d"] = df["high"].rolling(20).max().shift(1)
    df["platform_high_60d"] = df["high"].rolling(60).max().shift(1)
    df["rsi"] = calc_rsi(df["close"])
    macd, signal, hist = calc_macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    df["obv"] = calc_obv(df["close"], df["volume_lot"])
    df["obv_trend"] = df["obv"].diff(5)
    boll_upper, boll_mid, boll_lower = calc_bollinger(df["close"])
    df["boll_upper"] = boll_upper
    df["boll_mid"] = boll_mid
    df["boll_lower"] = boll_lower
    return df


def fetch_one_stock(info):
    stock_id = info.code
    name = info.name
    market = getattr(info, "market", "") or ""
    group = getattr(info, "group", "") or ""

    for _ in range(RETRY_TIMES + 1):
        try:
            df = fetch_history_df(stock_id)
            if df.empty or len(df) < 60:
                return None

            latest = df.iloc[-1]
            volume_ratio = to_float(latest["volume_lot"]) / max(to_float(latest["volume_avg20"]), 1e-9)
            pct_from_ma20 = ((to_float(latest["close"]) - to_float(latest["ma20"])) / max(to_float(latest["ma20"]), 1e-9)) * 100

            return {
                "stock_id": stock_id,
                "name": name,
                "market": market,
                "group": group,
                "close": safe_round(latest["close"]),
                "ma20": safe_round(latest["ma20"]),
                "ma60": safe_round(latest["ma60"]),
                "volume": safe_round(latest["volume_lot"]),
                "volume_avg20": safe_round(latest["volume_avg20"]),
                "volume_ratio": safe_round(volume_ratio),
                "turnover_100m": safe_round(latest["turnover_100m"]),
                "pct_from_ma20": safe_round(pct_from_ma20),
                "pct_5d": safe_round(latest["pct_5d"]),
                "pct_20d": safe_round(latest["pct_20d"]),
                "rsi": safe_round(latest["rsi"]),
                "macd": safe_round(latest["macd"], 4),
                "macd_signal": safe_round(latest["macd_signal"], 4),
                "macd_hist": safe_round(latest["macd_hist"], 4),
                "obv_trend": safe_round(latest["obv_trend"]),
                "boll_upper": safe_round(latest["boll_upper"]),
                "boll_mid": safe_round(latest["boll_mid"]),
                "boll_lower": safe_round(latest["boll_lower"]),
                "platform_high_20d": safe_round(latest["platform_high_20d"]),
                "platform_high_60d": safe_round(latest["platform_high_60d"]),
            }
        except Exception:
            continue
    return None


def fetch_market_snapshot_parallel(progress_every: int = 50) -> pd.DataFrame:
    codes = get_all_taiwan_stocks()
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(fetch_one_stock, info): info for info in codes}
        for future in as_completed(future_map):
            done += 1
            try:
                result = future.result()
                if result:
                    rows.append(result)
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
