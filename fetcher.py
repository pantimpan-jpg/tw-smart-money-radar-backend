from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import twstock

from config import LOOKBACK_MONTHS, MAX_WORKERS


def is_common_stock(info: Any) -> bool:
    code = str(getattr(info, "code", "") or "")
    market = str(getattr(info, "market", "") or "")
    type_ = str(getattr(info, "type", "") or "")
    name = str(getattr(info, "name", "") or "")

    if not code.isdigit():
        return False

    if len(code) != 4:
        return False

    if market not in {"上市", "上櫃"}:
        return False

    bad_keywords = ["ETF", "ETN", "特別股", "受益證券", "認購", "認售", "牛證", "熊證"]
    if any(k in name for k in bad_keywords):
        return False

    # 常見股票類型過濾
    if type_ and ("股票" not in type_) and ("ETF" in type_ or "ETN" in type_):
        return False

    return True


def get_all_taiwan_stocks() -> list[Any]:
    codes = []
    for _, info in twstock.codes.items():
        if is_common_stock(info):
            codes.append(info)
    codes.sort(key=lambda x: x.code)
    return codes


def _months_ago(months: int) -> tuple[int, int]:
    now = datetime.now()
    year = now.year
    month = now.month - months
    while month <= 0:
        month += 12
        year -= 1
    return year, month


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _calc_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    direction = direction.replace(0, np.nan).ffill().fillna(0)
    obv = (direction * volume).cumsum()
    return obv.fillna(0)


def _build_hist_df(stock: twstock.Stock, info: Any) -> pd.DataFrame:
    start_year, start_month = _months_ago(LOOKBACK_MONTHS)
    raw = stock.fetch_from(start_year, start_month)

    if not raw:
        return pd.DataFrame()

    rows = []
    for r in raw:
        close = _safe_float(getattr(r, "close", 0))
        volume = _safe_float(getattr(r, "capacity", 0))  # 股數
        high = _safe_float(getattr(r, "high", close))
        low = _safe_float(getattr(r, "low", close))
        open_ = _safe_float(getattr(r, "open", close))
        date_ = getattr(r, "date", None)

        if close <= 0 or volume <= 0:
            continue

        rows.append(
            {
                "date": date_,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True)

    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=1).mean()

    df["volume_avg20_shares"] = df["volume"].rolling(20, min_periods=1).mean()
    df["volume_avg20_lot"] = df["volume_avg20_shares"] / 1000

    df["rsi"] = _calc_rsi(df["close"], 14)

    macd, macd_signal, macd_hist = _calc_macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    df["boll_mid"] = df["close"].rolling(20, min_periods=1).mean()
    boll_std = df["close"].rolling(20, min_periods=1).std().fillna(0)
    df["boll_upper"] = df["boll_mid"] + boll_std * 2
    df["boll_lower"] = df["boll_mid"] - boll_std * 2

    df["obv"] = _calc_obv(df["close"], df["volume"])
    df["obv_ma5"] = df["obv"].rolling(5, min_periods=1).mean()
    df["obv_ma20"] = df["obv"].rolling(20, min_periods=1).mean()

    df["platform_high_20d"] = df["high"].shift(1).rolling(20, min_periods=1).max()
    df["platform_high_60d"] = df["high"].shift(1).rolling(60, min_periods=1).max()

    return df


def fetch_one_stock(info: Any) -> dict[str, Any] | None:
    code = str(getattr(info, "code", "") or "")
    name = str(getattr(info, "name", "") or "")
    group = str(getattr(info, "group", "") or getattr(info, "market", "") or "")

    try:
        stock = twstock.Stock(code)
        df = _build_hist_df(stock, info)

        if df.empty or len(df) < 20:
            return None

        last = df.iloc[-1]

        close = _safe_float(last["close"])
        volume_shares = _safe_float(last["volume"])
        volume_lot = volume_shares / 1000
        volume_avg20_lot = _safe_float(last["volume_avg20_lot"], 0)

        if close <= 0 or volume_lot <= 0:
            return None

        volume_ratio = volume_lot / volume_avg20_lot if volume_avg20_lot > 0 else 0
        turnover_100m = close * volume_shares / 100000000

        ma20 = _safe_float(last["ma20"], close)
        ma60 = _safe_float(last["ma60"], close)

        pct_from_ma20 = ((close - ma20) / ma20 * 100) if ma20 > 0 else 0

        obv_trend = 1 if _safe_float(last["obv_ma5"]) >= _safe_float(last["obv_ma20"]) else -1

        return {
            "stock_id": code,
            "name": name,
            "group": group,
            "close": round(close, 4),
            "volume": round(volume_lot, 2),  # 張
            "volume_ratio": round(volume_ratio, 4),
            "volume_avg20": round(volume_avg20_lot, 2),
            "turnover_100m": round(turnover_100m, 4),
            "ma20": round(ma20, 4),
            "ma60": round(ma60, 4),
            "pct_from_ma20": round(pct_from_ma20, 4),
            "rsi": round(_safe_float(last["rsi"], 50), 4),
            "macd": round(_safe_float(last["macd"], 0), 6),
            "macd_signal": round(_safe_float(last["macd_signal"], 0), 6),
            "macd_hist": round(_safe_float(last["macd_hist"], 0), 6),
            "boll_mid": round(_safe_float(last["boll_mid"], close), 4),
            "boll_upper": round(_safe_float(last["boll_upper"], close), 4),
            "boll_lower": round(_safe_float(last["boll_lower"], close), 4),
            "obv_trend": int(obv_trend),
            "platform_high_20d": round(_safe_float(last["platform_high_20d"], 0), 4),
            "platform_high_60d": round(_safe_float(last["platform_high_60d"], 0), 4),
        }

    except Exception as e:
        print(f"[FETCH][ERROR] {code} {name}: {e}")
        return None


def fetch_market_snapshot_parallel(progress_every: int = 50) -> pd.DataFrame:
    print("[FETCH] Loading Taiwan stock list...")
    codes = get_all_taiwan_stocks()
    total = len(codes)

    print(f"[FETCH] Total stock universe: {total}")
    print(f"[FETCH] Using max_workers={MAX_WORKERS}")

    rows: list[dict[str, Any]] = []
    success = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_one_stock, info): info
            for info in codes
        }

        for idx, future in enumerate(as_completed(future_map), start=1):
            info = future_map[future]

            try:
                result = future.result()
                if result:
                    rows.append(result)
                    success += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                print(f"[FETCH][ERROR] {info.code} {info.name}: {e}")

            if idx % progress_every == 0 or idx == total:
                elapsed = round(time.time() - start_time, 1)
                print(
                    f"[FETCH] Progress {idx}/{total} | "
                    f"success={success} skipped={skipped} failed={failed} | "
                    f"elapsed={elapsed}s"
                )

    total_elapsed = round(time.time() - start_time, 1)

    print("[FETCH] Completed market snapshot")
    print(f"[FETCH] Success rows: {success}")
    print(f"[FETCH] Skipped rows: {skipped}")
    print(f"[FETCH] Failed rows: {failed}")
    print(f"[FETCH] Total elapsed: {total_elapsed}s")

    if not rows:
        print("[FETCH] No rows returned")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"[FETCH] Final dataframe rows: {len(df)}")

    return df
