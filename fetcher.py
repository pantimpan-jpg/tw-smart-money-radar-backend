from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import twstock

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))

# 預設改成全市場掃描
# 若之後要快速測試，可把環境變數 USE_TEST_STOCKS=true
USE_TEST_STOCKS = os.getenv("USE_TEST_STOCKS", "false").lower() == "true"

# 測試用股票池（只有在 USE_TEST_STOCKS=true 時才會用到）
TEST_STOCKS = ["2330", "2317", "2454", "2382", "1301"]


def get_all_taiwan_stocks():
    return [
        info
        for _, info in twstock.codes.items()
        if getattr(info, "market", "") in ["上市", "上櫃"]
        and str(getattr(info, "code", "")).isdigit()
        and len(str(getattr(info, "code", ""))) == 4
    ]


def get_test_stocks():
    all_codes = get_all_taiwan_stocks()
    return [x for x in all_codes if str(x.code) in TEST_STOCKS]


def get_scan_stocks():
    if USE_TEST_STOCKS:
        print("[FETCH] TEST MODE ON: using test stocks only")
        return get_test_stocks()

    print("[FETCH] FULL MARKET MODE ON: using all Taiwan listed/OTC stocks")
    return get_all_taiwan_stocks()


def fetch_one_stock(info):
    code = getattr(info, "code", "unknown")
    name = getattr(info, "name", "")

    try:
        print(f"[FETCH][ONE] Start {code} {name}")

        stock = twstock.Stock(code)
        print(f"[FETCH][ONE] Created stock object {code}")

        hist = stock.fetch_from(2025, 1)
        print(f"[FETCH][ONE] Fetched history {code}, rows={0 if not hist else len(hist)}")

        if not hist or len(hist) < 60:
            print(f"[FETCH][ONE] Skip {code}: not enough history")
            return None

        df = pd.DataFrame(
            [
                {
                    "date": x.date,
                    "open": float(x.open or 0),
                    "high": float(x.high or 0),
                    "low": float(x.low or 0),
                    "close": float(x.close or 0),
                    "capacity": float(x.capacity or 0),
                }
                for x in hist
                if x.close is not None and x.capacity is not None
            ]
        )

        if df.empty or len(df) < 60:
            print(f"[FETCH][ONE] Skip {code}: dataframe too short")
            return None

        df = df.sort_values("date").reset_index(drop=True)

        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["volume_avg20"] = df["capacity"].rolling(20).mean() / 1000

        latest = df.iloc[-1]

        close = float(latest["close"])
        ma20 = float(latest["ma20"]) if pd.notna(latest["ma20"]) else 0
        ma60 = float(latest["ma60"]) if pd.notna(latest["ma60"]) else 0
        volume = float(latest["capacity"]) / 1000
        volume_avg20 = float(latest["volume_avg20"]) if pd.notna(latest["volume_avg20"]) else 0

        volume_ratio = volume / volume_avg20 if volume_avg20 > 0 else 0
        turnover_100m = close * volume * 1000 / 100000000
        pct_from_ma20 = ((close - ma20) / ma20 * 100) if ma20 > 0 else 0

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        boll_mid = df["close"].rolling(20).mean()
        boll_std = df["close"].rolling(20).std()
        boll_upper = boll_mid + 2 * boll_std
        boll_lower = boll_mid - 2 * boll_std

        obv = [0]
        for i in range(1, len(df)):
            if df.loc[i, "close"] > df.loc[i - 1, "close"]:
                obv.append(obv[-1] + df.loc[i, "capacity"])
            elif df.loc[i, "close"] < df.loc[i - 1, "close"]:
                obv.append(obv[-1] - df.loc[i, "capacity"])
            else:
                obv.append(obv[-1])
        df["obv"] = obv
        obv_trend = 1 if len(df) >= 5 and df["obv"].iloc[-1] > df["obv"].iloc[-5] else 0

        platform_high_20d = float(df["high"].rolling(20).max().iloc[-2]) if len(df) >= 21 else 0
        platform_high_60d = float(df["high"].rolling(60).max().iloc[-2]) if len(df) >= 61 else 0

        result = {
            "stock_id": str(code),
            "name": str(name),
            "group": str(getattr(info, "group", "")),
            "close": close,
            "volume": volume,
            "volume_ratio": volume_ratio,
            "volume_avg20": volume_avg20,
            "turnover_100m": turnover_100m,
            "ma20": ma20,
            "ma60": ma60,
            "pct_from_ma20": pct_from_ma20,
            "rsi": latest_rsi,
            "macd": float(macd.iloc[-1]) if pd.notna(macd.iloc[-1]) else 0,
            "macd_signal": float(macd_signal.iloc[-1]) if pd.notna(macd_signal.iloc[-1]) else 0,
            "macd_hist": float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0,
            "boll_mid": float(boll_mid.iloc[-1]) if pd.notna(boll_mid.iloc[-1]) else 0,
            "boll_upper": float(boll_upper.iloc[-1]) if pd.notna(boll_upper.iloc[-1]) else 0,
            "boll_lower": float(boll_lower.iloc[-1]) if pd.notna(boll_lower.iloc[-1]) else 0,
            "obv_trend": obv_trend,
            "platform_high_20d": platform_high_20d,
            "platform_high_60d": platform_high_60d,
        }

        print(f"[FETCH][ONE] Done {code} {name}")
        return result

    except Exception as e:
        print(f"[FETCH][ERROR] {code} {name}: {e}")
        return None


def fetch_market_snapshot_parallel(progress_every: int = 50, batch_size: int = 20) -> pd.DataFrame:
    print("[FETCH] Loading Taiwan stock list...")
    codes = get_scan_stocks()
    total = len(codes)

    print(f"[FETCH] Total stock universe: {total}")
    print(f"[FETCH] Using max_workers={MAX_WORKERS}")
    print(f"[FETCH] Using batch_size={batch_size}")

    rows: list[dict[str, Any]] = []
    success = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch = codes[batch_start: batch_start + batch_size]
        batch_end = min(batch_start + batch_size, total)

        print(f"[FETCH] Starting batch {batch_start + 1}-{batch_end}/{total}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(fetch_one_stock, info): info
                for info in batch
            }

            batch_timeout = 120
            batch_begin_time = time.time()

            while future_map and (time.time() - batch_begin_time) < batch_timeout:
                done_futures = [f for f in list(future_map.keys()) if f.done()]

                if not done_futures:
                    time.sleep(0.5)
                    continue

                for future in done_futures:
                    info = future_map.pop(future)

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

        if future_map:
            for _, info in future_map.items():
                failed += 1
                print(f"[FETCH][TIMEOUT] {info.code} {info.name}: batch timeout")
            future_map.clear()

        processed = success + skipped + failed
        elapsed = round(time.time() - start_time, 1)

        if processed % progress_every == 0 or processed == total:
            print(
                f"[FETCH] Progress {processed}/{total} | "
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
