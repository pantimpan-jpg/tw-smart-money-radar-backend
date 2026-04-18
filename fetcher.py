from __future__ import annotations

import gc
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config import MAX_WORKERS
from finmind_client import finmind_get


PRICE_LOOKBACK_DAYS = 150
MIN_UNIVERSE_SIZE = 300
SAFE_MAX_WORKERS = 4
FETCH_BATCH_SIZE_MULTIPLIER = 25

EXCLUDED_NAME_KEYWORDS = [
    "ETF",
    "ETN",
    "槓桿",
    "反向",
    "債券",
    "期貨",
    "受益",
    "基金",
    "存託",
]

EXCLUDED_GROUP_KEYWORDS = [
    "金融保險",
    "金融",
    "銀行",
    "保險",
    "證券",
    "金控",
]

EXCLUDED_STOCK_ID_PREFIXES: list[str] = []

EXCLUDED_NAME_PATTERN = "|".join(re.escape(keyword.upper()) for keyword in EXCLUDED_NAME_KEYWORDS)
EXCLUDED_GROUP_PATTERN = "|".join(re.escape(keyword) for keyword in EXCLUDED_GROUP_KEYWORDS)

SNAPSHOT_NUMERIC_COLUMNS = [
    "close",
    "change_pct",
    "volume",
    "turnover_100m",
    "volume_ratio",
    "volume_avg20",
    "ma20",
    "ma60",
    "pct_from_ma20",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "boll_mid",
    "boll_upper",
    "obv_trend",
    "platform_high_20d",
    "platform_high_60d",
    "low_5d",
    "low_20d",
    "high_5d",
    "high_20d",
    "pct_5d",
    "pct_10d",
    "drawdown_5d",
    "vol5_over_vol20",
    "ma20_slope",
    "near_high20_ratio",
]

SNAPSHOT_COLUMNS = [
    "stock_id",
    "name",
    "group",
    *SNAPSHOT_NUMERIC_COLUMNS,
    "trade_warning",
    "is_restricted",
    "fetch_skipped_reason",
]


def _to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        num = float(value)
        if pd.isna(num):
            return 0.0
        return num
    except Exception:
        return 0.0


def _to_float_series(series: pd.Series | None, fill_value: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float32")
    out = pd.to_numeric(series, errors="coerce").fillna(fill_value)
    return pd.to_numeric(out, downcast="float")


def _safe_series(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(index=df.index, dtype="float32")


def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = pd.to_numeric(df[col], downcast="float")

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    for col in ("stock_id", "name", "group", "trade_warning", "fetch_skipped_reason"):
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + std * num_std
    lower = mid - std * num_std
    return mid, upper, lower


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume.fillna(0)).cumsum()


def _pct_change_from_n_days(close: pd.Series, n: int) -> float:
    if len(close) <= n:
        return 0.0
    base = _to_float(close.iloc[-(n + 1)])
    latest = _to_float(close.iloc[-1])
    if base == 0:
        return 0.0
    return (latest - base) / base * 100.0


def _build_empty_row(stock_id: str, name: str, group: str, skipped_reason: str | None = None) -> dict[str, Any]:
    return {
        "stock_id": stock_id,
        "name": name,
        "group": group,
        "close": 0.0,
        "change_pct": 0.0,
        "volume": 0.0,
        "turnover_100m": 0.0,
        "volume_ratio": 0.0,
        "volume_avg20": 0.0,
        "ma20": 0.0,
        "ma60": 0.0,
        "pct_from_ma20": 999.0,
        "rsi": 50.0,
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        "boll_mid": 0.0,
        "boll_upper": 0.0,
        "obv_trend": 0.0,
        "platform_high_20d": 0.0,
        "platform_high_60d": 0.0,
        "low_5d": 0.0,
        "low_20d": 0.0,
        "high_5d": 0.0,
        "high_20d": 0.0,
        "pct_5d": 0.0,
        "pct_10d": 0.0,
        "drawdown_5d": 0.0,
        "vol5_over_vol20": 0.0,
        "ma20_slope": 0.0,
        "near_high20_ratio": 0.0,
        "trade_warning": "",
        "is_restricted": False,
        "fetch_skipped_reason": skipped_reason,
    }


def _build_stock_snapshot(stock_id: str, name: str, group: str, price_df: pd.DataFrame) -> dict[str, Any]:
    if price_df.empty:
        return _build_empty_row(stock_id, name, group, skipped_reason="empty_price")

    if "date" not in price_df.columns:
        return _build_empty_row(stock_id, name, group, skipped_reason="missing_date")

    keep_cols = [
        col
        for col in [
            "date",
            "open",
            "close",
            "max",
            "high",
            "min",
            "low",
            "Trading_Volume",
            "trading_volume",
            "volume",
            "Trading_money",
            "trading_money",
        ]
        if col in price_df.columns
    ]
    df = price_df.loc[:, keep_cols].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return _build_empty_row(stock_id, name, group, skipped_reason="invalid_date")

    close = _safe_series(df, "close")
    close = _to_float_series(close)
    valid_close_mask = close.notna()

    if not valid_close_mask.any():
        return _build_empty_row(stock_id, name, group, skipped_reason="invalid_close")

    df = df.loc[valid_close_mask].reset_index(drop=True)
    close = close.loc[valid_close_mask].reset_index(drop=True)

    if len(close) < 70:
        return _build_empty_row(stock_id, name, group, skipped_reason="not_enough_bars")

    open_ = _to_float_series(_safe_series(df, "open"), fill_value=np.nan).fillna(close)
    high = _to_float_series(_safe_series(df, "max", "high"), fill_value=np.nan).fillna(close)
    low = _to_float_series(_safe_series(df, "min", "low"), fill_value=np.nan).fillna(close)
    volume = _to_float_series(_safe_series(df, "Trading_Volume", "trading_volume", "volume"))
    trading_money = _to_float_series(_safe_series(df, "Trading_money", "trading_money"), fill_value=np.nan)
    trading_money = trading_money.fillna(close * volume)

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    vol_avg20 = volume.rolling(20).mean()
    vol_avg5 = volume.rolling(5).mean()

    rsi14 = _rsi(close, 14)
    macd_line, macd_signal, macd_hist = _macd(close)
    boll_mid, boll_upper, _ = _bollinger(close, 20, 2)
    obv_line = _obv(close, volume)

    latest_close = _to_float(close.iloc[-1])
    latest_open = _to_float(open_.iloc[-1])
    latest_volume = _to_float(volume.iloc[-1])
    latest_money = _to_float(trading_money.iloc[-1])

    latest_ma20 = _to_float(ma20.iloc[-1])
    latest_ma60 = _to_float(ma60.iloc[-1])
    latest_vol_avg20 = _to_float(vol_avg20.iloc[-1])
    latest_vol_avg5 = _to_float(vol_avg5.iloc[-1])

    latest_rsi = _to_float(rsi14.iloc[-1])
    latest_macd = _to_float(macd_line.iloc[-1])
    latest_macd_signal = _to_float(macd_signal.iloc[-1])
    latest_macd_hist = _to_float(macd_hist.iloc[-1])

    latest_boll_mid = _to_float(boll_mid.iloc[-1])
    latest_boll_upper = _to_float(boll_upper.iloc[-1])

    latest_change_pct = ((latest_close - latest_open) / latest_open * 100) if latest_open else 0.0
    latest_turnover_100m = latest_money / 100000000.0
    latest_volume_ratio = (latest_volume / latest_vol_avg20) if latest_vol_avg20 else 0.0
    latest_pct_from_ma20 = ((latest_close - latest_ma20) / latest_ma20 * 100) if latest_ma20 else 999.0

    obv_recent = obv_line.tail(5)
    obv_trend = 1.0 if len(obv_recent) >= 2 and obv_recent.iloc[-1] > obv_recent.iloc[0] else 0.0

    platform_high_20d = _to_float(high.tail(20).max())
    platform_high_60d = _to_float(high.tail(60).max())
    low_5d = _to_float(low.tail(5).min())
    low_20d = _to_float(low.tail(20).min())
    high_5d = _to_float(high.tail(5).max())
    high_20d = _to_float(high.tail(20).max())

    pct_5d = _pct_change_from_n_days(close, 5)
    pct_10d = _pct_change_from_n_days(close, 10)
    drawdown_5d = ((high_5d - latest_close) / high_5d * 100) if high_5d else 0.0
    vol5_over_vol20 = (latest_vol_avg5 / latest_vol_avg20) if latest_vol_avg20 else 0.0

    ma20_slope = 0.0
    ma20_valid = ma20.dropna()
    if len(ma20_valid) >= 6:
        ma20_now = _to_float(ma20.iloc[-1])
        ma20_prev5 = _to_float(ma20.iloc[-6])
        if ma20_prev5:
            ma20_slope = (ma20_now - ma20_prev5) / ma20_prev5 * 100.0

    near_high20_ratio = (latest_close / high_20d) if high_20d else 0.0

    return {
        "stock_id": stock_id,
        "name": name,
        "group": group,
        "close": round(latest_close, 2),
        "change_pct": round(latest_change_pct, 2),
        "volume": latest_volume,
        "turnover_100m": round(latest_turnover_100m, 3),
        "volume_ratio": round(latest_volume_ratio, 3),
        "volume_avg20": round(latest_vol_avg20 / 1000.0, 3),
        "ma20": round(latest_ma20, 2),
        "ma60": round(latest_ma60, 2),
        "pct_from_ma20": round(latest_pct_from_ma20, 2),
        "rsi": round(latest_rsi, 2),
        "macd": round(latest_macd, 4),
        "macd_signal": round(latest_macd_signal, 4),
        "macd_hist": round(latest_macd_hist, 4),
        "boll_mid": round(latest_boll_mid, 2),
        "boll_upper": round(latest_boll_upper, 2),
        "obv_trend": obv_trend,
        "platform_high_20d": round(platform_high_20d, 2),
        "platform_high_60d": round(platform_high_60d, 2),
        "low_5d": round(low_5d, 2),
        "low_20d": round(low_20d, 2),
        "high_5d": round(high_5d, 2),
        "high_20d": round(high_20d, 2),
        "pct_5d": round(pct_5d, 2),
        "pct_10d": round(pct_10d, 2),
        "drawdown_5d": round(drawdown_5d, 2),
        "vol5_over_vol20": round(vol5_over_vol20, 3),
        "ma20_slope": round(ma20_slope, 2),
        "near_high20_ratio": round(near_high20_ratio, 4),
        "trade_warning": "",
        "is_restricted": False,
        "fetch_skipped_reason": None,
    }


def _fetch_one_stock(stock_id: str, name: str, group: str, start_date: str) -> dict[str, Any]:
    try:
        price_df = finmind_get("TaiwanStockPrice", stock_id, start_date)
        result = _build_stock_snapshot(stock_id, name, group, price_df)
        del price_df
        return result
    except Exception:
        return _build_empty_row(stock_id, name, group, skipped_reason="fetch_error")


def _normalize_stock_info_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map: dict[str, str] = {}

    if "stock_id" not in out.columns:
        for c in ["stock_id", "stockId", "data_id"]:
            if c in out.columns:
                rename_map[c] = "stock_id"
                break

    if "stock_name" in out.columns and "name" not in out.columns:
        rename_map["stock_name"] = "name"

    if "industry_category" in out.columns and "group" not in out.columns:
        rename_map["industry_category"] = "group"

    if "industry" in out.columns and "group" not in out.columns:
        rename_map["industry"] = "group"

    if rename_map:
        out = out.rename(columns=rename_map)

    if "name" not in out.columns:
        out["name"] = ""
    if "group" not in out.columns:
        out["group"] = ""
    if "stock_id" not in out.columns:
        raise ValueError("TaiwanStockInfo 缺少 stock_id 欄位")

    out = out[["stock_id", "name", "group"]].copy()
    out["stock_id"] = out["stock_id"].astype(str)
    out["name"] = out["name"].astype(str)
    out["group"] = out["group"].fillna("").astype(str)

    return out


def _try_fetch_stock_info(stock_id_arg: Any, start_date_arg: Any) -> pd.DataFrame:
    try:
        df = finmind_get("TaiwanStockInfo", stock_id_arg, start_date_arg)
        if isinstance(df, pd.DataFrame):
            return df.copy()
    except Exception as e:
        print(
            f"[FETCH] TaiwanStockInfo failed | stock_id={stock_id_arg!r} "
            f"start_date={start_date_arg!r} err={e}"
        )
    return pd.DataFrame()


def _fetch_stock_info_full() -> pd.DataFrame:
    candidates: list[tuple[str, pd.DataFrame]] = [
        ('("", "")', _try_fetch_stock_info("", "")),
        ('("", "2024-01-01")', _try_fetch_stock_info("", "2024-01-01")),
        ("(None, None)", _try_fetch_stock_info(None, None)),
        ('(None, "2024-01-01")', _try_fetch_stock_info(None, "2024-01-01")),
    ]

    for label, df in candidates:
        print(f"[FETCH] TaiwanStockInfo candidate {label}: {len(df)} rows")

    best_label, best_df = max(candidates, key=lambda item: len(item[1]))
    print(f"[FETCH] TaiwanStockInfo selected candidate {best_label}: {len(best_df)} rows")

    if best_df.empty:
        raise RuntimeError("抓不到 TaiwanStockInfo")

    return best_df


def fetch_stock_universe() -> pd.DataFrame:
    info_df = _fetch_stock_info_full()
    info_df = _normalize_stock_info_df(info_df)

    print(f"[FETCH] TaiwanStockInfo raw rows: {len(info_df)}")

    info_df = info_df[info_df["stock_id"].str.match(r"^\d{4}$", na=False)].copy()
    print(f"[FETCH] 4-digit stock rows: {len(info_df)}")

    info_df = info_df.drop_duplicates(subset=["stock_id"], keep="first").copy()
    print(f"[FETCH] deduped 4-digit stock rows: {len(info_df)}")

    if len(info_df) < MIN_UNIVERSE_SIZE:
        sample_ids = info_df["stock_id"].head(20).tolist()
        raise RuntimeError(
            f"TaiwanStockInfo 股票母體異常，只有 {len(info_df)} 檔。sample={sample_ids}"
        )

    if EXCLUDED_STOCK_ID_PREFIXES:
        prefix_mask = info_df["stock_id"].astype(str).str.startswith(tuple(EXCLUDED_STOCK_ID_PREFIXES), na=False)
    else:
        prefix_mask = pd.Series(False, index=info_df.index)

    name_mask = info_df["name"].fillna("").astype(str).str.upper().str.contains(EXCLUDED_NAME_PATTERN, regex=True, na=False)
    group_mask = info_df["group"].fillna("").astype(str).str.contains(EXCLUDED_GROUP_PATTERN, regex=True, na=False)

    info_df["is_excluded"] = prefix_mask | name_mask | group_mask

    excluded_count = int(info_df["is_excluded"].sum())
    print(f"[FETCH] excluded ETF/financial count: {excluded_count}")

    info_df = info_df[~info_df["is_excluded"]].copy()
    info_df = info_df.reset_index(drop=True)

    print(f"[FETCH] final universe rows: {len(info_df)}")

    if len(info_df) < MIN_UNIVERSE_SIZE:
        sample_ids = info_df["stock_id"].head(20).tolist()
        raise RuntimeError(
            f"排除 ETF/金融後股票母體異常，只有 {len(info_df)} 檔。sample={sample_ids}"
        )

    return info_df[["stock_id", "name", "group"]]


def _iter_universe_batches(universe_df: pd.DataFrame, batch_size: int):
    total = len(universe_df)
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total), universe_df.iloc[start:start + batch_size]


def fetch_market_snapshot_parallel(progress_callback=None) -> pd.DataFrame:
    universe_df = fetch_stock_universe()
    total = len(universe_df)

    if total == 0:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    start_date = (date.today() - timedelta(days=PRICE_LOOKBACK_DAYS)).isoformat()

    results: list[dict[str, Any]] = []
    processed = 0
    success = 0
    failed = 0
    skipped = 0

    worker_count = min(max(1, int(MAX_WORKERS)), SAFE_MAX_WORKERS)
    batch_size = max(worker_count * FETCH_BATCH_SIZE_MULTIPLIER, worker_count)

    print(
        f"[FETCH] start market snapshot | universe={total} "
        f"| lookback_days={PRICE_LOOKBACK_DAYS} | workers={worker_count} | batch_size={batch_size}"
    )

    if progress_callback:
        progress_callback(
            {
                "scan_running": True,
                "stage": "fetch",
                "percent": 0,
                "message": "開始抓全市場資料",
                "processed": 0,
                "total": total,
                "success": 0,
                "failed": 0,
                "skipped": 0,
            }
        )

    for batch_start, batch_end, batch_df in _iter_universe_batches(universe_df, batch_size):
        print(f"[FETCH] batch {batch_start + 1}-{batch_end}/{total}")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _fetch_one_stock,
                    str(row.stock_id),
                    str(row.name),
                    str(row.group),
                    start_date,
                ): str(row.stock_id)
                for row in batch_df.itertuples(index=False)
            }

            for future in as_completed(future_map):
                result = future.result()
                results.append(result)

                processed += 1

                reason = result.get("fetch_skipped_reason")
                if reason is None:
                    success += 1
                elif reason == "fetch_error":
                    failed += 1
                else:
                    skipped += 1

                if processed % 100 == 0 or processed == total:
                    print(
                        f"[FETCH] progress processed={processed}/{total} "
                        f"success={success} failed={failed} skipped={skipped}"
                    )

                if progress_callback:
                    percent = min(87, int(processed / total * 87))
                    progress_callback(
                        {
                            "scan_running": True,
                            "stage": "fetch",
                            "percent": percent,
                            "message": "抓取全市場資料中",
                            "processed": processed,
                            "total": total,
                            "success": success,
                            "failed": failed,
                            "skipped": skipped,
                        }
                    )

        gc.collect()

    if not results:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    df = pd.DataFrame.from_records(results, columns=SNAPSHOT_COLUMNS)
    print(f"[FETCH] raw snapshot results rows: {len(df)}")

    if "fetch_skipped_reason" in df.columns:
        reason_counts = df["fetch_skipped_reason"].fillna("ok").value_counts(dropna=False).to_dict()
        print(f"[FETCH] skipped reason counts: {reason_counts}")
        df = df[df["fetch_skipped_reason"].isna()].copy()

    print(f"[FETCH] usable snapshot rows: {len(df)}")

    if df.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    for col in SNAPSHOT_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = pd.to_numeric(df[col], downcast="float")

    if "is_restricted" in df.columns:
        df["is_restricted"] = df["is_restricted"].astype(bool)

    df = df.sort_values(
        ["near_high20_ratio", "turnover_100m", "volume_ratio"],
        ascending=False,
    ).reset_index(drop=True)

    return _downcast_dataframe(df)
