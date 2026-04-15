from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from broker_history import (
    append_daily_broker_data,
    normalize_broker_df,
)


SUPPORTED_ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "cp950",
    "big5",
]


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    """
    自動嘗試常見台股 CSV 編碼。
    """
    path = Path(path)

    last_error: Exception | None = None

    for encoding in SUPPORTED_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"無法讀取 CSV：{path}\n最後錯誤：{last_error}")


def detect_market_from_filename(path: str | Path) -> str:
    """
    根據檔名推測市場：
    - twse / tse / 上市 -> TWSE
    - tpex / otc / 上櫃 -> TPEX
    """
    name = Path(path).name.lower()

    if any(k in name for k in ["twse", "tse", "上市"]):
        return "TWSE"

    if any(k in name for k in ["tpex", "otc", "上櫃"]):
        return "TPEX"

    return "UNKNOWN"


def import_broker_csv(
    csv_path: str | Path,
    trade_date: str,
    market: str | None = None,
    source: str = "manual_csv",
) -> pd.DataFrame:
    """
    匯入單一分點 CSV，轉成標準格式並寫入 broker_history。
    """
    csv_path = Path(csv_path)

    raw_df = read_csv_flexible(csv_path)

    final_market = market or detect_market_from_filename(csv_path)

    normalized = normalize_broker_df(
        raw_df,
        trade_date=trade_date,
        market=final_market,
        source=source,
    )

    append_daily_broker_data(normalized, trade_date)

    return normalized


def import_multiple_broker_csvs(
    file_paths: list[str | Path],
    trade_date: str,
    source: str = "manual_csv",
) -> pd.DataFrame:
    """
    一次匯入多個 CSV，例如：
    - TWSE 上市
    - TPEX 上櫃
    """
    all_frames: list[pd.DataFrame] = []

    for path in file_paths:
        try:
            df = import_broker_csv(
                csv_path=path,
                trade_date=trade_date,
                source=source,
            )
            if not df.empty:
                all_frames.append(df)
                print(f"[BROKER_IMPORT] 成功匯入 {path}，共 {len(df)} 筆")
            else:
                print(f"[BROKER_IMPORT] {path} 無資料")
        except Exception as e:
            print(f"[BROKER_IMPORT] 匯入失敗 {path}: {e}")

    if not all_frames:
        return pd.DataFrame()

    merged = pd.concat(all_frames, ignore_index=True)
    return merged


if __name__ == "__main__":
    """
    範例：

    python broker_importer.py
    """

    sample_files = [
        "data/raw_broker/twse_2026-04-15.csv",
        "data/raw_broker/tpex_2026-04-15.csv",
    ]

    result = import_multiple_broker_csvs(
        file_paths=sample_files,
        trade_date="2026-04-15",
    )

    print(result.head())
    print(f"總共匯入 {len(result)} 筆")
