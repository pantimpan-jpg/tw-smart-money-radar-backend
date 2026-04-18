from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import LATEST_JSON, LATEST_MARKET_CSV, LATEST_SELECTED_CSV


def _to_json_safe_value(value: Any) -> Any:
    if value is None:
        return 0

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, str):
        return value

    if pd.isna(value):
        return 0

    if isinstance(value, (np.floating, float)):
        if np.isinf(value) or np.isnan(value):
            return 0
        return float(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if hasattr(value, "item"):
        try:
            item = value.item()
            return _to_json_safe_value(item)
        except Exception:
            pass

    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    cols = list(df.columns)
    records: list[dict[str, Any]] = []

    for row in df.itertuples(index=False, name=None):
        record = {col: _to_json_safe_value(value) for col, value in zip(cols, row)}
        records.append(record)

    return records


def ensure_parent_dir(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def save_snapshot(
    payload: dict[str, Any],
    raw_df: pd.DataFrame | None = None,
    selected_df: pd.DataFrame | None = None,
) -> None:
    ensure_parent_dir(Path(LATEST_JSON))
    ensure_parent_dir(Path(LATEST_MARKET_CSV))
    ensure_parent_dir(Path(LATEST_SELECTED_CSV))

    wrapped_payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }

    Path(LATEST_JSON).write_text(
        json.dumps(wrapped_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        raw_df.to_csv(LATEST_MARKET_CSV, index=False, encoding="utf-8-sig")

    if isinstance(selected_df, pd.DataFrame) and not selected_df.empty:
        selected_df.to_csv(LATEST_SELECTED_CSV, index=False, encoding="utf-8-sig")


def load_snapshot() -> dict[str, Any] | None:
    path = Path(LATEST_JSON)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if "data" not in data:
        return {
            "updated_at": data.get("generated_at") or datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

    if "updated_at" not in data:
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

    return data
