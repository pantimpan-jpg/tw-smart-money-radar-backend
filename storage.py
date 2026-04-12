from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import LATEST_JSON, LATEST_MARKET_CSV, LATEST_SELECTED_CSV


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    clean = df.copy().replace([float("inf"), float("-inf")], 0).fillna(0)
    return clean.to_dict(orient="records")


def save_snapshot(payload: dict[str, Any], raw_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    wrapped_payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }

    LATEST_JSON.write_text(
        json.dumps(wrapped_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not raw_df.empty:
        raw_df.to_csv(LATEST_MARKET_CSV, index=False, encoding="utf-8-sig")

    if not selected_df.empty:
        selected_df.to_csv(LATEST_SELECTED_CSV, index=False, encoding="utf-8-sig")


def load_snapshot() -> dict[str, Any] | None:
    path = Path(LATEST_JSON)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # 向下相容：如果舊格式沒有包 data / updated_at，就自動補成新格式
    if "data" not in data:
        return {
            "updated_at": data.get("generated_at") or datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

    if "updated_at" not in data:
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

    return data
