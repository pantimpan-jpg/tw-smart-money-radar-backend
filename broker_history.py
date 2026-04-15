from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


BROKER_HISTORY_DIR = Path("data/broker_history")
BROKER_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BrokerDailyRecord:
    trade_date: str
    market: str          # TWSE / TPEX
    stock_id: str
    broker_name: str
    branch_name: str
    buy_lot: float
    sell_lot: float
    net_lot: float
    source: str          # twse / tpex / other

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _day_path(trade_date: str) -> Path:
    return BROKER_HISTORY_DIR / f"{trade_date}.csv"


def normalize_broker_df(
    df: pd.DataFrame,
    *,
    trade_date: str,
    market: str,
    source: str,
) -> pd.DataFrame:
    """
    將不同來源的分點資料統一成內部格式：
    trade_date, market, stock_id, broker_name, branch_name, buy_lot, sell_lot, net_lot, source
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "market",
                "stock_id",
                "broker_name",
                "branch_name",
                "buy_lot",
                "sell_lot",
                "net_lot",
                "source",
            ]
        )

    out = df.copy()

    rename_map = {}

    # 常見欄位映射
    for col in out.columns:
        c = str(col).strip()

        if c in {"stock_id", "code", "股票代號", "代號"}:
            rename_map[col] = "stock_id"
        elif c in {"broker_name", "券商", "證券商名稱", "證券商"}:
            rename_map[col] = "broker_name"
        elif c in {"branch_name", "分點", "分公司名稱", "營業據點"}:
            rename_map[col] = "branch_name"
        elif c in {"buy_lot", "buy", "買進股數", "買進張數", "買進"}:
            rename_map[col] = "buy_lot"
        elif c in {"sell_lot", "sell", "賣出股數", "賣出張數", "賣出"}:
            rename_map[col] = "sell_lot"
        elif c in {"net_lot", "net", "買賣超", "買賣超張數"}:
            rename_map[col] = "net_lot"

    out = out.rename(columns=rename_map)

    required = ["stock_id", "broker_name", "buy_lot", "sell_lot"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"normalize_broker_df 缺少必要欄位: {col}")

    if "branch_name" not in out.columns:
        out["branch_name"] = ""

    out["stock_id"] = out["stock_id"].map(_normalize_text)
    out["broker_name"] = out["broker_name"].map(_normalize_text)
    out["branch_name"] = out["branch_name"].map(_normalize_text)
    out["buy_lot"] = out["buy_lot"].map(_normalize_float)
    out["sell_lot"] = out["sell_lot"].map(_normalize_float)

    if "net_lot" not in out.columns:
        out["net_lot"] = out["buy_lot"] - out["sell_lot"]
    else:
        out["net_lot"] = out["net_lot"].map(_normalize_float)

    out["trade_date"] = trade_date
    out["market"] = market
    out["source"] = source

    out = out[
        [
            "trade_date",
            "market",
            "stock_id",
            "broker_name",
            "branch_name",
            "buy_lot",
            "sell_lot",
            "net_lot",
            "source",
        ]
    ].copy()

    out = out[(out["stock_id"] != "") & (out["broker_name"] != "")]
    out = out.reset_index(drop=True)
    return out


def append_daily_broker_data(df: pd.DataFrame, trade_date: str) -> Path:
    """
    將某一天的分點資料寫入 data/broker_history/YYYY-MM-DD.csv
    若同一日期已存在，會做覆蓋去重。
    """
    path = _day_path(trade_date)

    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        for col in ["buy_lot", "sell_lot", "net_lot"]:
            if col in existing.columns:
                existing[col] = pd.to_numeric(existing[col], errors="coerce").fillna(0.0)
        merged = pd.concat([existing, df], ignore_index=True)
    else:
        merged = df.copy()

    merged["trade_date"] = merged["trade_date"].astype(str)
    merged["market"] = merged["market"].astype(str)
    merged["stock_id"] = merged["stock_id"].astype(str)
    merged["broker_name"] = merged["broker_name"].astype(str)
    merged["branch_name"] = merged["branch_name"].astype(str)
    merged["source"] = merged["source"].astype(str)

    for col in ["buy_lot", "sell_lot", "net_lot"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged = merged.drop_duplicates(
        subset=["trade_date", "market", "stock_id", "broker_name", "branch_name"],
        keep="last",
    ).sort_values(["stock_id", "broker_name", "branch_name"])

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_broker_day(trade_date: str) -> pd.DataFrame:
    path = _day_path(trade_date)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str)
    for col in ["buy_lot", "sell_lot", "net_lot"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def list_available_dates() -> list[str]:
    files = sorted(BROKER_HISTORY_DIR.glob("*.csv"))
    return [f.stem for f in files]


def load_broker_history(
    stock_id: str,
    days: int = 20,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    讀某股票最近 N 個交易日已累積的分點資料
    """
    available = list_available_dates()
    if not available:
        return pd.DataFrame()

    available = sorted(available)
    if end_date:
        available = [d for d in available if d <= end_date]

    selected_dates = available[-days:]
    frames: list[pd.DataFrame] = []

    for d in selected_dates:
        df = load_broker_day(d)
        if df.empty:
            continue
        sub = df[df["stock_id"].astype(str) == str(stock_id)].copy()
        if not sub.empty:
            frames.append(sub)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.sort_values(["trade_date", "net_lot"], ascending=[False, False]).reset_index(drop=True)
    return out


def summarize_broker_windows(stock_id: str, end_date: str | None = None) -> dict[str, Any]:
    """
    將你自己累積的日分點資料，算成 D1 / D5 / D10 / D20。
    單位維持「張」。
    """
    history = load_broker_history(stock_id, days=20, end_date=end_date)
    if history.empty:
        return {
            "latest_date": None,
            "d1_top_buy": [],
            "d5_top_buy": [],
            "d10_top_buy": [],
            "d20_top_buy": [],
        }

    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce")
    dates = sorted(history["trade_date"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())

    def aggregate_for_last_n(n: int) -> list[dict[str, Any]]:
        use_dates = dates[-n:]
        sub = history[history["trade_date"].dt.strftime("%Y-%m-%d").isin(use_dates)].copy()
        if sub.empty:
            return []

        grouped = (
            sub.groupby(["broker_name", "branch_name"], as_index=False)[["buy_lot", "sell_lot", "net_lot"]]
            .sum()
            .sort_values(["net_lot", "buy_lot"], ascending=[False, False])
        )

        result = []
        for _, row in grouped.head(15).iterrows():
            result.append(
                {
                    "broker_name": _normalize_text(row.get("broker_name")),
                    "branch_name": _normalize_text(row.get("branch_name")),
                    "buy_lot": round(float(row.get("buy_lot", 0.0)), 2),
                    "sell_lot": round(float(row.get("sell_lot", 0.0)), 2),
                    "net_lot": round(float(row.get("net_lot", 0.0)), 2),
                }
            )
        return result

    return {
        "latest_date": dates[-1] if dates else None,
        "d1_top_buy": aggregate_for_last_n(1),
        "d5_top_buy": aggregate_for_last_n(5),
        "d10_top_buy": aggregate_for_last_n(10),
        "d20_top_buy": aggregate_for_last_n(20),
    }


def save_summary_json(stock_id: str, summary: dict[str, Any]) -> Path:
    out_dir = BROKER_HISTORY_DIR / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stock_id}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
