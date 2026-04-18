from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import (
    BROKER_CSV,
    FINMIND_API_TOKEN,
    FINMIND_BASE_URL,
    INSTITUTIONAL_CSV,
    MAX_WORKERS,
    REQUEST_TIMEOUT,
    RETRY_TIMES,
    REVENUE_CSV,
)

SESSION = requests.Session()

SPECIAL_BASE_URL = (
    FINMIND_BASE_URL[:-5] if FINMIND_BASE_URL.endswith("/data") else FINMIND_BASE_URL
)


def _auth_headers() -> dict[str, str]:
    if not FINMIND_API_TOKEN:
        return {}
    return {"Authorization": f"Bearer {FINMIND_API_TOKEN}"}


def _request_json(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    last_error: Exception | None = None

    for attempt in range(RETRY_TIMES + 1):
        try:
            response = SESSION.get(
                url,
                headers=_auth_headers(),
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            payload = response.json()
            status = payload.get("status")
            if status not in (200, None):
                msg = payload.get("msg", "unknown error")
                raise RuntimeError(f"FinMind API error: status={status}, msg={msg}")

            data = payload.get("data", [])
            if not isinstance(data, list):
                return []
            return data

        except Exception as e:
            last_error = e
            if attempt < RETRY_TIMES:
                time.sleep(0.8 * (attempt + 1))

    raise RuntimeError(f"FinMind request failed: url={url}, params={params} | err={last_error}")


def _request_finmind_data(params: dict[str, Any]) -> list[dict[str, Any]]:
    return _request_json(FINMIND_BASE_URL, params)


def _request_finmind_special(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    url = f"{SPECIAL_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    return _request_json(url, params)


def finmind_get(
    dataset: str,
    data_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {"dataset": dataset}

    if data_id not in (None, ""):
        params["data_id"] = data_id
    if start_date not in (None, ""):
        params["start_date"] = start_date
    if end_date not in (None, ""):
        params["end_date"] = end_date

    data = _request_finmind_data(params)
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def _read_local_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalize_stock_ids(stock_ids: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return sorted({str(x).strip() for x in stock_ids if str(x).strip()})


def _recent_trading_dates(n: int) -> list[str]:
    df = finmind_get("TaiwanStockTradingDate")
    if df.empty or "date" not in df.columns:
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return []

    today = pd.Timestamp(date.today())
    df = df[df["date"] <= today]

    return df.tail(n)["date"].dt.strftime("%Y-%m-%d").tolist()


def _recent_month_starts(n: int) -> list[str]:
    today = date.today()
    year = today.year
    month = today.month

    out: list[str] = []
    for _ in range(n):
        out.append(f"{year:04d}-{month:02d}-01")
        month -= 1
        if month == 0:
            month = 12
            year -= 1

    return list(reversed(out))


def _classify_institutional_name(name: str) -> str:
    text = str(name or "")
    if "外資" in text or "Foreign" in text:
        return "foreign"
    if "投信" in text or "Investment_Trust" in text:
        return "trust"
    if "自營商" in text or "Dealer" in text:
        return "dealer"
    return "other"


def get_institutional_data(stock_ids: list[str]) -> pd.DataFrame:
    ids = _normalize_stock_ids(stock_ids)
    if not ids:
        return pd.DataFrame(
            columns=[
                "stock_id",
                "foreign_buy_days",
                "investment_buy_days",
                "dealer_buy_days",
                "foreign_buy",
                "trust_buy",
                "dealer_buy",
                "trust_holding_pct",
                "estimated_inst_cost",
            ]
        )

    trading_dates = _recent_trading_dates(12)
    if not trading_dates:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for d in trading_dates:
        try:
            daily = finmind_get(
                "TaiwanStockInstitutionalInvestorsBuySell",
                start_date=d,
            )
            if daily.empty or "stock_id" not in daily.columns:
                continue

            daily["stock_id"] = daily["stock_id"].astype(str)
            daily = daily[daily["stock_id"].isin(ids)].copy()
            if not daily.empty:
                frames.append(daily)
        except Exception as e:
            print(f"[FINMIND] institutional daily fetch failed | date={d} | err={e}")

    if not frames:
        local = _read_local_csv(INSTITUTIONAL_CSV)
        if local.empty:
            return pd.DataFrame()

        if "stock_id" in local.columns:
            local["stock_id"] = local["stock_id"].astype(str)
            return local[local["stock_id"].isin(ids)].copy()

        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    name_col = None
    for candidate in ["name", "investors", "institutional_investors"]:
        if candidate in df.columns:
            name_col = candidate
            break

    if name_col is None or "buy" not in df.columns or "sell" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    df["stock_id"] = df["stock_id"].astype(str)
    df["kind"] = df[name_col].astype(str).map(_classify_institutional_name)
    df["buy_num"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0)
    df["sell_num"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0)
    df["net_lot"] = (df["buy_num"] - df["sell_num"]) / 1000.0

    grouped = (
        df[df["kind"].isin(["foreign", "trust", "dealer"])]
        .groupby(["stock_id", "date", "kind"], as_index=False)["net_lot"]
        .sum()
        .sort_values(["stock_id", "date", "kind"])
    )

    rows: list[dict[str, Any]] = []

    for stock_id, stock_df in grouped.groupby("stock_id"):
        row = {
            "stock_id": stock_id,
            "foreign_buy_days": 0.0,
            "investment_buy_days": 0.0,
            "dealer_buy_days": 0.0,
            "foreign_buy": 0.0,
            "trust_buy": 0.0,
            "dealer_buy": 0.0,
            "trust_holding_pct": 0.0,
            "estimated_inst_cost": 0.0,
        }

        for kind, prefix in [
            ("foreign", "foreign"),
            ("trust", "trust"),
            ("dealer", "dealer"),
        ]:
            sub = stock_df[stock_df["kind"] == kind].sort_values("date")
            if sub.empty:
                continue

            recent10 = sub.tail(10)
            row[f"{prefix}_buy_days"] = float((recent10["net_lot"] > 0).sum())
            row[f"{prefix}_buy"] = float(recent10["net_lot"].sum())

        row["investment_buy_days"] = row["trust_buy_days"]
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out[
        [
            "stock_id",
            "foreign_buy_days",
            "investment_buy_days",
            "dealer_buy_days",
            "foreign_buy",
            "trust_buy",
            "dealer_buy",
            "trust_holding_pct",
            "estimated_inst_cost",
        ]
    ]


def get_revenue_data(stock_ids: list[str]) -> pd.DataFrame:
    ids = _normalize_stock_ids(stock_ids)
    if not ids:
        return pd.DataFrame(columns=["stock_id", "revenue_yoy", "revenue_mom"])

    month_starts = _recent_month_starts(14)
    frames: list[pd.DataFrame] = []

    for start in month_starts:
        try:
            monthly = finmind_get("TaiwanStockMonthRevenue", start_date=start)
            if monthly.empty or "stock_id" not in monthly.columns:
                continue

            monthly["stock_id"] = monthly["stock_id"].astype(str)
            monthly = monthly[monthly["stock_id"].isin(ids)].copy()
            if not monthly.empty:
                frames.append(monthly)
        except Exception as e:
            print(f"[FINMIND] revenue monthly fetch failed | start={start} | err={e}")

    if not frames:
        local = _read_local_csv(REVENUE_CSV)
        if local.empty:
            return pd.DataFrame(columns=["stock_id", "revenue_yoy", "revenue_mom"])

        if "stock_id" in local.columns:
            local["stock_id"] = local["stock_id"].astype(str)
            return local[local["stock_id"].isin(ids)].copy()

        return pd.DataFrame(columns=["stock_id", "revenue_yoy", "revenue_mom"])

    df = pd.concat(frames, ignore_index=True)

    if "date" not in df.columns or "revenue" not in df.columns:
        return pd.DataFrame(columns=["stock_id", "revenue_yoy", "revenue_mom"])

    df["stock_id"] = df["stock_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["date", "revenue"]).copy()

    df = (
        df.sort_values(["stock_id", "date"])
        .drop_duplicates(subset=["stock_id", "date"], keep="last")
        .copy()
    )

    df["revenue_mom"] = df.groupby("stock_id")["revenue"].pct_change() * 100
    df["revenue_yoy"] = df.groupby("stock_id")["revenue"].pct_change(12) * 100

    latest = (
        df.sort_values(["stock_id", "date"])
        .groupby("stock_id", as_index=False)
        .tail(1)
        .copy()
    )

    latest["revenue_mom"] = pd.to_numeric(latest["revenue_mom"], errors="coerce").fillna(0)
    latest["revenue_yoy"] = pd.to_numeric(latest["revenue_yoy"], errors="coerce").fillna(0)

    return latest[["stock_id", "revenue_yoy", "revenue_mom"]].reset_index(drop=True)


def _coerce_trading_daily_report_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "stock_id" in out.columns:
        out["stock_id"] = out["stock_id"].astype(str)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if "buy" in out.columns:
        out["buy"] = pd.to_numeric(out["buy"], errors="coerce").fillna(0)
    if "sell" in out.columns:
        out["sell"] = pd.to_numeric(out["sell"], errors="coerce").fillna(0)

    out = out.dropna(subset=["date"]).copy()
    return out


def _finmind_trading_daily_report(
    stock_id: str | None = None,
    trade_date: str | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {}
    if stock_id not in (None, ""):
        params["data_id"] = stock_id
    if trade_date not in (None, ""):
        params["date"] = trade_date

    data = _request_finmind_special("taiwan_stock_trading_daily_report", params)
    if not data:
        return pd.DataFrame()

    return _coerce_trading_daily_report_df(pd.DataFrame(data))


def _fetch_broker_day_all(trade_date: str) -> pd.DataFrame:
    try:
        return _finmind_trading_daily_report(trade_date=trade_date)
    except Exception as e:
        print(f"[FINMIND] broker daily all fetch failed | date={trade_date} | err={e}")
        return pd.DataFrame()


def _fetch_broker_day_by_stock(stock_id: str, trade_date: str) -> pd.DataFrame:
    try:
        return _finmind_trading_daily_report(stock_id=stock_id, trade_date=trade_date)
    except Exception as e:
        print(f"[FINMIND] broker daily stock fetch failed | stock_id={stock_id} | date={trade_date} | err={e}")
        return pd.DataFrame()


def _fetch_broker_range_for_stock(stock_id: str, trading_dates: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in trading_dates:
        daily = _fetch_broker_day_by_stock(stock_id, d)
        if not daily.empty:
            frames.append(daily)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "stock_id" not in df.columns:
        df["stock_id"] = stock_id
    return df


def _build_broker_summary(stock_id: str, df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "stock_id": stock_id,
            "main_force_10d": 0.0,
            "broker_buy_5d": 0.0,
        }

    out = df.copy()
    out["net"] = pd.to_numeric(out["buy"], errors="coerce").fillna(0) - pd.to_numeric(out["sell"], errors="coerce").fillna(0)

    all_dates = sorted(out["date"].dropna().unique().tolist())
    recent10 = all_dates[-10:]
    recent5 = all_dates[-5:]

    by_broker_10 = (
        out[out["date"].isin(recent10)]
        .groupby("securities_trader", as_index=False)["net"]
        .sum()
        .sort_values("net", ascending=False)
    )
    by_broker_5 = (
        out[out["date"].isin(recent5)]
        .groupby("securities_trader", as_index=False)["net"]
        .sum()
        .sort_values("net", ascending=False)
    )

    main_force_10d = float(by_broker_10[by_broker_10["net"] > 0]["net"].head(5).sum())
    broker_buy_5d = float(by_broker_5[by_broker_5["net"] > 0]["net"].head(3).sum())

    return {
        "stock_id": stock_id,
        "main_force_10d": round(main_force_10d, 3),
        "broker_buy_5d": round(broker_buy_5d, 3),
    }


def _build_broker_detail_rows(stock_id: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "stock_id",
                "broker_name",
                "branch_name",
                "buy",
                "sell",
                "net",
                "buy_5d",
                "sell_5d",
                "net_5d",
                "buy_10d",
                "sell_10d",
                "net_10d",
            ]
        )

    out = df.copy()
    out["buy"] = pd.to_numeric(out["buy"], errors="coerce").fillna(0)
    out["sell"] = pd.to_numeric(out["sell"], errors="coerce").fillna(0)
    out["net"] = out["buy"] - out["sell"]

    all_dates = sorted(out["date"].dropna().unique().tolist())
    recent10 = all_dates[-10:]
    recent5 = all_dates[-5:]

    grouped10 = (
        out[out["date"].isin(recent10)]
        .groupby(["stock_id", "securities_trader"], as_index=False)[["buy", "sell", "net"]]
        .sum()
        .rename(columns={"buy": "buy_10d", "sell": "sell_10d", "net": "net_10d"})
    )

    grouped5 = (
        out[out["date"].isin(recent5)]
        .groupby(["stock_id", "securities_trader"], as_index=False)[["buy", "sell", "net"]]
        .sum()
        .rename(columns={"buy": "buy_5d", "sell": "sell_5d", "net": "net_5d"})
    )

    merged = grouped10.merge(
        grouped5,
        on=["stock_id", "securities_trader"],
        how="outer",
    ).fillna(0)

    merged["broker_name"] = merged["securities_trader"]
    merged["branch_name"] = None

    # 預設前端顯示 10 日聚合
    merged["buy"] = merged["buy_10d"]
    merged["sell"] = merged["sell_10d"]
    merged["net"] = merged["net_10d"]

    merged = merged.sort_values("net_10d", ascending=False).copy()

    return merged[
        [
            "stock_id",
            "broker_name",
            "branch_name",
            "buy",
            "sell",
            "net",
            "buy_5d",
            "sell_5d",
            "net_5d",
            "buy_10d",
            "sell_10d",
            "net_10d",
        ]
    ]


def get_broker_data(stock_ids: list[str]) -> pd.DataFrame:
    ids = _normalize_stock_ids(stock_ids)
    if not ids:
        return pd.DataFrame()

    trading_dates = _recent_trading_dates(10)
    if not trading_dates:
        local = _read_local_csv(BROKER_CSV)
        if local.empty:
            return pd.DataFrame()

        if "stock_id" in local.columns:
            local["stock_id"] = local["stock_id"].astype(str)
            return local[local["stock_id"].isin(ids)].copy()

        return pd.DataFrame()

    # 單檔：回傳券商明細，給個股頁用
    if len(ids) == 1:
        df = _fetch_broker_range_for_stock(ids[0], trading_dates)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "stock_id",
                    "broker_name",
                    "branch_name",
                    "buy",
                    "sell",
                    "net",
                    "buy_5d",
                    "sell_5d",
                    "net_5d",
                    "buy_10d",
                    "sell_10d",
                    "net_10d",
                ]
            )
        return _build_broker_detail_rows(ids[0], df)

    # 多檔：優先用「單日全市場分點」下載後過濾 stage1 股票，避免 320 檔 * 10 天逐檔打
    frames: list[pd.DataFrame] = []
    for d in trading_dates:
        daily = _fetch_broker_day_all(d)
        if daily.empty or "stock_id" not in daily.columns:
            continue
        daily["stock_id"] = daily["stock_id"].astype(str)
        daily = daily[daily["stock_id"].isin(ids)].copy()
        if not daily.empty:
            frames.append(daily)

    # fallback：若 date-only 全市場抓法失敗，再退回逐檔抓
    if not frames:
        print("[FINMIND] broker all-stock daily fetch unavailable, fallback to per-stock mode")
        worker_count = max(1, min(int(MAX_WORKERS), 8))
        rows: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_fetch_broker_range_for_stock, stock_id, trading_dates): stock_id
                for stock_id in ids
            }

            for future in as_completed(future_map):
                stock_id = future_map[future]
                try:
                    df = future.result()
                    rows.append(_build_broker_summary(stock_id, df))
                except Exception as e:
                    print(f"[FINMIND] broker summary failed | stock_id={stock_id} | err={e}")
                    rows.append(
                        {
                            "stock_id": stock_id,
                            "main_force_10d": 0.0,
                            "broker_buy_5d": 0.0,
                        }
                    )

        return pd.DataFrame(rows)

    merged = pd.concat(frames, ignore_index=True)

    rows: list[dict[str, Any]] = []
    for stock_id, sub in merged.groupby("stock_id"):
        rows.append(_build_broker_summary(str(stock_id), sub.copy()))

    return pd.DataFrame(rows)
