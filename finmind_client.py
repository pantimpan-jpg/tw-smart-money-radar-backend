from __future__ import annotations

import time
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import (
    BROKER_CSV,
    FINMIND_API_TOKEN,
    FINMIND_BASE_URL,
    INSTITUTIONAL_CSV,
    REQUEST_TIMEOUT,
    RETRY_TIMES,
    REVENUE_CSV,
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "tw-smart-money-radar/1.0"})


class FinMindRequestError(RuntimeError):
    pass


class FinMindBadRequest(FinMindRequestError):
    pass


def _auth_headers() -> dict[str, str]:
    if not FINMIND_API_TOKEN:
        return {}
    return {"Authorization": f"Bearer {FINMIND_API_TOKEN}"}


def _sleep_backoff(attempt: int) -> None:
    time.sleep(0.8 * (attempt + 1))


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
        except requests.RequestException as e:
            last_error = e
            if attempt < RETRY_TIMES:
                _sleep_backoff(attempt)
                continue
            raise FinMindRequestError(
                f"FinMind request failed: url={url}, params={params} | err={e}"
            ) from e

        # 400/401/403/404 這類通常不是 retry 能解的，直接停
        if response.status_code in {400, 401, 403, 404}:
            raise FinMindBadRequest(
                f"FinMind bad request: url={url}, params={params} | "
                f"status={response.status_code} | text={response.text[:200]}"
            )

        # 429 / 5xx 才做 retry
        if response.status_code == 429 or 500 <= response.status_code < 600:
            last_error = FinMindRequestError(
                f"FinMind retryable error: url={url}, params={params} | "
                f"status={response.status_code} | text={response.text[:200]}"
            )
            if attempt < RETRY_TIMES:
                _sleep_backoff(attempt)
                continue
            raise last_error

        try:
            payload = response.json()
        except Exception as e:
            last_error = e
            if attempt < RETRY_TIMES:
                _sleep_backoff(attempt)
                continue
            raise FinMindRequestError(
                f"FinMind json decode failed: url={url}, params={params} | err={e}"
            ) from e

        if isinstance(payload, dict):
            status = payload.get("status")
            if status not in (200, None):
                msg = payload.get("msg", "unknown error")
                err = FinMindRequestError(
                    f"FinMind API error: url={url}, params={params} | status={status}, msg={msg}"
                )
                last_error = err
                if attempt < RETRY_TIMES:
                    _sleep_backoff(attempt)
                    continue
                raise err

            data = payload.get("data", [])
            if not isinstance(data, list):
                return []
            return data

        if isinstance(payload, list):
            return payload

        return []

    raise FinMindRequestError(
        f"FinMind request failed: url={url}, params={params} | err={last_error}"
    )


def _request_finmind_data(params: dict[str, Any]) -> list[dict[str, Any]]:
    return _request_json(FINMIND_BASE_URL, params)


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


@lru_cache(maxsize=1)
def _recent_trading_dates_cache() -> tuple[str, ...]:
    df = finmind_get("TaiwanStockTradingDate")
    if df.empty or "date" not in df.columns:
        return tuple()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return tuple()

    today = pd.Timestamp(date.today())
    df = df[df["date"] <= today]

    return tuple(df["date"].dt.strftime("%Y-%m-%d").tolist())


def _recent_trading_dates(n: int) -> list[str]:
    dates = list(_recent_trading_dates_cache())
    if not dates:
        return []
    return dates[-n:]


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
                end_date=d,
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


def _fetch_broker_day_by_stock(stock_id: str, trade_date: str) -> pd.DataFrame:
    try:
        df = finmind_get(
            "TaiwanStockTradingDailyReport",
            data_id=stock_id,
            start_date=trade_date,
            end_date=trade_date,
        )
        if df.empty:
            return pd.DataFrame()
        return _coerce_trading_daily_report_df(df)
    except FinMindBadRequest as e:
        print(f"[FINMIND] broker daily stock bad request | stock_id={stock_id} | date={trade_date} | err={e}")
        return pd.DataFrame()
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


def _zero_broker_summary_rows(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"stock_id": stock_id, "main_force_10d": 0.0, "broker_buy_5d": 0.0} for stock_id in ids]
    )


def _build_broker_summary(stock_id: str, df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "stock_id": stock_id,
            "main_force_10d": 0.0,
            "broker_buy_5d": 0.0,
        }

    out = df.copy()

    if "buy" not in out.columns:
        out["buy"] = 0
    if "sell" not in out.columns:
        out["sell"] = 0

    out["buy"] = pd.to_numeric(out["buy"], errors="coerce").fillna(0)
    out["sell"] = pd.to_numeric(out["sell"], errors="coerce").fillna(0)
    out["net"] = out["buy"] - out["sell"]

    if "securities_trader" not in out.columns:
        if "securities_trader_id" in out.columns:
            out["securities_trader"] = out["securities_trader_id"].astype(str)
        else:
            out["securities_trader"] = "UNKNOWN"

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

    if "buy" not in out.columns:
        out["buy"] = 0
    if "sell" not in out.columns:
        out["sell"] = 0

    out["buy"] = pd.to_numeric(out["buy"], errors="coerce").fillna(0)
    out["sell"] = pd.to_numeric(out["sell"], errors="coerce").fillna(0)
    out["net"] = out["buy"] - out["sell"]

    if "securities_trader" not in out.columns:
        if "securities_trader_id" in out.columns:
            out["securities_trader"] = out["securities_trader_id"].astype(str)
        else:
            out["securities_trader"] = "UNKNOWN"

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
            return _zero_broker_summary_rows(ids)

        if "stock_id" in local.columns:
            local["stock_id"] = local["stock_id"].astype(str)
            local = local[local["stock_id"].isin(ids)].copy()
            if not local.empty:
                return local

        return _zero_broker_summary_rows(ids)

    # 單檔：給個股頁用，真的去抓 stock_id + 單日
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

    # 多檔批量掃描：
    # 官方文件的 TaiwanStockTradingDailyReport 是 query by 股票代碼 / 券商代碼，且單次一天。
    # 這裡不再走 date-only 全市場特別端點，也不做逐檔 * 多天暴力 fallback。
    # 有本地 broker cache 就用，沒有就回零，讓 bulk scan 穩定完成。
    local = _read_local_csv(BROKER_CSV)
    if not local.empty and "stock_id" in local.columns:
        local["stock_id"] = local["stock_id"].astype(str)
        local = local[local["stock_id"].isin(ids)].copy()
        if not local.empty:
            print(f"[FINMIND] broker bulk scan use local cache rows={len(local)}")
            return local

    print("[FINMIND] broker bulk scan disabled in multi-stock mode; fill zero summary")
    return _zero_broker_summary_rows(ids)
