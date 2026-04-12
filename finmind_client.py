from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import (
    BROKER_CSV,
    CACHE_DIR,
    FINMIND_API_TOKEN,
    FINMIND_BASE_URL,
    INSTITUTIONAL_CSV,
    REQUEST_TIMEOUT,
    REVENUE_CSV,
)


def load_optional_csv(file_name: str) -> pd.DataFrame:
    path = Path(file_name)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _cache_path(prefix: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    return CACHE_DIR / f"{prefix}_{today}.csv"


def finmind_get(dataset: str, data_id: Optional[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    if not FINMIND_API_TOKEN:
        return pd.DataFrame()

    headers = {"Authorization": f"Bearer {FINMIND_API_TOKEN}"}
    params = {"dataset": dataset, "start_date": start_date}
    if data_id:
        params["data_id"] = data_id
    if end_date:
        params["end_date"] = end_date

    try:
        resp = requests.get(
            FINMIND_BASE_URL,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        payload = resp.json()
        return pd.DataFrame(payload.get("data", []))
    except Exception:
        return pd.DataFrame()


def _normalize_institutional(df: pd.DataFrame, stock_id: str) -> Optional[dict]:
    if df.empty:
        return None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("investors") or cols.get("institutional_investors")
    buy_col = cols.get("buy") or cols.get("buy_volume") or cols.get("buy_shares")
    sell_col = cols.get("sell") or cols.get("sell_volume") or cols.get("sell_shares")
    if not name_col or not buy_col or not sell_col:
        return None

    sub = df[[name_col, buy_col, sell_col]].copy()
    sub.columns = ["investor", "buy", "sell"]
    sub["investor"] = sub["investor"].astype(str)
    sub["buy"] = pd.to_numeric(sub["buy"], errors="coerce").fillna(0)
    sub["sell"] = pd.to_numeric(sub["sell"], errors="coerce").fillna(0)
    sub["net"] = sub["buy"] - sub["sell"]

    foreign = sub[sub["investor"].str.contains("外資|Foreign", case=False, na=False)].copy()
    trust = sub[sub["investor"].str.contains("投信|Investment_Trust", case=False, na=False)].copy()
    dealer = sub[sub["investor"].str.contains("自營商|Dealer", case=False, na=False)].copy()

    return {
        "stock_id": stock_id,
        "foreign_buy_days": int((foreign["net"] > 0).sum()) if not foreign.empty else 0,
        "investment_buy_days": int((trust["net"] > 0).sum()) if not trust.empty else 0,
        "dealer_buy_days": int((dealer["net"] > 0).sum()) if not dealer.empty else 0,
        "foreign_buy": float(foreign["net"].tail(5).sum()) if not foreign.empty else 0.0,
        "trust_buy": float(trust["net"].tail(5).sum()) if not trust.empty else 0.0,
        "dealer_buy": float(dealer["net"].tail(5).sum()) if not dealer.empty else 0.0,
        "trust_holding_pct": 0.0,
        "estimated_inst_cost": 0.0,
    }


def get_institutional_data(stock_ids: list[str]) -> pd.DataFrame:
    cache = _cache_path("institutional")
    if cache.exists():
        return pd.read_csv(cache, encoding="utf-8-sig")

    start_date = (date.today() - timedelta(days=20)).isoformat()
    rows = []
    for stock_id in stock_ids:
        df = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_date)
        row = _normalize_institutional(df, stock_id)
        if row:
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        out = load_optional_csv(INSTITUTIONAL_CSV)
    if not out.empty:
        out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def get_revenue_data(stock_ids: list[str]) -> pd.DataFrame:
    cache = _cache_path("revenue")
    if cache.exists():
        return pd.read_csv(cache, encoding="utf-8-sig")

    start_date = (date.today() - timedelta(days=400)).isoformat()
    rows = []
    for stock_id in stock_ids:
        df = finmind_get("TaiwanStockMonthRevenue", stock_id, start_date)
        if df.empty:
            continue
        revenue_col = None
        for c in df.columns:
            if c.lower() in {"revenue", "month_revenue"}:
                revenue_col = c
                break
        if revenue_col is None:
            continue
        date_col = "date" if "date" in df.columns else df.columns[0]
        x = df[[date_col, revenue_col]].copy()
        x.columns = ["date", "revenue"]
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x["revenue"] = pd.to_numeric(x["revenue"], errors="coerce").fillna(0)
        x = x.dropna().sort_values("date")
        if len(x) < 13:
            continue
        cur = x.iloc[-1]["revenue"]
        prev = x.iloc[-2]["revenue"] if len(x) >= 2 else 0
        yoy_base = x.iloc[-13]["revenue"] if len(x) >= 13 else 0
        mom = ((cur - prev) / prev * 100) if prev else 0
        yoy = ((cur - yoy_base) / yoy_base * 100) if yoy_base else 0
        rows.append({"stock_id": stock_id, "revenue_yoy": yoy, "revenue_mom": mom})

    out = pd.DataFrame(rows)
    if out.empty:
        out = load_optional_csv(REVENUE_CSV)
    if not out.empty:
        out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def get_broker_data(stock_ids: list[str]) -> pd.DataFrame:
    broker = load_optional_csv(BROKER_CSV)
    if broker.empty:
        return pd.DataFrame()
    broker["stock_id"] = broker["stock_id"].astype(str)
    return broker[broker["stock_id"].isin(stock_ids)].copy()
