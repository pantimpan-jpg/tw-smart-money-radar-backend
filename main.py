from __future__ import annotations

import threading
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import APP_NAME, LATEST_MARKET_CSV
from finmind_client import finmind_get, get_broker_data
from scanner import run_scan
from storage import load_snapshot

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = BackgroundScheduler(timezone=ZoneInfo("Asia/Taipei"))

scan_running = False
last_scan_error: str | None = None
last_finished_at: str | None = None
scan_lock = threading.Lock()
progress_lock = threading.Lock()

scan_progress: dict[str, Any] = {
    "scan_running": False,
    "percent": 0,
    "stage": "idle",
    "message": "尚未開始掃描",
    "processed": 0,
    "total": 0,
    "success": 0,
    "failed": 0,
    "skipped": 0,
    "last_scan_error": None,
    "last_updated": None,
}


def update_scan_progress(**kwargs) -> None:
    with progress_lock:
        scan_progress.update(kwargs)


def get_scan_progress_snapshot() -> dict[str, Any]:
    with progress_lock:
        return dict(scan_progress)


def progress_callback(payload: dict[str, Any]) -> None:
    update_scan_progress(**payload)


def _run_scan_job() -> None:
    global scan_running, last_scan_error, last_finished_at

    with scan_lock:
        if scan_running:
            print("[SCAN] Skip: already running")
            return
        scan_running = True

    last_scan_error = None
    update_scan_progress(
        scan_running=True,
        percent=0,
        stage="prepare",
        message="準備開始掃描",
        processed=0,
        total=0,
        success=0,
        failed=0,
        skipped=0,
        last_scan_error=None,
    )

    try:
        print("[SCAN] === Starting scan job ===")
        result = run_scan(save=True, progress_callback=progress_callback)
        last_finished_at = datetime.now(timezone.utc).isoformat()

        update_scan_progress(
            scan_running=False,
            percent=100,
            stage="completed",
            message="掃描完成",
            last_updated=last_finished_at,
            last_scan_error=None,
        )

        print("[SCAN] === Scan completed successfully ===")
        if isinstance(result, dict):
            print(f"[SCAN] Result keys: {list(result.keys())}")

    except Exception as e:
        last_scan_error = str(e)
        update_scan_progress(
            scan_running=False,
            stage="error",
            message=f"掃描失敗：{e}",
            last_scan_error=str(e),
        )
        print("[SCAN] === Scan failed ===")
        print(f"[SCAN] Error: {e}")
        print(traceback.format_exc())

    finally:
        scan_running = False
        print("[SCAN] === Scan thread finished ===")


def get_snapshot_or_404() -> dict[str, Any]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果，請先執行掃描")
    return snapshot


def load_market_snapshot_df() -> pd.DataFrame:
    path = Path(LATEST_MARKET_CSV)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)

    return df


def get_all_selected_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return snapshot.get("data", {}).get("all_selected", [])


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        num = float(value)
        if pd.isna(num):
            return None
        return num
    except Exception:
        return None


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def to_lot_from_shares(value: Any) -> float | None:
    num = safe_float(value)
    if num is None:
        return None
    return num / 1000.0


def latest_date_text(df: pd.DataFrame) -> str | None:
    if df.empty or "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    if d.empty:
        return None
    return d.max().date().isoformat()


def find_stock_context(snapshot: dict[str, Any], stock_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    stock_id = str(stock_id)

    selected_rows = get_all_selected_rows(snapshot)
    for row in selected_rows:
        if str(row.get("stock_id")) == stock_id:
            return row, {
                "in_selected": True,
                "source": "selected",
            }

    market_df = load_market_snapshot_df()
    if not market_df.empty and "stock_id" in market_df.columns:
        sub = market_df[market_df["stock_id"].astype(str) == stock_id]
        if not sub.empty:
            row = sub.iloc[0].to_dict()
            return row, {
                "in_selected": False,
                "source": "market_snapshot",
            }

    raise HTTPException(status_code=404, detail="找不到該股票")


def build_overview_from_row(row: dict[str, Any], *, in_selected: bool) -> dict[str, Any]:
    return {
        "stock_id": str(row.get("stock_id", "")),
        "stock_name": row.get("name") or row.get("stock_name") or str(row.get("stock_id", "")),
        "industry": row.get("group"),
        "theme": row.get("theme"),
        "close": safe_float(row.get("close")),
        "change": None,
        "change_pct": safe_float(row.get("change_pct")),
        "volume": safe_float(row.get("volume")),
        "turnover_100m": safe_float(row.get("turnover_100m")),
        "market_cap_100m": None,
        "pe_ratio": None,
        "pb_ratio": None,
        "dividend_yield": None,
        "foreign_buy_sell": None,
        "investment_trust_buy_sell": None,
        "dealer_buy_sell": None,
        "margin_balance": None,
        "short_balance": None,
        "margin_change": None,
        "short_change": None,
        "institution_score": safe_float(row.get("institution_score")) if in_selected else None,
        "broker_score": safe_float(row.get("broker_score")) if in_selected else None,
        "main_force_score": safe_float(row.get("main_force_score")) if in_selected else None,
        "final_score": safe_float(row.get("score_total") or row.get("score")) if in_selected else None,
        "technical_tag": row.get("tag") if in_selected else None,
        "radar_tag": row.get("radar_tag") if in_selected else None,
        "support_price": None,
        "pressure_price": None,
    }


def get_recent_price_df(stock_id: str) -> pd.DataFrame:
    start_date = (date.today() - timedelta(days=120)).isoformat()
    return finmind_get("TaiwanStockPrice", stock_id, start_date)


def enrich_overview_with_price(overview: dict[str, Any], stock_id: str) -> None:
    df = get_recent_price_df(stock_id)
    if df.empty:
        return

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    latest = df.iloc[-1]

    close_col = "close" if "close" in df.columns else None
    open_col = "open" if "open" in df.columns else None
    volume_col = "Trading_Volume" if "Trading_Volume" in df.columns else None
    money_col = "Trading_money" if "Trading_money" in df.columns else None

    latest_close = safe_float(latest.get(close_col)) if close_col else None
    latest_open = safe_float(latest.get(open_col)) if open_col else None

    if latest_close is not None:
        overview["close"] = latest_close

    if latest_close is not None and latest_open is not None:
        overview["change"] = latest_close - latest_open
        if latest_open != 0:
            overview["change_pct"] = (latest_close - latest_open) / latest_open * 100

    if volume_col:
        latest_volume = safe_float(latest.get(volume_col))
        overview["volume"] = latest_volume

    if money_col:
        latest_money = safe_float(latest.get(money_col))
        if latest_money is not None:
            overview["turnover_100m"] = latest_money / 100000000

    if close_col and len(df) >= 20:
        close_series = pd.to_numeric(df[close_col], errors="coerce")
        overview["pressure_price"] = safe_float(close_series.rolling(20).max().iloc[-1])
        overview["support_price"] = safe_float(close_series.rolling(20).min().iloc[-1])


def enrich_overview_with_per(stock_id: str, overview: dict[str, Any]) -> None:
    start_date = (date.today() - timedelta(days=30)).isoformat()
    df = finmind_get("TaiwanStockPER", stock_id, start_date)
    if df.empty:
        return

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    latest = df.iloc[-1]
    overview["pe_ratio"] = safe_float(latest.get("PER"))
    overview["pb_ratio"] = safe_float(latest.get("PBR"))
    overview["dividend_yield"] = safe_float(latest.get("dividend_yield"))


def _normalize_institutional_name(name: str) -> str:
    text = str(name)
    if "外資" in text or "Foreign" in text:
        return "foreign"
    if "投信" in text or "Investment_Trust" in text:
        return "trust"
    if "自營商" in text or "Dealer" in text:
        return "dealer"
    return "other"


def get_institutional_summary(stock_id: str) -> dict[str, Any]:
    start_date = (date.today() - timedelta(days=40)).isoformat()
    df = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_date)

    empty_result = {
        "latest_date": None,
        "foreign": {"d1": None, "d5": None, "d10": None, "d20": None},
        "trust": {"d1": None, "d5": None, "d10": None, "d20": None},
        "dealer": {"d1": None, "d5": None, "d10": None, "d20": None},
    }

    if df.empty or "date" not in df.columns:
        return empty_result

    name_col = None
    for candidate in ["name", "investors", "institutional_investors"]:
        if candidate in df.columns:
            name_col = candidate
            break
    if not name_col or "buy" not in df.columns or "sell" not in df.columns:
        return empty_result

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return empty_result

    df["kind"] = df[name_col].astype(str).map(_normalize_institutional_name)
    df["buy_num"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0)
    df["sell_num"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0)
    df["net_lot"] = (df["buy_num"] - df["sell_num"]) / 1000.0

    grouped = (
        df[df["kind"].isin(["foreign", "trust", "dealer"])]
        .groupby(["date", "kind"], as_index=False)["net_lot"]
        .sum()
        .sort_values("date")
    )

    latest_date = grouped["date"].max()
    all_dates = sorted(grouped["date"].drop_duplicates().tolist())

    def calc_window(kind: str, window: int) -> int | None:
        sub = grouped[grouped["kind"] == kind].copy()
        if sub.empty:
            return None
        latest_sub = sub[sub["date"] <= latest_date].sort_values("date")
        if latest_sub.empty:
            return None
        if window == 1:
            return int(round(latest_sub.iloc[-1]["net_lot"]))
        recent_dates = all_dates[-window:]
        recent = latest_sub[latest_sub["date"].isin(recent_dates)]
        if recent.empty:
            return None
        return int(round(recent["net_lot"].sum()))

    return {
        "latest_date": latest_date.date().isoformat() if pd.notna(latest_date) else None,
        "foreign": {
            "d1": calc_window("foreign", 1),
            "d5": calc_window("foreign", 5),
            "d10": calc_window("foreign", 10),
            "d20": calc_window("foreign", 20),
        },
        "trust": {
            "d1": calc_window("trust", 1),
            "d5": calc_window("trust", 5),
            "d10": calc_window("trust", 10),
            "d20": calc_window("trust", 20),
        },
        "dealer": {
            "d1": calc_window("dealer", 1),
            "d5": calc_window("dealer", 5),
            "d10": calc_window("dealer", 10),
            "d20": calc_window("dealer", 20),
        },
    }


def enrich_overview_with_institutional(stock_id: str, overview: dict[str, Any]) -> dict[str, Any]:
    summary = get_institutional_summary(stock_id)
    overview["foreign_buy_sell"] = summary["foreign"]["d1"]
    overview["investment_trust_buy_sell"] = summary["trust"]["d1"]
    overview["dealer_buy_sell"] = summary["dealer"]["d1"]
    return summary


def get_margin_summary(stock_id: str) -> dict[str, Any]:
    start_date = (date.today() - timedelta(days=40)).isoformat()
    df = finmind_get("TaiwanStockMarginPurchaseShortSale", stock_id, start_date)

    empty_result = {
        "latest_date": None,
        "margin_balance": None,
        "short_balance": None,
        "margin": {"d1": None, "d5": None, "d10": None, "d20": None},
        "short": {"d1": None, "d5": None, "d10": None, "d20": None},
    }

    if df.empty or "date" not in df.columns:
        return empty_result

    if "MarginPurchaseTodayBalance" not in df.columns or "ShortSaleTodayBalance" not in df.columns:
        return empty_result

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return empty_result

    df["margin_balance"] = pd.to_numeric(df["MarginPurchaseTodayBalance"], errors="coerce")
    df["short_balance"] = pd.to_numeric(df["ShortSaleTodayBalance"], errors="coerce")
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    latest_date = df.iloc[-1]["date"]

    def calc_change(col: str, days: int) -> int | None:
        sub = df[["date", col]].dropna().copy().sort_values("date").reset_index(drop=True)
        if sub.empty:
            return None
        latest_value = sub.iloc[-1][col]

        if days == 1:
            if len(sub) < 2:
                return None
            prev_value = sub.iloc[-2][col]
            return int(round((latest_value - prev_value) / 1000.0))

        if len(sub) <= days:
            return None

        base_value = sub.iloc[-(days + 1)][col]
        return int(round((latest_value - base_value) / 1000.0))

    latest_margin_balance = df.iloc[-1]["margin_balance"]
    latest_short_balance = df.iloc[-1]["short_balance"]

    return {
        "latest_date": latest_date.date().isoformat() if pd.notna(latest_date) else None,
        "margin_balance": int(round(latest_margin_balance / 1000.0)) if pd.notna(latest_margin_balance) else None,
        "short_balance": int(round(latest_short_balance / 1000.0)) if pd.notna(latest_short_balance) else None,
        "margin": {
            "d1": calc_change("margin_balance", 1),
            "d5": calc_change("margin_balance", 5),
            "d10": calc_change("margin_balance", 10),
            "d20": calc_change("margin_balance", 20),
        },
        "short": {
            "d1": calc_change("short_balance", 1),
            "d5": calc_change("short_balance", 5),
            "d10": calc_change("short_balance", 10),
            "d20": calc_change("short_balance", 20),
        },
    }


def enrich_overview_with_margin(stock_id: str, overview: dict[str, Any]) -> dict[str, Any]:
    summary = get_margin_summary(stock_id)
    overview["margin_balance"] = summary["margin_balance"]
    overview["short_balance"] = summary["short_balance"]
    overview["margin_change"] = summary["margin"]["d1"]
    overview["short_change"] = summary["short"]["d1"]
    return summary


def get_revenues_list(stock_id: str) -> list[dict[str, Any]]:
    start_date = (date.today() - timedelta(days=500)).isoformat()
    df = finmind_get("TaiwanStockMonthRevenue", stock_id, start_date)
    if df.empty:
        return []

    if "date" not in df.columns:
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    revenue_col = "revenue" if "revenue" in df.columns else None
    if revenue_col is None:
        return []

    df["revenue"] = pd.to_numeric(df[revenue_col], errors="coerce")
    df["revenue_mom_calc"] = df["revenue"].pct_change() * 100
    df["revenue_yoy_calc"] = df["revenue"].pct_change(12) * 100

    out: list[dict[str, Any]] = []
    for _, row in df.tail(24).sort_values("date", ascending=False).iterrows():
        out.append(
            {
                "date": row["date"].strftime("%Y/%m") if pd.notna(row["date"]) else None,
                "revenue": safe_float(row.get("revenue")),
                "revenue_mom": safe_float(row.get("revenue_mom_calc")),
                "revenue_yoy": safe_float(row.get("revenue_yoy_calc")),
            }
        )
    return out


def get_eps_list(stock_id: str) -> list[dict[str, Any]]:
    start_date = (date.today() - timedelta(days=1200)).isoformat()
    df = finmind_get("TaiwanStockFinancialStatements", stock_id, start_date)
    if df.empty:
        return []

    if "type" not in df.columns or "value" not in df.columns or "date" not in df.columns:
        return []

    eps_df = df[df["type"].astype(str).str.contains("基本每股盈餘", na=False)].copy()
    if eps_df.empty:
        return []

    eps_df["date"] = pd.to_datetime(eps_df["date"], errors="coerce")
    eps_df["value_num"] = pd.to_numeric(eps_df["value"], errors="coerce")
    eps_df = eps_df.dropna(subset=["date"]).sort_values("date")

    eps_df["yoy"] = eps_df["value_num"].pct_change(4) * 100
    eps_df["qoq"] = eps_df["value_num"].pct_change(1) * 100

    out: list[dict[str, Any]] = []
    for _, row in eps_df.tail(12).sort_values("date", ascending=False).iterrows():
        quarter = f"{row['date'].year}Q{((row['date'].month - 1) // 3) + 1}"
        out.append(
            {
                "quarter": quarter,
                "eps": safe_float(row.get("value_num")),
                "yoy": safe_float(row.get("yoy")),
                "qoq": safe_float(row.get("qoq")),
            }
        )
    return out


def get_dividends_list(stock_id: str) -> list[dict[str, Any]]:
    start_date = (date.today() - timedelta(days=2500)).isoformat()
    df = finmind_get("TaiwanStockDividend", stock_id, start_date)
    if df.empty:
        return []

    if "AnnouncementDate" in df.columns:
        df["sort_date"] = pd.to_datetime(df["AnnouncementDate"], errors="coerce")
    elif "date" in df.columns:
        df["sort_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["sort_date"] = pd.NaT

    df = df.sort_values("sort_date")

    out: list[dict[str, Any]] = []
    for _, row in df.tail(12).sort_values("sort_date", ascending=False).iterrows():
        out.append(
            {
                "year": safe_str(row.get("year")) or safe_str(row.get("date")) or "待補",
                "ex_dividend_date": safe_str(row.get("CashExDividendTradingDate")),
                "payment_date": safe_str(row.get("CashDividendPaymentDate")),
                "cash_dividend": safe_float(row.get("CashEarningsDistribution")),
                "stock_dividend": safe_float(row.get("StockEarningsDistribution")),
                "dividend_yield": None,
            }
        )
    return out


def get_financials_list(stock_id: str) -> list[dict[str, Any]]:
    start_date = (date.today() - timedelta(days=1200)).isoformat()
    df = finmind_get("TaiwanStockFinancialStatements", stock_id, start_date)
    if df.empty:
        return []

    if "date" not in df.columns or "type" not in df.columns or "value" not in df.columns:
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"])

    wanted_map = {
        "營業收入": "revenue",
        "營業毛利": "gross_profit",
        "營業利益": "operating_income",
        "本期淨利": "net_income",
        "基本每股盈餘": "eps",
    }

    out: list[dict[str, Any]] = []
    for d in sorted(df["date"].dropna().unique(), reverse=True)[:8]:
        sub = df[df["date"] == d].copy()
        item: dict[str, Any] = {
            "period": f"{pd.Timestamp(d).year}Q{((pd.Timestamp(d).month - 1) // 3) + 1}",
            "revenue": None,
            "gross_profit": None,
            "operating_income": None,
            "net_income": None,
            "eps": None,
        }

        for keyword, field in wanted_map.items():
            match = sub[sub["type"].astype(str).str.contains(keyword, na=False)]
            if not match.empty:
                item[field] = safe_float(match.iloc[0]["value_num"])

        out.append(item)

    return out


def get_news_list(stock_id: str) -> list[dict[str, Any]]:
    start_date = (date.today() - timedelta(days=14)).isoformat()
    df = finmind_get("TaiwanStockNews", stock_id, start_date)
    if df.empty:
        return []

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False)

    out: list[dict[str, Any]] = []
    for _, row in df.head(20).iterrows():
        published_at = None
        row_date = row.get("date")
        if pd.notna(row_date):
            try:
                published_at = pd.Timestamp(row_date).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                published_at = None

        out.append(
            {
                "id": None,
                "title": safe_str(row.get("title")) or "待補",
                "summary": safe_str(row.get("description")),
                "source": safe_str(row.get("source")),
                "published_at": published_at,
                "url": safe_str(row.get("link")),
            }
        )
    return out


def get_broker_branches_list(stock_id: str) -> list[dict[str, Any]]:
    try:
        df = get_broker_data([stock_id])
    except Exception:
        return []

    if df.empty:
        return []

    out: list[dict[str, Any]] = []
    for _, row in df.head(20).iterrows():
        out.append(
            {
                "broker_name": safe_str(row.get("broker_name")) or safe_str(row.get("securities_trader")) or "待補",
                "branch_name": safe_str(row.get("branch_name")),
                "buy": safe_float(row.get("buy")),
                "sell": safe_float(row.get("sell")),
                "net": safe_float(row.get("net")),
            }
        )
    return out


def build_stock_meta(stock_id: str, *, in_selected: bool, source: str) -> dict[str, Any]:
    return {
        "stock_id": stock_id,
        "in_selected": in_selected,
        "source": source,
        "has_scoring": in_selected,
        "label": "今日榜單股票" if in_selected else "未進今日榜單",
    }


@app.on_event("startup")
def startup_event() -> None:
    if not scheduler.running:
        print("[APP] Starting scheduler")
        scheduler.add_job(
            _run_scan_job,
            CronTrigger(day_of_week="mon-fri", hour=17, minute=0),
            id="daily_scan_tw",
            replace_existing=True,
        )
        scheduler.start()
        print("[APP] Scheduler started: Mon-Fri 17:00 Asia/Taipei")


@app.on_event("shutdown")
def shutdown_event() -> None:
    if scheduler.running:
        print("[APP] Shutting down scheduler")
        scheduler.shutdown()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": APP_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scheduler_running": scheduler.running,
        "schedule": "Mon-Fri 17:00 Asia/Taipei",
        "scan_running": scan_running,
        "last_scan_error": last_scan_error,
        "last_finished_at": last_finished_at,
    }


@app.get("/api/scan/latest")
def get_latest_scan() -> dict[str, Any]:
    snapshot = get_snapshot_or_404()
    return snapshot


@app.post("/api/scan/run")
def trigger_scan() -> dict[str, Any]:
    if scan_running:
        current = get_scan_progress_snapshot()
        return {
            "ok": True,
            "message": "掃描已在執行中",
            "status": current,
        }

    thread = threading.Thread(target=_run_scan_job, daemon=True)
    thread.start()

    return {
        "ok": True,
        "message": "已啟動背景掃描，請稍後查看 /api/scan/status",
    }


@app.get("/api/scan/status")
def get_scan_status() -> dict[str, Any]:
    snapshot = load_snapshot()
    updated_at = snapshot.get("updated_at") if snapshot else None

    status = get_scan_progress_snapshot()
    status["scan_running"] = scan_running
    status["last_scan_error"] = last_scan_error
    status["last_updated"] = updated_at or last_finished_at or status.get("last_updated")

    return status


@app.get("/api/scan/top30")
def get_top30() -> list[dict[str, Any]]:
    snapshot = get_snapshot_or_404()
    return snapshot.get("data", {}).get("top30", [])


@app.get("/api/scan/watchlist")
def get_watchlist() -> list[dict[str, Any]]:
    snapshot = get_snapshot_or_404()
    return snapshot.get("data", {}).get("watchlist", [])


@app.get("/api/stocks/{stock_id}")
def get_stock_detail(stock_id: str) -> dict[str, Any]:
    snapshot = get_snapshot_or_404()
    row, ctx = find_stock_context(snapshot, stock_id)

    in_selected = bool(ctx["in_selected"])
    overview = build_overview_from_row(row, in_selected=in_selected)

    enrich_overview_with_price(overview, stock_id)
    enrich_overview_with_per(stock_id, overview)

    institutional_summary = enrich_overview_with_institutional(stock_id, overview)
    margin_summary = enrich_overview_with_margin(stock_id, overview)

    return {
        "meta": build_stock_meta(stock_id, in_selected=in_selected, source=ctx["source"]),
        "overview": overview,
        "revenues": get_revenues_list(stock_id),
        "eps_list": get_eps_list(stock_id),
        "dividends": get_dividends_list(stock_id),
        "financials": get_financials_list(stock_id),
        "news": get_news_list(stock_id),
        "broker_branches": get_broker_branches_list(stock_id),
        "institutional_summary": institutional_summary,
        "margin_summary": margin_summary,
    }


@app.get("/api/stocks/{stock_id}/revenues")
def get_stock_revenues(stock_id: str) -> list[dict[str, Any]]:
    return get_revenues_list(stock_id)


@app.get("/api/stocks/{stock_id}/eps")
def get_stock_eps(stock_id: str) -> list[dict[str, Any]]:
    return get_eps_list(stock_id)


@app.get("/api/stocks/{stock_id}/dividends")
def get_stock_dividends(stock_id: str) -> list[dict[str, Any]]:
    return get_dividends_list(stock_id)


@app.get("/api/stocks/{stock_id}/financials")
def get_stock_financials(stock_id: str) -> list[dict[str, Any]]:
    return get_financials_list(stock_id)


@app.get("/api/stocks/{stock_id}/news")
def get_stock_news(stock_id: str) -> list[dict[str, Any]]:
    return get_news_list(stock_id)


@app.get("/api/stocks/{stock_id}/broker-branches")
def get_stock_broker_branches(stock_id: str) -> list[dict[str, Any]]:
    return get_broker_branches_list(stock_id)


@app.get("/api/stocks")
def search_stocks(
    q: str = Query(default="", description="股號、名稱、題材關鍵字"),
    tag: str = Query(default=""),
    theme: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    snapshot = get_snapshot_or_404()
    selected_rows = get_all_selected_rows(snapshot)
    selected_ids = {str(r.get("stock_id")) for r in selected_rows}

    market_df = load_market_snapshot_df()
    if market_df.empty:
        base_rows = selected_rows
    else:
        base_rows = market_df.to_dict(orient="records")

    ql = q.lower().strip()
    filtered: list[dict[str, Any]] = []

    for row in base_rows:
        stock_id = str(row.get("stock_id", ""))
        stock_name = str(row.get("name", "") or row.get("stock_name", ""))
        group = str(row.get("group", ""))
        row_theme = str(row.get("theme", ""))

        hay = f"{stock_id} {stock_name} {group} {row_theme}".lower()

        if ql and ql not in hay:
            continue

        if tag:
            row_tag = str(row.get("radar_tag", "") or row.get("tag", ""))
            if row_tag != tag:
                continue

        if theme and row_theme != theme:
            continue

        filtered.append(
            {
                "stock_id": stock_id,
                "name": stock_name,
                "close": safe_float(row.get("close")),
                "group": group or None,
                "theme": row_theme or None,
                "turnover_100m": safe_float(row.get("turnover_100m")),
                "in_selected": stock_id in selected_ids,
                "radar_tag": row.get("radar_tag") if stock_id in selected_ids else None,
                "score_total": safe_float(row.get("score_total")) if stock_id in selected_ids else None,
            }
        )

    filtered = sorted(
        filtered,
        key=lambda x: (
            0 if x["in_selected"] else 1,
            -(x["score_total"] or 0),
            -(x["turnover_100m"] or 0),
        ),
    )

    return filtered[:limit]
