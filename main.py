from __future__ import annotations

import threading
import traceback
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import APP_NAME
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


def get_all_selected_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return snapshot.get("data", {}).get("all_selected", [])


def find_stock_row(snapshot: dict[str, Any], stock_id: str) -> dict[str, Any]:
    rows = get_all_selected_rows(snapshot)
    for row in rows:
        if str(row.get("stock_id")) == str(stock_id):
            return row
    raise HTTPException(status_code=404, detail="找不到該股票")


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


def latest_date_text(df: pd.DataFrame) -> str | None:
    if df.empty or "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    if d.empty:
        return None
    return d.max().date().isoformat()


def build_overview_from_snapshot(row: dict[str, Any]) -> dict[str, Any]:
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
        "institution_score": safe_float(row.get("institution_score")),
        "broker_score": safe_float(row.get("broker_score")),
        "main_force_score": safe_float(row.get("main_force_score")),
        "final_score": safe_float(row.get("score_total") or row.get("score")),
        "technical_tag": row.get("tag"),
        "radar_tag": row.get("radar_tag") or row.get("tag"),
        "support_price": None,
        "pressure_price": None,
    }


def get_recent_price_df(stock_id: str) -> pd.DataFrame:
    start_date = (date.today() - timedelta(days=45)).isoformat()
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
        rolling_high = pd.to_numeric(df[close_col], errors="coerce").rolling(20).max().iloc[-1]
        rolling_low = pd.to_numeric(df[close_col], errors="coerce").rolling(20).min().iloc[-1]
        overview["pressure_price"] = safe_float(rolling_high)
        overview["support_price"] = safe_float(rolling_low)


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


def enrich_overview_with_institutional(stock_id: str, overview: dict[str, Any]) -> None:
    start_date = (date.today() - timedelta(days=10)).isoformat()
    df = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_date)
    if df.empty:
        return

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    name_col = None
    for candidate in ["name", "investors", "institutional_investors"]:
        if candidate in df.columns:
            name_col = candidate
            break
    if not name_col:
        return

    buy_col = "buy" if "buy" in df.columns else None
    sell_col = "sell" if "sell" in df.columns else None
    if not buy_col or not sell_col:
        return

    latest_date = df["date"].dropna().max()
    if pd.isna(latest_date):
        return

    latest_df = df[df["date"] == latest_date].copy()
    latest_df["buy"] = pd.to_numeric(latest_df[buy_col], errors="coerce").fillna(0)
    latest_df["sell"] = pd.to_numeric(latest_df[sell_col], errors="coerce").fillna(0)
    latest_df["net"] = latest_df["buy"] - latest_df["sell"]
    latest_df["name_clean"] = latest_df[name_col].astype(str)

    foreign = latest_df[latest_df["name_clean"].str.contains("外資|Foreign", case=False, na=False)]
    trust = latest_df[latest_df["name_clean"].str.contains("投信|Investment_Trust", case=False, na=False)]
    dealer = latest_df[latest_df["name_clean"].str.contains("自營商|Dealer", case=False, na=False)]

    overview["foreign_buy_sell"] = float(foreign["net"].sum()) if not foreign.empty else 0.0
    overview["investment_trust_buy_sell"] = float(trust["net"].sum()) if not trust.empty else 0.0
    overview["dealer_buy_sell"] = float(dealer["net"].sum()) if not dealer.empty else 0.0


def enrich_overview_with_margin(stock_id: str, overview: dict[str, Any]) -> None:
    start_date = (date.today() - timedelta(days=10)).isoformat()
    df = finmind_get("TaiwanStockMarginPurchaseShortSale", stock_id, start_date)
    if df.empty:
        return

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    today_margin = safe_float(latest.get("MarginPurchaseTodayBalance"))
    today_short = safe_float(latest.get("ShortSaleTodayBalance"))

    prev_margin = safe_float(prev.get("MarginPurchaseTodayBalance")) if prev is not None else None
    prev_short = safe_float(prev.get("ShortSaleTodayBalance")) if prev is not None else None

    overview["margin_balance"] = today_margin
    overview["short_balance"] = today_short

    if today_margin is not None and prev_margin is not None:
        overview["margin_change"] = today_margin - prev_margin

    if today_short is not None and prev_short is not None:
        overview["short_change"] = today_short - prev_short


def get_revenues_list(stock_id: str) -> list[dict[str, Any]]:
    start_date = (date.today() - timedelta(days=500)).isoformat()
    df = finmind_get("TaiwanStockMonthRevenue", stock_id, start_date)
    if df.empty:
        return []

    if "date" in df.columns:
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

    type_col = "type" if "type" in df.columns else None
    value_col = "value" if "value" in df.columns else None
    date_col = "date" if "date" in df.columns else None
    if not type_col or not value_col or not date_col:
        return []

    eps_df = df[df[type_col].astype(str).str.contains("基本每股盈餘", na=False)].copy()
    if eps_df.empty:
        return []

    eps_df["date"] = pd.to_datetime(eps_df[date_col], errors="coerce")
    eps_df["value_num"] = pd.to_numeric(eps_df[value_col], errors="coerce")
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
        out.append(
            {
                "id": None,
                "title": safe_str(row.get("title")) or "待補",
                "summary": safe_str(row.get("description")),
                "source": safe_str(row.get("source")),
                "published_at": row["date"].strftime("%Y-%m-%d %H:%M:%S") if "date" in row and pd.notna(row["date"]) else None,
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


@app.on_event("startup")
def startup_event() -> None:
    if not scheduler.running:
        print("[APP] Starting scheduler")
        scheduler.add_job(
            _run_scan_job,
            CronTrigger(day_of_week="mon-fri", hour=21, minute=0),
            id="daily_scan_tw",
            replace_existing=True,
        )
        scheduler.start()
        print("[APP] Scheduler started: Mon-Fri 21:00 Asia/Taipei")


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
        "schedule": "Mon-Fri 21:00 Asia/Taipei",
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

    print("[API] Manual trigger received")

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
    row = find_stock_row(snapshot, stock_id)

    overview = build_overview_from_snapshot(row)
    enrich_overview_with_price(stock_id, overview)
    enrich_overview_with_per(stock_id, overview)
    enrich_overview_with_institutional(stock_id, overview)
    enrich_overview_with_margin(stock_id, overview)

    return {
        "overview": overview,
        "revenues": get_revenues_list(stock_id),
        "eps_list": get_eps_list(stock_id),
        "dividends": get_dividends_list(stock_id),
        "financials": get_financials_list(stock_id),
        "news": get_news_list(stock_id),
        "broker_branches": get_broker_branches_list(stock_id),
    }


@app.get("/api/stocks/{stock_id}/revenues")
def get_stock_revenues(stock_id: str) -> list[dict[str, Any]]:
    _ = get_snapshot_or_404()
    return get_revenues_list(stock_id)


@app.get("/api/stocks/{stock_id}/eps")
def get_stock_eps(stock_id: str) -> list[dict[str, Any]]:
    _ = get_snapshot_or_404()
    return get_eps_list(stock_id)


@app.get("/api/stocks/{stock_id}/dividends")
def get_stock_dividends(stock_id: str) -> list[dict[str, Any]]:
    _ = get_snapshot_or_404()
    return get_dividends_list(stock_id)


@app.get("/api/stocks/{stock_id}/financials")
def get_stock_financials(stock_id: str) -> list[dict[str, Any]]:
    _ = get_snapshot_or_404()
    return get_financials_list(stock_id)


@app.get("/api/stocks/{stock_id}/news")
def get_stock_news(stock_id: str) -> list[dict[str, Any]]:
    _ = get_snapshot_or_404()
    return get_news_list(stock_id)


@app.get("/api/stocks/{stock_id}/broker-branches")
def get_stock_broker_branches(stock_id: str) -> list[dict[str, Any]]:
    _ = get_snapshot_or_404()
    return get_broker_branches_list(stock_id)


@app.get("/api/stocks")
def search_stocks(
    q: str = Query(default="", description="股號、名稱、題材關鍵字"),
    tag: str = Query(default=""),
    theme: str = Query(default=""),
):
    snapshot = get_snapshot_or_404()

    rows = snapshot.get("data", {}).get("all_selected", [])
    ql = q.lower().strip()
    filtered = []

    for row in rows:
        hay = f"{row.get('stock_id','')} {row.get('name','')} {row.get('group','')} {row.get('theme','')}".lower()
        if ql and ql not in hay:
            continue
        if tag and row.get("radar_tag") != tag and row.get("tag") != tag:
            continue
        if theme and row.get("theme") != theme:
            continue
        filtered.append(row)

    return filtered
