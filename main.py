from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import APP_NAME
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

scheduler = BackgroundScheduler(timezone="Asia/Taipei")


def format_snapshot_response(snapshot: dict[str, Any]) -> dict[str, Any]:
    data = snapshot.get("data", {})
    updated_at = snapshot.get("updated_at")

    return {
        "last_updated": updated_at,
        **data,
    }


def scheduled_scan_job() -> None:
    try:
        print("Starting scheduled scan...")
        run_scan(save=True)
        print("Scheduled scan finished.")
    except Exception as e:
        print(f"Scheduled scan failed: {e}")


@app.on_event("startup")
def startup_event() -> None:
    if not scheduler.running:
        scheduler.add_job(
            scheduled_scan_job,
            CronTrigger(day_of_week="mon-fri", hour=21, minute=0),
            id="weekday_night_scan",
            replace_existing=True,
        )
        scheduler.start()
        print("Scheduler started: Mon-Fri 21:00 Asia/Taipei")


@app.on_event("shutdown")
def shutdown_event() -> None:
    if scheduler.running:
        scheduler.shutdown()


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": APP_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scheduler_running": scheduler.running,
        "schedule": "Mon-Fri 21:00 Asia/Taipei",
    }


@app.get("/api/scan/latest")
def get_latest_scan() -> dict[str, Any]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果，請先執行 /api/scan/run")
    return format_snapshot_response(snapshot)


@app.post("/api/scan/run")
def trigger_scan() -> dict[str, Any]:
    try:
        result = run_scan(save=True)
        snapshot = load_snapshot()
        if snapshot:
            return format_snapshot_response(snapshot)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/scan/top30")
def get_top30() -> list[dict[str, Any]]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")
    data = snapshot.get("data", {})
    return data.get("top30", [])


@app.get("/api/scan/watchlist")
def get_watchlist() -> list[dict[str, Any]]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")
    data = snapshot.get("data", {})
    return data.get("watchlist", [])


@app.get("/api/stocks/{stock_id}")
def get_stock(stock_id: str):
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")

    data = snapshot.get("data", {})
    all_selected = data.get("all_selected", [])

    for row in all_selected:
        if str(row.get("stock_id")) == str(stock_id):
            return row

    raise HTTPException(status_code=404, detail="找不到該股票")


@app.get("/api/stocks")
def search_stocks(
    q: str = Query(default="", description="股號、名稱、題材關鍵字"),
    tag: str = Query(default=""),
    theme: str = Query(default=""),
):
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")

    data = snapshot.get("data", {})
    rows = data.get("all_selected", [])

    ql = q.lower().strip()
    filtered = []

    for row in rows:
        hay = f"{row.get('stock_id','')} {row.get('name','')} {row.get('group','')} {row.get('theme','')}".lower()

        if ql and ql not in hay:
            continue
        if tag and row.get("radar_tag") != tag:
            continue
        if theme and row.get("theme") != theme:
            continue

        filtered.append(row)

    return filtered
