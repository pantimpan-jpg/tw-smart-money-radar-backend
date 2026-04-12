from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

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

scheduler = BackgroundScheduler(timezone=ZoneInfo("Asia/Taipei"))

scan_running = False
last_scan_error: str | None = None
last_finished_at: str | None = None
scan_lock = threading.Lock()


def _run_scan_job() -> None:
    global scan_running, last_scan_error, last_finished_at

    with scan_lock:
        if scan_running:
            print("[SCAN] Skip: already running")
            return
        scan_running = True

    last_scan_error = None

    try:
        print("[SCAN] === Starting scan job ===")
        result = run_scan(save=True)
        last_finished_at = datetime.now(timezone.utc).isoformat()
        print("[SCAN] === Scan completed successfully ===")
        if isinstance(result, dict):
            print(f"[SCAN] Result keys: {list(result.keys())}")
    except Exception as e:
        last_scan_error = str(e)
        print("[SCAN] === Scan failed ===")
        print(f"[SCAN] Error: {e}")
        print(traceback.format_exc())
    finally:
        scan_running = False
        print("[SCAN] === Scan thread finished ===")


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
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果，請先執行掃描")
    return snapshot


@app.post("/api/scan/run")
def trigger_scan() -> dict[str, Any]:
    if scan_running:
        return {
            "ok": True,
            "message": "掃描已在執行中",
        }

    print("[API] Manual trigger received")

    thread = threading.Thread(target=_run_scan_job, daemon=True)
    thread.start()

    return {
        "ok": True,
        "message": "已啟動背景掃描，請稍後查看 /api/scan/latest",
    }


@app.get("/api/scan/status")
def get_scan_status() -> dict[str, Any]:
    snapshot = load_snapshot()
    updated_at = snapshot.get("updated_at") if snapshot else None

    return {
        "scan_running": scan_running,
        "last_scan_error": last_scan_error,
        "last_updated": updated_at or last_finished_at,
    }


@app.get("/api/scan/top30")
def get_top30() -> list[dict[str, Any]]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")
    return snapshot.get("data", {}).get("top30", [])


@app.get("/api/scan/watchlist")
def get_watchlist() -> list[dict[str, Any]]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")
    return snapshot.get("data", {}).get("watchlist", [])


@app.get("/api/stocks/{stock_id}")
def get_stock(stock_id: str):
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")

    all_selected = snapshot.get("data", {}).get("all_selected", [])
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

    rows = snapshot.get("data", {}).get("all_selected", [])
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
