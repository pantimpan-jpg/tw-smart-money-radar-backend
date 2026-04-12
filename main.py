from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import APP_NAME
from .scanner import run_scan
from .storage import load_snapshot

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": APP_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/scan/latest")
def get_latest_scan() -> dict[str, Any]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果，請先執行 /api/scan/run")
    return snapshot


@app.post("/api/scan/run")
def trigger_scan() -> dict[str, Any]:
    try:
        return run_scan(save=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/scan/top30")
def get_top30() -> list[dict[str, Any]]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")
    return snapshot.get("top30", [])


@app.get("/api/scan/watchlist")
def get_watchlist() -> list[dict[str, Any]]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")
    return snapshot.get("watchlist", [])


@app.get("/api/stocks/{stock_id}")
def get_stock(stock_id: str):
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果")
    all_selected = snapshot.get("all_selected", [])
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
    rows = snapshot.get("all_selected", [])
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
