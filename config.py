from __future__ import annotations

import os
from pathlib import Path

# =========================
# Basic settings
# =========================
APP_NAME = "TW Smart Money Radar API"
APP_ENV = os.getenv("APP_ENV", "development")

TOP30_COUNT = int(os.getenv("TOP30_COUNT", 30))
WATCHLIST_COUNT = int(os.getenv("WATCHLIST_COUNT", 20))
BROKER_TRACK_COUNT = int(os.getenv("BROKER_TRACK_COUNT", 20))
SECOND_SCAN_LIMIT = int(os.getenv("SECOND_SCAN_LIMIT", 260))
LOOKBACK_MONTHS = int(os.getenv("LOOKBACK_MONTHS", 4))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 12))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
RETRY_TIMES = int(os.getenv("RETRY_TIMES", 2))

# =========================
# Scan thresholds
# =========================
MIN_PRICE = float(os.getenv("MIN_PRICE", 10))
MAX_PRICE = float(os.getenv("MAX_PRICE", 500))
MIN_TURNOVER_100M = float(os.getenv("MIN_TURNOVER_100M", 1.5))
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", 1.2))
MIN_AVG_VOLUME_LOT = float(os.getenv("MIN_AVG_VOLUME_LOT", 800))
MAX_DISTANCE_FROM_MA20 = float(os.getenv("MAX_DISTANCE_FROM_MA20", 12))

# =========================
# Exclusions
# =========================
EXCLUDE_ETF_ETN = os.getenv("EXCLUDE_ETF_ETN", "true").lower() == "true"
EXCLUDE_WARRANT = os.getenv("EXCLUDE_WARRANT", "true").lower() == "true"
EXCLUDE_INDEX = os.getenv("EXCLUDE_INDEX", "true").lower() == "true"
EXCLUDE_FINANCIAL = os.getenv("EXCLUDE_FINANCIAL", "true").lower() == "true"

# =========================
# Optional local fallback files
# =========================
INSTITUTIONAL_CSV = os.getenv("INSTITUTIONAL_CSV", "institutional_chip.csv")
BROKER_CSV = os.getenv("BROKER_CSV", "broker_chip.csv")
REVENUE_CSV = os.getenv("REVENUE_CSV", "revenue_data.csv")

# =========================
# FinMind
# =========================
FINMIND_API_TOKEN = os.getenv("FINMIND_API_TOKEN", "")
FINMIND_BASE_URL = os.getenv("FINMIND_BASE_URL", "https://api.finmindtrade.com/api/v4/data")

# =========================
# Storage
# Render persistent disk mount path: /var/data
# Local dev fallback: project_root/data
# =========================
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = Path("/var/data")
if not DEFAULT_DATA_DIR.exists():
    DEFAULT_DATA_DIR = BASE_DIR / "data"

DATA_DIR = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(DATA_DIR / "cache")))
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", str(DATA_DIR / "snapshots")))

for p in [DATA_DIR, CACHE_DIR, SNAPSHOT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

LATEST_JSON = SNAPSHOT_DIR / "latest_scan.json"
LATEST_MARKET_CSV = SNAPSHOT_DIR / "latest_market_snapshot.csv"
LATEST_SELECTED_CSV = SNAPSHOT_DIR / "latest_selected_stocks.csv"

LATEST_JSON = SNAPSHOT_DIR / "latest_scan.json"
LATEST_MARKET_CSV = SNAPSHOT_DIR / "latest_market_snapshot.csv"
LATEST_SELECTED_CSV = SNAPSHOT_DIR / "latest_selected_stocks.csv"
