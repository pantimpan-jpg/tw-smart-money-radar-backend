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
SECOND_SCAN_LIMIT = int(os.getenv("SECOND_SCAN_LIMIT", 320))
LOOKBACK_MONTHS = int(os.getenv("LOOKBACK_MONTHS", 4))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 12))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
RETRY_TIMES = int(os.getenv("RETRY_TIMES", 2))

# =========================
# Stage 1 scan thresholds
# 先保留最低流動性門檻，不再用太硬的成交值 / 均量去卡死波段股
# =========================
MIN_PRICE = float(os.getenv("MIN_PRICE", 10))
MAX_PRICE = float(os.getenv("MAX_PRICE", 1500))

# 改成低流動性底線，後面 scanner 會把成交值更多改成排序因子
MIN_TURNOVER_100M = float(os.getenv("MIN_TURNOVER_100M", 0.8))
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", 1.2))
MIN_AVG_VOLUME_LOT = float(os.getenv("MIN_AVG_VOLUME_LOT", 300))

# 剛啟動主模型先放寬到 12
MAX_DISTANCE_FROM_MA20 = float(os.getenv("MAX_DISTANCE_FROM_MA20", 12))

# =========================
# Starting / Breakout model
# 剛啟動｜爆量突破
# =========================
STARTING_BREAKOUT_MIN_VOLUME_RATIO = float(
    os.getenv("STARTING_BREAKOUT_MIN_VOLUME_RATIO", 1.2)
)
STARTING_BREAKOUT_MAX_DISTANCE_FROM_MA20 = float(
    os.getenv("STARTING_BREAKOUT_MAX_DISTANCE_FROM_MA20", 12)
)
STARTING_BREAKOUT_RSI_MIN = float(os.getenv("STARTING_BREAKOUT_RSI_MIN", 50))
STARTING_BREAKOUT_RSI_MAX = float(os.getenv("STARTING_BREAKOUT_RSI_MAX", 74))

# =========================
# Starting / Accumulation model
# 剛啟動｜收籌墊高
# =========================
STARTING_ACCUM_10D_PCT_MIN = float(os.getenv("STARTING_ACCUM_10D_PCT_MIN", 5))
STARTING_ACCUM_10D_PCT_MAX = float(os.getenv("STARTING_ACCUM_10D_PCT_MAX", 12))
STARTING_ACCUM_5D_PULLBACK_MAX = float(os.getenv("STARTING_ACCUM_5D_PULLBACK_MAX", 4))
STARTING_ACCUM_CLOSE_OVER_MA20 = float(os.getenv("STARTING_ACCUM_CLOSE_OVER_MA20", 1.01))
STARTING_ACCUM_VOL5_OVER_VOL20_MIN = float(
    os.getenv("STARTING_ACCUM_VOL5_OVER_VOL20_MIN", 1.05)
)
STARTING_ACCUM_NEAR_HIGH20_RATIO = float(
    os.getenv("STARTING_ACCUM_NEAR_HIGH20_RATIO", 0.96)
)

# =========================
# Second wave model
# 可能第二波
# =========================
SECOND_WAVE_MAX_DISTANCE_FROM_MA20 = float(
    os.getenv("SECOND_WAVE_MAX_DISTANCE_FROM_MA20", 20)
)
SECOND_WAVE_MIN_RSI = float(os.getenv("SECOND_WAVE_MIN_RSI", 58))
SECOND_WAVE_MAX_RSI = float(os.getenv("SECOND_WAVE_MAX_RSI", 82))
SECOND_WAVE_MIN_VOLUME_RATIO = float(os.getenv("SECOND_WAVE_MIN_VOLUME_RATIO", 1.0))

# =========================
# Strong trend model
# 強者恆強
# =========================
STRONG_TREND_MIN_RSI = float(os.getenv("STRONG_TREND_MIN_RSI", 68))
STRONG_TREND_MAX_RSI = float(os.getenv("STRONG_TREND_MAX_RSI", 88))
STRONG_TREND_MAX_DISTANCE_FROM_MA20 = float(
    os.getenv("STRONG_TREND_MAX_DISTANCE_FROM_MA20", 25)
)
STRONG_TREND_MIN_VOLUME_RATIO = float(os.getenv("STRONG_TREND_MIN_VOLUME_RATIO", 1.0))
STRONG_TREND_NEAR_HIGH20_RATIO = float(
    os.getenv("STRONG_TREND_NEAR_HIGH20_RATIO", 0.97)
)

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
FINMIND_BASE_URL = os.getenv(
    "FINMIND_BASE_URL",
    "https://api.finmindtrade.com/api/v4/data",
)

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
