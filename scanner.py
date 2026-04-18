from __future__ import annotations

import gc
import re

import numpy as np
import pandas as pd

from config import (
    BROKER_TRACK_COUNT,
    INSTITUTIONAL_CSV,
    MAX_PRICE,
    MIN_AVG_VOLUME_LOT,
    MIN_PRICE,
    MIN_TURNOVER_100M,
    MIN_VOLUME_RATIO,
    REVENUE_CSV,
    SECOND_SCAN_LIMIT,
    TOP30_COUNT,
    WATCHLIST_COUNT,
    SECOND_WAVE_MAX_DISTANCE_FROM_MA20,
    SECOND_WAVE_MAX_RSI,
    SECOND_WAVE_MIN_RSI,
    SECOND_WAVE_MIN_VOLUME_RATIO,
    STARTING_ACCUM_10D_PCT_MAX,
    STARTING_ACCUM_10D_PCT_MIN,
    STARTING_ACCUM_5D_PULLBACK_MAX,
    STARTING_ACCUM_CLOSE_OVER_MA20,
    STARTING_ACCUM_NEAR_HIGH20_RATIO,
    STARTING_ACCUM_VOL5_OVER_VOL20_MIN,
    STARTING_BREAKOUT_MAX_DISTANCE_FROM_MA20,
    STARTING_BREAKOUT_MIN_VOLUME_RATIO,
    STARTING_BREAKOUT_RSI_MAX,
    STARTING_BREAKOUT_RSI_MIN,
    STRONG_TREND_MAX_DISTANCE_FROM_MA20,
    STRONG_TREND_MAX_RSI,
    STRONG_TREND_MIN_RSI,
    STRONG_TREND_MIN_VOLUME_RATIO,
    STRONG_TREND_NEAR_HIGH20_RATIO,
)
from fetcher import fetch_market_snapshot_parallel
from finmind_client import get_broker_data, get_institutional_data, get_revenue_data
from storage import dataframe_to_records, save_snapshot

EXCLUDED_NAME_KEYWORDS = [
    "ETF",
    "ETN",
    "槓桿",
    "反向",
    "債券",
    "期貨",
    "受益",
    "基金",
    "存託",
]

EXCLUDED_GROUP_KEYWORDS = [
    "金融保險",
    "金融",
    "銀行",
    "保險",
    "證券",
    "金控",
]

EXCLUDED_NAME_PATTERN = "|".join(re.escape(keyword.upper()) for keyword in EXCLUDED_NAME_KEYWORDS)
EXCLUDED_GROUP_PATTERN = "|".join(re.escape(keyword) for keyword in EXCLUDED_GROUP_KEYWORDS)

# =========================
# 細題材 mapping
# 判斷順序：
# 1. 股號 override
# 2. 股名 / 關鍵字
# 3. 產業 group fallback
# =========================
STOCK_THEME_OVERRIDES: dict[str, str] = {
    # CPO / 矽光子 / 光通訊
    "4979": "CPO/矽光子",
    "4971": "CPO/矽光子",
    "4977": "CPO/矽光子",
    "3363": "CPO/矽光子",
    "3163": "CPO/矽光子",
    "3081": "CPO/矽光子",
    "3450": "CPO/矽光子",
    "6442": "CPO/矽光子",
    "3234": "CPO/矽光子",
    "4908": "光通訊",
    "2345": "網通",

    # AI 伺服器 / BMC / ODM
    "5274": "AI伺服器/BMC",
    "6669": "AI伺服器/OEM",
    "3231": "AI伺服器/OEM",
    "2382": "AI伺服器/OEM",
    "2357": "AI伺服器/OEM",
    "4938": "AI伺服器/OEM",

    # 散熱
    "3017": "散熱",
    "3324": "散熱",
    "3653": "散熱",
    "2421": "散熱",
    "2241": "散熱",

    # BBU / 電池備援 / 電源
    "4931": "BBU/電池備援",
    "3211": "BBU/電池備援",
    "6781": "BBU/電池備援",
    "2308": "電源/UPS",
    "2301": "電源/UPS",
    "8043": "電源/UPS",
    "6409": "電源/UPS",

    # 低軌衛星 / 網通
    "3491": "低軌衛星",
    "2314": "低軌衛星",
    "3596": "網通",
    "6285": "網通",
    "5388": "網通",
    "4906": "網通",

    # PCB / CCL / 載板
    "6274": "PCB/CCL",
    "2383": "PCB/CCL",
    "6672": "PCB/CCL",
    "2368": "PCB/CCL",
    "5469": "PCB/CCL",
    "8046": "ABF載板",
    "3189": "ABF載板",
    "3037": "ABF載板",

    # 記憶體
    "8299": "記憶體",
    "2408": "記憶體",
    "2344": "記憶體",
    "4967": "記憶體",
    "3260": "記憶體",
    "2451": "記憶體",

    # 面板 / 觸控
    "3673": "面板/觸控",
    "3481": "面板/觸控",
    "2409": "面板/觸控",
    "4976": "面板/觸控",

    # MLCC / 被動元件
    "2327": "MLCC/被動元件",
    "2492": "MLCC/被動元件",
    "3026": "MLCC/被動元件",
    "6173": "MLCC/被動元件",
    "6284": "MLCC/被動元件",

    # 半導體設備 / 測試
    "2360": "半導體設備/測試",
    "3583": "半導體設備/測試",
    "3131": "半導體設備/測試",
    "6640": "半導體設備/測試",
    "3413": "半導體設備/測試",
    "5536": "半導體設備/測試",
    "6217": "半導體設備/測試",

    # 銅箔 / 金屬材料
    "1785": "銅箔/材料",
    "2031": "銅箔/材料",
    "2009": "銅箔/材料",
    "9958": "銅箔/材料",

    # 軸承 / 傳動元件
    "4562": "軸承/傳動元件",
    "4577": "軸承/傳動元件",
    "3376": "軸承/傳動元件",
    "1597": "軸承/傳動元件",

    # 工業電腦
    "2395": "工業電腦",
    "2468": "工業電腦",
    "3022": "工業電腦",
    "6166": "工業電腦",

    # 軍工 / 航太
    "2634": "軍工/航太",
    "4572": "軍工/航太",
    "8033": "軍工/航太",
    "2208": "軍工/航太",

    # 生技
    "4743": "生技",
    "6589": "生技",
    "1795": "生技",
    "6472": "生技",

    # 重電 / 電網
    "1519": "重電/電網",
    "1503": "重電/電網",
    "1513": "重電/電網",
    "1609": "重電/電網",
}

THEME_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    (
        "CPO/矽光子",
        [
            "華星光", "IET", "IET-KY", "眾達", "眾達-KY", "上詮", "波若威", "聯亞", "聯鈞",
            "光聖", "光環", "CPO", "矽光子", "共同封裝光學", "光模組", "高速光通訊"
        ],
    ),
    ("光通訊", ["光通訊", "光纖", "收發模組", "光纖模組", "光器件"]),
    ("AI伺服器/BMC", ["信驊", "BMC", "伺服器管理晶片", "AST2600", "AST2700"]),
    (
        "AI伺服器/OEM",
        [
            "廣達", "緯創", "緯穎", "華碩", "和碩", "英業達", "OEM", "ODM", "伺服器",
            "AI伺服器", "資料中心", "GPU伺服器"
        ],
    ),
    ("散熱", ["奇鋐", "雙鴻", "健策", "建準", "散熱", "熱導管", "均熱片", "風扇", "散熱模組"]),
    ("BBU/電池備援", ["新盛力", "順達", "AES", "AES-KY", "BBU", "電池備援", "備援電池", "鋰電池模組"]),
    ("電源/UPS", ["台達電", "光寶科", "群電", "電源供應器", "UPS", "不斷電", "電源模組"]),
    ("低軌衛星", ["昇達科", "台揚", "低軌", "衛星", "LEO", "衛星通訊", "天線"]),
    ("網通", ["智易", "啟碁", "中磊", "正文", "網通", "交換器", "路由器", "乙太網路", "WiFi"]),
    ("PCB/CCL", ["台燿", "台光電", "騰輝", "騰輝電子", "金像電", "瀚宇博", "高技", "PCB", "CCL", "銅箔基板", "印刷電路板"]),
    ("ABF載板", ["欣興", "景碩", "南電", "ABF", "載板", "IC載板"]),
    ("記憶體", ["群聯", "南亞科", "華邦電", "威剛", "創見", "十銓", "記憶體", "DRAM", "NAND", "NOR", "SSD控制晶片"]),
    ("面板/觸控", ["TPK", "TPK-KY", "群創", "友達", "面板", "觸控", "觸控模組", "顯示面板"]),
    ("MLCC/被動元件", ["國巨", "華新科", "禾伸堂", "信昌電", "MLCC", "被動元件", "電容", "電感", "電阻"]),
    ("半導體設備/測試", ["致茂", "辛耘", "弘塑", "均華", "京鼎", "測試", "ATE", "探針卡", "晶圓設備", "半導體設備", "封測設備"]),
    ("銅箔/材料", ["光洋科", "銅箔", "材料", "金屬材料", "濺鍍靶材", "靶材"]),
    ("軸承/傳動元件", ["軸承", "傳動", "線性滑軌", "滾珠螺桿", "傳動元件"]),
    ("工業電腦", ["研華", "凌華", "振樺電", "飛捷", "工業電腦", "IPC"]),
    ("軍工/航太", ["軍工", "航太", "雷虎", "漢翔", "長榮航太", "軍用", "無人機"]),
    ("生技", ["生技", "新藥", "醫材", "疫苗", "藥華藥", "保瑞"]),
    ("重電/電網", ["重電", "電網", "華城", "士電", "中興電", "亞力", "變壓器"]),
]

INDUSTRY_FALLBACK_RULES: list[tuple[str, list[str]]] = [
    ("半導體", ["半導體"]),
    ("光電", ["光電"]),
    ("網通", ["通信網路", "網路通訊"]),
    ("電子零組件", ["電子零組件"]),
    ("其他電子", ["其他電子"]),
    ("電腦及週邊", ["電腦及週邊"]),
    ("電機機械", ["電機機械"]),
    ("生技", ["生技醫療"]),
    ("航運", ["航運"]),
    ("鋼鐵", ["鋼鐵"]),
]

THEME_SORT_PRIORITY = {
    "CPO/矽光子": 1,
    "光通訊": 2,
    "AI伺服器/BMC": 3,
    "AI伺服器/OEM": 4,
    "散熱": 5,
    "BBU/電池備援": 6,
    "電源/UPS": 7,
    "低軌衛星": 8,
    "網通": 9,
    "PCB/CCL": 10,
    "ABF載板": 11,
    "記憶體": 12,
    "面板/觸控": 13,
    "MLCC/被動元件": 14,
    "半導體設備/測試": 15,
    "銅箔/材料": 16,
    "軸承/傳動元件": 17,
    "工業電腦": 18,
    "軍工/航太": 19,
    "重電/電網": 20,
    "生技": 21,
    "半導體": 22,
    "光電": 23,
    "電子零組件": 24,
    "其他電子": 25,
    "電腦及週邊": 26,
    "電機機械": 27,
    "航運": 28,
    "鋼鐵": 29,
    "其他": 999,
}

NUMERIC_DEFAULTS: dict[str, float] = {
    "close": 0.0,
    "turnover_100m": 0.0,
    "volume_ratio": 0.0,
    "volume_avg20": 0.0,
    "volume": 0.0,
    "ma20": 0.0,
    "ma60": 0.0,
    "pct_from_ma20": 999.0,
    "platform_high_20d": 0.0,
    "platform_high_60d": 0.0,
    "rsi": 0.0,
    "macd": 0.0,
    "macd_signal": 0.0,
    "macd_hist": 0.0,
    "boll_mid": 0.0,
    "boll_upper": 0.0,
    "obv_trend": 0.0,
    "low_5d": 0.0,
    "low_20d": 0.0,
    "high_5d": 0.0,
    "high_20d": 0.0,
    "pct_5d": 0.0,
    "pct_10d": 0.0,
    "drawdown_5d": 0.0,
    "vol5_over_vol20": 0.0,
    "ma20_slope": 0.0,
    "near_high20_ratio": 0.0,
}

RAW_SNAPSHOT_COLUMNS = [
    "stock_id",
    "name",
    "group",
    "close",
    "turnover_100m",
    "volume_ratio",
    "volume_avg20",
    "ma20",
    "ma60",
]

SELECTED_SNAPSHOT_COLUMNS = [
    "stock_id",
    "name",
    "group",
    "theme",
    "close",
    "turnover_100m",
    "volume_ratio",
    "rsi",
    "ma20",
    "ma60",
    "score_total",
    "overall_priority_score",
    "radar_tag",
    "radar_tag_main",
    "radar_tag_sub",
    "institution_score",
    "main_force_score",
    "broker_score",
    "breakout_score",
    "revenue_score",
    "near_support",
    "strong_support",
    "near_resistance",
    "strong_resistance",
]

TABLE_EXPORT_COLUMNS = [
    "stock_id",
    "name",
    "group",
    "theme",
    "close",
    "turnover_100m",
    "volume_ratio",
    "volume_avg20",
    "rsi",
    "ma20",
    "ma60",
    "platform_high_20d",
    "platform_high_60d",
    "near_high20_ratio",
    "pct_from_ma20",
    "vol5_over_vol20",
    "institution_score",
    "main_force_score",
    "broker_score",
    "breakout_score",
    "revenue_score",
    "tech_score",
    "score_total",
    "overall_priority_score",
    "score_starting",
    "score_second_wave",
    "score_strong_trend",
    "starting_breakout_score",
    "starting_accum_score",
    "second_wave_score",
    "strong_trend_score",
    "radar_tag",
    "radar_tag_main",
    "radar_tag_sub",
    "foreign_buy_days",
    "investment_buy_days",
    "dealer_buy_days",
    "foreign_buy",
    "trust_buy",
    "dealer_buy",
    "trust_holding_pct",
    "estimated_inst_cost",
    "main_force_10d",
    "broker_buy_5d",
    "revenue_yoy",
    "revenue_mom",
    "institution_force",
    "trust_force",
    "near_support",
    "strong_support",
    "near_resistance",
    "strong_resistance",
    "trade_warning",
    "is_restricted",
]

EXTERNAL_ZERO_COLUMNS = [
    "foreign_buy_days",
    "investment_buy_days",
    "dealer_buy_days",
    "foreign_buy",
    "trust_buy",
    "dealer_buy",
    "trust_holding_pct",
    "estimated_inst_cost",
    "main_force_10d",
    "broker_buy_5d",
    "revenue_yoy",
    "revenue_mom",
]

SCORE_NUMERIC_COLUMNS = [
    *EXTERNAL_ZERO_COLUMNS,
    "institution_force",
    "trust_force",
]

SCORE_OUTPUT_COLUMNS = [
    "institution_score",
    "main_force_score",
    "broker_score",
    "breakout_score",
    "revenue_score",
    "tech_score",
    "score_total",
    "starting_breakout_score",
    "starting_accum_score",
    "second_wave_score",
    "strong_trend_score",
    "score_starting",
    "score_second_wave",
    "score_strong_trend",
    "overall_priority_score",
]

BOOL_COLUMNS = [
    "is_above_ma20",
    "is_above_ma60",
    "is_near_breakout",
    "is_strong_near_high",
    "is_macd_positive",
    "is_macd_hist_positive",
    "is_excluded",
    "is_restricted",
]


def log_count(label: str, df: pd.DataFrame) -> None:
    print(f"[SCAN] {label}: {len(df)}")


def normalize_text(value: str) -> str:
    text = str(value or "").upper()
    text = text.replace("－", "-").replace("—", "-").replace("–", "-")
    text = re.sub(r"[\s_/()（）．.]+", "", text)
    return text


def downcast_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def shrink_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    if numeric_cols:
        df = downcast_numeric_columns(df, numeric_cols)

    for col in bool_cols:
        df[col] = df[col].astype(bool)

    for col in ("stock_id", "name", "group", "theme", "radar_tag", "radar_tag_main", "radar_tag_sub", "tag", "trade_warning"):
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def ensure_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "stock_id" not in out.columns:
        out["stock_id"] = ""
    if "name" not in out.columns:
        out["name"] = ""
    if "group" not in out.columns:
        out["group"] = ""

    out["stock_id"] = out["stock_id"].astype(str)
    out["name"] = out["name"].astype(str)
    out["group"] = out["group"].astype(str)

    for col, default in NUMERIC_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(default)
        )

    return downcast_numeric_columns(out, list(NUMERIC_DEFAULTS.keys()))


def derive_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_market_columns(df)

    out["near_high20_ratio"] = np.where(
        out["high_20d"] > 0,
        out["close"] / out["high_20d"],
        out["near_high20_ratio"],
    )

    out["drawdown_5d"] = np.where(
        out["high_5d"] > 0,
        np.maximum(0, (out["high_5d"] - out["close"]) / out["high_5d"] * 100),
        out["drawdown_5d"],
    )

    out["pct_5d"] = np.where(
        out["pct_5d"] != 0,
        out["pct_5d"],
        np.where(out["ma20"] > 0, np.clip(out["pct_from_ma20"] * 0.45, -20, 20), 0),
    )
    out["pct_10d"] = np.where(
        out["pct_10d"] != 0,
        out["pct_10d"],
        np.where(out["ma20"] > 0, np.clip(out["pct_from_ma20"] * 0.85, -30, 30), 0),
    )

    out["vol5_over_vol20"] = np.where(
        out["vol5_over_vol20"] > 0,
        out["vol5_over_vol20"],
        np.where(out["volume_ratio"] > 0, np.maximum(out["volume_ratio"] * 0.92, 0), 0),
    )

    out["ma20_slope"] = np.where(
        out["ma20_slope"] != 0,
        out["ma20_slope"],
        np.where(out["ma60"] > 0, (out["ma20"] - out["ma60"]) / out["ma60"] * 100, 0),
    )

    out["is_above_ma20"] = out["close"] > out["ma20"]
    out["is_above_ma60"] = out["close"] > out["ma60"]
    out["is_near_breakout"] = out["near_high20_ratio"] >= STARTING_ACCUM_NEAR_HIGH20_RATIO
    out["is_strong_near_high"] = out["near_high20_ratio"] >= STRONG_TREND_NEAR_HIGH20_RATIO
    out["is_macd_positive"] = out["macd"] > out["macd_signal"]
    out["is_macd_hist_positive"] = out["macd_hist"] > 0

    return shrink_dataframe_memory(out)


def is_excluded_stock(name: str, group: str) -> bool:
    name_text = str(name or "").upper()
    group_text = str(group or "")

    if any(keyword.upper() in name_text for keyword in EXCLUDED_NAME_KEYWORDS):
        return True

    if any(keyword in group_text for keyword in EXCLUDED_GROUP_KEYWORDS):
        return True

    return False


def classify_theme(stock_id: str, name: str, group: str) -> str:
    sid = str(stock_id or "").strip()
    if sid in STOCK_THEME_OVERRIDES:
        return STOCK_THEME_OVERRIDES[sid]

    combined_raw = f"{name} {group}"
    combined_norm = normalize_text(combined_raw)

    for theme, keywords in THEME_KEYWORD_RULES:
        for keyword in keywords:
            if normalize_text(keyword) in combined_norm:
                return theme

    group_text = str(group or "")
    for theme, keywords in INDUSTRY_FALLBACK_RULES:
        if any(keyword in group_text for keyword in keywords):
            return theme

    return "其他"


def add_exclusion_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    name_series = out["name"].fillna("").astype(str).str.upper()
    group_series = out["group"].fillna("").astype(str)

    name_mask = name_series.str.contains(EXCLUDED_NAME_PATTERN, regex=True, na=False)
    group_mask = group_series.str.contains(EXCLUDED_GROUP_PATTERN, regex=True, na=False)

    out["is_excluded"] = name_mask | group_mask
    return out


def sort_stage1_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.assign(
        sort_turnover=pd.to_numeric(df["turnover_100m"], errors="coerce").fillna(0),
        sort_volume_ratio=pd.to_numeric(df["volume_ratio"], errors="coerce").fillna(0),
        sort_vol5_over_vol20=pd.to_numeric(df["vol5_over_vol20"], errors="coerce").fillna(0),
        sort_near_high=pd.to_numeric(df["near_high20_ratio"], errors="coerce").fillna(0),
        sort_trend_bonus=np.where(
            (df["close"] > df["ma20"])
            | (df["close"] > df["ma60"])
            | (df["macd"] > df["macd_signal"])
            | (df["obv_trend"] > 0),
            1,
            0,
        ),
    )

    out = out.sort_values(
        [
            "sort_trend_bonus",
            "sort_near_high",
            "sort_turnover",
            "sort_volume_ratio",
            "sort_vol5_over_vol20",
        ],
        ascending=False,
    ).head(SECOND_SCAN_LIMIT)

    return out.drop(
        columns=[
            "sort_turnover",
            "sort_volume_ratio",
            "sort_vol5_over_vol20",
            "sort_near_high",
            "sort_trend_bonus",
        ]
    ).copy()


def first_stage_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = derive_pattern_features(df)
    out = add_exclusion_flag(out)

    log_count("Raw universe", out)

    universe = out.loc[~out["is_excluded"]].copy()
    del out
    gc.collect()

    log_count("After exclude ETF/financial", universe)

    universe = universe.loc[
        (universe["close"] >= MIN_PRICE)
        & (universe["close"] <= MAX_PRICE)
    ].copy()
    log_count("After price range", universe)

    if universe.empty:
        return universe

    print(
        "[SCAN] Dist stats | turnover>=min:",
        int((universe["turnover_100m"] >= MIN_TURNOVER_100M).sum()),
        "| volume_ratio>=min:",
        int((universe["volume_ratio"] >= MIN_VOLUME_RATIO).sum()),
        "| volume_avg20>=min:",
        int((universe["volume_avg20"] >= MIN_AVG_VOLUME_LOT).sum()),
        "| above_ma20_or_ma60:",
        int(((universe["close"] > universe["ma20"]) | (universe["close"] > universe["ma60"])).sum()),
        "| near_high20:",
        int((universe["near_high20_ratio"] >= STARTING_ACCUM_NEAR_HIGH20_RATIO).sum()),
    )

    liquidity_floor = (
        (universe["turnover_100m"] >= MIN_TURNOVER_100M)
        | (universe["volume_avg20"] >= MIN_AVG_VOLUME_LOT)
        | (
            (universe["turnover_100m"] >= max(MIN_TURNOVER_100M * 0.5, 0.5))
            & (universe["volume_avg20"] >= max(MIN_AVG_VOLUME_LOT * 0.6, 80))
        )
    )
    base_trend_floor = (
        (universe["close"] > universe["ma20"] * 0.99)
        | (universe["close"] > universe["ma60"] * 0.99)
        | (universe["macd_hist"] > -0.05)
        | (universe["obv_trend"] > 0)
        | (universe["near_high20_ratio"] >= 0.95)
    )

    stage1_base = universe.loc[liquidity_floor & base_trend_floor].copy()
    log_count("Stage1 base gate", stage1_base)

    if stage1_base.empty:
        return universe.head(0).copy()

    stage1_focus = stage1_base.loc[
        (stage1_base["pct_from_ma20"] <= max(STRONG_TREND_MAX_DISTANCE_FROM_MA20, 25))
        & (
            (stage1_base["volume_ratio"] >= STARTING_BREAKOUT_MIN_VOLUME_RATIO)
            | (stage1_base["vol5_over_vol20"] >= STARTING_ACCUM_VOL5_OVER_VOL20_MIN)
            | (stage1_base["near_high20_ratio"] >= STARTING_ACCUM_NEAR_HIGH20_RATIO)
            | stage1_base["is_above_ma20"]
            | stage1_base["is_above_ma60"]
        )
    ].copy()
    log_count("Stage1 focus gate", stage1_focus)

    if not stage1_focus.empty:
        return sort_stage1_candidates(stage1_focus)

    relaxed_gate_1 = stage1_base.loc[
        (
            (stage1_base["near_high20_ratio"] >= 0.94)
            | (stage1_base["close"] > stage1_base["boll_mid"])
            | (stage1_base["macd_hist"] > -0.03)
        )
        & (stage1_base["pct_from_ma20"] <= max(STRONG_TREND_MAX_DISTANCE_FROM_MA20 + 5, 30))
    ].copy()
    log_count("Stage1 relaxed gate 1", relaxed_gate_1)

    if not relaxed_gate_1.empty:
        print("[SCAN] Fallback used: relaxed gate 1")
        return sort_stage1_candidates(relaxed_gate_1)

    relaxed_gate_2 = universe.loc[
        (
            (universe["turnover_100m"] >= max(MIN_TURNOVER_100M * 0.35, 0.3))
            | (universe["volume_avg20"] >= max(MIN_AVG_VOLUME_LOT * 0.45, 60))
        )
        & (
            (universe["volume_ratio"] >= 0.85)
            | (universe["near_high20_ratio"] >= 0.95)
            | (universe["obv_trend"] > 0)
        )
    ].copy()
    log_count("Stage1 relaxed gate 2", relaxed_gate_2)

    if not relaxed_gate_2.empty:
        print("[SCAN] Fallback used: relaxed gate 2")
        return sort_stage1_candidates(relaxed_gate_2)

    emergency_pool = universe.sort_values(
        ["near_high20_ratio", "turnover_100m", "volume_ratio", "volume_avg20"],
        ascending=False,
    ).head(min(SECOND_SCAN_LIMIT, 100)).copy()
    log_count("Stage1 emergency pool", emergency_pool)

    if not emergency_pool.empty:
        print("[SCAN] Emergency pool used")
        return emergency_pool

    return universe.head(0).copy()


def safe_merge_external(
    base_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    required_cols: list[str],
    source_name: str,
) -> pd.DataFrame:
    if ext_df.empty:
        print(f"[SCAN] {source_name} empty, fill zero")
        out = base_df.copy()
        for col in required_cols:
            if col != "stock_id":
                out[col] = 0.0
        return out

    if "stock_id" not in ext_df.columns:
        print(f"[SCAN] {source_name} missing stock_id, fill zero")
        out = base_df.copy()
        for col in required_cols:
            if col != "stock_id":
                out[col] = 0.0
        return out

    ext = ext_df.copy()
    ext["stock_id"] = ext["stock_id"].astype(str)
    ext = ext.drop_duplicates(subset=["stock_id"], keep="last")

    keep_cols = ["stock_id"]
    for col in required_cols:
        if col == "stock_id":
            continue
        if col not in ext.columns:
            print(f"[SCAN] {source_name} missing col {col}, fill zero")
            ext[col] = 0.0
        keep_cols.append(col)

    ext = ext[keep_cols]
    out = base_df.merge(ext, on="stock_id", how="left", copy=False)

    for col in required_cols:
        if col == "stock_id":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return downcast_numeric_columns(out, [col for col in required_cols if col != "stock_id"])


def merge_external_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[SCAN] merge_external_data start")

    out = df.copy()
    stock_ids = out["stock_id"].astype(str).tolist()
    print(f"[SCAN] merge_external_data stock count: {len(stock_ids)}")

    print("[SCAN] Fetch institutional data")
    institutional = get_institutional_data(stock_ids)
    print(f"[SCAN] Institutional rows: {len(institutional)}")

    out = safe_merge_external(
        out,
        institutional,
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
        ],
        INSTITUTIONAL_CSV,
    )
    del institutional
    gc.collect()

    print("[SCAN] Fetch broker data")
    broker = get_broker_data(stock_ids)
    print(f"[SCAN] Broker rows: {len(broker)}")

    out = safe_merge_external(
        out,
        broker,
        ["stock_id", "main_force_10d", "broker_buy_5d"],
        "broker",
    )
    del broker
    gc.collect()

    print("[SCAN] Fetch revenue data")
    revenue = get_revenue_data(stock_ids)
    print(f"[SCAN] Revenue rows: {len(revenue)}")

    out = safe_merge_external(
        out,
        revenue,
        ["stock_id", "revenue_yoy", "revenue_mom"],
        REVENUE_CSV,
    )
    del revenue
    gc.collect()

    for col in EXTERNAL_ZERO_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    volume_base = pd.to_numeric(out["volume"], errors="coerce").replace(0, np.nan)

    out["institution_force"] = (out["foreign_buy"] + out["trust_buy"]) / volume_base
    out["trust_force"] = out["trust_buy"] / volume_base
    out["institution_force"] = out["institution_force"].replace([np.inf, -np.inf], 0).fillna(0)
    out["trust_force"] = out["trust_force"].replace([np.inf, -np.inf], 0).fillna(0)

    out = downcast_numeric_columns(out, EXTERNAL_ZERO_COLUMNS + ["institution_force", "trust_force"])

    print("[SCAN] merge_external_data done")
    return shrink_dataframe_memory(out)


def calc_institution_score(row: pd.Series) -> float:
    score = 0.0

    investment_buy_days = float(row.get("investment_buy_days", 0) or 0)
    foreign_buy_days = float(row.get("foreign_buy_days", 0) or 0)
    dealer_buy_days = float(row.get("dealer_buy_days", 0) or 0)
    institution_force = float(row.get("institution_force", 0) or 0)
    trust_force = float(row.get("trust_force", 0) or 0)
    trust_holding_pct = float(row.get("trust_holding_pct", 0) or 0)
    estimated_inst_cost = float(row.get("estimated_inst_cost", 0) or 0)
    close = float(row.get("close", 0) or 0)

    if investment_buy_days >= 5:
        score += 12
    elif investment_buy_days >= 3:
        score += 8
    elif investment_buy_days >= 2:
        score += 4

    if foreign_buy_days >= 5:
        score += 8
    elif foreign_buy_days >= 3:
        score += 5
    elif foreign_buy_days >= 2:
        score += 2

    if dealer_buy_days >= 5:
        score += 3
    elif dealer_buy_days >= 3:
        score += 2
    elif dealer_buy_days >= 2:
        score += 1

    if institution_force >= 0.12:
        score += 10
    elif institution_force >= 0.08:
        score += 7
    elif institution_force >= 0.05:
        score += 4

    if trust_force >= 0.08:
        score += 8
    elif trust_force >= 0.05:
        score += 5
    elif trust_force >= 0.03:
        score += 2

    if 1 <= trust_holding_pct <= 3:
        score += 6
    elif 3 < trust_holding_pct <= 8:
        score += 4
    elif trust_holding_pct > 15:
        score -= 5

    if estimated_inst_cost > 0 and close > 0:
        diff_pct = abs(close - estimated_inst_cost) / estimated_inst_cost * 100
        if diff_pct <= 3:
            score += 6
        elif diff_pct <= 5:
            score += 3

    return score


def calc_main_force_score(row: pd.Series) -> float:
    value = float(row.get("main_force_10d", 0) or 0)
    if value >= 30000:
        return 15
    if value >= 15000:
        return 10
    if value >= 5000:
        return 6
    return 0.0


def calc_broker_score(row: pd.Series) -> float:
    value = float(row.get("broker_buy_5d", 0) or 0)
    if value >= 10000:
        return 12
    if value >= 5000:
        return 8
    if value >= 2000:
        return 5
    return 0.0


def calc_breakout_score(row: pd.Series) -> float:
    score = 0.0
    close = float(row.get("close", 0) or 0)
    p20 = float(row.get("platform_high_20d", 0) or 0)
    p60 = float(row.get("platform_high_60d", 0) or 0)
    volume_ratio = float(row.get("volume_ratio", 0) or 0)

    if close > p20 > 0:
        score += 8
    if close > p60 > 0:
        score += 12
    if volume_ratio >= 2:
        score += 4
    return score


def calc_revenue_score(row: pd.Series) -> float:
    score = 0.0
    revenue_yoy = float(row.get("revenue_yoy", 0) or 0)
    revenue_mom = float(row.get("revenue_mom", 0) or 0)

    if revenue_yoy >= 30:
        score += 10
    elif revenue_yoy >= 15:
        score += 6

    if revenue_mom >= 10:
        score += 5
    elif revenue_mom >= 5:
        score += 3

    return score


def calculate_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["near_support"] = np.round(
        np.where(out["ma20"] > 0, np.maximum(out["low_5d"], out["ma20"]), out["low_5d"]),
        2,
    )
    out["strong_support"] = np.round(
        np.where(out["ma60"] > 0, np.minimum(out["low_20d"], out["ma60"]), out["low_20d"]),
        2,
    )
    out["near_resistance"] = np.round(out["high_5d"], 2)
    out["strong_resistance"] = np.round(np.maximum(out["high_20d"], out["high_5d"] * 1.03), 2)

    return downcast_numeric_columns(out, ["near_support", "strong_support", "near_resistance", "strong_resistance"])


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col, default in NUMERIC_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default

    for col in SCORE_NUMERIC_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "is_above_ma20" not in out.columns:
        out["is_above_ma20"] = out["close"] > out["ma20"]
    if "is_above_ma60" not in out.columns:
        out["is_above_ma60"] = out["close"] > out["ma60"]
    if "is_near_breakout" not in out.columns:
        out["is_near_breakout"] = out["near_high20_ratio"] >= STARTING_ACCUM_NEAR_HIGH20_RATIO
    if "is_strong_near_high" not in out.columns:
        out["is_strong_near_high"] = out["near_high20_ratio"] >= STRONG_TREND_NEAR_HIGH20_RATIO
    if "is_macd_positive" not in out.columns:
        out["is_macd_positive"] = out["macd"] > out["macd_signal"]
    if "is_macd_hist_positive" not in out.columns:
        out["is_macd_hist_positive"] = out["macd_hist"] > 0

    out["theme"] = [
        classify_theme(stock_id, name, group)
        for stock_id, name, group in zip(out["stock_id"], out["name"], out["group"])
    ]
    out["theme_priority"] = out["theme"].map(THEME_SORT_PRIORITY).fillna(999)

    out["institution_score"] = out.apply(calc_institution_score, axis=1)
    out["main_force_score"] = out.apply(calc_main_force_score, axis=1)
    out["broker_score"] = out.apply(calc_broker_score, axis=1)
    out["breakout_score"] = out.apply(calc_breakout_score, axis=1)
    out["revenue_score"] = out.apply(calc_revenue_score, axis=1)

    out["tech_score"] = 0.0
    out.loc[out["is_above_ma20"], "tech_score"] += 4
    out.loc[out["is_above_ma60"], "tech_score"] += 6
    out.loc[out["ma20"] > out["ma60"], "tech_score"] += 4

    out.loc[(out["volume_ratio"] >= 1.2) & (out["volume_ratio"] < 1.5), "tech_score"] += 4
    out.loc[(out["volume_ratio"] >= 1.5) & (out["volume_ratio"] < 2.0), "tech_score"] += 6
    out.loc[out["volume_ratio"] >= 2.0, "tech_score"] += 8

    out.loc[(out["rsi"] >= 50) & (out["rsi"] <= 72), "tech_score"] += 6
    out.loc[(out["rsi"] > 72) & (out["rsi"] <= 80), "tech_score"] += 3
    out.loc[out["rsi"] > 80, "tech_score"] -= 2
    out.loc[out["rsi"] < 35, "tech_score"] -= 2

    out.loc[out["is_macd_positive"], "tech_score"] += 5
    out.loc[out["is_macd_hist_positive"], "tech_score"] += 4
    out.loc[out["close"] > out["boll_mid"], "tech_score"] += 3
    out.loc[out["close"] > out["boll_upper"], "tech_score"] += 2
    out.loc[out["obv_trend"] > 0, "tech_score"] += 5

    out["score_total"] = (
        out["tech_score"]
        + out["institution_score"]
        + out["main_force_score"]
        + out["broker_score"]
        + out["breakout_score"]
        + out["revenue_score"]
    )

    out["starting_breakout_score"] = 0.0
    out.loc[out["volume_ratio"] >= STARTING_BREAKOUT_MIN_VOLUME_RATIO, "starting_breakout_score"] += 7
    out.loc[out["close"] > out["platform_high_20d"], "starting_breakout_score"] += 8
    out.loc[out["close"] > out["ma20"], "starting_breakout_score"] += 5
    out.loc[
        (out["rsi"] >= STARTING_BREAKOUT_RSI_MIN) & (out["rsi"] <= STARTING_BREAKOUT_RSI_MAX),
        "starting_breakout_score",
    ] += 6
    out.loc[
        out["pct_from_ma20"] <= STARTING_BREAKOUT_MAX_DISTANCE_FROM_MA20,
        "starting_breakout_score",
    ] += 5
    out.loc[out["is_macd_positive"], "starting_breakout_score"] += 3
    out.loc[out["obv_trend"] > 0, "starting_breakout_score"] += 2
    out.loc[out["pct_from_ma20"] > STARTING_BREAKOUT_MAX_DISTANCE_FROM_MA20 + 3, "starting_breakout_score"] -= 6
    out.loc[out["rsi"] > 80, "starting_breakout_score"] -= 4

    out["starting_accum_score"] = 0.0
    out.loc[
        (out["pct_10d"] >= STARTING_ACCUM_10D_PCT_MIN) & (out["pct_10d"] <= STARTING_ACCUM_10D_PCT_MAX),
        "starting_accum_score",
    ] += 7
    out.loc[out["drawdown_5d"] <= STARTING_ACCUM_5D_PULLBACK_MAX, "starting_accum_score"] += 6
    out.loc[out["close"] >= out["ma20"] * STARTING_ACCUM_CLOSE_OVER_MA20, "starting_accum_score"] += 5
    out.loc[out["ma20_slope"] > 0, "starting_accum_score"] += 5
    out.loc[out["vol5_over_vol20"] >= STARTING_ACCUM_VOL5_OVER_VOL20_MIN, "starting_accum_score"] += 6
    out.loc[out["obv_trend"] > 0, "starting_accum_score"] += 4
    out.loc[out["near_high20_ratio"] >= STARTING_ACCUM_NEAR_HIGH20_RATIO, "starting_accum_score"] += 6
    out.loc[out["is_macd_positive"], "starting_accum_score"] += 3
    out.loc[out["pct_from_ma20"] > 15, "starting_accum_score"] -= 4

    out["second_wave_score"] = 0.0
    out.loc[out["close"] > out["ma60"], "second_wave_score"] += 6
    out.loc[out["close"] > out["ma20"] * 1.02, "second_wave_score"] += 4
    out.loc[out["obv_trend"] > 0, "second_wave_score"] += 5
    out.loc[out["is_macd_hist_positive"], "second_wave_score"] += 5
    out.loc[
        (out["rsi"] >= SECOND_WAVE_MIN_RSI) & (out["rsi"] <= SECOND_WAVE_MAX_RSI),
        "second_wave_score",
    ] += 5
    out.loc[out["volume_ratio"] >= SECOND_WAVE_MIN_VOLUME_RATIO, "second_wave_score"] += 3
    out.loc[out["near_high20_ratio"] >= 0.96, "second_wave_score"] += 5
    out.loc[out["drawdown_5d"] <= 6, "second_wave_score"] += 3
    out.loc[out["broker_score"] >= 5, "second_wave_score"] += 3
    out.loc[out["main_force_score"] >= 6, "second_wave_score"] += 4
    out.loc[out["revenue_score"] >= 5, "second_wave_score"] += 3
    out.loc[out["pct_from_ma20"] > SECOND_WAVE_MAX_DISTANCE_FROM_MA20, "second_wave_score"] -= 6

    out["strong_trend_score"] = 0.0
    out.loc[out["close"] > out["ma20"] * 1.08, "strong_trend_score"] += 7
    out.loc[out["close"] > out["ma60"], "strong_trend_score"] += 5
    out.loc[
        (out["rsi"] >= STRONG_TREND_MIN_RSI) & (out["rsi"] <= STRONG_TREND_MAX_RSI),
        "strong_trend_score",
    ] += 6
    out.loc[out["obv_trend"] > 0, "strong_trend_score"] += 5
    out.loc[out["is_macd_hist_positive"], "strong_trend_score"] += 5
    out.loc[out["near_high20_ratio"] >= STRONG_TREND_NEAR_HIGH20_RATIO, "strong_trend_score"] += 7
    out.loc[out["volume_ratio"] >= STRONG_TREND_MIN_VOLUME_RATIO, "strong_trend_score"] += 3
    out.loc[out["close"] > out["platform_high_60d"], "strong_trend_score"] += 5
    out.loc[out["revenue_score"] >= 5, "strong_trend_score"] += 2
    out.loc[out["pct_from_ma20"] > STRONG_TREND_MAX_DISTANCE_FROM_MA20, "strong_trend_score"] -= 5
    out.loc[out["rsi"] > 90, "strong_trend_score"] -= 4

    out["score_starting"] = np.maximum(out["starting_breakout_score"], out["starting_accum_score"])
    out["score_second_wave"] = out["second_wave_score"]
    out["score_strong_trend"] = out["strong_trend_score"]

    out["radar_tag_main"] = "剛啟動"
    out["radar_tag_sub"] = np.where(
        out["starting_breakout_score"] >= out["starting_accum_score"],
        "爆量突破",
        "收籌墊高",
    )

    strong_mask = (
        (out["strong_trend_score"] >= out["score_starting"])
        & (out["strong_trend_score"] >= out["second_wave_score"])
        & (out["strong_trend_score"] >= 17)
    )
    second_wave_mask = (
        ~strong_mask
        & (out["second_wave_score"] >= out["score_starting"])
        & (out["second_wave_score"] >= 15)
    )

    out.loc[strong_mask, "radar_tag_main"] = "強者恆強"
    out.loc[strong_mask, "radar_tag_sub"] = ""
    out.loc[second_wave_mask, "radar_tag_main"] = "可能第二波"
    out.loc[second_wave_mask, "radar_tag_sub"] = ""

    out["radar_tag"] = np.where(
        out["radar_tag_main"] == "剛啟動",
        out["radar_tag_main"] + "｜" + out["radar_tag_sub"],
        out["radar_tag_main"],
    )
    out["tag"] = out["radar_tag"]

    out["overall_priority_score"] = (
        out["score_total"]
        + np.where(out["radar_tag_main"] == "強者恆強", out["strong_trend_score"], 0)
        + np.where(out["radar_tag_main"] == "可能第二波", out["second_wave_score"], 0)
        + np.where(out["radar_tag_main"] == "剛啟動", out["score_starting"], 0)
    )

    out = calculate_support_resistance(out)

    if "trade_warning" not in out.columns:
        out["trade_warning"] = ""
    if "is_restricted" not in out.columns:
        out["is_restricted"] = False

    out = downcast_numeric_columns(out, SCORE_OUTPUT_COLUMNS + ["theme_priority"])
    return shrink_dataframe_memory(out)


def build_reason_and_targets(row: dict) -> dict:
    reasons: list[str] = []

    main_tag = str(row.get("radar_tag_main") or "")
    sub_tag = str(row.get("radar_tag_sub") or "")

    if main_tag == "剛啟動":
        reasons.append(f"型態偏剛啟動，子類型為{sub_tag or '待補'}")
    elif main_tag == "可能第二波":
        reasons.append("型態偏第二波，屬整理後再攻候選")
    elif main_tag == "強者恆強":
        reasons.append("型態偏強者恆強，屬主升段延續候選")

    theme = str(row.get("theme") or "")
    if theme and theme != "其他":
        reasons.append(f"細題材歸類：{theme}")

    if float(row.get("institution_score", 0) or 0) >= 12:
        reasons.append("法人籌碼加分明顯")
    elif float(row.get("institution_score", 0) or 0) >= 6:
        reasons.append("法人買盤有延續")

    if float(row.get("main_force_score", 0) or 0) >= 10:
        reasons.append("主力近10日累積明顯")
    elif float(row.get("main_force_score", 0) or 0) >= 6:
        reasons.append("主力有偏多痕跡")

    if float(row.get("broker_score", 0) or 0) >= 8:
        reasons.append("分點買盤偏強")
    elif float(row.get("broker_score", 0) or 0) >= 5:
        reasons.append("分點有短線加分")

    if float(row.get("revenue_score", 0) or 0) >= 10:
        reasons.append("營收成長動能強")
    elif float(row.get("revenue_score", 0) or 0) >= 5:
        reasons.append("營收成長具支撐")

    if float(row.get("volume_ratio", 0) or 0) >= 2:
        reasons.append("量比放大，資金關注提高")
    elif float(row.get("volume_ratio", 0) or 0) >= 1.3:
        reasons.append("量能優於均值")
    elif float(row.get("vol5_over_vol20", 0) or 0) >= STARTING_ACCUM_VOL5_OVER_VOL20_MIN:
        reasons.append("5日均量高於20日均量，偏收籌墊高")

    if float(row.get("close", 0) or 0) > float(row.get("platform_high_20d", 0) or 0) > 0:
        reasons.append("突破短期平台高點")
    if float(row.get("close", 0) or 0) > float(row.get("platform_high_60d", 0) or 0) > 0:
        reasons.append("突破中期平台高點")
    if float(row.get("close", 0) or 0) > float(row.get("ma20", 0) or 0) > 0:
        reasons.append("站上 MA20")
    if float(row.get("close", 0) or 0) > float(row.get("ma60", 0) or 0) > 0:
        reasons.append("站上 MA60")
    if float(row.get("near_high20_ratio", 0) or 0) >= STRONG_TREND_NEAR_HIGH20_RATIO:
        reasons.append("股價貼近20日高點")

    short_reasons = reasons[:4]
    reason_text = "；".join(short_reasons) if short_reasons else "符合模型條件"

    return {
        "score": float(row.get("score_total", 0) or 0),
        "tag": str(row.get("radar_tag") or ""),
        "reason_text": reason_text,
        "reason_list": reasons[:8],
        "near_support": row.get("near_support"),
        "strong_support": row.get("strong_support"),
        "near_resistance": row.get("near_resistance"),
        "strong_resistance": row.get("strong_resistance"),
    }


def build_table_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    export_cols = [col for col in TABLE_EXPORT_COLUMNS if col in df.columns]
    slim_df = df[export_cols].copy()
    records = dataframe_to_records(slim_df)

    for row in records:
        row.update(build_reason_and_targets(row))

    return records


def build_payload(market_scanned: int, analyzed_df: pd.DataFrame) -> dict:
    top30_df = analyzed_df.head(TOP30_COUNT)
    watchlist_df = analyzed_df.iloc[TOP30_COUNT:TOP30_COUNT + WATCHLIST_COUNT]

    starting_df = analyzed_df[analyzed_df["radar_tag_main"] == "剛啟動"].sort_values(
        ["score_starting", "overall_priority_score", "turnover_100m"],
        ascending=False,
    ).head(TOP30_COUNT)

    starting_breakout_df = analyzed_df[
        (analyzed_df["radar_tag_main"] == "剛啟動")
        & (analyzed_df["radar_tag_sub"] == "爆量突破")
    ].sort_values(
        ["starting_breakout_score", "overall_priority_score", "turnover_100m"],
        ascending=False,
    ).head(TOP30_COUNT)

    starting_accum_df = analyzed_df[
        (analyzed_df["radar_tag_main"] == "剛啟動")
        & (analyzed_df["radar_tag_sub"] == "收籌墊高")
    ].sort_values(
        ["starting_accum_score", "overall_priority_score", "turnover_100m"],
        ascending=False,
    ).head(TOP30_COUNT)

    second_wave_df = analyzed_df[analyzed_df["radar_tag_main"] == "可能第二波"].sort_values(
        ["second_wave_score", "overall_priority_score", "turnover_100m"],
        ascending=False,
    ).head(TOP30_COUNT)

    strong_trend_df = analyzed_df[analyzed_df["radar_tag_main"] == "強者恆強"].sort_values(
        ["strong_trend_score", "overall_priority_score", "turnover_100m"],
        ascending=False,
    ).head(TOP30_COUNT)

    broker_track_df = analyzed_df.sort_values(
        ["broker_score", "main_force_score", "overall_priority_score"],
        ascending=False,
    ).head(BROKER_TRACK_COUNT)

    risk_overheated_df = analyzed_df[
        (analyzed_df["volume_ratio"] >= 2.5) & (analyzed_df["rsi"] >= 82)
    ].head(20)

    high_turnover_df = analyzed_df.sort_values(
        ["turnover_100m", "volume_ratio"],
        ascending=False,
    ).head(20)

    all_selected_df = analyzed_df.sort_values(
        ["theme_priority", "overall_priority_score", "turnover_100m"],
        ascending=[True, False, False],
    )

    payload = {
        "summary": {
            "market_scanned": int(market_scanned),
            "selected": int(len(analyzed_df)),
            "starting_count": int((analyzed_df["radar_tag_main"] == "剛啟動").sum()),
            "starting_breakout_count": int((
                (analyzed_df["radar_tag_main"] == "剛啟動")
                & (analyzed_df["radar_tag_sub"] == "爆量突破")
            ).sum()),
            "starting_accum_count": int((
                (analyzed_df["radar_tag_main"] == "剛啟動")
                & (analyzed_df["radar_tag_sub"] == "收籌墊高")
            ).sum()),
            "second_wave_count": int((analyzed_df["radar_tag_main"] == "可能第二波").sum()),
            "strong_trend_count": int((analyzed_df["radar_tag_main"] == "強者恆強").sum()),
            "overheated_count": int(len(risk_overheated_df)),
        },
        "top30": build_table_rows(top30_df),
        "watchlist": build_table_rows(watchlist_df),
        "starting": build_table_rows(starting_df),
        "starting_breakout": build_table_rows(starting_breakout_df),
        "starting_accum": build_table_rows(starting_accum_df),
        "second_wave": build_table_rows(second_wave_df),
        "strong_trend": build_table_rows(strong_trend_df),
        "broker_track": build_table_rows(broker_track_df),
        "overheated": build_table_rows(risk_overheated_df),
        "high_turnover": build_table_rows(high_turnover_df),
        "all_selected": build_table_rows(all_selected_df),
    }
    return payload


def make_snapshot_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.head(0).copy()

    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = ""

    return shrink_dataframe_memory(out.copy())


def run_scan(save: bool = True, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback(
            {
                "stage": "prepare",
                "percent": 0,
                "message": "準備掃描",
                "processed": 0,
                "total": 0,
                "success": 0,
                "failed": 0,
                "skipped": 0,
            }
        )

    print("[SCAN] Step 1: fetch market snapshot")
    raw_df = fetch_market_snapshot_parallel(progress_callback=progress_callback)
    print(f"[SCAN] Market snapshot rows: {len(raw_df)}")

    if raw_df.empty:
        raise RuntimeError("抓不到市場資料")

    market_scanned = len(raw_df)
    raw_snapshot_df = make_snapshot_df(raw_df, RAW_SNAPSHOT_COLUMNS) if save else pd.DataFrame()

    if progress_callback:
        progress_callback(
            {
                "stage": "filter",
                "percent": 88,
                "message": "第一層快篩中",
            }
        )

    print("[SCAN] Step 2: first stage filter")
    stage1_df = first_stage_filter(raw_df)
    print(f"[SCAN] First stage selected: {len(stage1_df)}")

    del raw_df
    gc.collect()

    if stage1_df.empty:
        raise RuntimeError("第一層快篩後沒有股票（請檢查欄位來源或門檻設定）")

    if progress_callback:
        progress_callback(
            {
                "stage": "merge",
                "percent": 92,
                "message": "合併法人 / 分點 / 營收資料",
            }
        )

    print("[SCAN] Step 3: merge external data")
    stage2_input = merge_external_data(stage1_df)
    print(f"[SCAN] Stage2 rows: {len(stage2_input)}")

    del stage1_df
    gc.collect()

    if progress_callback:
        progress_callback(
            {
                "stage": "score",
                "percent": 96,
                "message": "計算分數中",
            }
        )

    print("[SCAN] Step 4: calculate scores")
    analyzed_df = calculate_scores(stage2_input)
    analyzed_df = analyzed_df.sort_values(
        ["overall_priority_score", "score_total", "turnover_100m", "volume_ratio"],
        ascending=False,
    ).reset_index(drop=True)
    print(f"[SCAN] Final analyzed rows: {len(analyzed_df)}")

    del stage2_input
    gc.collect()

    if analyzed_df.empty:
        raise RuntimeError("分數計算後沒有股票（請檢查 merge / score 欄位）")

    if progress_callback:
        progress_callback(
            {
                "stage": "payload",
                "percent": 98,
                "message": "建立結果資料中",
            }
        )

    print("[SCAN] Step 5: build payload")
    payload = build_payload(market_scanned, analyzed_df)

    if save:
        if progress_callback:
            progress_callback(
                {
                    "stage": "save",
                    "percent": 99,
                    "message": "儲存掃描結果中",
                }
            )

        selected_snapshot_df = make_snapshot_df(analyzed_df, SELECTED_SNAPSHOT_COLUMNS)

        print("[SCAN] Step 6: save snapshot")
        save_snapshot(payload, raw_snapshot_df, selected_snapshot_df)
        print("[SCAN] Snapshot saved")

        del selected_snapshot_df
        gc.collect()

    del raw_snapshot_df
    del analyzed_df
    gc.collect()

    if progress_callback:
        progress_callback(
            {
                "stage": "done",
                "percent": 100,
                "message": "掃描完成",
            }
        )

    print("[SCAN] run_scan completed")
    return payload
