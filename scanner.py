from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BROKER_TRACK_COUNT,
    INSTITUTIONAL_CSV,
    MAX_DISTANCE_FROM_MA20,
    MAX_PRICE,
    MIN_AVG_VOLUME_LOT,
    MIN_PRICE,
    MIN_TURNOVER_100M,
    MIN_VOLUME_RATIO,
    REVENUE_CSV,
    SECOND_SCAN_LIMIT,
    TOP30_COUNT,
    WATCHLIST_COUNT,
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

THEME_RULES: dict[str, list[str]] = {
    "記憶體": ["記憶體", "DRAM", "NAND", "NOR", "快閃記憶體", "SSD"],
    "ABF載板": ["ABF", "載板", "IC載板"],
    "散熱": ["散熱", "熱導管", "均熱片", "散熱模組", "風扇"],
    "PCB": ["PCB", "印刷電路板", "銅箔基板", "CCL", "HDI", "軟板"],
    "AI伺服器": ["AI", "伺服器", "SERVER", "GPU", "ASIC", "資料中心"],
    "CPO/矽光子": ["CPO", "矽光子", "光通訊", "光模組", "高速光"],
    "網通": ["網通", "網路", "交換器", "路由器", "WIFI", "乙太網路"],
    "低軌衛星": ["低軌", "衛星", "通訊天線", "太空", "衛星通訊"],
    "被動元件": ["被動元件", "MLCC", "電感", "電容", "電阻"],
    "電源/BBU": ["電源", "BBU", "UPS", "電池備援", "電源供應器"],
    "面板": ["面板", "顯示器", "LCD", "OLED", "觸控"],
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
}


def log_count(label: str, df: pd.DataFrame) -> None:
    print(f"[SCAN] {label}: {len(df)}")


def ensure_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "stock_id" not in out.columns:
        out["stock_id"] = ""

    if "name" not in out.columns:
        out["name"] = ""

    if "group" not in out.columns:
        out["group"] = ""

    for col, default in NUMERIC_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    out["stock_id"] = out["stock_id"].astype(str)
    out["name"] = out["name"].astype(str)
    out["group"] = out["group"].astype(str)

    for col, default in NUMERIC_DEFAULTS.items():
        out[col] = out[col].fillna(default)

    return out


def is_excluded_stock(name: str, group: str) -> bool:
    name_text = str(name or "").upper()
    group_text = str(group or "")

    if any(keyword.upper() in name_text for keyword in EXCLUDED_NAME_KEYWORDS):
        return True

    if any(keyword in group_text for keyword in EXCLUDED_GROUP_KEYWORDS):
        return True

    return False


def classify_theme(name: str, group: str) -> str:
    text = f"{name} {group}".upper()

    for theme, keywords in THEME_RULES.items():
        for keyword in keywords:
            if keyword.upper() in text:
                return theme

    return "其他"


def add_exclusion_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_excluded"] = out.apply(
        lambda x: is_excluded_stock(str(x["name"]), str(x["group"])),
        axis=1,
    )
    return out


def sort_stage1_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if out.empty:
        return out

    out["sort_turnover"] = pd.to_numeric(out["turnover_100m"], errors="coerce").fillna(0)
    out["sort_volume_ratio"] = pd.to_numeric(out["volume_ratio"], errors="coerce").fillna(0)
    out["sort_rsi_bonus"] = np.where((out["rsi"] >= 45) & (out["rsi"] <= 78), 1, 0)
    out["sort_trend_bonus"] = np.where(
        (out["close"] > out["ma20"]) | (out["close"] > out["ma60"]) | (out["macd"] > out["macd_signal"]),
        1,
        0,
    )

    out = out.sort_values(
        ["sort_turnover", "sort_volume_ratio", "sort_rsi_bonus", "sort_trend_bonus"],
        ascending=False,
    ).head(SECOND_SCAN_LIMIT).copy()

    return out.drop(columns=["sort_turnover", "sort_volume_ratio", "sort_rsi_bonus", "sort_trend_bonus"])


def first_stage_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_market_columns(df)
    out = add_exclusion_flag(out)

    log_count("Raw universe", out)

    universe = out.loc[~out["is_excluded"]].copy()
    log_count("After exclude ETF/financial", universe)

    universe = universe.loc[
        (universe["close"] >= MIN_PRICE)
        & (universe["close"] <= MAX_PRICE)
    ].copy()
    log_count("After price range", universe)

    if universe.empty:
        return universe

    # 先檢查最常出問題的欄位分布
    print(
        "[SCAN] Dist stats | turnover>=min:",
        int((universe["turnover_100m"] >= MIN_TURNOVER_100M).sum()),
        "| volume_ratio>=min:",
        int((universe["volume_ratio"] >= MIN_VOLUME_RATIO).sum()),
        "| volume_avg20>=min:",
        int((universe["volume_avg20"] >= MIN_AVG_VOLUME_LOT).sum()),
        "| above_ma20_or_ma60:",
        int(((universe["close"] > universe["ma20"]) | (universe["close"] > universe["ma60"])).sum()),
        "| within_ma20_distance:",
        int((universe["pct_from_ma20"] <= MAX_DISTANCE_FROM_MA20).sum()),
    )

    liquidity_gate = universe.loc[
        (universe["turnover_100m"] >= max(MIN_TURNOVER_100M * 0.7, 1.5))
        & (universe["volume_avg20"] >= max(MIN_AVG_VOLUME_LOT * 0.6, 80))
    ].copy()
    log_count("Stage1 liquidity gate", liquidity_gate)

    normal_gate = liquidity_gate.loc[
        (liquidity_gate["volume_ratio"] >= max(min(MIN_VOLUME_RATIO, 1.0), 0.95))
        & (
            (liquidity_gate["close"] > liquidity_gate["ma20"])
            | (liquidity_gate["close"] > liquidity_gate["ma60"])
            | (liquidity_gate["macd"] > liquidity_gate["macd_signal"])
        )
        & (liquidity_gate["pct_from_ma20"] <= max(MAX_DISTANCE_FROM_MA20, 12))
    ].copy()
    log_count("Stage1 normal gate", normal_gate)

    if not normal_gate.empty:
        return sort_stage1_candidates(normal_gate)

    relaxed_gate_1 = liquidity_gate.loc[
        (liquidity_gate["volume_ratio"] >= 0.9)
        & (
            (liquidity_gate["close"] > liquidity_gate["ma20"] * 0.985)
            | (liquidity_gate["close"] > liquidity_gate["ma60"] * 0.985)
            | (liquidity_gate["macd_hist"] > -0.03)
            | (liquidity_gate["close"] > liquidity_gate["boll_mid"])
        )
        & (liquidity_gate["pct_from_ma20"] <= max(MAX_DISTANCE_FROM_MA20 + 8, 20))
    ].copy()
    log_count("Stage1 relaxed gate 1", relaxed_gate_1)

    if not relaxed_gate_1.empty:
        print("[SCAN] Fallback used: relaxed gate 1")
        return sort_stage1_candidates(relaxed_gate_1)

    relaxed_gate_2 = universe.loc[
        (universe["turnover_100m"] >= max(MIN_TURNOVER_100M * 0.45, 1.0))
        & (universe["volume_avg20"] >= max(MIN_AVG_VOLUME_LOT * 0.4, 50))
        & (universe["volume_ratio"] >= 0.8)
    ].copy()
    log_count("Stage1 relaxed gate 2", relaxed_gate_2)

    if not relaxed_gate_2.empty:
        print("[SCAN] Fallback used: relaxed gate 2")
        return sort_stage1_candidates(relaxed_gate_2)

    emergency_pool = universe.sort_values(
        ["turnover_100m", "volume_ratio", "volume_avg20"],
        ascending=False,
    ).head(min(SECOND_SCAN_LIMIT, 80)).copy()
    log_count("Stage1 emergency pool", emergency_pool)

    if not emergency_pool.empty:
        print("[SCAN] Emergency pool used: please inspect first-stage thresholds or source fields")
        return emergency_pool

    return universe.head(0).copy()


def safe_merge_external(
    base_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    required_cols: list[str],
    source_name: str,
) -> pd.DataFrame:
    out = base_df.copy()

    if ext_df.empty:
        print(f"[SCAN] {source_name} empty, fill zero")
        for col in required_cols:
            if col != "stock_id":
                out[col] = 0.0
        return out

    ext = ext_df.copy()
    if "stock_id" not in ext.columns:
        print(f"[SCAN] {source_name} missing stock_id, fill zero")
        for col in required_cols:
            if col != "stock_id":
                out[col] = 0.0
        return out

    ext["stock_id"] = ext["stock_id"].astype(str)

    for col in required_cols:
        if col not in ext.columns:
            print(f"[SCAN] {source_name} missing col {col}, fill zero")
            if col != "stock_id":
                ext[col] = 0.0

    out = out.merge(ext[required_cols], on="stock_id", how="left")

    for col in required_cols:
        if col != "stock_id":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


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

    print("[SCAN] Fetch broker data")
    broker = get_broker_data(stock_ids)
    print(f"[SCAN] Broker rows: {len(broker)}")

    out = safe_merge_external(
        out,
        broker,
        ["stock_id", "main_force_10d", "broker_buy_5d"],
        "broker",
    )

    print("[SCAN] Fetch revenue data")
    revenue = get_revenue_data(stock_ids)
    print(f"[SCAN] Revenue rows: {len(revenue)}")

    out = safe_merge_external(
        out,
        revenue,
        ["stock_id", "revenue_yoy", "revenue_mom"],
        REVENUE_CSV,
    )

    fill_zero_cols = [
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
    for col in fill_zero_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    volume_base = pd.to_numeric(out["volume"], errors="coerce").replace(0, np.nan)

    out["institution_force"] = (out["foreign_buy"] + out["trust_buy"]) / volume_base
    out["trust_force"] = out["trust_buy"] / volume_base

    out["institution_force"] = out["institution_force"].replace([np.inf, -np.inf], 0).fillna(0)
    out["trust_force"] = out["trust_force"].replace([np.inf, -np.inf], 0).fillna(0)

    print("[SCAN] merge_external_data done")
    return out


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
        score += 14
    elif investment_buy_days >= 3:
        score += 10
    elif investment_buy_days >= 2:
        score += 6

    if foreign_buy_days >= 5:
        score += 10
    elif foreign_buy_days >= 3:
        score += 7
    elif foreign_buy_days >= 2:
        score += 4

    if dealer_buy_days >= 5:
        score += 4
    elif dealer_buy_days >= 3:
        score += 2
    elif dealer_buy_days >= 2:
        score += 1

    if institution_force >= 0.12:
        score += 12
    elif institution_force >= 0.08:
        score += 8
    elif institution_force >= 0.05:
        score += 4

    if trust_force >= 0.08:
        score += 10
    elif trust_force >= 0.05:
        score += 6
    elif trust_force >= 0.03:
        score += 3

    if 1 <= trust_holding_pct <= 3:
        score += 8
    elif 3 < trust_holding_pct <= 8:
        score += 5
    elif trust_holding_pct > 15:
        score -= 6

    if estimated_inst_cost > 0 and close > 0:
        diff_pct = abs(close - estimated_inst_cost) / estimated_inst_cost * 100
        if diff_pct <= 3:
            score += 8
        elif diff_pct <= 5:
            score += 4

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
        score += 10
    if close > p60 > 0:
        score += 15
    if volume_ratio >= 2:
        score += 5
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

    return out


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_market_columns(df)

    for col in [
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
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["theme"] = out.apply(lambda x: classify_theme(str(x["name"]), str(x["group"])), axis=1)

    out["institution_score"] = out.apply(calc_institution_score, axis=1)
    out["main_force_score"] = out.apply(calc_main_force_score, axis=1)
    out["broker_score"] = out.apply(calc_broker_score, axis=1)
    out["breakout_score"] = out.apply(calc_breakout_score, axis=1)
    out["revenue_score"] = out.apply(calc_revenue_score, axis=1)

    out["tech_score"] = 0.0
    out.loc[out["close"] > out["ma20"], "tech_score"] += 5
    out.loc[out["close"] > out["ma60"], "tech_score"] += 8
    out.loc[out["ma20"] > out["ma60"], "tech_score"] += 5

    out.loc[(out["volume_ratio"] >= 1.1) & (out["volume_ratio"] < 1.3), "tech_score"] += 4
    out.loc[(out["volume_ratio"] >= 1.3) & (out["volume_ratio"] < 1.8), "tech_score"] += 7
    out.loc[out["volume_ratio"] >= 1.8, "tech_score"] += 9

    out.loc[(out["turnover_100m"] >= 5) & (out["turnover_100m"] < 10), "tech_score"] += 5
    out.loc[out["turnover_100m"] >= 10, "tech_score"] += 8

    out.loc[(out["rsi"] >= 50) & (out["rsi"] <= 72), "tech_score"] += 6
    out.loc[(out["rsi"] > 72) & (out["rsi"] <= 78), "tech_score"] += 3
    out.loc[out["rsi"] > 78, "tech_score"] -= 2
    out.loc[out["rsi"] < 35, "tech_score"] -= 2

    out.loc[out["macd"] > out["macd_signal"], "tech_score"] += 6
    out.loc[out["macd_hist"] > 0, "tech_score"] += 4
    out.loc[out["close"] > out["boll_mid"], "tech_score"] += 4
    out.loc[out["close"] > out["boll_upper"], "tech_score"] += 3
    out.loc[out["obv_trend"] > 0, "tech_score"] += 6

    out["score_starting"] = 0.0
    out.loc[(out["close"] > out["ma20"]) & (out["close"] <= out["ma20"] * 1.08), "score_starting"] += 8
    out.loc[(out["volume_ratio"] >= 1.1) & (out["volume_ratio"] < 1.3), "score_starting"] += 6
    out.loc[(out["volume_ratio"] >= 1.3) & (out["volume_ratio"] < 1.8), "score_starting"] += 3
    out.loc[out["macd"] > out["macd_signal"], "score_starting"] += 4
    out.loc[(out["rsi"] >= 50) & (out["rsi"] <= 68), "score_starting"] += 5
    out.loc[(out["rsi"] > 68) & (out["rsi"] <= 74), "score_starting"] += 2
    out.loc[out["institution_force"] >= 0.05, "score_starting"] += 4
    out.loc[out["turnover_100m"] >= 5, "score_starting"] += 3
    out.loc[out["close"] > out["platform_high_20d"], "score_starting"] += 4
    out.loc[out["pct_from_ma20"] > 8, "score_starting"] -= 6
    out.loc[out["rsi"] > 78, "score_starting"] -= 6

    out["score_second_wave"] = 0.0
    out.loc[out["close"] > out["ma20"] * 1.03, "score_second_wave"] += 5
    out.loc[out["close"] > out["ma60"], "score_second_wave"] += 4
    out.loc[out["obv_trend"] > 0, "score_second_wave"] += 5
    out.loc[out["macd_hist"] > 0, "score_second_wave"] += 5
    out.loc[(out["rsi"] >= 60) & (out["rsi"] <= 78), "score_second_wave"] += 5
    out.loc[(out["rsi"] > 78) & (out["rsi"] <= 84), "score_second_wave"] += 2
    out.loc[out["investment_buy_days"] >= 3, "score_second_wave"] += 4
    out.loc[out["foreign_buy_days"] >= 3, "score_second_wave"] += 3
    out.loc[out["main_force_score"] >= 6, "score_second_wave"] += 4
    out.loc[out["broker_score"] >= 5, "score_second_wave"] += 3
    out.loc[out["pct_from_ma20"] > 15, "score_second_wave"] -= 5

    out["score_total"] = (
        out["tech_score"]
        + out["institution_score"]
        + out["main_force_score"]
        + out["broker_score"]
        + out["breakout_score"]
        + out["revenue_score"]
    )

    starting_bias = (
        out["score_starting"]
        + np.where(out["pct_from_ma20"] <= 8, 3, 0)
        + np.where(out["volume_ratio"] <= 1.35, 2, 0)
        - np.where(out["rsi"] >= 75, 4, 0)
    )

    second_wave_bias = (
        out["score_second_wave"]
        + np.where(out["close"] > out["ma60"], 2, 0)
        + np.where(out["investment_buy_days"] >= 3, 2, 0)
        + np.where(out["broker_score"] >= 5, 2, 0)
        + np.where(out["rsi"] >= 70, 1, 0)
    )

    out["radar_tag"] = np.where(starting_bias >= second_wave_bias, "剛啟動", "可能第二波")

    out = calculate_support_resistance(out)

    if "trade_warning" not in out.columns:
        out["trade_warning"] = ""
    if "is_restricted" not in out.columns:
        out["is_restricted"] = False

    return out


def build_reason_and_targets(row: pd.Series) -> dict:
    reasons: list[str] = []

    if row.get("radar_tag") == "剛啟動":
        reasons.append("型態偏剛啟動，偏向較早期轉強")
    else:
        reasons.append("型態偏第二波，屬強勢整理後再攻候選")

    if row.get("institution_score", 0) >= 12:
        reasons.append("法人籌碼加分明顯")
    elif row.get("institution_score", 0) >= 6:
        reasons.append("法人買盤有延續")

    if row.get("main_force_score", 0) >= 10:
        reasons.append("主力近10日累積明顯")
    elif row.get("main_force_score", 0) >= 6:
        reasons.append("主力有偏多痕跡")

    if row.get("broker_score", 0) >= 8:
        reasons.append("分點買盤偏強")
    elif row.get("broker_score", 0) >= 5:
        reasons.append("分點有短線加分")

    if row.get("breakout_score", 0) >= 15:
        reasons.append("突破中期平台高點")
    elif row.get("breakout_score", 0) >= 10:
        reasons.append("突破短期平台高點")

    if row.get("revenue_score", 0) >= 10:
        reasons.append("營收成長動能強")
    elif row.get("revenue_score", 0) >= 5:
        reasons.append("營收成長具支撐")

    if row.get("volume_ratio", 0) >= 2:
        reasons.append("量比放大，資金關注提高")
    elif row.get("volume_ratio", 0) >= 1.3:
        reasons.append("量能優於均值")
    elif row.get("volume_ratio", 0) >= 1.1:
        reasons.append("量能剛開始轉強")

    if row.get("close", 0) > row.get("ma20", 0) > 0:
        reasons.append("站上 MA20")
    if row.get("close", 0) > row.get("ma60", 0) > 0:
        reasons.append("站上 MA60")
    if row.get("macd", 0) > row.get("macd_signal", 0):
        reasons.append("MACD 偏多")
    if row.get("obv_trend", 0) > 0:
        reasons.append("OBV 走升")

    short_reasons = reasons[:4]
    reason_text = "；".join(short_reasons) if short_reasons else "符合模型條件"

    return {
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

    out = df.copy()
    out["score"] = out["score_total"]
    out["tag"] = out["radar_tag"]

    records = dataframe_to_records(out)

    for row in records:
        extra = build_reason_and_targets(pd.Series(row))
        row.update(extra)

    return records


def build_payload(raw_df: pd.DataFrame, analyzed_df: pd.DataFrame) -> dict:
    top30_df = analyzed_df.head(TOP30_COUNT).copy()
    watchlist_df = analyzed_df.iloc[TOP30_COUNT:TOP30_COUNT + WATCHLIST_COUNT].copy()

    starting_df = analyzed_df[analyzed_df["radar_tag"] == "剛啟動"].sort_values(
        ["score_starting", "score_total", "turnover_100m"],
        ascending=False,
    ).head(TOP30_COUNT).copy()

    second_wave_df = analyzed_df[analyzed_df["radar_tag"] == "可能第二波"].sort_values(
        ["score_second_wave", "score_total", "turnover_100m"],
        ascending=False,
    ).head(TOP30_COUNT).copy()

    broker_track_df = analyzed_df.sort_values(
        ["broker_score", "main_force_score", "score_total"],
        ascending=False,
    ).head(BROKER_TRACK_COUNT).copy()

    risk_overheated_df = analyzed_df[
        (analyzed_df["volume_ratio"] >= 2.5) & (analyzed_df["rsi"] >= 80)
    ].head(20).copy()

    high_turnover_df = analyzed_df.sort_values(
        ["turnover_100m", "volume_ratio"],
        ascending=False,
    ).head(20).copy()

    payload = {
        "summary": {
            "market_scanned": int(len(raw_df)),
            "selected": int(len(analyzed_df)),
            "starting_count": int(len(starting_df)),
            "second_wave_count": int(len(second_wave_df)),
            "overheated_count": int(len(risk_overheated_df)),
        },
        "top30": build_table_rows(top30_df),
        "watchlist": build_table_rows(watchlist_df),
        "starting": build_table_rows(starting_df),
        "second_wave": build_table_rows(second_wave_df),
        "broker_track": build_table_rows(broker_track_df),
        "overheated": build_table_rows(risk_overheated_df),
        "high_turnover": build_table_rows(high_turnover_df),
        "all_selected": build_table_rows(analyzed_df),
    }
    return payload


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
        ["score_total", "turnover_100m", "volume_ratio"],
        ascending=False,
    ).reset_index(drop=True)
    print(f"[SCAN] Final analyzed rows: {len(analyzed_df)}")

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
    payload = build_payload(raw_df, analyzed_df)

    if save:
        if progress_callback:
            progress_callback(
                {
                    "stage": "save",
                    "percent": 99,
                    "message": "儲存掃描結果中",
                }
            )

        print("[SCAN] Step 6: save snapshot")
        save_snapshot(payload, raw_df, analyzed_df)
        print("[SCAN] Snapshot saved")

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
