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


def first_stage_filter(df: pd.DataFrame) -> pd.DataFrame:
    cond = (
        (df["close"] >= MIN_PRICE)
        & (df["close"] <= MAX_PRICE)
        & (df["turnover_100m"] >= MIN_TURNOVER_100M)
        & (df["volume_ratio"] >= MIN_VOLUME_RATIO)
        & (df["volume_avg20"] >= MIN_AVG_VOLUME_LOT)
        & ((df["close"] > df["ma20"]) | (df["close"] > df["ma60"]))
        & (df["pct_from_ma20"] <= MAX_DISTANCE_FROM_MA20)
    )
    filtered = df.loc[cond].copy()
    filtered = filtered.sort_values(["turnover_100m", "volume_ratio"], ascending=False)
    return filtered.head(SECOND_SCAN_LIMIT).copy()


def classify_theme(name: str, group: str) -> str:
    text = f"{name} {group}".upper()
    pcb_keywords = ["PCB", "載板", "ABF", "銅箔", "CCL"]
    thermal_keywords = ["散熱", "均熱片", "熱導管", "風扇"]
    cpo_keywords = ["CPO", "矽光子", "光通訊", "光模組"]
    ai_keywords = ["AI", "伺服器", "GPU", "ASIC"]
    satellite_keywords = ["低軌", "衛星", "天線", "通訊"]
    if any(k.upper() in text for k in pcb_keywords):
        return "PCB"
    if any(k.upper() in text for k in thermal_keywords):
        return "散熱"
    if any(k.upper() in text for k in cpo_keywords):
        return "CPO"
    if any(k.upper() in text for k in ai_keywords):
        return "AI"
    if any(k.upper() in text for k in satellite_keywords):
        return "衛星"
    return "其他"


def merge_external_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[SCAN] merge_external_data start")

    out = df.copy()
    stock_ids = out["stock_id"].astype(str).tolist()
    print(f"[SCAN] merge_external_data stock count: {len(stock_ids)}")

    print("[SCAN] Fetch institutional data")
    institutional = get_institutional_data(stock_ids)
    print(f"[SCAN] Institutional rows: {len(institutional)}")

    if not institutional.empty:
        required = [
            "stock_id", "foreign_buy_days", "investment_buy_days", "dealer_buy_days",
            "foreign_buy", "trust_buy", "dealer_buy", "trust_holding_pct", "estimated_inst_cost",
        ]
        missing = [c for c in required if c not in institutional.columns]
        if missing:
            raise ValueError(f"{INSTITUTIONAL_CSV} / API 缺少欄位：{missing}")
        institutional["stock_id"] = institutional["stock_id"].astype(str)
        out = out.merge(institutional[required], on="stock_id", how="left")
    else:
        for col in [
            "foreign_buy_days", "investment_buy_days", "dealer_buy_days",
            "foreign_buy", "trust_buy", "dealer_buy", "trust_holding_pct", "estimated_inst_cost"
        ]:
            out[col] = 0.0

    print("[SCAN] Fetch broker data")
    broker = get_broker_data(stock_ids)
    print(f"[SCAN] Broker rows: {len(broker)}")

    if not broker.empty:
        required = ["stock_id", "main_force_10d", "broker_buy_5d"]
        missing = [c for c in required if c not in broker.columns]
        if missing:
            raise ValueError(f"broker CSV 缺少欄位：{missing}")
        broker["stock_id"] = broker["stock_id"].astype(str)
        out = out.merge(broker[required], on="stock_id", how="left")
    else:
        out["main_force_10d"] = 0.0
        out["broker_buy_5d"] = 0.0

    print("[SCAN] Fetch revenue data")
    revenue = get_revenue_data(stock_ids)
    print(f"[SCAN] Revenue rows: {len(revenue)}")

    if not revenue.empty:
        required = ["stock_id", "revenue_yoy", "revenue_mom"]
        missing = [c for c in required if c not in revenue.columns]
        if missing:
            raise ValueError(f"{REVENUE_CSV} / API 缺少欄位：{missing}")
        revenue["stock_id"] = revenue["stock_id"].astype(str)
        out = out.merge(revenue[required], on="stock_id", how="left")
    else:
        out["revenue_yoy"] = 0.0
        out["revenue_mom"] = 0.0

    fill_zero_cols = [
        "foreign_buy_days", "investment_buy_days", "dealer_buy_days",
        "foreign_buy", "trust_buy", "dealer_buy", "trust_holding_pct", "estimated_inst_cost",
        "main_force_10d", "broker_buy_5d", "revenue_yoy", "revenue_mom",
    ]
    for col in fill_zero_cols:
        out[col] = out[col].fillna(0)

    out["institution_force"] = (out["foreign_buy"] + out["trust_buy"]) / out["volume"].replace(0, np.nan)
    out["trust_force"] = out["trust_buy"] / out["volume"].replace(0, np.nan)
    out["institution_force"] = out["institution_force"].replace([np.inf, -np.inf], 0).fillna(0)
    out["trust_force"] = out["trust_force"].replace([np.inf, -np.inf], 0).fillna(0)

    print("[SCAN] merge_external_data done")
    return out


def calc_institution_score(row: pd.Series) -> float:
    score = 0.0
    if row["investment_buy_days"] >= 5:
        score += 14
    elif row["investment_buy_days"] >= 3:
        score += 10
    elif row["investment_buy_days"] >= 2:
        score += 6
    if row["foreign_buy_days"] >= 5:
        score += 10
    elif row["foreign_buy_days"] >= 3:
        score += 7
    elif row["foreign_buy_days"] >= 2:
        score += 4
    if row["dealer_buy_days"] >= 5:
        score += 4
    elif row["dealer_buy_days"] >= 3:
        score += 2
    elif row["dealer_buy_days"] >= 2:
        score += 1
    if row["institution_force"] >= 0.12:
        score += 12
    elif row["institution_force"] >= 0.08:
        score += 8
    elif row["institution_force"] >= 0.05:
        score += 4
    if row["trust_force"] >= 0.08:
        score += 10
    elif row["trust_force"] >= 0.05:
        score += 6
    elif row["trust_force"] >= 0.03:
        score += 3
    if 1 <= row["trust_holding_pct"] <= 3:
        score += 8
    elif 3 < row["trust_holding_pct"] <= 8:
        score += 5
    elif row["trust_holding_pct"] > 15:
        score -= 6
    if row["estimated_inst_cost"] > 0:
        diff_pct = abs(row["close"] - row["estimated_inst_cost"]) / row["estimated_inst_cost"] * 100
        if diff_pct <= 3:
            score += 8
        elif diff_pct <= 5:
            score += 4
    return score


def calc_main_force_score(row: pd.Series) -> float:
    if row["main_force_10d"] >= 30000:
        return 15
    if row["main_force_10d"] >= 15000:
        return 10
    if row["main_force_10d"] >= 5000:
        return 6
    return 0.0


def calc_broker_score(row: pd.Series) -> float:
    if row["broker_buy_5d"] >= 10000:
        return 12
    if row["broker_buy_5d"] >= 5000:
        return 8
    if row["broker_buy_5d"] >= 2000:
        return 5
    return 0.0


def calc_breakout_score(row: pd.Series) -> float:
    score = 0.0
    if row["close"] > row["platform_high_20d"] > 0:
        score += 10
    if row["close"] > row["platform_high_60d"] > 0:
        score += 15
    if row["volume_ratio"] >= 2:
        score += 5
    return score


def calc_revenue_score(row: pd.Series) -> float:
    score = 0.0
    if row["revenue_yoy"] >= 30:
        score += 10
    elif row["revenue_yoy"] >= 15:
        score += 6
    if row["revenue_mom"] >= 10:
        score += 5
    elif row["revenue_mom"] >= 5:
        score += 3
    return score


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tech_score"] = 0.0
    out.loc[out["close"] > out["ma20"], "tech_score"] += 5
    out.loc[out["close"] > out["ma60"], "tech_score"] += 8
    out.loc[out["ma20"] > out["ma60"], "tech_score"] += 5
    out.loc[(out["volume_ratio"] >= 1.5) & (out["volume_ratio"] < 2), "tech_score"] += 6
    out.loc[out["volume_ratio"] >= 2, "tech_score"] += 10
    out.loc[(out["turnover_100m"] >= 5) & (out["turnover_100m"] < 10), "tech_score"] += 5
    out.loc[out["turnover_100m"] >= 10, "tech_score"] += 8
    out.loc[(out["rsi"] >= 50) & (out["rsi"] <= 70), "tech_score"] += 6
    out.loc[out["rsi"] > 70, "tech_score"] -= 3
    out.loc[out["rsi"] < 35, "tech_score"] -= 2
    out.loc[out["macd"] > out["macd_signal"], "tech_score"] += 6
    out.loc[out["macd_hist"] > 0, "tech_score"] += 4
    out.loc[out["close"] > out["boll_mid"], "tech_score"] += 4
    out.loc[out["close"] > out["boll_upper"], "tech_score"] += 3
    out.loc[out["obv_trend"] > 0, "tech_score"] += 6

    out["score_starting"] = 0.0
    out.loc[(out["close"] > out["ma20"]) & (out["close"] <= out["ma20"] * 1.08), "score_starting"] += 8
    out.loc[out["volume_ratio"] >= 1.5, "score_starting"] += 5
    out.loc[out["macd"] > out["macd_signal"], "score_starting"] += 4
    out.loc[out["rsi"] < 68, "score_starting"] += 3
    out.loc[out["institution_force"] >= 0.05, "score_starting"] += 4

    out["score_second_wave"] = 0.0
    out.loc[out["close"] > out["ma20"] * 1.08, "score_second_wave"] += 8
    out.loc[out["close"] > out["ma60"], "score_second_wave"] += 4
    out.loc[out["obv_trend"] > 0, "score_second_wave"] += 5
    out.loc[out["macd_hist"] > 0, "score_second_wave"] += 5
    out.loc[out["investment_buy_days"] >= 3, "score_second_wave"] += 4

    out["theme"] = out.apply(lambda x: classify_theme(str(x["name"]), str(x["group"])), axis=1)
    out["institution_score"] = out.apply(calc_institution_score, axis=1)
    out["main_force_score"] = out.apply(calc_main_force_score, axis=1)
    out["broker_score"] = out.apply(calc_broker_score, axis=1)
    out["breakout_score"] = out.apply(calc_breakout_score, axis=1)
    out["revenue_score"] = out.apply(calc_revenue_score, axis=1)
    out["score_total"] = (
        out["tech_score"]
        + out["institution_score"]
        + out["main_force_score"]
        + out["broker_score"]
        + out["breakout_score"]
        + out["revenue_score"]
    )
    out["radar_tag"] = np.where(out["score_starting"] >= out["score_second_wave"], "剛啟動", "可能第二波")
    return out


def build_payload(raw_df: pd.DataFrame, analyzed_df: pd.DataFrame) -> dict:
    top30_df = analyzed_df.head(TOP30_COUNT).copy()
    watchlist_df = analyzed_df.iloc[TOP30_COUNT:TOP30_COUNT + WATCHLIST_COUNT].copy()
    starting_df = analyzed_df[analyzed_df["radar_tag"] == "剛啟動"].sort_values(
        ["score_starting", "score_total"], ascending=False
    ).head(TOP30_COUNT).copy()
    second_wave_df = analyzed_df[analyzed_df["radar_tag"] == "可能第二波"].sort_values(
        ["score_second_wave", "score_total"], ascending=False
    ).head(TOP30_COUNT).copy()
    broker_track_df = analyzed_df.sort_values(
        ["broker_score", "main_force_score", "score_total"], ascending=False
    ).head(BROKER_TRACK_COUNT).copy()

    risk_overheated_df = analyzed_df[
        (analyzed_df["volume_ratio"] >= 2.5) & (analyzed_df["rsi"] >= 70)
    ].head(20).copy()
    high_turnover_df = analyzed_df.sort_values(
        ["turnover_100m", "volume_ratio"], ascending=False
    ).head(20).copy()

    payload = {
        "summary": {
            "market_scanned": int(len(raw_df)),
            "selected": int(len(analyzed_df)),
            "starting_count": int(len(starting_df)),
            "second_wave_count": int(len(second_wave_df)),
            "overheated_count": int(len(risk_overheated_df)),
        },
        "top30": dataframe_to_records(top30_df),
        "watchlist": dataframe_to_records(watchlist_df),
        "starting": dataframe_to_records(starting_df),
        "second_wave": dataframe_to_records(second_wave_df),
        "broker_track": dataframe_to_records(broker_track_df),
        "overheated": dataframe_to_records(risk_overheated_df),
        "high_turnover": dataframe_to_records(high_turnover_df),
        "all_selected": dataframe_to_records(analyzed_df),
    }
    return payload


def run_scan(save: bool = True) -> dict:
    print("[SCAN] Step 1: fetch market snapshot")
    raw_df = fetch_market_snapshot_parallel()
    print(f"[SCAN] Market snapshot rows: {len(raw_df)}")

    if raw_df.empty:
        raise RuntimeError("抓不到市場資料")

    print("[SCAN] Step 2: first stage filter")
    stage1_df = first_stage_filter(raw_df)
    print(f"[SCAN] First stage selected: {len(stage1_df)}")

    if stage1_df.empty:
        raise RuntimeError("第一層快篩後沒有股票")

    print("[SCAN] Step 3: merge external data")
    stage2_input = merge_external_data(stage1_df)
    print(f"[SCAN] Stage2 rows: {len(stage2_input)}")

    print("[SCAN] Step 4: calculate scores")
    analyzed_df = calculate_scores(stage2_input)
    analyzed_df = analyzed_df.sort_values(
        ["score_total", "turnover_100m", "volume_ratio"],
        ascending=False
    ).reset_index(drop=True)
    print(f"[SCAN] Final analyzed rows: {len(analyzed_df)}")

    print("[SCAN] Step 5: build payload")
    payload = build_payload(raw_df, analyzed_df)

    if save:
        print("[SCAN] Step 6: save snapshot")
        save_snapshot(payload, raw_df, analyzed_df)
        print("[SCAN] Snapshot saved")

    print("[SCAN] run_scan completed")
    return payload
