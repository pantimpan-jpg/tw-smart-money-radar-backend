from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from fetcher import fetch_market_snapshot_parallel
from storage import save_snapshot

ProgressCallback = Callable[[dict[str, Any]], None] | None


THEME_BY_STOCK_ID: dict[str, str] = {
    # 記憶體
    "2408": "記憶體",
    "2344": "記憶體",
    "2337": "記憶體",
    "3260": "記憶體",
    "8299": "記憶體",
    "4967": "記憶體",
    # ABF / 載板
    "3189": "ABF載板",
    "8046": "ABF載板",
    "3037": "ABF載板",
    # 散熱
    "3017": "散熱",
    "3324": "散熱",
    "6230": "散熱",
    "3653": "散熱",
    # PCB / CCL
    "2383": "PCB",
    "2368": "PCB",
    "5469": "PCB",
    "6274": "PCB",
    "4958": "PCB",
    "6191": "PCB",
    "8046": "ABF載板",
    "3037": "ABF載板",
    "3189": "ABF載板",
    # 半導體設備 / 測試
    "3131": "半導體設備",
    "3413": "半導體設備",
    "3583": "半導體設備",
    "8028": "半導體設備",
    "6640": "半導體設備",
    # 矽光子 / 光通訊
    "4979": "光通訊",
    "3081": "光通訊",
    "3363": "光通訊",
    "3234": "光通訊",
    "4908": "光通訊",
    # 網通
    "5388": "網通",
    "3596": "網通",
    "2345": "網通",
    "2419": "網通",
    "6285": "網通",
    # AI 伺服器 / ODM
    "3231": "AI伺服器",
    "6669": "AI伺服器",
    "2356": "AI伺服器",
    "2382": "AI伺服器",
    "2317": "AI伺服器",
    "6669": "AI伺服器",
    # 機殼 / 組裝
    "8210": "機殼",
    "3013": "機殼",
    "6117": "機殼",
    # 電源 / UPS
    "2308": "電源",
    "3023": "電源",
    "6121": "電源",
    # 被動元件 / MLCC
    "2327": "MLCC",
    "6173": "MLCC",
    "2492": "被動元件",
    "3042": "被動元件",
    # 面板
    "2409": "面板",
    "3481": "面板",
    # 塑化
    "1301": "塑化",
    "1303": "塑化",
    "1326": "塑化",
    "6505": "塑化",
    # 重電 / 電力設備
    "1519": "重電",
    "1503": "重電",
    "1513": "重電",
    "1584": "重電",
    "4526": "重電",
    # 工具機 / 自動化
    "2049": "工具機",
    "4510": "工具機",
    "1536": "自動化",
    "1597": "自動化",
    # 航運
    "2603": "航運",
    "2609": "航運",
    "2615": "航運",
    # 鋼鐵
    "2002": "鋼鐵",
    "2027": "鋼鐵",
    "2034": "鋼鐵",
    # 軍工 / 航太
    "2634": "軍工航太",
    "8222": "軍工航太",
    "4572": "軍工航太",
    # 生技
    "4743": "生技",
    "6446": "生技",
    "6472": "生技",
}

THEME_GROUP_RULES: list[tuple[list[str], str]] = [
    (["半導體"], "半導體"),
    (["電子零組件"], "電子零組件"),
    (["通信網路"], "網通"),
    (["光電"], "光電"),
    (["電腦及週邊"], "AI伺服器"),
    (["其他電子"], "其他電子"),
    (["塑膠"], "塑化"),
    (["電機機械"], "電機機械"),
    (["鋼鐵"], "鋼鐵"),
    (["航運"], "航運"),
    (["生技"], "生技"),
]

THEME_NAME_RULES: list[tuple[list[str], str]] = [
    (["南亞科", "華邦電", "旺宏", "威剛", "創見", "品安"], "記憶體"),
    (["欣興", "南電", "景碩"], "ABF載板"),
    (["奇鋐", "雙鴻", "力致", "建準", "泰碩"], "散熱"),
    (["金像電", "台光電", "台燿", "高技", "敬鵬", "華通", "欣興"], "PCB"),
    (["聯亞", "上詮", "光聖", "眾達", "波若威"], "光通訊"),
    (["智邦", "中磊", "啟碁", "明泰", "正文"], "網通"),
    (["英業達", "緯創", "廣達", "技嘉", "微星", "仁寶", "鴻海"], "AI伺服器"),
    (["國巨", "華新科", "禾伸堂", "旺詮"], "被動元件"),
    (["群創", "友達"], "面板"),
    (["南亞", "台塑", "台化"], "塑化"),
    (["亞力", "士電", "華城", "中興電"], "重電"),
    (["東台", "高鋒", "瀧澤科"], "工具機"),
]


def _update_progress(progress_callback: ProgressCallback, payload: dict[str, Any]) -> None:
    if progress_callback:
        progress_callback(payload)


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        x = float(value)
        if pd.isna(x):
            return 0.0
        return x
    except Exception:
        return 0.0


def _clean_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _assign_theme(row: pd.Series) -> str:
    stock_id = str(row.get("stock_id", ""))
    name = str(row.get("name", "") or "")
    group = str(row.get("group", "") or "")

    if stock_id in THEME_BY_STOCK_ID:
        return THEME_BY_STOCK_ID[stock_id]

    for keywords, theme in THEME_NAME_RULES:
        if any(k in name for k in keywords):
            return theme

    for keywords, theme in THEME_GROUP_RULES:
        if any(k in group for k in keywords):
            return theme

    return "其他"


def _compute_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "low_5d" not in out.columns:
        out["low_5d"] = 0.0
    if "low_20d" not in out.columns:
        out["low_20d"] = 0.0
    if "high_5d" not in out.columns:
        out["high_5d"] = 0.0
    if "high_20d" not in out.columns:
        out["high_20d"] = 0.0
    if "ma20" not in out.columns:
        out["ma20"] = 0.0
    if "ma60" not in out.columns:
        out["ma60"] = 0.0

    out["near_support"] = np.where(
        out["ma20"] > 0,
        np.maximum(out["low_5d"], out["ma20"] * 0.985),
        out["low_5d"],
    )
    out["strong_support"] = np.where(
        out["ma60"] > 0,
        np.minimum(out["low_20d"], out["ma60"]),
        out["low_20d"],
    )
    out["near_resistance"] = out["high_5d"]
    out["strong_resistance"] = np.where(
        out["high_20d"] > 0,
        np.maximum(out["high_20d"], out["high_5d"] * 1.03),
        out["high_5d"] * 1.03,
    )

    out["near_support"] = out["near_support"].round(2)
    out["strong_support"] = out["strong_support"].round(2)
    out["near_resistance"] = out["near_resistance"].round(2)
    out["strong_resistance"] = out["strong_resistance"].round(2)

    return out


def _first_filter_with_fallback(raw_df: pd.DataFrame, progress_callback: ProgressCallback) -> pd.DataFrame:
    df = raw_df.copy()

    _update_progress(
        progress_callback,
        {
            "scan_running": True,
            "stage": "filter",
            "percent": 90,
            "message": f"進入第一層快篩，原始候選 {len(df)} 檔",
        },
    )

    # 第一層：正常偏嚴
    f1 = df[
        (df["close"] >= 10)
        & (df["turnover_100m"] >= 1.0)
        & (
            ((df["volume_ratio"] >= 1.15) & (df["pct_from_ma20"] > -8))
            | (df["turnover_100m"] >= 8.0)
        )
    ].copy()

    _update_progress(
        progress_callback,
        {
            "scan_running": True,
            "stage": "filter",
            "percent": 91,
            "message": f"第一層快篩 A 後剩 {len(f1)} 檔",
        },
    )

    if len(f1) > 0:
        return f1

    # 第二層：放寬
    f2 = df[
        (df["close"] >= 8)
        & (df["turnover_100m"] >= 0.5)
        & (
            ((df["volume_ratio"] >= 1.0) & (df["pct_from_ma20"] > -12))
            | (df["turnover_100m"] >= 5.0)
        )
    ].copy()

    _update_progress(
        progress_callback,
        {
            "scan_running": True,
            "stage": "filter",
            "percent": 92,
            "message": f"第一層快篩 B 後剩 {len(f2)} 檔",
        },
    )

    if len(f2) > 0:
        return f2

    # 第三層：再放寬，至少不要整個炸掉
    f3 = df[
        (df["close"] >= 5)
        & (
            (df["turnover_100m"] >= 0.3)
            | (df["volume_ratio"] >= 0.9)
        )
    ].copy()

    _update_progress(
        progress_callback,
        {
            "scan_running": True,
            "stage": "filter",
            "percent": 93,
            "message": f"第一層快篩 C 後剩 {len(f3)} 檔",
        },
    )

    if len(f3) > 0:
        return f3

    # 最後保底：直接取成交值前 300 檔，不要讓整個掃描失敗
    fallback = df.sort_values(["turnover_100m", "volume_ratio"], ascending=False).head(300).copy()

    _update_progress(
        progress_callback,
        {
            "scan_running": True,
            "stage": "filter",
            "percent": 94,
            "message": f"快篩全空，改用成交值 fallback {len(fallback)} 檔",
        },
    )

    return fallback


def _classify_starting_second_wave(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 剛啟動：靠近 20 日高點 / 量比有放大 / 還沒離均線太遠 / MACD 轉強
    starting_mask = (
        (out["turnover_100m"] >= 0.8)
        & (out["volume_ratio"] >= 1.1)
        & (out["pct_from_ma20"] >= -3)
        & (out["pct_from_ma20"] <= 18)
        & (out["macd_hist"] >= -0.02)
        & (out["close"] >= out["ma20"] * 0.98)
        & (out["close"] >= out["platform_high_20d"] * 0.96)
    )

    # 第二波：第一波後整理，再轉強，但不要離月線太遠
    second_wave_mask = (
        (out["turnover_100m"] >= 0.8)
        & (out["volume_ratio"] >= 0.95)
        & (out["pct_from_ma20"] >= -4)
        & (out["pct_from_ma20"] <= 25)
        & (out["close"] >= out["ma20"] * 0.99)
        & (out["close"] < out["platform_high_60d"] * 1.02)
        & (out["close"] >= out["high_20d"] * 0.92)
        & (
            (out["rsi"] >= 52)
            | (out["macd_hist"] > 0)
            | (out["obv_trend"] > 0)
        )
    )

    out["radar_tag"] = np.select(
        [starting_mask, second_wave_mask],
        ["剛啟動", "可能第二波"],
        default="觀察",
    )
    out["tag"] = out["radar_tag"]

    return out


def _score_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 法人分數：目前先用量價 / 趨勢強弱近似，等你之後接更完整法人資料再替換
    out["institution_score"] = (
        np.clip(out["turnover_100m"] * 2.5, 0, 40)
        + np.clip((out["volume_ratio"] - 1) * 18, 0, 25)
        + np.where(out["obv_trend"] > 0, 10, 0)
    )

    # 券商分數：先保留 0，等後面接分點資料
    out["broker_score"] = 0.0

    # 主力分數：技術結構 + 價格位置
    out["main_force_score"] = (
        np.where(out["close"] >= out["ma20"], 10, 0)
        + np.where(out["close"] >= out["ma60"], 8, 0)
        + np.where(out["macd_hist"] > 0, 10, 0)
        + np.where((out["rsi"] >= 52) & (out["rsi"] <= 78), 10, 0)
        + np.where(out["pct_from_ma20"].between(-2, 15), 12, 0)
        + np.where(out["radar_tag"] == "剛啟動", 14, 0)
        + np.where(out["radar_tag"] == "可能第二波", 16, 0)
    )

    out["score_total"] = (
        out["institution_score"]
        + out["broker_score"]
        + out["main_force_score"]
    ).round(1)

    return out


def _build_payload(selected_df: pd.DataFrame, raw_df: pd.DataFrame) -> dict[str, Any]:
    selected_df = selected_df.copy()
    raw_df = raw_df.copy()

    top30 = (
        selected_df.sort_values(["score_total", "turnover_100m"], ascending=False)
        .head(30)
        .to_dict(orient="records")
    )

    starting_df = selected_df[selected_df["radar_tag"] == "剛啟動"].copy()
    second_wave_df = selected_df[selected_df["radar_tag"] == "可能第二波"].copy()

    starting = (
        starting_df.sort_values(["score_total", "turnover_100m"], ascending=False)
        .head(30)
        .to_dict(orient="records")
    )

    second_wave = (
        second_wave_df.sort_values(["score_total", "turnover_100m"], ascending=False)
        .head(30)
        .to_dict(orient="records")
    )

    watchlist_df = (
        selected_df.sort_values(["score_total", "turnover_100m"], ascending=False)
        .head(20)
        .copy()
    )
    watchlist = watchlist_df.to_dict(orient="records")

    high_turnover = (
        selected_df.sort_values(["turnover_100m", "score_total"], ascending=False)
        .head(20)
        .to_dict(orient="records")
    )

    overheated_df = selected_df[
        (selected_df["rsi"] >= 82)
        & (selected_df["pct_from_ma20"] >= 18)
        & (selected_df["turnover_100m"] >= 3)
    ].copy()

    overheated = (
        overheated_df.sort_values(["turnover_100m", "score_total"], ascending=False)
        .head(20)
        .to_dict(orient="records")
    )

    broker_track: list[dict[str, Any]] = []

    payload = {
        "summary": {
            "market_scanned": int(len(raw_df)),
            "selected": int(len(selected_df)),
            "starting_count": int(len(starting_df)),
            "second_wave_count": int(len(second_wave_df)),
            "overheated_count": int(len(overheated_df)),
        },
        "top30": top30,
        "watchlist": watchlist,
        "starting": starting,
        "second_wave": second_wave,
        "broker_track": broker_track,
        "overheated": overheated,
        "high_turnover": high_turnover,
        "all_selected": selected_df.to_dict(orient="records"),
    }
    return payload


def run_scan(save: bool = True, progress_callback: ProgressCallback = None) -> dict[str, Any]:
    _update_progress(
        progress_callback,
        {
            "scan_running": True,
            "stage": "fetch",
            "percent": 0,
            "message": "開始掃描",
            "processed": 0,
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "last_scan_error": None,
        },
    )

    raw_df = fetch_market_snapshot_parallel(progress_callback=progress_callback)

    if raw_df.empty:
        raise ValueError("抓完市場資料後是空的")

    raw_df = raw_df.copy()

    numeric_cols = [
        "close",
        "change_pct",
        "volume",
        "turnover_100m",
        "volume_ratio",
        "volume_avg20",
        "ma20",
        "ma60",
        "pct_from_ma20",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "boll_mid",
        "boll_upper",
        "obv_trend",
        "platform_high_20d",
        "platform_high_60d",
        "low_5d",
        "low_20d",
        "high_5d",
        "high_20d",
    ]
    raw_df = _clean_numeric(raw_df, numeric_cols)

    if "trade_warning" not in raw_df.columns:
        raw_df["trade_warning"] = ""
    if "is_restricted" not in raw_df.columns:
        raw_df["is_restricted"] = False

    raw_df["theme"] = raw_df.apply(_assign_theme, axis=1)
    raw_df = _compute_support_resistance(raw_df)

    filtered_df = _first_filter_with_fallback(raw_df, progress_callback)

    if filtered_df.empty:
        # 這裡理論上不會發生，因為前面已有 fallback
        filtered_df = raw_df.sort_values(["turnover_100m"], ascending=False).head(100).copy()

    _update_progress(
        progress_callback,
        {
            "scan_running": True,
            "stage": "classify",
            "percent": 95,
            "message": f"開始分類剛啟動 / 第二波，候選 {len(filtered_df)} 檔",
        },
    )

    classified_df = _classify_starting_second_wave(filtered_df)
    selected_df = classified_df[classified_df["radar_tag"].isin(["剛啟動", "可能第二波"])].copy()

    if selected_df.empty:
        # 保底：用成交值前 60 檔，再用較寬鬆規則指定分類
        fallback_df = filtered_df.sort_values(["turnover_100m", "volume_ratio"], ascending=False).head(60).copy()
        fallback_df["radar_tag"] = np.where(
            fallback_df["pct_from_ma20"] <= 8,
            "剛啟動",
            "可能第二波",
        )
        fallback_df["tag"] = fallback_df["radar_tag"]
        selected_df = fallback_df.copy()

        _update_progress(
            progress_callback,
            {
                "scan_running": True,
                "stage": "classify",
                "percent": 96,
                "message": f"原分類為空，啟用保底分類 {len(selected_df)} 檔",
            },
        )

    selected_df = _score_rows(selected_df)

    # 重新排序
    selected_df = selected_df.sort_values(
        ["score_total", "turnover_100m", "volume_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # 統一補齊欄位，避免前端缺欄位
    required_cols_with_defaults: dict[str, Any] = {
        "stock_id": "",
        "name": "",
        "group": "",
        "theme": "其他",
        "close": 0.0,
        "change_pct": 0.0,
        "volume": 0.0,
        "turnover_100m": 0.0,
        "volume_ratio": 0.0,
        "institution_score": 0.0,
        "broker_score": 0.0,
        "main_force_score": 0.0,
        "score_total": 0.0,
        "radar_tag": "觀察",
        "tag": "觀察",
        "near_support": 0.0,
        "strong_support": 0.0,
        "near_resistance": 0.0,
        "strong_resistance": 0.0,
        "trade_warning": "",
        "is_restricted": False,
    }
    for col, default_value in required_cols_with_defaults.items():
        if col not in selected_df.columns:
            selected_df[col] = default_value
        if col not in raw_df.columns:
            raw_df[col] = default_value

    payload = _build_payload(selected_df, raw_df)

    if save:
        save_snapshot(payload=payload, raw_df=raw_df, selected_df=selected_df)

    _update_progress(
        progress_callback,
        {
            "scan_running": False,
            "stage": "completed",
            "percent": 100,
            "message": f"掃描完成，入選 {len(selected_df)} 檔",
            "processed": int(len(raw_df)),
            "total": int(len(raw_df)),
            "success": int(len(raw_df)),
            "failed": 0,
            "skipped": 0,
            "last_scan_error": None,
        },
    )

    return payload
