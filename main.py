from __future__ import annotations

import threading
import html
import re
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import APP_NAME, LATEST_MARKET_CSV
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


def is_empty_scan_exception(exc: Exception | str) -> bool:
    text = str(exc).strip()
    empty_markers = [
        "第一層快篩後沒有股票",
        "快篩後沒有股票",
        "沒有股票通過第一層快篩",
    ]
    return any(marker in text for marker in empty_markers)


def result_indicates_empty(result: Any) -> bool:
    if not isinstance(result, dict):
        return False

    data = result.get("data")
    if not isinstance(data, dict):
        return False

    summary = data.get("summary")
    if isinstance(summary, dict):
        selected = summary.get("selected")
        try:
            if selected is not None and int(selected) == 0:
                return True
        except Exception:
            pass

    all_selected = data.get("all_selected")
    if isinstance(all_selected, list) and len(all_selected) == 0:
        return True

    return False


def mark_scan_empty(message: str) -> None:
    global last_scan_error, last_finished_at
    last_scan_error = None
    last_finished_at = datetime.now(timezone.utc).isoformat()
    update_scan_progress(
        scan_running=False,
        percent=100,
        stage="empty",
        message=message,
        last_scan_error=None,
        last_updated=last_finished_at,
    )


def mark_scan_completed() -> None:
    global last_scan_error, last_finished_at
    last_scan_error = None
    last_finished_at = datetime.now(timezone.utc).isoformat()
    update_scan_progress(
        scan_running=False,
        percent=100,
        stage="completed",
        message="掃描完成",
        last_updated=last_finished_at,
        last_scan_error=None,
    )


def mark_scan_error(message: str) -> None:
    global last_scan_error, last_finished_at
    last_scan_error = message
    last_finished_at = datetime.now(timezone.utc).isoformat()
    update_scan_progress(
        scan_running=False,
        percent=100,
        stage="error",
        message=f"掃描失敗：{message}",
        last_scan_error=message,
        last_updated=last_finished_at,
    )


def _run_scan_job() -> None:
    global scan_running, last_scan_error

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
        last_updated=datetime.now(timezone.utc).isoformat(),
    )

    try:
        result = run_scan(save=True, progress_callback=progress_callback)

        current = get_scan_progress_snapshot()
        current_stage = str(current.get("stage") or "").strip().lower()

        if current_stage == "empty" or result_indicates_empty(result):
            mark_scan_empty("第一層快篩後沒有股票")
        else:
            mark_scan_completed()

        if isinstance(result, dict):
            print(f"[SCAN] Result keys: {list(result.keys())}")

    except Exception as e:
        err = str(e).strip()

        if is_empty_scan_exception(err):
            mark_scan_empty("第一層快篩後沒有股票")
        else:
            mark_scan_error(err)
            print(traceback.format_exc())

    finally:
        scan_running = False


def get_snapshot_or_404() -> dict[str, Any]:
    snapshot = load_snapshot()
    if not snapshot:
        raise HTTPException(status_code=404, detail="尚未產生掃描結果，請先執行掃描")
    return snapshot


def load_market_snapshot_df() -> pd.DataFrame:
    path = Path(LATEST_MARKET_CSV)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)

    return df


def get_all_selected_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return snapshot.get("data", {}).get("all_selected", [])


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


def safe_int(value: Any) -> int | None:
    num = safe_float(value)
    if num is None:
        return None
    return int(round(num))


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None

def _calc_main_force_score_value(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if v >= 30000:
        return 15.0
    if v >= 15000:
        return 10.0
    if v >= 5000:
        return 6.0
    return 0.0


def _calc_broker_score_value(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if v >= 10000:
        return 12.0
    if v >= 5000:
        return 8.0
    if v >= 2000:
        return 5.0
    return 0.0


def _get_financial_statement_df(stock_id: str) -> pd.DataFrame:
    start_date = (date.today() - timedelta(days=2200)).isoformat()
    df = finmind_get("TaiwanStockFinancialStatements", stock_id, start_date)
    if df.empty or "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value_num"] = pd.to_numeric(out["value"], errors="coerce")

    if "type" not in out.columns:
        out["type"] = ""
    if "origin_name" not in out.columns:
        out["origin_name"] = ""

    out["type"] = out["type"].astype(str)
    out["origin_name"] = out["origin_name"].astype(str)

    out = out.dropna(subset=["date", "value_num"]).sort_values("date")
    return out


def _match_financial_value(
    sub: pd.DataFrame,
    *,
    type_codes: list[str],
    origin_keywords: list[str],
) -> float | None:
    if sub.empty:
        return None

    exact = sub[sub["type"].isin(type_codes)].copy()
    if not exact.empty:
        return safe_float(exact.iloc[0]["value_num"])

    fuzzy = sub[
        sub["origin_name"].apply(lambda x: any(keyword in str(x) for keyword in origin_keywords))
    ].copy()
    if not fuzzy.empty:
        return safe_float(fuzzy.iloc[0]["value_num"])

    return None


def _quarter_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}Q{((ts.month - 1) // 3) + 1}"


def _get_live_broker_df(stock_id: str) -> pd.DataFrame:
    try:
        return get_broker_data([stock_id])
    except Exception:
        return pd.DataFrame()


def _enrich_overview_with_live_broker(overview: dict[str, Any], broker_df: pd.DataFrame) -> None:
    if broker_df.empty:
        return

    net10_col = "net_10d" if "net_10d" in broker_df.columns else "net"
    net5_col = "net_5d" if "net_5d" in broker_df.columns else "net"

    net10 = pd.to_numeric(broker_df[net10_col], errors="coerce").fillna(0)
    net5 = pd.to_numeric(broker_df[net5_col], errors="coerce").fillna(0)

    main_force_10d = float(net10[net10 > 0].nlargest(5).sum())
    broker_buy_5d = float(net5[net5 > 0].nlargest(3).sum())

    overview["main_force_score"] = _calc_main_force_score_value(main_force_10d)
    overview["broker_score"] = _calc_broker_score_value(broker_buy_5d)

def find_stock_context(snapshot: dict[str, Any], stock_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    stock_id = str(stock_id)

    selected_rows = get_all_selected_rows(snapshot)
    for row in selected_rows:
        if str(row.get("stock_id")) == stock_id:
            return row, {
                "in_selected": True,
                "source": "selected",
            }

    market_df = load_market_snapshot_df()
    if not market_df.empty and "stock_id" in market_df.columns:
        sub = market_df[market_df["stock_id"].astype(str) == stock_id]
        if not sub.empty:
            row = sub.iloc[0].to_dict()
            return row, {
                "in_selected": False,
                "source": "market_snapshot",
            }

    raise HTTPException(status_code=404, detail="找不到該股票")


def build_stock_meta(stock_id: str, *, in_selected: bool, source: str) -> dict[str, Any]:
    return {
        "stock_id": stock_id,
        "in_selected": in_selected,
        "source": source,
        "has_scoring": in_selected,
        "label": "今日榜單股票" if in_selected else "未進今日榜單",
    }


def build_overview_from_row(row: dict[str, Any], *, in_selected: bool) -> dict[str, Any]:
    trade_warning = safe_str(row.get("trade_warning")) or safe_str(row.get("restriction_note"))

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
        "institution_score": safe_float(row.get("institution_score")) if in_selected else None,
        "broker_score": safe_float(row.get("broker_score")) if in_selected else None,
        "main_force_score": safe_float(row.get("main_force_score")) if in_selected else None,
        "final_score": safe_float(row.get("score_total") or row.get("score")) if in_selected else None,
        "technical_tag": row.get("tag") if in_selected else None,
        "radar_tag": row.get("radar_tag") if in_selected else None,
        "near_support": safe_float(row.get("near_support")),
        "strong_support": safe_float(row.get("strong_support")),
        "near_pressure": safe_float(row.get("near_resistance") or row.get("near_pressure")),
        "strong_pressure": safe_float(row.get("strong_resistance") or row.get("strong_pressure")),
        "trade_warning": trade_warning,
        "is_restricted": bool(row.get("is_restricted", False)),
    }

COMPANY_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "8046": {
        "industry_role": "ABF載板代表廠商 / 載板供應鏈中游",
        "one_liner": "IC 載板與 ABF 載板供應商",
        "positioning": "高階封裝與高速運算載板供應鏈",
        "core_tech": ["ABF載板", "高密度互連", "細線路製程"],
        "main_applications": ["AI/HPC", "伺服器", "網通晶片"],
    },
    "3037": {
        "industry_role": "ABF載板代表廠商 / 載板供應鏈中游",
        "one_liner": "IC 載板與高階 ABF 載板供應商",
        "positioning": "先進封裝與高階運算載板供應鏈",
        "core_tech": ["ABF載板", "IC載板", "高層數載板"],
        "main_applications": ["AI/HPC", "伺服器", "網通"],
    },
    "3189": {
        "industry_role": "ABF載板代表廠商 / 載板供應鏈中游",
        "one_liner": "IC 載板與 ABF 載板供應商",
        "positioning": "高階封裝與高速傳輸載板供應鏈",
        "core_tech": ["ABF載板", "封裝載板", "高速訊號傳輸"],
        "main_applications": ["AI/HPC", "伺服器", "消費電子"],
    },
    "3504": {
        "industry_role": "光學元件代表廠商 / 光學供應鏈中游",
        "one_liner": "投影光學與精密鏡頭模組廠",
        "positioning": "精密光學零組件與光學模組供應商",
        "core_tech": ["精密光學設計", "鏡頭模組", "投影光學系統"],
        "main_applications": ["投影設備", "安控鏡頭", "車載光學"],
    },
    "4979": {
        "industry_role": "光通訊代表廠商 / 光模組供應鏈中游",
        "one_liner": "高速光通訊模組與傳輸元件廠",
        "positioning": "資料中心高速互連供應鏈",
        "core_tech": ["高速光模組", "光收發", "矽光子互連"],
        "main_applications": ["AI資料中心", "高速網通", "雲端伺服器"],
    },
    "3163": {
        "industry_role": "光通訊代表廠商 / 光模組供應鏈中游",
        "one_liner": "光通訊模組與被動元件供應商",
        "positioning": "高速光纖傳輸供應鏈",
        "core_tech": ["光纖被動元件", "WDM", "高速傳輸模組"],
        "main_applications": ["資料中心", "電信網路", "高速互連"],
    },
    "6442": {
        "industry_role": "光通訊代表廠商 / 光模組供應鏈中游",
        "one_liner": "高速光通訊模組供應商",
        "positioning": "AI伺服器與資料中心高速互連供應鏈",
        "core_tech": ["高速光模組", "光電轉換", "高速傳輸封裝"],
        "main_applications": ["AI資料中心", "雲端伺服器", "高速網通"],
    },
    "3017": {
        "industry_role": "散熱代表廠商 / 散熱供應鏈中游",
        "one_liner": "AI 伺服器散熱廠",
        "positioning": "高功耗運算與伺服器散熱供應鏈",
        "core_tech": ["均熱板", "熱導管", "液冷/氣冷散熱模組"],
        "main_applications": ["AI伺服器", "GPU伺服器", "高功耗運算"],
    },
    "3324": {
        "industry_role": "散熱代表廠商 / 散熱供應鏈中游",
        "one_liner": "AI 伺服器與高階散熱模組供應商",
        "positioning": "高效能運算散熱供應鏈",
        "core_tech": ["均熱板", "熱導管", "高階散熱模組"],
        "main_applications": ["AI伺服器", "顯示卡", "高效能運算"],
    },
    "2383": {
        "industry_role": "PCB/CCL代表廠商 / 材料供應鏈中游",
        "one_liner": "高階 PCB/CCL 材料供應商",
        "positioning": "高速傳輸與伺服器板材供應鏈",
        "core_tech": ["高速低損耗材料", "CCL", "高階板材"],
        "main_applications": ["AI伺服器", "交換器", "高頻高速板"],
    },
    "6274": {
        "industry_role": "PCB/CCL代表廠商 / 材料供應鏈中游",
        "one_liner": "高階 CCL 與伺服器板材供應商",
        "positioning": "高速傳輸與網通板材供應鏈",
        "core_tech": ["低損耗材料", "CCL", "高頻高速板材"],
        "main_applications": ["交換器", "伺服器", "網通設備"],
    },
    "5274": {
        "industry_role": "BMC晶片代表廠商 / 半導體供應鏈中游",
        "one_liner": "伺服器管理晶片與 BMC 晶片供應商",
        "positioning": "AI伺服器管理控制供應鏈",
        "core_tech": ["BMC", "遠端管理", "伺服器控制晶片"],
        "main_applications": ["AI伺服器", "雲端資料中心", "企業伺服器"],
    },
}

THEME_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "ABF載板": {
        "industry_role": "ABF載板代表廠商 / 載板供應鏈中游",
        "one_liner": "IC 載板與 ABF 載板供應商",
        "positioning": "先進封裝與高階運算載板供應鏈",
        "core_tech": ["ABF載板", "封裝載板", "高密度互連"],
        "main_applications": ["AI/HPC", "伺服器", "網通晶片"],
    },
    "CPO/矽光子": {
        "industry_role": "矽光子 / 光通訊代表廠商 / 光模組供應鏈中游",
        "one_liner": "高速光通訊與矽光子互連供應商",
        "positioning": "AI資料中心高速互連供應鏈",
        "core_tech": ["矽光子", "CPO", "高速光模組"],
        "main_applications": ["AI資料中心", "雲端伺服器", "高速交換器"],
    },
    "光通訊": {
        "industry_role": "光通訊代表廠商 / 光模組供應鏈中游",
        "one_liner": "高速光通訊模組供應商",
        "positioning": "資料中心與電信高速傳輸供應鏈",
        "core_tech": ["光模組", "光收發", "高速傳輸"],
        "main_applications": ["資料中心", "電信網路", "高速網通"],
    },
    "AI伺服器/BMC": {
        "industry_role": "AI伺服器控制晶片代表廠商 / 半導體供應鏈中游",
        "one_liner": "伺服器管理晶片與控制晶片供應商",
        "positioning": "資料中心與AI伺服器管理供應鏈",
        "core_tech": ["BMC", "伺服器管理控制", "遠端管理晶片"],
        "main_applications": ["AI伺服器", "資料中心", "企業伺服器"],
    },
    "AI伺服器/OEM": {
        "industry_role": "AI伺服器代表廠商 / 組裝供應鏈下游",
        "one_liner": "AI 伺服器整機與系統供應商",
        "positioning": "雲端與高效能運算系統供應鏈",
        "core_tech": ["伺服器整機", "系統整合", "高密度運算平台"],
        "main_applications": ["AI伺服器", "資料中心", "雲端運算"],
    },
    "散熱": {
        "industry_role": "散熱代表廠商 / 散熱供應鏈中游",
        "one_liner": "AI 伺服器散熱廠",
        "positioning": "高功耗運算散熱供應鏈",
        "core_tech": ["均熱板", "熱導管", "散熱模組"],
        "main_applications": ["AI伺服器", "GPU伺服器", "高效能運算"],
    },
    "PCB/CCL": {
        "industry_role": "PCB/CCL代表廠商 / 材料供應鏈中游",
        "one_liner": "高階 PCB/CCL 材料供應商",
        "positioning": "高速傳輸板材供應鏈",
        "core_tech": ["CCL", "低損耗材料", "高階板材"],
        "main_applications": ["AI伺服器", "交換器", "高頻高速板"],
    },
    "網通": {
        "industry_role": "網通代表廠商 / 網通供應鏈中下游",
        "one_liner": "網路通訊設備與模組供應商",
        "positioning": "高速傳輸與資料交換供應鏈",
        "core_tech": ["交換器", "路由器", "高速傳輸模組"],
        "main_applications": ["資料中心", "企業網路", "高速網通"],
    },
    "低軌衛星": {
        "industry_role": "低軌衛星概念代表廠商 / 通訊供應鏈中游",
        "one_liner": "衛星通訊零組件與天線供應商",
        "positioning": "衛星與長距通訊供應鏈",
        "core_tech": ["衛星通訊", "高頻天線", "射頻模組"],
        "main_applications": ["低軌衛星", "通訊設備", "長距傳輸"],
    },
    "MLCC/被動元件": {
        "industry_role": "被動元件代表廠商 / 零組件供應鏈中游",
        "one_liner": "MLCC 與被動元件供應商",
        "positioning": "電子零組件基礎供應鏈",
        "core_tech": ["MLCC", "電容", "高可靠度被動元件"],
        "main_applications": ["手機", "車用電子", "伺服器"],
    },
    "半導體設備/測試": {
        "industry_role": "半導體設備代表廠商 / 設備供應鏈中游",
        "one_liner": "半導體設備與測試方案供應商",
        "positioning": "晶圓製程與封測設備供應鏈",
        "core_tech": ["測試設備", "晶圓製程設備", "封測設備"],
        "main_applications": ["晶圓製造", "封裝測試", "先進製程"],
    },
    "記憶體": {
        "industry_role": "記憶體代表廠商 / 半導體供應鏈中游",
        "one_liner": "記憶體與儲存解決方案供應商",
        "positioning": "DRAM / NAND 與儲存供應鏈",
        "core_tech": ["DRAM", "NAND", "儲存控制"],
        "main_applications": ["伺服器", "PC", "消費電子"],
    },
}

INDUSTRY_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "光電": {
        "industry_role": "光電供應鏈中游",
        "one_liner": "光學與光電零組件供應商",
        "positioning": "精密光學與光電供應鏈",
        "core_tech": ["光學設計", "光電模組", "精密製程"],
        "main_applications": ["鏡頭", "顯示", "感測"],
    },
    "電子零組件": {
        "industry_role": "電子零組件供應鏈中游",
        "one_liner": "電子零組件供應商",
        "positioning": "基礎電子零組件供應鏈",
        "core_tech": ["零組件製造", "高可靠度元件", "精密製程"],
        "main_applications": ["伺服器", "車用", "消費電子"],
    },
    "電腦及週邊": {
        "industry_role": "電腦及週邊供應鏈中下游",
        "one_liner": "電腦與系統周邊供應商",
        "positioning": "系統整合與周邊設備供應鏈",
        "core_tech": ["系統整合", "板卡/周邊", "硬體設計"],
        "main_applications": ["PC", "伺服器", "企業設備"],
    },
    "半導體": {
        "industry_role": "半導體供應鏈中游",
        "one_liner": "半導體元件與晶片供應商",
        "positioning": "IC 設計 / 製造 / 應用供應鏈",
        "core_tech": ["IC設計", "先進製程", "高效能晶片"],
        "main_applications": ["伺服器", "網通", "消費電子"],
    },
}


def _ensure_list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = safe_str(item)
            if text:
                out.append(text)
        return out

    text = safe_str(value)
    if not text:
        return []

    parts = re.split(r"[、,/|]+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _merge_profile(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    if not extra:
        return base

    for key in ["industry_role", "one_liner", "positioning"]:
        if extra.get(key):
            base[key] = extra[key]

    for key in ["core_tech", "main_applications"]:
        merged = _ensure_list_text(base.get(key)) + _ensure_list_text(extra.get(key))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in merged:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        base[key] = deduped[:8]

    return base


def build_company_profile(stock_id: str, overview: dict[str, Any]) -> dict[str, Any]:
    industry = safe_str(overview.get("industry")) or "待補"
    theme = safe_str(overview.get("theme")) or None

    profile: dict[str, Any] = {
        "stock_id": stock_id,
        "industry": industry,
        "theme": theme,
        "industry_role": None,
        "one_liner": None,
        "positioning": None,
        "core_tech": [],
        "main_applications": [],
    }

    profile = _merge_profile(profile, INDUSTRY_PROFILE_DEFAULTS.get(industry))
    if theme:
        profile = _merge_profile(profile, THEME_PROFILE_DEFAULTS.get(theme))
    profile = _merge_profile(profile, COMPANY_PROFILE_OVERRIDES.get(stock_id))

    if not profile.get("industry_role"):
        profile["industry_role"] = f"{theme or industry} 供應鏈代表廠商"

    if not profile.get("one_liner"):
        if theme:
            profile["one_liner"] = f"{theme} 相關供應商"
        else:
            profile["one_liner"] = f"{industry} 相關供應商"

    if not profile.get("positioning"):
        if theme and industry:
            profile["positioning"] = f"{theme} / {industry} 供應鏈"
        elif theme:
            profile["positioning"] = f"{theme} 供應鏈"
        else:
            profile["positioning"] = f"{industry} 供應鏈"

    if not profile.get("core_tech"):
        profile["core_tech"] = ["待補"]

    if not profile.get("main_applications"):
        profile["main_applications"] = ["待補"]

    return profile


def build_price_history_180(price_df: pd.DataFrame) -> list[dict[str, Any]]:
    if price_df.empty or "date" not in price_df.columns or "close" not in price_df.columns:
        return []

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").tail(180)

    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "close": float(row["close"]),
            }
        )
    return out

def get_recent_price_df(stock_id: str) -> pd.DataFrame:
    # 180 個交易日大概需要接近 9~12 個月的日曆天數，這裡多抓一點再 tail(180)
    start_date = (date.today() - timedelta(days=320)).isoformat()
    return finmind_get("TaiwanStockPrice", stock_id, start_date)


def enrich_overview_with_price(overview: dict[str, Any], stock_id: str) -> pd.DataFrame:
    df = get_recent_price_df(stock_id)
    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    latest = df.iloc[-1]

    latest_close = safe_float(latest.get("close"))
    latest_open = safe_float(latest.get("open"))

    if latest_close is not None:
        overview["close"] = latest_close

    if latest_close is not None and latest_open is not None and latest_open != 0:
        overview["change"] = latest_close - latest_open
        overview["change_pct"] = (latest_close - latest_open) / latest_open * 100

    latest_volume = safe_float(latest.get("Trading_Volume"))
    if latest_volume is not None:
        overview["volume"] = latest_volume

    latest_money = safe_float(latest.get("Trading_money"))
    if latest_money is not None:
        overview["turnover_100m"] = latest_money / 100000000

    if len(df) >= 20:
        close_series = pd.to_numeric(df["close"], errors="coerce")
        low_series = pd.to_numeric(df.get("min", df.get("low", df["close"])), errors="coerce")
        high_series = pd.to_numeric(df.get("max", df.get("high", df["close"])), errors="coerce")

        ma20 = close_series.rolling(20).mean().iloc[-1]
        ma60 = close_series.rolling(60).mean().iloc[-1] if len(df) >= 60 else np.nan
        low_5 = low_series.tail(5).min()
        low_20 = low_series.tail(20).min()
        high_5 = high_series.tail(5).max()
        high_20 = high_series.tail(20).max()

        near_support = max(low_5, ma20) if pd.notna(ma20) else low_5
        strong_support = min(low_20, ma60) if pd.notna(ma60) else low_20

        overview["near_support"] = round(float(near_support), 2) if pd.notna(near_support) else overview.get("near_support")
        overview["strong_support"] = round(float(strong_support), 2) if pd.notna(strong_support) else overview.get("strong_support")
        overview["near_pressure"] = round(float(high_5), 2) if pd.notna(high_5) else overview.get("near_pressure")
        overview["strong_pressure"] = round(float(max(high_20, high_5 * 1.03)), 2) if pd.notna(high_20) and pd.notna(high_5) else overview.get("strong_pressure")

    return df


def enrich_overview_with_per(stock_id: str, overview: dict[str, Any]) -> None:
    start_date = (date.today() - timedelta(days=60)).isoformat()
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


def _normalize_institutional_name(name: str) -> str:
    text = str(name)
    if "外資" in text or "Foreign" in text:
        return "foreign"
    if "投信" in text or "Investment_Trust" in text:
        return "trust"
    if "自營商" in text or "Dealer" in text:
        return "dealer"
    return "other"


def get_institutional_summary(stock_id: str) -> dict[str, Any]:
    start_date = (date.today() - timedelta(days=50)).isoformat()
    df = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_date)

    empty_result = {
        "latest_date": None,
        "foreign": {"d1": None, "d5": None, "d10": None, "d20": None},
        "trust": {"d1": None, "d5": None, "d10": None, "d20": None},
        "dealer": {"d1": None, "d5": None, "d10": None, "d20": None},
    }

    if df.empty or "date" not in df.columns:
        return empty_result

    name_col = None
    for candidate in ["name", "investors", "institutional_investors"]:
        if candidate in df.columns:
            name_col = candidate
            break
    if not name_col or "buy" not in df.columns or "sell" not in df.columns:
        return empty_result

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return empty_result

    df["kind"] = df[name_col].astype(str).map(_normalize_institutional_name)
    df["buy_num"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0)
    df["sell_num"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0)
    df["net_lot"] = (df["buy_num"] - df["sell_num"]) / 1000.0

    grouped = (
        df[df["kind"].isin(["foreign", "trust", "dealer"])]
        .groupby(["date", "kind"], as_index=False)["net_lot"]
        .sum()
        .sort_values("date")
    )

    latest_date = grouped["date"].max()
    all_dates = sorted(grouped["date"].drop_duplicates().tolist())

    def calc_window(kind: str, window: int) -> int | None:
        sub = grouped[grouped["kind"] == kind].copy()
        if sub.empty:
            return None
        latest_sub = sub[sub["date"] <= latest_date].sort_values("date")
        if latest_sub.empty:
            return None
        if window == 1:
            return int(round(latest_sub.iloc[-1]["net_lot"]))
        recent_dates = all_dates[-window:]
        recent = latest_sub[latest_sub["date"].isin(recent_dates)]
        if recent.empty:
            return None
        return int(round(recent["net_lot"].sum()))

    return {
        "latest_date": latest_date.date().isoformat() if pd.notna(latest_date) else None,
        "foreign": {
            "d1": calc_window("foreign", 1),
            "d5": calc_window("foreign", 5),
            "d10": calc_window("foreign", 10),
            "d20": calc_window("foreign", 20),
        },
        "trust": {
            "d1": calc_window("trust", 1),
            "d5": calc_window("trust", 5),
            "d10": calc_window("trust", 10),
            "d20": calc_window("trust", 20),
        },
        "dealer": {
            "d1": calc_window("dealer", 1),
            "d5": calc_window("dealer", 5),
            "d10": calc_window("dealer", 10),
            "d20": calc_window("dealer", 20),
        },
    }


def enrich_overview_with_institutional(stock_id: str, overview: dict[str, Any]) -> dict[str, Any]:
    summary = get_institutional_summary(stock_id)
    overview["foreign_buy_sell"] = summary["foreign"]["d1"]
    overview["investment_trust_buy_sell"] = summary["trust"]["d1"]
    overview["dealer_buy_sell"] = summary["dealer"]["d1"]
    return summary


def get_margin_summary(stock_id: str) -> dict[str, Any]:
    start_date = (date.today() - timedelta(days=50)).isoformat()
    df = finmind_get("TaiwanStockMarginPurchaseShortSale", stock_id, start_date)

    empty_result = {
        "latest_date": None,
        "margin_balance": None,
        "short_balance": None,
        "margin": {"d1": None, "d5": None, "d10": None, "d20": None},
        "short": {"d1": None, "d5": None, "d10": None, "d20": None},
    }

    if df.empty or "date" not in df.columns:
        return empty_result

    margin_col = "MarginPurchaseTodayBalance" if "MarginPurchaseTodayBalance" in df.columns else None
    short_col = "ShortSaleTodayBalance" if "ShortSaleTodayBalance" in df.columns else None
    margin_yesterday_col = "MarginPurchaseYesterdayBalance" if "MarginPurchaseYesterdayBalance" in df.columns else None
    short_yesterday_col = "ShortSaleYesterdayBalance" if "ShortSaleYesterdayBalance" in df.columns else None

    if margin_col is None or short_col is None:
        return empty_result

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return empty_result

    df["margin_balance"] = pd.to_numeric(df[margin_col], errors="coerce")
    df["short_balance"] = pd.to_numeric(df[short_col], errors="coerce")

    if margin_yesterday_col:
        df["margin_yesterday_balance"] = pd.to_numeric(df[margin_yesterday_col], errors="coerce")
    else:
        df["margin_yesterday_balance"] = np.nan

    if short_yesterday_col:
        df["short_yesterday_balance"] = pd.to_numeric(df[short_yesterday_col], errors="coerce")
    else:
        df["short_yesterday_balance"] = np.nan

    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    def calc_change(col: str, days: int, yesterday_col: str | None = None) -> int | None:
        sub = df[["date", col] + ([yesterday_col] if yesterday_col else [])].dropna(subset=[col]).copy()
        if sub.empty:
            return None

        latest_value = sub.iloc[-1][col]

        if days == 1:
            if yesterday_col and yesterday_col in sub.columns and pd.notna(sub.iloc[-1][yesterday_col]):
                return int(round(latest_value - sub.iloc[-1][yesterday_col]))
            if len(sub) < 2:
                return None
            return int(round(latest_value - sub.iloc[-2][col]))

        if len(sub) <= days:
            return None

        base_value = sub.iloc[-(days + 1)][col]
        return int(round(latest_value - base_value))

    latest_date = df.iloc[-1]["date"]
    latest_margin_balance = df.iloc[-1]["margin_balance"]
    latest_short_balance = df.iloc[-1]["short_balance"]

    return {
        "latest_date": latest_date.date().isoformat() if pd.notna(latest_date) else None,
        "margin_balance": int(round(latest_margin_balance)) if pd.notna(latest_margin_balance) else None,
        "short_balance": int(round(latest_short_balance)) if pd.notna(latest_short_balance) else None,
        "margin": {
            "d1": calc_change("margin_balance", 1, "margin_yesterday_balance"),
            "d5": calc_change("margin_balance", 5),
            "d10": calc_change("margin_balance", 10),
            "d20": calc_change("margin_balance", 20),
        },
        "short": {
            "d1": calc_change("short_balance", 1, "short_yesterday_balance"),
            "d5": calc_change("short_balance", 5),
            "d10": calc_change("short_balance", 10),
            "d20": calc_change("short_balance", 20),
        },
    }


def enrich_overview_with_margin(stock_id: str, overview: dict[str, Any]) -> dict[str, Any]:
    summary = get_margin_summary(stock_id)
    overview["margin_balance"] = summary["margin_balance"]
    overview["short_balance"] = summary["short_balance"]
    overview["margin_change"] = summary["margin"]["d1"]
    overview["short_change"] = summary["short"]["d1"]
    return summary


def get_revenues_list(stock_id: str) -> list[dict[str, Any]]:
    start_date = (date.today() - timedelta(days=550)).isoformat()
    df = finmind_get("TaiwanStockMonthRevenue", stock_id, start_date)
    if df.empty or "date" not in df.columns:
        return []

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
    df = _get_financial_statement_df(stock_id)
    if df.empty:
        return []

    eps_df = df[
        (df["type"] == "EPS")
        | df["origin_name"].apply(lambda x: "基本每股盈餘" in str(x) or "每股盈餘" in str(x))
    ].copy()

    if eps_df.empty:
        return []

    eps_df = (
        eps_df.sort_values(["date"])
        .drop_duplicates(subset=["date"], keep="first")
        .copy()
    )

    eps_df["yoy"] = eps_df["value_num"].pct_change(4) * 100
    eps_df["qoq"] = eps_df["value_num"].pct_change(1) * 100

    out: list[dict[str, Any]] = []
    for _, row in eps_df.tail(12).sort_values("date", ascending=False).iterrows():
        ts = pd.Timestamp(row["date"])
        out.append(
            {
                "quarter": _quarter_label(ts),
                "eps": safe_float(row.get("value_num")),
                "yoy": safe_float(row.get("yoy")),
                "qoq": safe_float(row.get("qoq")),
            }
        )
    return out

def get_dividends_list(stock_id: str) -> list[dict[str, Any]]:
    start_date = (date.today() - timedelta(days=2600)).isoformat()
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
    df = _get_financial_statement_df(stock_id)
    if df.empty:
        return []

    wanted_specs = {
        "revenue": {
            "type_codes": ["Revenue"],
            "origin_keywords": ["營業收入", "收入"],
        },
        "gross_profit": {
            "type_codes": ["GrossProfit"],
            "origin_keywords": ["營業毛利", "毛利"],
        },
        "operating_income": {
            "type_codes": ["OperatingIncome"],
            "origin_keywords": ["營業利益", "營業利益（損失）", "營業利益（損失）淨額"],
        },
        "net_income": {
            "type_codes": ["IncomeAfterTaxes"],
            "origin_keywords": ["本期淨利", "本期稅後淨利", "本期淨利（淨損）"],
        },
        "eps": {
            "type_codes": ["EPS"],
            "origin_keywords": ["基本每股盈餘", "每股盈餘"],
        },
    }

    out: list[dict[str, Any]] = []

    for d in sorted(df["date"].dropna().unique(), reverse=True)[:8]:
        sub = df[df["date"] == d].copy()
        ts = pd.Timestamp(d)

        item: dict[str, Any] = {
            "period": _quarter_label(ts),
            "revenue": None,
            "gross_profit": None,
            "operating_income": None,
            "net_income": None,
            "eps": None,
        }

        for field, spec in wanted_specs.items():
            item[field] = _match_financial_value(
                sub,
                type_codes=spec["type_codes"],
                origin_keywords=spec["origin_keywords"],
            )

        out.append(item)

    return out


def get_news_list(stock_id: str) -> list[dict[str, Any]]:
    def clean_html_text(value: Any) -> str | None:
        text = safe_str(value)
        if not text:
            return None
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    # FinMind v4 的 TaiwanStockNews 一次只提供一天資料，所以逐日抓
    for offset in range(0, 20):
        query_date = (date.today() - timedelta(days=offset)).isoformat()

        try:
            df = finmind_get("TaiwanStockNews", stock_id, query_date)
        except Exception:
            continue

        if df.empty:
            continue

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date", ascending=False)

        for _, row in df.iterrows():
            title = clean_html_text(row.get("title")) or "待補"
            link = safe_str(row.get("link")) or ""

            dedupe_key = (title, link)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            published_at = None
            row_date = row.get("date")
            if pd.notna(row_date):
                published_at = pd.Timestamp(row_date).strftime("%Y-%m-%d %H:%M:%S")

            rows.append(
                {
                    "id": None,
                    "title": title,
                    "summary": clean_html_text(row.get("description")),
                    "source": clean_html_text(row.get("source")),
                    "published_at": published_at,
                    "url": safe_str(row.get("link")),
                }
            )

            if len(rows) >= 20:
                break

        if len(rows) >= 20:
            break

    rows.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return rows

def get_broker_branches_list(stock_id: str) -> list[dict[str, Any]]:
    df = _get_live_broker_df(stock_id)
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
        scheduler.add_job(
            _run_scan_job,
            CronTrigger(day_of_week="mon-fri", hour=17, minute=0),
            id="daily_scan_tw",
            replace_existing=True,
        )
        scheduler.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    if scheduler.running:
        scheduler.shutdown()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": APP_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scheduler_running": scheduler.running,
        "schedule": "Mon-Fri 17:00 Asia/Taipei",
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

    thread = threading.Thread(target=_run_scan_job, daemon=True)
    thread.start()

    return {
        "ok": True,
        "message": "已啟動背景掃描，請稍後查看 /api/scan/status",
    }


@app.get("/api/scan/status")
def get_scan_status() -> dict[str, Any]:
    snapshot = load_snapshot()
    snapshot_updated_at = snapshot.get("updated_at") if snapshot else None

    status = get_scan_progress_snapshot()
    stage = str(status.get("stage") or "").strip().lower()

    status["scan_running"] = scan_running

    if stage == "completed":
        status["last_updated"] = snapshot_updated_at or last_finished_at or status.get("last_updated")
        status["last_scan_error"] = None
    elif stage == "empty":
        status["last_updated"] = last_finished_at or status.get("last_updated") or snapshot_updated_at
        status["last_scan_error"] = None
    elif stage == "error":
        status["last_updated"] = last_finished_at or status.get("last_updated") or snapshot_updated_at
        status["last_scan_error"] = last_scan_error or status.get("last_scan_error")
    else:
        status["last_updated"] = snapshot_updated_at or last_finished_at or status.get("last_updated")
        status["last_scan_error"] = last_scan_error or status.get("last_scan_error")

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
    row, ctx = find_stock_context(snapshot, stock_id)

    in_selected = bool(ctx["in_selected"])
    overview = build_overview_from_row(row, in_selected=in_selected)

    enrich_overview_with_price(overview, stock_id)
    enrich_overview_with_per(stock_id, overview)
    institutional_summary = enrich_overview_with_institutional(stock_id, overview)
    margin_summary = enrich_overview_with_margin(stock_id, overview)

    broker_df = _get_live_broker_df(stock_id)
    _enrich_overview_with_live_broker(overview, broker_df)

    broker_branches: list[dict[str, Any]] = []
    if not broker_df.empty:
        for _, broker_row in broker_df.head(20).iterrows():
            broker_branches.append(
                {
                    "broker_name": safe_str(broker_row.get("broker_name")) or safe_str(broker_row.get("securities_trader")) or "待補",
                    "branch_name": safe_str(broker_row.get("branch_name")),
                    "buy": safe_float(broker_row.get("buy")),
                    "sell": safe_float(broker_row.get("sell")),
                    "net": safe_float(broker_row.get("net")),
                }
            )

    return {
        "meta": build_stock_meta(stock_id, in_selected=in_selected, source=ctx["source"]),
        "overview": overview,
        "revenues": get_revenues_list(stock_id),
        "eps_list": get_eps_list(stock_id),
        "dividends": get_dividends_list(stock_id),
        "financials": get_financials_list(stock_id),
        "news": get_news_list(stock_id),
        "broker_branches": broker_branches,
        "institutional_summary": institutional_summary,
        "margin_summary": margin_summary,
    }


@app.get("/api/stocks")
def search_stocks(
    q: str = Query(default="", description="股號、名稱、題材關鍵字"),
    tag: str = Query(default=""),
    theme: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    snapshot = get_snapshot_or_404()
    selected_rows = get_all_selected_rows(snapshot)
    selected_ids = {str(r.get("stock_id")) for r in selected_rows}

    market_df = load_market_snapshot_df()
    if market_df.empty:
        base_rows = selected_rows
    else:
        base_rows = market_df.to_dict(orient="records")

    ql = q.lower().strip()
    filtered: list[dict[str, Any]] = []

    for row in base_rows:
        stock_id = str(row.get("stock_id", ""))
        stock_name = str(row.get("name", "") or row.get("stock_name", ""))
        group = str(row.get("group", ""))
        row_theme = str(row.get("theme", ""))
        trade_warning = str(row.get("trade_warning", "") or "")

        hay = f"{stock_id} {stock_name} {group} {row_theme}".lower()

        if ql and ql not in hay:
            continue

        if tag:
            row_tag = str(row.get("radar_tag", "") or row.get("tag", ""))
            if row_tag != tag:
                continue

        if theme and row_theme != theme:
            continue

        filtered.append(
            {
                "stock_id": stock_id,
                "name": stock_name,
                "close": safe_float(row.get("close")),
                "group": group or None,
                "theme": row_theme or None,
                "turnover_100m": safe_float(row.get("turnover_100m")),
                "in_selected": stock_id in selected_ids,
                "radar_tag": row.get("radar_tag") if stock_id in selected_ids else None,
                "score_total": safe_float(row.get("score_total")) if stock_id in selected_ids else None,
                "trade_warning": trade_warning or None,
            }
        )

    filtered = sorted(
        filtered,
        key=lambda x: (
            0 if x["in_selected"] else 1,
            -(x["score_total"] or 0),
            -(x["turnover_100m"] or 0),
        ),
    )

    return filtered[:limit]
