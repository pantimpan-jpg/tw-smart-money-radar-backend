import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import twstock

MAX_WORKERS = 12

def get_all_taiwan_stocks():
    return [
        info
        for _, info in twstock.codes.items()
        if getattr(info, "market", "") in ["上市", "上櫃"]
        and str(getattr(info, "code", "")).isdigit()
        and len(str(getattr(info, "code", ""))) == 4
    ]


def fetch_market_snapshot_parallel(progress_every: int = 20, batch_size: int = 20) -> pd.DataFrame:
    print("[FETCH] Loading Taiwan stock list...")
    codes = get_all_taiwan_stocks()
    total = len(codes)

    print(f"[FETCH] Total stock universe: {total}")
    print(f"[FETCH] Using max_workers={MAX_WORKERS}")
    print(f"[FETCH] Using batch_size={batch_size}")

    rows: list[dict[str, Any]] = []
    success = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch = codes[batch_start: batch_start + batch_size]
        batch_end = min(batch_start + batch_size, total)

        print(f"[FETCH] Starting batch {batch_start + 1}-{batch_end}/{total}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(fetch_one_stock, info): info
                for info in batch
            }

            batch_timeout = 60
            batch_begin_time = time.time()

            while future_map and (time.time() - batch_begin_time) < batch_timeout:
                done_futures = [f for f in list(future_map.keys()) if f.done()]

                if not done_futures:
                    time.sleep(0.5)
                    continue

                for future in done_futures:
                    info = future_map.pop(future)

                    try:
                        result = future.result()
                        if result:
                            rows.append(result)
                            success += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        failed += 1
                        print(f"[FETCH][ERROR] {info.code} {info.name}: {e}")

        if future_map:
            for future, info in future_map.items():
                failed += 1
                print(f"[FETCH][TIMEOUT] {info.code} {info.name}: batch timeout")
            future_map.clear()

        elapsed = round(time.time() - start_time, 1)
        processed = success + skipped + failed
        print(
            f"[FETCH] Progress {processed}/{total} | "
            f"success={success} skipped={skipped} failed={failed} | "
            f"elapsed={elapsed}s"
        )

    total_elapsed = round(time.time() - start_time, 1)

    print("[FETCH] Completed market snapshot")
    print(f"[FETCH] Success rows: {success}")
    print(f"[FETCH] Skipped rows: {skipped}")
    print(f"[FETCH] Failed rows: {failed}")
    print(f"[FETCH] Total elapsed: {total_elapsed}s")

    if not rows:
        print("[FETCH] No rows returned")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"[FETCH] Final dataframe rows: {len(df)}")

    return df
