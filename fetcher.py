import time


def fetch_market_snapshot_parallel(progress_every: int = 20) -> pd.DataFrame:
    print("[FETCH] Loading Taiwan stock list...")
    codes = get_all_taiwan_stocks()
    total = len(codes)

    print(f"[FETCH] Total stock universe: {total}")
    print(f"[FETCH] Using max_workers={MAX_WORKERS}")

    rows = []
    success = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_one_stock, info): info
            for info in codes
        }

        for idx, future in enumerate(as_completed(future_map), start=1):
            info = future_map[future]

            try:
                result = future.result(timeout=20)

                if result:
                    rows.append(result)
                    success += 1
                else:
                    skipped += 1

            except Exception as e:
                failed += 1
                print(f"[FETCH][ERROR] {info.code} {info.name}: {str(e)}")

            if idx % progress_every == 0 or idx == total:
                elapsed = round(time.time() - start_time, 1)
                print(
                    f"[FETCH] Progress {idx}/{total} | "
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
