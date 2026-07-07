"""Audit benchmark CSV checkpoints and report missing jobs."""
import glob
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

EXPECTED = [
    ("time n-sweep", "time_profile_benchmark_vimpDS.csv", ["n", "rep"], 90,
     lambda: [(int(n), r) for n in [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000] for r in range(10)]),
    ("time p-sweep", "time_profile_benchmark_vimpDS_by_p.csv", ["p", "rep"], 50,
     lambda: [(int(p), r) for p in [20, 50, 100, 200, 500] for r in range(10)]),
    ("video1234", "video1234_benchmark_metrics.csv", ["dataset_id", "rep"], 30, None),
    ("ChronoBerg cs30", "ChronoBerg_cs30_benchmark_metrics.csv", ["dataset_id"], 15, None),
    ("ChronoBerg adv", "ChronoBerg_benchmark_metrics.csv", ["dataset_id"], 115, None),
    ("FMoW", "FMoW_benchmark_metrics.csv", ["dataset_id", "shift_type"], 100, None),
    ("WILDS", "wilds_vimp_metrics.csv", ["shift_type"], 5, None),
    ("WhyShift adv CD50", "whyshift_adv_cd50_vimp_metrics.csv", ["dataset_id", "shift_type"], 250, None),
]


def try_recover_tmp(path):
    tmp = f"{path}.tmp"
    if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        return False
    if os.path.exists(path) and os.path.getsize(path) > os.path.getsize(tmp):
        return False
    os.replace(tmp, path)
    print(f"  [recover] restored {path} from .tmp", flush=True)
    return True


def read_csv_safe(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"  [warn] cannot read {path}: {exc}", flush=True)
        return None


def audit_one(name, path, keys, total, key_fn):
    print(f"=== {name} ({path}) ===", flush=True)
    missing_file = not os.path.exists(path)
    if missing_file:
        print(f"  MISSING file (expected {total} jobs)", flush=True)
    try_recover_tmp(path)
    if missing_file and not os.path.exists(path):
        tmp = f"{path}.tmp"
        if os.path.exists(tmp):
            print(f"  [hint] {tmp} exists ({os.path.getsize(tmp)} bytes) — run restore_benchmark_helpers.py", flush=True)
        return total, 0

    df = read_csv_safe(path)
    if df is None:
        print(f"  EMPTY or corrupt (expected {total} jobs)", flush=True)
        return total, 0

    cols = [c for c in keys if c in df.columns]
    if cols:
        done = df[cols].drop_duplicates()
        n_done = len(done)
    else:
        n_done = len(df)
    n_missing = max(total - n_done, 0)
    print(
        f"  jobs: {n_done}/{total} | rows: {len(df)} | "
        f"size: {os.path.getsize(path)/1024:.1f} KB",
        flush=True,
    )
    if key_fn is not None and n_missing:
        missing = set(key_fn()) - set(map(tuple, done.to_numpy()))
        if missing:
            print(f"  missing: {len(missing)} e.g. {list(missing)[:3]}", flush=True)
    elif n_missing:
        print(f"  missing: ~{n_missing} jobs (resume script will skip finished rows)", flush=True)
    print(flush=True)
    return n_missing, n_done


def main():
    print(f"[audit] cwd={os.getcwd()}\n", flush=True)

    orphan_tmp = [
        p for p in glob.glob("*.csv.tmp")
        if os.path.getsize(p) > 0
    ]
    if orphan_tmp:
        print("=== recoverable .tmp files ===", flush=True)
        for p in sorted(orphan_tmp):
            print(f"  {p} ({os.path.getsize(p)/1024:.1f} KB)", flush=True)
        print(flush=True)

    total_missing = 0
    total_done = 0
    for item in EXPECTED:
        n_miss, n_done = audit_one(*item)
        total_missing += n_miss
        total_done += n_done

    if os.path.exists("realdata_vimp_metrics.csv"):
        df = read_csv_safe("realdata_vimp_metrics.csv")
        if df is not None:
            ws = df[(df.get("scenario") == "whyshift") & (df.get("shift_type") == "linear")]
            tab = df[df.get("scenario") == "tabular"]
            print("=== WhyShift linear (in realdata_vimp_metrics.csv) ===", flush=True)
            ws_pairs = ws["dataset_id"].nunique() if len(ws) and "dataset_id" in ws.columns else 0
            ws_miss = max(50 - ws_pairs, 0)
            print(f"  pairs: {ws_pairs}/50 | rows: {len(ws)} | missing pairs: {ws_miss}", flush=True)
            total_missing += ws_miss
            print("=== Tabular (in realdata_vimp_metrics.csv) ===", flush=True)
            if len(tab):
                print(tab.groupby("shift_type").size().to_string(), flush=True)
            else:
                print("  no tabular rows found", flush=True)
            print(flush=True)

    print("=== summary ===", flush=True)
    print(f"  recorded jobs (above): {total_done}", flush=True)
    print(f"  missing jobs (estimate): {total_missing}", flush=True)
    if total_missing == 0 and total_done > 0:
        print("  all tracked benchmarks look complete", flush=True)
    elif total_done == 0:
        print("  no CSV checkpoints found — restore from iCloud Versions or Time Machine first", flush=True)
    else:
        print("  run: RESTORE_YES=1 ./restore_all_benchmarks.sh  (to resume incomplete jobs)", flush=True)


if __name__ == "__main__":
    main()
