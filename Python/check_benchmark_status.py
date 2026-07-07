"""Audit benchmark CSV checkpoints and report missing jobs."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

EXPECTED = [
    ("time n-sweep", "time_profile_benchmark_vimpDS.csv", ["n", "rep"], 90,
     lambda: [(int(n), r) for n in [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000] for r in range(10)]),
    ("time p-sweep", "time_profile_benchmark_vimpDS_by_p.csv", ["p", "rep"], 50,
     lambda: [(int(p), r) for p in [20, 50, 100, 200, 500] for r in range(10)]),
    ("video1234", "video1234_benchmark_metrics.csv", ["dataset_id", "rep"], 30, None),
    ("ChronoBerg cs30", "ChronoBerg_cs30_benchmark_metrics.csv", ["dataset_id"], 15, None),
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


def main():
    print(f"[audit] cwd={os.getcwd()}\n", flush=True)
    for name, path, keys, total, key_fn in EXPECTED:
        print(f"=== {name} ({path}) ===", flush=True)
        if not os.path.exists(path):
            print(f"  MISSING file (expected {total} jobs)", flush=True)
            try_recover_tmp(path)
            continue
        try_recover_tmp(path)
        df = read_csv_safe(path)
        if df is None:
            print(f"  EMPTY or corrupt (expected {total} jobs)", flush=True)
            continue
        cols = [c for c in keys if c in df.columns]
        if cols:
            done = df[cols].drop_duplicates()
            n_done = len(done)
        else:
            n_done = len(df)
        print(f"  jobs: {n_done}/{total} | rows: {len(df)} | size: {os.path.getsize(path)/1024:.1f} KB", flush=True)
        if key_fn is not None:
            missing = set(key_fn()) - set(map(tuple, done.to_numpy()))
            if missing:
                print(f"  missing: {len(missing)} e.g. {list(missing)[:3]}", flush=True)
        print(flush=True)

    if os.path.exists("realdata_vimp_metrics.csv"):
        df = read_csv_safe("realdata_vimp_metrics.csv")
        if df is not None:
            ws = df[(df.get("scenario") == "whyshift") & (df.get("shift_type") == "linear")]
            tab = df[df.get("scenario") == "tabular"]
            print("=== WhyShift linear (in realdata_vimp_metrics.csv) ===", flush=True)
            print(f"  pairs: {ws['dataset_id'].nunique() if len(ws) else 0}/50 | rows: {len(ws)}", flush=True)
            print("=== Tabular ===", flush=True)
            if len(tab):
                print(tab.groupby("shift_type").size().to_string(), flush=True)
            print(flush=True)


if __name__ == "__main__":
    main()
