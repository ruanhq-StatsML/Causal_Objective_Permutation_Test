"""Quick snapshot: running jobs + latest log lines + CSV progress."""
import glob
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

JOBS = [
    ("video1234_benchmark.py", "video1234_benchmark.log", "video1234_benchmark_metrics.csv", 30),
    ("audio1234_benchmark.py", "audio1234_benchmark.log", "audio1234_benchmark_metrics.csv", 30),
    ("ChronoBerg_benchmark.py", "ChronoBerg_benchmark.log", "ChronoBerg_benchmark_metrics.csv", 115),
    ("ChronoBerg_covariate_shift_benchmark.py", "ChronoBerg_cs30_benchmark.log", "ChronoBerg_cs30_benchmark_metrics.csv", 15),
    ("wilds_data_benchmark.py", "wilds_data_benchmark.log", "wilds_vimp_metrics.csv", 5),
    ("run_fgsm30_benchmark.py", "fgsm30_benchmark.log", "fgsm30_vimp_metrics.csv", 8),
    ("whyshift_benchmark_run.py", "whyshift_benchmark_run.log", "realdata_vimp_metrics.csv", None),
]


def running(script):
    try:
        out = subprocess.check_output(["pgrep", "-fl", script], stderr=subprocess.DEVNULL, text=True)
        return out.strip().splitlines()[0] if out.strip() else None
    except subprocess.CalledProcessError:
        return None


def tail(path, n=2):
    if not os.path.exists(path):
        return ["(no log yet)"]
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [ln.rstrip() for ln in lines[-n:]] if lines else ["(empty log)"]


def csv_jobs(path, expected):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0, expected
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception as exc:
        return -1, expected
    keys = [c for c in ["dataset_id", "shift_type", "rep", "method", "scenario"] if c in df.columns]
    if keys:
        n = len(df[keys].drop_duplicates())
    else:
        n = len(df)
    return n, expected


def main():
    print(f"[monitor] cwd={os.getcwd()}\n")
    for script, log, csv_path, expected in JOBS:
        print(f"=== {script} ===")
        proc = running(script)
        print(f"  process: {proc or 'not running'}")
        n, exp = csv_jobs(csv_path, expected)
        if n < 0:
            print(f"  csv: {csv_path} (unreadable)")
        elif exp is None:
            print(f"  csv: {csv_path} rows={n}")
        else:
            print(f"  csv: {n}/{exp} jobs in {csv_path}")
        for ln in tail(log):
            print(f"  log: {ln}")
        print()

    if os.path.exists("check_benchmark_status.py"):
        print("--- check_benchmark_status.py ---")
        subprocess.run([sys.executable, "check_benchmark_status.py"], check=False)


if __name__ == "__main__":
    main()
