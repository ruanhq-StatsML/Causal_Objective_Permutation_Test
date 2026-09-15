"""Restore helper files and CSV checkpoints after iCloud corruption."""
import glob
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

UTILS_HELPERS = '''
def _as_1d(y):
    return np.asarray(y, dtype=float).ravel()

def _as_2d(X):
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        return Xa.reshape(-1, 1)
    return Xa
'''

ADV_PERT_DS = "from FGSM_adversarial import adv_pert_reg as adv_pert\n"

EXPECTED_CSVS = [
    "time_profile_benchmark_vimpDS.csv",
    "time_profile_benchmark_vimpDS_by_p.csv",
    "video1234_benchmark_metrics.csv",
    "ChronoBerg_cs30_benchmark_metrics.csv",
    "ChronoBerg_benchmark_metrics.csv",
    "FMoW_benchmark_metrics.csv",
    "wilds_vimp_metrics.csv",
    "whyshift_adv_cd50_vimp_metrics.csv",
    "realdata_vimp_metrics.csv",
]


def restore_if_empty(path, content, min_bytes=50):
    if os.path.exists(path) and os.path.getsize(path) > min_bytes:
        print(f"[skip] {path} looks ok ({os.path.getsize(path)} bytes)", flush=True)
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[restore] wrote {path}", flush=True)
    return True


def ensure_utils_helpers():
    path = os.path.join(ROOT, "utils.py")
    if not os.path.exists(path) or os.path.getsize(path) <= 50:
        restore_if_empty(path, "import numpy as np\n" + UTILS_HELPERS + UTILS_TAIL, min_bytes=0)
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if "def _as_1d(" in text:
        print(f"[skip] utils.py already defines _as_1d", flush=True)
        return
    marker = "import numpy as np\n"
    if marker in text:
        text = text.replace(marker, marker + UTILS_HELPERS + "\n", 1)
    else:
        text = "import numpy as np\n" + UTILS_HELPERS + "\n" + text
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("[restore] injected _as_1d/_as_2d into utils.py", flush=True)


UTILS_TAIL = '''import numpy as np

def _as_1d(y):
    return np.asarray(y, dtype=float).ravel()

def _as_2d(X):
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        return Xa.reshape(-1, 1)
    return Xa

def make_folds(n, n_folds=5, seed=1):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, n_folds)
'''


def recover_csv_from_tmp(path):
    tmp = f"{path}.tmp"
    if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        return False
    if os.path.exists(path) and os.path.getsize(path) > os.path.getsize(tmp):
        print(f"[skip] {path} larger than .tmp; keeping CSV", flush=True)
        return False
    os.replace(tmp, path)
    print(f"[recover] {path} <- {tmp} ({os.path.getsize(path)} bytes)", flush=True)
    return True


def recover_all_csv_tmp():
    seen = set()
    candidates = list(EXPECTED_CSVS)
    candidates.extend(glob.glob(os.path.join(ROOT, "*.csv")))
    for path in sorted(set(candidates)):
        base = os.path.basename(path)
        if base.endswith(".tmp"):
            continue
        full = path if os.path.isabs(path) else os.path.join(ROOT, base)
        if full in seen:
            continue
        seen.add(full)
        recover_csv_from_tmp(full)

    for tmp_path in sorted(glob.glob(os.path.join(ROOT, "*.csv.tmp"))):
        csv_path = tmp_path[:-4]
        if csv_path not in seen:
            recover_csv_from_tmp(csv_path)


if __name__ == "__main__":
    os.chdir(ROOT)
    print("========== recover CSV from .tmp ==========", flush=True)
    recover_all_csv_tmp()
    print("\n========== restore helper files ==========", flush=True)
    ensure_utils_helpers()
    restore_if_empty("adversarial_pert_DS.py", ADV_PERT_DS)
