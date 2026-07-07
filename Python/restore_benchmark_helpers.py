"""Restore minimal helper files if iCloud zeroed them out."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

UTILS = '''import numpy as np

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

ADV_PERT_DS = "from FGSM_adversarial import adv_pert_reg as adv_pert\n"


def restore_if_empty(path, content):
    if os.path.exists(path) and os.path.getsize(path) > 50:
        print(f"[skip] {path} looks ok ({os.path.getsize(path)} bytes)", flush=True)
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[restore] wrote {path}", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    restore_if_empty("utils.py", UTILS)
    restore_if_empty("adversarial_pert_DS.py", ADV_PERT_DS)
