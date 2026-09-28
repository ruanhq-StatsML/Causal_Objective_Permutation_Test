"""Run the online PFI and online PO-risk monitors and write one figure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from online_monitor import monitor


def _stream(n: int, shift: float, delta: float, seed: int):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, 8, 2))
    tokens[n // 2:, 5, :] += shift
    y = tokens.reshape(n, -1)[:, :4] @ np.ones(4) + rng.normal(size=n)
    beta = np.arange(4) * float(delta)
    y[n // 2:] = y[n // 2:] + tokens.reshape(n, -1)[n // 2:, :4] @ beta
    return tokens, y


def plot_pack(panels, path: str, enter_m: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(panels), figsize=(6.0 * len(panels), 7.2), sharex="col")
    if len(panels) == 1:
        axes = np.array(axes).reshape(2, 1)
    for col, (title, pack) in enumerate(panels):
        pfi, porisk = pack["pfi"], pack["porisk"]
        m = np.array([row["m"] for row in pfi])
        top = np.array([row["top_pfi"] for row in pfi])
        axes[0, col].plot(m, top, color="#ff7f0e", lw=1.8, marker="o")
        axes[0, col].axvline(enter_m, color="#d62728", ls="--", lw=1.1)
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("online PFI of the top token")
        m2 = np.array([row["m"] for row in porisk])
        y = np.array([row["porisk"] for row in porisk])
        lo = np.array([row["ci_lo"] for row in porisk])
        hi = np.array([row["ci_hi"] for row in porisk])
        axes[1, col].fill_between(m2, lo, hi, color="#1f77b4", alpha=0.18)
        axes[1, col].plot(m2, y, color="#1f77b4", lw=1.8, marker="o")
        axes[1, col].axvline(enter_m, color="#d62728", ls="--", lw=1.1)
        axes[1, col].set_xlabel("stream step m")
        axes[1, col].set_ylabel("online PO-risk")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _public(pack) -> dict:
    pfi = []
    for row in pack["pfi"]:
        pfi.append({
            "m": row["m"],
            "top_token": row["top_token"],
            "top_patch": row["top_patch"],
            "top_pfi": row["top_pfi"],
        })
    return {"pfi": pfi, "porisk": pack["porisk"]}


def main() -> None:
    grid = (2, 2, 4)
    shift = monitor(*_stream(48, shift=2.5, delta=0.8, seed=3), grid, n_boot=8, seed=3)
    quiet = monitor(*_stream(48, shift=0.0, delta=0.0, seed=4), grid, n_boot=8, seed=4)
    for name, pack in (("shift", shift), ("quiet", quiet)):
        last = pack["pfi"][-1]
        risk = pack["porisk"][-1]
        print(
            f"{name} m={last['m']} top token {last['top_token']} "
            f"patch {last['top_patch']} pfi={last['top_pfi']:+.4f}  "
            f"PO-risk={risk['porisk']:.4f} CI=[{risk['ci_lo']:.4f}, {risk['ci_hi']:.4f}]"
        )
    out = Path("/opt/cursor/artifacts/online_monitor_pack.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_pack([("token shift and concept drift", shift), ("no shift", quiet)], str(out), enter_m=32)
    payload = {"shift": _public(shift), "quiet": _public(quiet)}
    Path("/opt/cursor/artifacts/online_monitor_pack.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[1] / "results" / "online_monitor_pack.json"
    repo.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert shift["pfi"][-1]["top_token"] == 5
    print("PACK_OK", out)


if __name__ == "__main__":
    main()
