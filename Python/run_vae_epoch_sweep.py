"""Sweep VAE training epochs (25/50/75/100) and early-stopping on fmow subset."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_EPOCH_GRID = [25, 50, 75, 100]


def run_one(
    epochs: int | None,
    output_dir: Path,
    train_dir: str,
    eval_dir: str,
    batch_size: int,
    cpu: bool,
    num_workers: int,
    early_stopping: bool = False,
    max_epochs: int = 100,
    patience: int = 5,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "early_stop" if early_stopping else f"ep{epochs}"
    log_path = output_dir / f"train_{label}.log"

    cmd = [
        sys.executable,
        "-u",
        "run_vae_domain_shift.py",
        "--train-dir",
        train_dir,
        "--eval-dir",
        eval_dir,
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--retrain",
        "--save-val-loss",
    ]
    if cpu:
        cmd.append("--cpu")
    if early_stopping:
        cmd.extend(
            [
                "--num-epochs",
                str(max_epochs),
                "--early-stopping",
                "--early-stopping-patience",
                str(patience),
            ]
        )
    else:
        cmd.extend(["--num-epochs", str(epochs)])

    print(f"\n{'=' * 60}\nRunning: {label}\nLog: {log_path}\n{'=' * 60}")
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, cwd=Path(__file__).parent, stdout=log_f, stderr=subprocess.STDOUT)

    summary_path = output_dir / "summary_vae.json"
    row: dict = {
        "label": label,
        "epochs_requested": max_epochs if early_stopping else epochs,
        "early_stopping": early_stopping,
        "returncode": proc.returncode,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
    }
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        row["oob_score"] = summary.get("oob_score")
        training = summary.get("training", {})
        row["epochs_trained"] = training.get("epochs_trained")
        row["stopped_early"] = training.get("stopped_early", False)
        row["best_epoch"] = training.get("best_epoch")
        row["best_val_loss"] = training.get("best_val_loss")
        if training.get("train_loss"):
            row["final_train_loss"] = training["train_loss"][-1]
        if training.get("val_loss"):
            row["final_val_loss"] = training["val_loss"][-1]
        region = summary.get("region_selection", {})
        if "10" in region:
            row["top10_ratio_k10"] = region["10"].get("top10_ratio")
            row["gini_k10"] = region["10"].get("gini_coeff")
    else:
        row["error"] = "summary_vae.json not found"

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAE epoch sweep on fmow subset")
    parser.add_argument("--train-dir", default="/workspace/data/fmow_subset/train")
    parser.add_argument("--eval-dir", default="/workspace/data/fmow_subset/eval")
    parser.add_argument("--output-root", default="vae_fmow_epoch_sweep")
    parser.add_argument("--epochs-grid", nargs="+", type=int, default=DEFAULT_EPOCH_GRID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-early-stopping", action="store_true")
    parser.add_argument("--early-stop-max-epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for epochs in args.epochs_grid:
        out_dir = root / f"ep{epochs}"
        results.append(
            run_one(
                epochs=epochs,
                output_dir=out_dir,
                train_dir=args.train_dir,
                eval_dir=args.eval_dir,
                batch_size=args.batch_size,
                cpu=args.cpu,
                num_workers=args.num_workers,
            )
        )

    if not args.skip_early_stopping:
        out_dir = root / "early_stop"
        results.append(
            run_one(
                epochs=None,
                output_dir=out_dir,
                train_dir=args.train_dir,
                eval_dir=args.eval_dir,
                batch_size=args.batch_size,
                cpu=args.cpu,
                num_workers=args.num_workers,
                early_stopping=True,
                max_epochs=args.early_stop_max_epochs,
                patience=args.early_stop_patience,
            )
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = root / f"epoch_sweep_summary_{stamp}.csv"
    summary_json = root / f"epoch_sweep_summary_{stamp}.json"

    import pandas as pd

    pd.DataFrame(results).to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEpoch sweep complete. Summary: {summary_csv}")
    for row in results:
        status = "OK" if row.get("returncode") == 0 else "FAIL"
        print(
            f"  [{status}] {row['label']:12s} | trained={row.get('epochs_trained')} "
            f"| OOB={row.get('oob_score')} | out={row.get('output_dir')}"
        )


if __name__ == "__main__":
    main()
