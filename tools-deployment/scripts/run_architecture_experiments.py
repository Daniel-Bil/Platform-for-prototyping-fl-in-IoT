#!/usr/bin/env python3
"""Run each JSON architecture as a separate benchmark campaign.

Default architecture study intentionally changes only the model architecture:
all FL methods, 7 clients, 20 rounds, 3 local epochs, batch 32, 5 repetitions.
Each architecture gets its own output directory.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_ALL = ROOT / "tools-deployment/scripts/run_all_experiments.py"


def slug(name: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "model"


def write_status(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["model", "model_path", "status", "started_utc", "finished_utc", "returncode", "output_root"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def campaign_complete(model_output: Path, expected_runs: int) -> tuple[bool, str]:
    campaigns = sorted(
        [p for p in model_output.glob("campaign-*") if (p / "campaign_status.csv").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if not campaigns:
        return False, "no campaign_status.csv found"
    status_path = campaigns[-1] / "campaign_status.csv"
    with status_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    completed_keys = {
        (r.get("algorithm"), r.get("requested_clients"), r.get("repetition"))
        for r in rows
        if r.get("status") == "completed"
    }
    if len(completed_keys) != expected_runs:
        return False, f"{len(completed_keys)}/{expected_runs} experiment configurations completed"
    return True, str(campaigns[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="tools-deployment/config/models")
    p.add_argument("--clients", type=int, default=7)
    p.add_argument("--algorithms", default="FedAvg,FedProx,FedPAQ,FedMA,HierFedAvg")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default="architecture-results")
    p.add_argument("--fail-fast", action="store_true")
    a = p.parse_args()

    models_dir = (ROOT / a.models_dir).resolve()
    model_files = sorted(models_dir.glob("*.json"))
    if not model_files:
        raise SystemExit(f"no *.json models found in {models_dir}")

    base_output = (ROOT / a.output_root).resolve()
    base_output.mkdir(parents=True, exist_ok=True)
    status_file = base_output / "architecture_status.csv"
    rows: list[dict[str, str]] = []

    for model in model_files:
        model_name = slug(model.stem)
        model_output = base_output / model_name
        model_relative = model.relative_to(ROOT)
        started = datetime.now(timezone.utc).isoformat()

        cmd = [
            sys.executable,
            str(RUN_ALL),
            "--model", str(model_relative),
            "--clients", str(a.clients),
            "--algorithms", a.algorithms,
            "--rounds", str(a.rounds),
            "--local-epochs", str(a.local_epochs),
            "--batch-size", str(a.batch_size),
            "--repetitions", str(a.repetitions),
            "--seed", str(a.seed),
            "--output-root", str(model_output),
        ]

        print("\n" + "=" * 78)
        print(f"ARCHITECTURE: {model.stem}")
        print("=" * 78, flush=True)
        proc = subprocess.run(cmd, cwd=ROOT)
        expected_runs = len([x for x in a.algorithms.split(",") if x.strip()]) * a.repetitions
        complete, detail = campaign_complete(model_output, expected_runs)
        ok = proc.returncode == 0 and complete
        if not ok:
            print(f"ARCHITECTURE INCOMPLETE: {model.stem}: {detail}", file=sys.stderr, flush=True)

        row = {
            "model": model.stem,
            "model_path": str(model_relative),
            "status": "completed" if ok else "failed",
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": str(proc.returncode if proc.returncode != 0 else (0 if complete else 2)),
            "output_root": str(model_output.relative_to(ROOT)),
        }
        rows.append(row)
        write_status(status_file, rows)

        if not ok and a.fail_fast:
            raise SystemExit(proc.returncode or 2)

    print(f"\nArchitecture study finished. Status: {status_file}")
    print(f"Results root: {base_output}")


if __name__ == "__main__":
    main()
