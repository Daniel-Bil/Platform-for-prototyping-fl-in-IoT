#!/usr/bin/env python3
"""Validate thesis FL dataset integrity before distributed experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("value_temp", "value_hum", "value_acid", "value_PV")
EXPECTED_CADENCE = pd.Timedelta(minutes=10)


def sequence_positive_rate(labels: np.ndarray, seq_len: int) -> float:
    """Positive rate after causal windowing where y[t] is inside X-window."""
    if len(labels) < seq_len:
        return float("nan")
    return float(np.asarray(labels[seq_len - 1 :], dtype=float).mean())


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("dataset", nargs="?", type=Path, default=here / "data" / "fl_dataset_real")
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--min-anomaly-rate", type=float, default=0.10)
    p.add_argument("--max-anomaly-rate", type=float, default=0.40)
    args = p.parse_args()

    root = args.dataset.resolve()
    clients = sorted(p for p in root.glob("client_*") if p.is_dir())
    if not clients:
        raise SystemExit(f"no client_* directories in {root}")

    errors: list[str] = []
    rows: list[dict] = []
    expected_rows: dict[str, int] = {}

    directional_profiles = {"temperature_drift", "temperature_bias", "humidity_flatline"}

    for client in clients:
        manifest_path = client / "dataset_manifest.json"
        client_manifest = None
        if not manifest_path.exists():
            errors.append(f"{client.name}: missing dataset_manifest.json")
        else:
            client_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if client_manifest.get("sequence_label_semantics") != "classify_last_sample_in_input_window":
                errors.append(f"{client.name}: wrong/missing sequence_label_semantics")
            if client_manifest.get("fault_direction_policy") != "balanced_positive_negative_per_split":
                errors.append(f"{client.name}: wrong/missing fault_direction_policy")

        for split in ("train", "val", "test"):
            path = client / f"{split}.csv"
            if not path.exists():
                errors.append(f"{client.name}/{split}: missing CSV")
                continue
            df = pd.read_csv(path, parse_dates=["time"])
            missing = (set(FEATURES) | {"label", "time"}) - set(df.columns)
            if missing:
                errors.append(f"{client.name}/{split}: missing columns {sorted(missing)}")
                continue
            if df[list(FEATURES)].isna().any().any():
                errors.append(f"{client.name}/{split}: contains NaN features")
            if not df["label"].isin([0, 1]).all():
                errors.append(f"{client.name}/{split}: labels outside {{0,1}}")
            classes = set(int(x) for x in df["label"].unique())
            if classes != {0, 1}:
                errors.append(f"{client.name}/{split}: expected both classes, got {sorted(classes)}")
            rate = float(df["label"].mean())
            if not args.min_anomaly_rate <= rate <= args.max_anomaly_rate:
                errors.append(
                    f"{client.name}/{split}: anomaly rate {rate:.3f} outside "
                    f"[{args.min_anomaly_rate:.3f},{args.max_anomaly_rate:.3f}]"
                )
            times = df["time"]
            deltas = times.diff().dropna()
            if len(deltas) and not (deltas == EXPECTED_CADENCE).all():
                errors.append(f"{client.name}/{split}: timestamps are not continuous 10-minute cadence")
            seq_rate = sequence_positive_rate(df["label"].to_numpy(), args.seq_len)

            if client_manifest is not None:
                profile = client_manifest.get("fault_profile")
                split_info = client_manifest.get("splits", {}).get(split, {})
                variants = split_info.get("fault_variants", [])
                directional = [v for v in variants if v.get("direction") in {"positive", "negative"}]
                if profile in directional_profiles:
                    directions = {v.get("direction") for v in directional}
                    if directions != {"positive", "negative"}:
                        errors.append(
                            f"{client.name}/{split}: {profile} must contain both fault directions; got {sorted(directions)}"
                        )
                elif profile == "mixed_flatline_dropout":
                    flatline_dirs = {
                        v.get("direction") for v in variants
                        if v.get("kind") == "humidity_flatline" and v.get("direction") in {"positive", "negative"}
                    }
                    if flatline_dirs != {"positive", "negative"}:
                        errors.append(
                            f"{client.name}/{split}: mixed flatline component must contain both directions; "
                            f"got {sorted(flatline_dirs)}"
                        )

            rows.append({
                "client": client.name,
                "split": split,
                "rows": len(df),
                "normal": int((df["label"] == 0).sum()),
                "anomaly": int((df["label"] == 1).sum()),
                "row_anomaly_rate": rate,
                "sequence_anomaly_rate": seq_rate,
            })
            expected_rows.setdefault(split, len(df))
            if expected_rows[split] != len(df):
                errors.append(
                    f"{client.name}/{split}: {len(df)} rows; expected equal client size {expected_rows[split]}"
                )

    table = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", 100, "display.width", 160):
        print(table.to_string(index=False, formatters={
            "row_anomaly_rate": lambda x: f"{x:.3f}",
            "sequence_anomaly_rate": lambda x: f"{x:.3f}",
        }))

    top_manifest = root / "dataset_manifest.json"
    if top_manifest.exists():
        manifest = json.loads(top_manifest.read_text(encoding="utf-8"))
        print(f"\nDataset version: {manifest.get('dataset_version')}")
        print(f"Source kind:     {manifest.get('source_kind')}")

    if errors:
        print("\nVALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(f"\nVALIDATION PASSED: {len(clients)} clients, every split contains both classes and no NaNs.")


if __name__ == "__main__":
    main()
