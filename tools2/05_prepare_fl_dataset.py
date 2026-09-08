#!/usr/bin/env python3
"""Prepare the FINAL thesis FL benchmark from CLEANED REAL RuralIoT data.

Why this script exists
----------------------
The original pipeline generated a synthetic 43,200-row timeline, injected one
long-lived fault, and only then split chronologically.  That accidentally made
some validation/test sets almost 100% anomalous.  It also caused the 40%
per-feature dropout profile to label ~87% of rows as anomalous.

This replacement keeps the synthetic/VAE pipeline available as an optional
platform feature, but the default thesis benchmark now uses real measurements:

    cleaned real RuralIoT -> select contiguous valid real interval
                          -> chronological train/val/test split
                          -> inject controlled fault EPISODES in each split
                          -> explicit ground-truth mask
                          -> local imputation of injected dropouts
                          -> tools2/data/fl_dataset_real/client_*/{train,val,test}.csv

Every client receives approximately the same anomaly prevalence in every split,
while the *fault type* and the underlying real sensor distribution remain
client-specific (non-IID).  This lets aggregation methods be compared without
class imbalance dominating the result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

FEATURES = ("value_temp", "value_hum", "value_acid", "value_PV")
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
EXPECTED_CADENCE = pd.Timedelta(minutes=10)

PHYSICAL_LIMITS = {
    "value_temp": (-30.0, 60.0),
    "value_hum": (0.0, 100.0),
    "value_acid": (0.0, 14.0),
    "value_PV": (0.0, 10.0),
}

# Same target prevalence for every client: non-IID comes from sensor/fault type,
# not from pathological 0%/100% class balance.
PROFILES = {
    "df_RuralIoT_001": "temperature_drift",
    "df_RuralIoT_002": "temperature_bias",
    "df_RuralIoT_003": "single_feature_dropout",
    "df_RuralIoT_010": "multi_feature_dropout",
    "df_RuralIoT_21": "burst_noise",
    "df_RuralIoT_22": "humidity_flatline",
    "df_RuralIoT_23": "mixed_flatline_dropout",
}


def stable_seed(base_seed: int, sensor_id: str, split_name: str) -> int:
    token = f"{sensor_id}:{split_name}".encode("utf-8")
    return int((base_seed + zlib.crc32(token)) % (2**32 - 1))


def _valid_contiguous_runs(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Return [start, end) runs with all features present and 10-minute cadence."""
    valid = df[list(FEATURES)].notna().all(axis=1).to_numpy()
    times = df.index
    runs: list[tuple[int, int]] = []
    start: int | None = None

    for i in range(len(df)):
        cadence_ok = i == 0 or (times[i] - times[i - 1] == EXPECTED_CADENCE)
        row_ok = bool(valid[i])

        if row_ok and (start is None):
            start = i
        elif row_ok and not cadence_ok:
            if start is not None and i > start:
                runs.append((start, i))
            start = i
        elif not row_ok and start is not None:
            runs.append((start, i))
            start = None

    if start is not None:
        runs.append((start, len(df)))
    return runs


def select_real_interval(df: pd.DataFrame, rows_per_client: int) -> tuple[pd.DataFrame, dict]:
    runs = _valid_contiguous_runs(df)
    if not runs:
        raise ValueError("no contiguous real interval with complete measurements")

    start, end = max(runs, key=lambda pair: pair[1] - pair[0])
    longest = end - start
    if longest < rows_per_client:
        raise ValueError(
            f"longest valid 10-minute interval has only {longest} rows; "
            f"requested {rows_per_client}"
        )

    # Centre crop so one client with a long recording does not always use the
    # beginning of the season.  The result remains deterministic.
    offset = start + (longest - rows_per_client) // 2
    selected = df.iloc[offset : offset + rows_per_client].copy()
    if selected[list(FEATURES)].isna().any().any():
        raise AssertionError("selected real interval unexpectedly contains NaN")

    metadata = {
        "source_valid_run_start": int(start),
        "source_valid_run_end_exclusive": int(end),
        "source_valid_run_rows": int(longest),
        "selected_start_row": int(offset),
        "selected_rows": int(rows_per_client),
        "selected_start_time": selected.index[0].isoformat(),
        "selected_end_time": selected.index[-1].isoformat(),
    }
    return selected, metadata


def split_clean_real(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    return {
        "train": df.iloc[:train_end].copy(),
        "val": df.iloc[train_end:val_end].copy(),
        "test": df.iloc[val_end:].copy(),
    }


def choose_fault_episodes(
    n_rows: int,
    anomaly_rate: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Create non-overlapping [start,end) episodes covering ~anomaly_rate rows."""
    if not 0 < anomaly_rate < 0.5:
        raise ValueError("anomaly_rate must be between 0 and 0.5")
    target = max(1, int(round(n_rows * anomaly_rate)))

    # Episode lengths scale with split size: several-hour to multi-day faults in
    # 10-minute data, while validation/test still get multiple separate episodes.
    min_len = max(4, min(12, n_rows // 18))
    max_len = max(min_len, min(36, n_rows // 4))
    occupied = np.zeros(n_rows, dtype=bool)
    episodes: list[tuple[int, int]] = []
    remaining = target
    attempts = 0

    while remaining > 0 and attempts < 10000:
        attempts += 1
        length = min(remaining, int(rng.integers(min_len, max_len + 1)))
        if remaining < min_len:
            length = remaining
        if length <= 0 or length > n_rows:
            break
        start = int(rng.integers(0, n_rows - length + 1))
        end = start + length

        # Keep a two-row healthy guard band between episodes where possible.
        guard_start = max(0, start - 2)
        guard_end = min(n_rows, end + 2)
        if occupied[guard_start:guard_end].any():
            continue

        occupied[start:end] = True
        episodes.append((start, end))
        remaining -= length

    # With a 25% target this should almost always be exact.  Deterministically
    # fill any tiny remainder without making the whole split unusable.
    if remaining > 0:
        free = np.flatnonzero(~occupied)
        if len(free) < remaining:
            raise RuntimeError("unable to place requested anomaly episodes")
        chosen = np.sort(rng.choice(free, size=remaining, replace=False))
        for idx in chosen:
            occupied[idx] = True
        # Merge the fallback points into contiguous intervals for fault application.
        positions = np.flatnonzero(occupied)
        episodes = []
        if len(positions):
            seg_start = prev = int(positions[0])
            for p in positions[1:]:
                p = int(p)
                if p != prev + 1:
                    episodes.append((seg_start, prev + 1))
                    seg_start = p
                prev = p
            episodes.append((seg_start, prev + 1))

    episodes.sort()
    return episodes


def mask_from_episodes(n_rows: int, episodes: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    for start, end in episodes:
        mask[start:end] = True
    return mask


def _local_reference(series: pd.Series, start: int, end: int) -> float:
    before = series.iloc[max(0, start - 12) : start]
    if len(before) and before.notna().any():
        return float(before.median())
    return float(series.iloc[start:end].median())


def _flatline_episode(df: pd.DataFrame, start: int, end: int, col: str) -> None:
    base = _local_reference(df[col], start, end)
    lo, hi = PHYSICAL_LIMITS[col]
    span = hi - lo
    # Deliberately wrong but physically possible stuck value.  Shift away from
    # the local baseline so labels correspond to a real corruption.
    shift = 0.15 * span
    stuck = base - shift if base > (lo + hi) / 2 else base + shift
    stuck = float(np.clip(stuck, lo, hi))
    df.iloc[start:end, df.columns.get_loc(col)] = stuck


def apply_profile(
    clean_split: pd.DataFrame,
    profile: str,
    episodes: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    corrupted = clean_split.copy()
    mask = mask_from_episodes(len(corrupted), episodes)

    if profile == "temperature_drift":
        for start, end in episodes:
            length = end - start
            sign = -1.0 if rng.random() < 0.5 else 1.0
            severity = float(rng.uniform(2.0, 4.0))
            drift = sign * np.linspace(0.4, severity, length)
            corrupted.iloc[start:end, corrupted.columns.get_loc("value_temp")] += drift

    elif profile == "temperature_bias":
        for start, end in episodes:
            sign = -1.0 if rng.random() < 0.5 else 1.0
            bias = sign * float(rng.uniform(2.0, 4.0))
            corrupted.iloc[start:end, corrupted.columns.get_loc("value_temp")] += bias

    elif profile == "single_feature_dropout":
        feature_choices = np.array(FEATURES, dtype=object)
        for start, end in episodes:
            col = str(rng.choice(feature_choices))
            corrupted.iloc[start:end, corrupted.columns.get_loc(col)] = np.nan

    elif profile == "multi_feature_dropout":
        feature_choices = np.array(FEATURES, dtype=object)
        for start, end in episodes:
            cols = rng.choice(feature_choices, size=2, replace=False)
            for col in cols:
                corrupted.iloc[start:end, corrupted.columns.get_loc(str(col))] = np.nan

    elif profile == "burst_noise":
        amplitudes = {
            "value_temp": 2.5,
            "value_hum": 12.0,
            "value_acid": 0.45,
            "value_PV": 1.1,
        }
        feature_choices = np.array(FEATURES, dtype=object)
        for start, end in episodes:
            col = str(rng.choice(feature_choices))
            length = end - start
            noise = rng.normal(0.0, amplitudes[col], size=length)
            # Ensure even low random draws represent a visible sensor disturbance.
            noise += np.sign(noise + 1e-9) * amplitudes[col] * 0.35
            corrupted.iloc[start:end, corrupted.columns.get_loc(col)] += noise

    elif profile == "humidity_flatline":
        for start, end in episodes:
            _flatline_episode(corrupted, start, end, "value_hum")

    elif profile == "mixed_flatline_dropout":
        feature_choices = np.array(FEATURES, dtype=object)
        for i, (start, end) in enumerate(episodes):
            if i % 2 == 0:
                _flatline_episode(corrupted, start, end, "value_hum")
            else:
                col = str(rng.choice(feature_choices))
                corrupted.iloc[start:end, corrupted.columns.get_loc(col)] = np.nan

    else:
        raise ValueError(f"unknown profile {profile!r}")

    # Keep non-dropout corruptions within broad physical sensor limits.
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        corrupted[col] = corrupted[col].clip(lower=lo, upper=hi)

    # The NN cannot ingest NaNs.  For packet-loss profiles, local forward-fill
    # creates the expected stale-value pattern; bfill is only a boundary fallback.
    corrupted[list(FEATURES)] = corrupted[list(FEATURES)].ffill().bfill()
    if corrupted[list(FEATURES)].isna().any().any():
        raise AssertionError("imputation left NaNs in prepared client data")

    corrupted["label"] = mask.astype(np.int8)
    return corrupted, mask


def write_client(
    sensor_id: str,
    source_path: Path,
    output_root: Path,
    rows_per_client: int,
    anomaly_rate: float,
    seed: int,
) -> dict:
    profile = PROFILES[sensor_id]
    df = pd.read_csv(source_path, parse_dates=["time"])
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"{source_path} missing features {sorted(missing)}")
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="first").set_index("time")

    selected, source_meta = select_real_interval(df, rows_per_client)
    clean_splits = split_clean_real(selected)

    client_dir = output_root / f"client_{sensor_id}"
    client_dir.mkdir(parents=True, exist_ok=True)
    split_meta: dict[str, dict] = {}

    for split_name, clean_split in clean_splits.items():
        rng = np.random.default_rng(stable_seed(seed, sensor_id, split_name))
        episodes = choose_fault_episodes(len(clean_split), anomaly_rate, rng)
        corrupted, mask = apply_profile(clean_split, profile, episodes, rng)
        corrupted.to_csv(client_dir / f"{split_name}.csv", index_label="time")

        positives = int(mask.sum())
        split_meta[split_name] = {
            "rows": int(len(corrupted)),
            "normal_rows": int(len(corrupted) - positives),
            "anomaly_rows": positives,
            "anomaly_rate": float(positives / len(corrupted)),
            "episodes": [[int(a), int(b)] for a, b in episodes],
            "start_time": corrupted.index[0].isoformat(),
            "end_time": corrupted.index[-1].isoformat(),
        }

    manifest = {
        "dataset_version": "cleaned-real-controlled-faults-v1",
        "sensor_id": sensor_id,
        "client_dir": client_dir.name,
        "source_kind": "cleaned_real_ruraliot",
        "source_file": str(source_path.as_posix()),
        "fault_profile": profile,
        "target_anomaly_rate": float(anomaly_rate),
        "seed": int(seed),
        "split_before_fault_injection": True,
        "features": list(FEATURES),
        "cadence_minutes": 10,
        **source_meta,
        "splits": split_meta,
    }
    (client_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_summary(output_root: Path, manifests: list[dict]) -> None:
    summary_path = output_root / "dataset_summary.csv"
    fields = [
        "dataset_version", "sensor_id", "client_dir", "fault_profile", "source_kind",
        "selected_rows", "selected_start_time", "selected_end_time",
        "train_rows", "train_anomaly_rate", "val_rows", "val_anomaly_rate",
        "test_rows", "test_anomaly_rate",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for m in manifests:
            writer.writerow({
                "dataset_version": m["dataset_version"],
                "sensor_id": m["sensor_id"],
                "client_dir": m["client_dir"],
                "fault_profile": m["fault_profile"],
                "source_kind": m["source_kind"],
                "selected_rows": m["selected_rows"],
                "selected_start_time": m["selected_start_time"],
                "selected_end_time": m["selected_end_time"],
                "train_rows": m["splits"]["train"]["rows"],
                "train_anomaly_rate": m["splits"]["train"]["anomaly_rate"],
                "val_rows": m["splits"]["val"]["rows"],
                "val_anomaly_rate": m["splits"]["val"]["anomaly_rate"],
                "test_rows": m["splits"]["test"]["rows"],
                "test_anomaly_rate": m["splits"]["test"]["anomaly_rate"],
            })

    campaign_manifest = {
        "dataset_version": "cleaned-real-controlled-faults-v1",
        "source_kind": "cleaned_real_ruraliot",
        "rows_per_client": manifests[0]["selected_rows"] if manifests else None,
        "target_anomaly_rate": manifests[0]["target_anomaly_rate"] if manifests else None,
        "clients": [m["client_dir"] for m in manifests],
        "profiles": {m["sensor_id"]: m["fault_profile"] for m in manifests},
        "notes": (
            "Chronological split is performed before controlled fault injection. "
            "Each split contains both classes; synthetic VAE data is not used by this benchmark."
        ),
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(campaign_manifest, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=here / "data")
    p.add_argument("--output-dir", type=Path, default=here / "data" / "fl_dataset_real")
    p.add_argument(
        "--rows-per-client", type=int, default=960,
        help="equal number of consecutive real 10-minute measurements per client",
    )
    p.add_argument("--anomaly-rate", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="replace an existing output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows_per_client < 200:
        raise SystemExit("--rows-per-client must be at least 200")
    if not 0.05 <= args.anomaly_rate <= 0.45:
        raise SystemExit("--anomaly-rate must be in [0.05, 0.45]")

    output_root = args.output_dir.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise SystemExit(f"{output_root} already exists and is not empty; use --force")
        import shutil
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifests: list[dict] = []
    print("Preparing thesis FL dataset from CLEANED REAL RuralIoT measurements")
    print(f"  output:       {output_root}")
    print(f"  rows/client:  {args.rows_per_client}")
    print(f"  anomaly rate: {args.anomaly_rate:.1%} in EACH split")
    print(f"  seed:         {args.seed}")

    for sensor_id in PROFILES:
        source = args.data_dir / f"{sensor_id}_CLEANED.csv"
        if not source.exists():
            raise FileNotFoundError(f"missing cleaned real dataset: {source}")
        manifest = write_client(
            sensor_id=sensor_id,
            source_path=source,
            output_root=output_root,
            rows_per_client=args.rows_per_client,
            anomaly_rate=args.anomaly_rate,
            seed=args.seed,
        )
        manifests.append(manifest)
        s = manifest["splits"]
        print(
            f"  {sensor_id:<18} {manifest['fault_profile']:<26} "
            f"train={s['train']['anomaly_rate']:.1%} "
            f"val={s['val']['anomaly_rate']:.1%} "
            f"test={s['test']['anomaly_rate']:.1%}"
        )

    write_summary(output_root, manifests)
    print(f"\nOK: wrote {len(manifests)} real-data FL clients to {output_root}")
    print("Synthetic/VAE files were not modified and remain available for optional experiments.")


if __name__ == "__main__":
    main()
