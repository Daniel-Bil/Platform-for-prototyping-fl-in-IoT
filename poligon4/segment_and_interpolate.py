from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TIME_COL = "time"
VALUE_COLS = ["value_temp", "value_hum", "value_acid", "value_PV"]

def load_df(preferred_path: Path, fallbacks: list[Path]) -> pd.DataFrame:
    if preferred_path.exists():
        path = preferred_path
    else:
        found = None
        for fb in fallbacks:
            if fb.exists():
                found = fb
                break
        if not found:
            raise FileNotFoundError(f"Brak pliku: {preferred_path.name} oraz {', '.join(p.name for p in fallbacks)}")
        path = found

    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Brak kolumny '{TIME_COL}' w {path}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL]).drop_duplicates(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    for c in VALUE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def lin_interp(v0: float, v1: float, frac: float) -> float:
    if pd.isna(v0) and pd.isna(v1):
        return np.nan
    if pd.isna(v0):
        return v1
    if pd.isna(v1):
        return v0
    return v0 + (v1 - v0) * frac

def plot_segment(seg_df: pd.DataFrame, out_png: Path, title: str):
    # X = sample index (nie czas)
    seg = seg_df.reset_index(drop=True)
    x = seg.index

    fig, axes = plt.subplots(len(VALUE_COLS), 1, figsize=(14, 9), sharex=True)
    for i, col in enumerate(VALUE_COLS):
        axes[i].plot(x, seg[col], lw=1, label="data")
        art = seg[seg["is_artificial"]]
        if not art.empty:
            axes[i].scatter(art.index, art[col], marker="x", c="red", label="artificial")
        axes[i].set_ylabel(col)
        axes[i].legend()
    axes[-1].set_xlabel("sample index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def process_sensor(sensor_id: str, data_dir: Path, out_root: Path,
                   step: pd.Timedelta, max_small_gap: pd.Timedelta):
    out_dir = out_root / f"sensor_{sensor_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = out_dir / "graphical_representation"
    graph_dir.mkdir(parents=True, exist_ok=True)

    in_path = data_dir / f"df_RuralIoT_{sensor_id}.csv"
    if not in_path.exists():
        in_path = data_dir / f"df_RuralIoT_{sensor_id}_truncated.csv"
    if not in_path.exists():
        in_path = data_dir / f"{sensor_id}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Nie znalazłem pliku dla {sensor_id} w {data_dir}")

    df = load_df(in_path, [])

    if df.empty:
        print(f"[WARN] Sensor {sensor_id}: pusty plik — pomijam")
        return []

    segments_meta = []
    seg_rows = []
    seg_idx = 0

    def flush_segment():
        nonlocal seg_rows, seg_idx
        if not seg_rows:
            return
        seg_idx += 1
        seg_df = pd.DataFrame(seg_rows)
        seg_df[VALUE_COLS] = seg_df[VALUE_COLS].round(2)
        seg_df["gap_threshold"] = str(max_small_gap)  # <---- nowa kolumna
        out_csv = out_dir / f"df_RuralIoT_{sensor_id}_segment_{seg_idx:03d}.csv"
        seg_df.to_csv(out_csv, index=False)
        plot_segment(seg_df, graph_dir / f"segment_{seg_idx:03d}.png",
                     f"Sensor {sensor_id} — segment {seg_idx:03d}")
        meta = {
            "sensor": sensor_id,
            "segment_id": seg_idx,
            "rows": len(seg_df),
            "artificial_rows": int(seg_df["is_artificial"].sum()),
            "t_start": seg_df[TIME_COL].iloc[0],
            "t_end": seg_df[TIME_COL].iloc[-1],
            "duration": seg_df[TIME_COL].iloc[-1] - seg_df[TIME_COL].iloc[0],
            "gap_threshold": str(max_small_gap)
        }
        segments_meta.append(meta)
        seg_rows = []

    # Start
    first_row = df.iloc[0]
    seg_rows.append({TIME_COL: first_row[TIME_COL], **{c: first_row[c] for c in VALUE_COLS}, "is_artificial": False})
    prev_time = first_row[TIME_COL]
    prev_vals = {c: first_row[c] for c in VALUE_COLS}

    for i in range(1, len(df)):
        row = df.iloc[i]
        t = row[TIME_COL]
        delta = t - prev_time

        if delta <= step:
            seg_rows.append({TIME_COL: t, **{c: row[c] for c in VALUE_COLS}, "is_artificial": False})
            prev_time = t; prev_vals = {c: row[c] for c in VALUE_COLS}
            continue

        n_steps = round(delta / step)  # <---- zmiana floor -> round
        n_missing = max(n_steps - 1, 0)

        if delta > max_small_gap:
            flush_segment()
            seg_rows.append({TIME_COL: t, **{c: row[c] for c in VALUE_COLS}, "is_artificial": False})
            prev_time = t; prev_vals = {c: row[c] for c in VALUE_COLS}
        else:
            for k in range(1, n_missing + 1):
                tk = prev_time + k * step
                frac = (tk - prev_time) / (t - prev_time)
                vals = {c: lin_interp(prev_vals[c], row[c], float(frac)) for c in VALUE_COLS}
                seg_rows.append({TIME_COL: tk, **vals, "is_artificial": True})
            seg_rows.append({TIME_COL: t, **{c: row[c] for c in VALUE_COLS}, "is_artificial": False})
            prev_time = t; prev_vals = {c: row[c] for c in VALUE_COLS}

    flush_segment()
    return segments_meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Katalog z danymi wejściowymi (np. truncated_v1 albo ../dane/humidity_ofset_fixed)")
    parser.add_argument("--out-dir", default="segmented_simple_v1")
    parser.add_argument("--sensors", nargs="*", default=["001","002","003","010","21","22","23"])
    parser.add_argument("--gap-threshold", default="3H", help="np. 3H, 180T, 10800S")
    parser.add_argument("--step", default="10T", help="nominalny krok np. 10T, 600S")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    max_small_gap = pd.to_timedelta(args.gap_threshold)
    step = pd.to_timedelta(args.step)

    manifest = []
    for sensor in args.sensors:
        try:
            print(f"[INFO] Sensor {sensor}")
            manifest.extend(process_sensor(sensor, data_dir, out_root, step, max_small_gap))
        except Exception as e:
            print(f"[ERR] Sensor {sensor}: {e}")

    if manifest:
        pd.DataFrame(manifest).to_csv(out_root / "segments_manifest.csv", index=False)
        print(f"[OK] Manifest -> {out_root/'segments_manifest.csv'}")


if __name__ == "__main__":
    main()
