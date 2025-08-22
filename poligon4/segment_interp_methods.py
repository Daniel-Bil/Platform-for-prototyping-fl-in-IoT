from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SciPy (opcjonalny); jeśli brak, metody inne niż linear spadną do linear
try:
    from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

TIME_COL = "time"
VALUE_COLS = ["value_temp", "value_hum", "value_acid", "value_PV"]

def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Brak kolumny '{TIME_COL}' w {path}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL]).drop_duplicates(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    for c in VALUE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def to_epoch_s(ts: pd.Series) -> np.ndarray:
    # sekundowa oś czasu (float)
    return ts.view("int64").to_numpy() / 1e9

def lin_interp(v0: float, v1: float, frac: float) -> float:
    if pd.isna(v0) and pd.isna(v1): return np.nan
    if pd.isna(v0): return v1
    if pd.isna(v1): return v0
    return float(v0 + (v1 - v0) * frac)

def build_context(df: pd.DataFrame, i_prev: int, i_curr: int, ctx: int):
    """Zwróć ramkę z sąsiadami: [i_prev-ctx+1 ... i_prev] + [i_curr ... i_curr+ctx-1] (przycięte do zakresu)."""
    lo = max(0, i_prev - ctx + 1)
    left = df.iloc[lo:i_prev+1]
    right = df.iloc[i_curr:min(len(df), i_curr + ctx)]
    return pd.concat([left, right], axis=0)

def interpolate_gap_rowwise(method: str,
                            df_ctx: pd.DataFrame,
                            tk_epoch: np.ndarray,
                            t0_epoch: float,
                            t1_epoch: float,
                            v0: dict, v1: dict) -> dict:
    """Zwraca słowniki wartości w tk dla wszystkich VALUE_COLS wg wybranej metody.
       Jeśli metoda niemożliwa (brak SciPy/za mało punktów), fallback = linear pomiędzy (t0,v0) i (t1,v1)."""
    out = {}
    if method == "linear" or not SCIPY_OK:
        # czysta linear pomiędzy końcami
        denom = (t1_epoch - t0_epoch) if (t1_epoch - t0_epoch) != 0 else 1.0
        for tk in tk_epoch:
            frac = (tk - t0_epoch) / denom
            for c in VALUE_COLS:
                out.setdefault(c, []).append(lin_interp(v0[c], v1[c], frac))
        return out

    # przygotuj kontekst
    x_all = to_epoch_s(df_ctx[TIME_COL])
    for c in VALUE_COLS:
        y_all = df_ctx[c].astype(float).to_numpy()
        mask = ~np.isnan(y_all)
        x = x_all[mask]
        y = y_all[mask]

        # wymagania minimalne
        ok = False
        interp = None
        if method == "pchip" and x.size >= 2:
            # zabezpieczenie na powtarzające się czasy
            ux, idx = np.unique(x, return_index=True)
            x, y = ux, y[idx]
            if x.size >= 2:
                interp = PchipInterpolator(x, y, extrapolate=True)
                ok = True
        elif method == "akima" and x.size >= 5:
            ux, idx = np.unique(x, return_index=True)
            x, y = ux, y[idx]
            if x.size >= 5:
                interp = Akima1DInterpolator(x, y)
                ok = True

        if ok and interp is not None:
            yk = interp(tk_epoch)
            out[c] = list(map(float, yk))
        else:
            # fallback: linear między końcami
            denom = (t1_epoch - t0_epoch) if (t1_epoch - t0_epoch) != 0 else 1.0
            out[c] = []
            for tk in tk_epoch:
                frac = (tk - t0_epoch) / denom
                out[c].append(lin_interp(v0[c], v1[c], frac))
    return out

def plot_segment_index(seg_df: pd.DataFrame, out_png: Path, title: str):
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

def process_sensor(sensor_id: str,
                   data_dir: Path,
                   out_root: Path,
                   step: pd.Timedelta,
                   gap_threshold: pd.Timedelta,
                   method: str,
                   context_points: int):
    out_dir = out_root / f"sensor_{sensor_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = out_dir / "graphical_representation"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # wybór pliku (pozwala używać zarówno truncated_v1 jak i surowych dirów)
    candidates = [
        data_dir / f"df_RuralIoT_{sensor_id}_truncated.csv",
        data_dir / f"df_RuralIoT_{sensor_id}.csv",
        data_dir / f"{sensor_id}.csv",
    ]
    in_path = next((p for p in candidates if p.exists()), None)
    if in_path is None:
        raise FileNotFoundError(f"Nie znaleziono pliku dla {sensor_id} w {data_dir}")

    df = load_df(in_path)
    if df.empty:
        print(f"[WARN] {sensor_id}: pusty plik")
        return []

    seg_idx = 0
    seg_rows = []
    manifest = []

    # start segmentu
    first = df.iloc[0]
    seg_rows.append({TIME_COL: first[TIME_COL], **{c: first[c] for c in VALUE_COLS}, "is_artificial": False})
    prev_time = first[TIME_COL]
    prev_vals = {c: first[c] for c in VALUE_COLS}

    def flush():
        nonlocal seg_idx, seg_rows, manifest
        if not seg_rows:
            return
        seg_idx += 1
        seg_df = pd.DataFrame(seg_rows)
        seg_df[VALUE_COLS] = seg_df[VALUE_COLS].round(2)
        seg_df["gap_threshold"] = str(gap_threshold)
        out_csv = out_dir / f"df_RuralIoT_{sensor_id}_segment_{seg_idx:03d}.csv"
        seg_df.to_csv(out_csv, index=False)
        plot_segment_index(seg_df, graph_dir / f"segment_{seg_idx:03d}.png",
                           f"Sensor {sensor_id} — segment {seg_idx:03d} ({method})")
        manifest.append({
            "sensor": sensor_id,
            "segment_id": seg_idx,
            "rows": len(seg_df),
            "artificial_rows": int(seg_df["is_artificial"].sum()),
            "t_start": seg_df[TIME_COL].iloc[0],
            "t_end": seg_df[TIME_COL].iloc[-1],
            "duration": seg_df[TIME_COL].iloc[-1] - seg_df[TIME_COL].iloc[0],
            "method": method,
            "gap_threshold": str(gap_threshold),
            "source": str(in_path),
        })
        seg_rows = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        t = row[TIME_COL]
        delta = t - prev_time

        if delta <= step:
            seg_rows.append({TIME_COL: t, **{c: row[c] for c in VALUE_COLS}, "is_artificial": False})
            prev_time = t
            prev_vals = {c: row[c] for c in VALUE_COLS}
            continue

        # ile kroków 10-minutowych w przerwie (zaokrąglenie do najbliższej liczby interwałów)
        n_steps = round(delta / step)
        n_missing = max(n_steps - 1, 0)

        if delta > gap_threshold:
            # duża luka — kończymy segment
            flush()
            # nowy segment zaczyna się bieżącą próbką
            seg_rows.append({TIME_COL: t, **{c: row[c] for c in VALUE_COLS}, "is_artificial": False})
            prev_time = t
            prev_vals = {c: row[c] for c in VALUE_COLS}
        else:
            # mała luka — wstaw co step, metodą wskazaną
            if n_missing > 0:
                tk = [prev_time + k * step for k in range(1, n_missing + 1)]
                tk_epoch = np.array([(tt.value // 10**9) for tt in pd.to_datetime(tk, utc=True)])
                t0_epoch = prev_time.value // 10**9
                t1_epoch = t.value // 10**9

                # kontekst dla metod nieliniowych
                df_ctx = build_context(df, i_prev=i-1, i_curr=i, ctx=context_points)
                vals_interp = interpolate_gap_rowwise(
                    method=method,
                    df_ctx=df_ctx,
                    tk_epoch=tk_epoch,
                    t0_epoch=t0_epoch,
                    t1_epoch=t1_epoch,
                    v0=prev_vals,
                    v1={c: row[c] for c in VALUE_COLS}
                )
                for j, tt in enumerate(tk):
                    seg_rows.append({
                        TIME_COL: tt,
                        **{c: vals_interp[c][j] for c in VALUE_COLS},
                        "is_artificial": True
                    })

            # dodaj prawdziwą próbkę kończącą lukę
            seg_rows.append({TIME_COL: t, **{c: row[c] for c in VALUE_COLS}, "is_artificial": False})
            prev_time = t
            prev_vals = {c: row[c] for c in VALUE_COLS}

    flush()
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Katalog z danymi (np. truncated_v1 albo ../dane/humidity_ofset_fixed)")
    parser.add_argument("--out-dir", default="segmented_methods_v1")
    parser.add_argument("--sensors", nargs="*", default=["001","002","003","010","21","22","23"])
    parser.add_argument("--gap-threshold", default="3H", help="np. 3H, 180T, 10800S")
    parser.add_argument("--step", default="10T", help="nominalny krok, np. 10T")
    parser.add_argument("--method", choices=["linear", "pchip", "akima"], default="linear",
                        help="metoda interpolacji luk ≤ próg")
    parser.add_argument("--context-points", type=int, default=3,
                        help="liczba punktów kontekstu po każdej stronie luki dla pchip/akima")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    step = pd.to_timedelta(args.step)
    gap_threshold = pd.to_timedelta(args.gap_threshold)

    if args.method != "linear" and not SCIPY_OK:
        print("[WARN] SciPy nie jest dostępne — metoda zostanie zredukowana do 'linear'.")

    manifest_all = []
    for sensor in args.sensors:
        try:
            print(f"[INFO] Sensor {sensor} — method={args.method}")
            manifest_all.extend(process_sensor(
                sensor_id=sensor,
                data_dir=data_dir,
                out_root=out_root,
                step=step,
                gap_threshold=gap_threshold,
                method=args.method if SCIPY_OK else "linear",
                context_points=args.context_points
            ))
        except Exception as e:
            print(f"[ERR] {sensor}: {e}")

    if manifest_all:
        pd.DataFrame(manifest_all).to_csv(out_root / "segments_manifest.csv", index=False)
        print(f"[OK] Manifest -> {out_root/'segments_manifest.csv'}")

if __name__ == "__main__":
    main()
