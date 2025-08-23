from pathlib import Path
import argparse, json, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ====== Konfiguracja bazowych kolumn ======
BASE_COLS = ["value_temp","value_hum","value_acid"]

# ====== Pomocnicze ======
def robust_z_mad(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return 0.6745 * (x - med) / mad  # ≈ z-score odporny

def detect_spike(series: pd.Series, abs_thr: float, rel_thr: float, z_mad_thr: float, max_width: int) -> bool:
    """Robust detekcja SPIKE na różnicach; warunki: duża |Δ| (abs lub względem IQR), wysoki z_MAD, wąska szerokość."""
    s = series.astype(float)
    d = s.diff().fillna(0.0)
    if len(d) == 0:
        return False
    iqr = float(d.quantile(0.75) - d.quantile(0.25))
    rel_cut = rel_thr * max(iqr, 1e-6)
    cond_mag = (d.abs() >= abs_thr) | (d.abs() >= rel_cut)
    z = robust_z_mad(d.abs())
    cond_robust = (z >= z_mad_thr)
    cand = (cond_mag & cond_robust).astype(int)
    run = 0
    for v in cand:
        run = run + 1 if v == 1 else 0
        if v == 0 and 1 <= run <= max_width:
            return True
    return 1 <= run <= max_width

def nanstd(x: pd.Series) -> float:
    v = x.to_numpy(dtype=float)
    return float(np.nanstd(v)) if v.size else np.nan

def unique_ratio(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.nunique()/len(x)) if len(x) else 0.0

def load_csv(path: Path, drop_artificial: bool) -> pd.DataFrame:
    df = pd.read_csv(path).reset_index(drop=True)
    # kolumny bazowe
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].astype(float)
    # usuń interpolacje, jeśli jest kolumna
    if drop_artificial and "is_artificial" in df.columns:
        df = df[df["is_artificial"] == False].reset_index(drop=True)
    return df

def extract_sensor_and_segment(path: Path):
    # sensor z nazwy pliku df_RuralIoT_<ID>_segment_<NNN>.csv
    m1 = re.search(r"df_RuralIoT_([^_]+)", path.name)
    sensor = m1.group(1) if m1 else "unknown"
    m2 = re.search(r"segment_(\d+)", path.name)
    seg_id = int(m2.group(1)) if m2 else None
    return sensor, seg_id

# ====== Labelowanie okna (bez CROSS_VIOLATION; z GAP z granic) ======
def label_window(win: pd.DataFrame, args, boundary_offsets_in_win: list[int]) -> dict:
    out = {}

    # GAP: jeśli w oknie są granice segmentów
    out["GAP"] = int(len(boundary_offsets_in_win) > 0)

    # STUCK: bardzo niska zmienność w dowolnym kanale
    out["STUCK"] = int(
        (nanstd(win["value_temp"]) <= args.thr_stuck_std) or
        (nanstd(win["value_hum"])  <= args.thr_stuck_std) or
        (nanstd(win["value_acid"]) <= args.thr_stuck_std)
    )

    # QUANT: mało unikalnych poziomów
    # out["QUANT"] = int(
    #     (unique_ratio(win["value_temp"]) <= args.thr_quant_uniq_ratio) or
    #     (unique_ratio(win["value_hum"])  <= args.thr_quant_uniq_ratio) or
    #     (unique_ratio(win["value_acid"]) <= args.thr_quant_uniq_ratio)
    # )

    # RANGE: poza zakresem fizycznym
    out["RANGE"] = int(
        (win["value_hum"].lt(0).any() or win["value_hum"].gt(100).any()) or
        (win["value_acid"].lt(0).any() or win["value_acid"].gt(14).any()) or
        (win["value_temp"].lt(args.temp_min).any() or win["value_temp"].gt(args.temp_max).any())
    )

    # RESET: duża różnica średnich połówek (w dowolnym kanale)
    # def reset_col(col, thr):
    #     n = len(win)
    #     if n < 8: return False
    #     m1 = float(win[col].iloc[: n//2].mean())
    #     m2 = float(win[col].iloc[n//2 :].mean())
    #     return abs(m2 - m1) >= thr
    # out["RESET"] = int(
    #     reset_col("value_temp", args.thr_reset_temp) or
    #     reset_col("value_hum",  args.thr_reset_hum)  or
    #     reset_col("value_acid", args.thr_reset_acid)
    # )

    # SPIKE: robust (MAD z-score + progi abs/rel + max szerokość)
    spike_temp = detect_spike(win["value_temp"], args.spike_abs_temp, args.spike_rel_temp, args.spike_zmad, args.spike_max_width)
    spike_hum  = detect_spike(win["value_hum"],  args.spike_abs_hum,  args.spike_rel_hum,  args.spike_zmad, args.spike_max_width)
    spike_acid = detect_spike(win["value_acid"], args.spike_abs_acid, args.spike_rel_acid, args.spike_zmad, args.spike_max_width)
    out["SPIKE"] = int(spike_temp or spike_hum or spike_acid)

    # OK = brak błędów
    any_err = any(out[k]==1 for k in ["GAP","STUCK","SPIKE","RANGE"])
    out["OK"] = int(not any_err)
    return out

# ====== Plot jednego okna (z pionowymi liniami granic) ======
def plot_window(win: pd.DataFrame, chan_cols: list[str], boundary_offsets: list[int], out_path: Path, title: str):
    x = np.arange(len(win))
    C = len(chan_cols)
    fig, axes = plt.subplots(C, 1, figsize=(12, max(6, 2*C)), sharex=True)
    if C == 1: axes = [axes]
    for i, col in enumerate(chan_cols):
        axes[i].plot(x, win[col].astype(float).values, lw=1)
        for b in boundary_offsets:
            axes[i].axvline(b+0.5, color="gray", linestyle="--", linewidth=1)
        axes[i].set_ylabel(col)
    axes[-1].set_xlabel("sample index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

# ====== Sklejanie segmentów per sensor i generowanie okien ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Katalog z segmentami (rekurencyjnie). Np. segmented_methods_pchip_4H")
    ap.add_argument("--pattern", default="*_segment_*.csv", help="Wzorzec plików segmentów")
    ap.add_argument("--out-dir", required=True, help="Katalog wyjściowy na dataset i ploty")
    ap.add_argument("--window", type=int, default=40, help="Długość okna (L)")
    ap.add_argument("--stride", type=int, default=10, help="Krok przesuwu")
    ap.add_argument("--std-win", type=int, default=11, help="Okno do rolling std (cecha)")
    ap.add_argument("--drop-artificial", action="store_true", help="Usuń punkty is_artificial==True przed obliczeniami")
    ap.add_argument("--min-short-seg", type=int, default=1, help="Próg tylko informacyjny; segmenty krótsze i tak są łączone kaskadowo")

    # Progi detekcji (BEZ cross-violation)
    ap.add_argument("--thr-stuck-std", type=float, default=0.02)
    # ap.add_argument("--thr-reset-temp", type=float, default=3.0)
    # ap.add_argument("--thr-reset-hum",  type=float, default=10.0)
    # ap.add_argument("--thr-reset-acid", type=float, default=0.3)
    # ap.add_argument("--thr-quant-uniq-ratio", type=float, default=0.05)
    ap.add_argument("--temp-min", type=float, default=-20.0)
    ap.add_argument("--temp-max", type=float, default=50.0)

    # SPIKE (robust)
    ap.add_argument("--spike-abs-temp", type=float, default=4.5)
    ap.add_argument("--spike-abs-hum",  type=float, default=8.0)
    ap.add_argument("--spike-abs-acid", type=float, default=1.0)
    ap.add_argument("--spike-rel-temp", type=float, default=0.35, help="część IQR(|Δ|)")
    ap.add_argument("--spike-rel-hum",  type=float, default=0.40, help="część IQR(|Δ|)")
    ap.add_argument("--spike-rel-acid", type=float, default=0.35, help="część IQR(|Δ|)")
    ap.add_argument("--spike-zmad", type=float, default=6.5)
    ap.add_argument("--spike-max-width", type=int, default=3)

    # Zapis tensora
    ap.add_argument("--order", choices=["torch","keras","both"], default="both")

    # Plotowanie
    ap.add_argument("--plot-label", choices=["ALL","OK","GAP","STUCK","SPIKE","RANGE","NONE"], default="GAP",
                    help="Które okna zapisać jako PNG (ALL/konkretna etykieta/NONE)")
    ap.add_argument("--plot-limit", type=int, default=0, help="Limit liczby PNG (0 = bez limitu)")

    args = ap.parse_args()

    in_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Zbierz segmenty per sensor
    per_sensor = defaultdict(list)
    files = sorted(in_dir.rglob(args.pattern))
    print(f"[INFO] Znaleziono {len(files)} plików z segmentami (pattern={args.pattern})")
    if not files:
        print("[WARN] Brak plików do przetworzenia."); return

    for p in files:
        sensor, seg_id = extract_sensor_and_segment(p)
        per_sensor[sensor].append((seg_id, p))
    # posortuj segmenty wg seg_id
    for s in per_sensor:
        per_sensor[s].sort(key=lambda x: (x[0] is None, x[0]))

    # 2) Dla każdego sensora — sklej, oznacz granice, generuj okna
    X_list, Y_rows = [], []
    total_windows = 0
    plots_written = 0

    for sensor, seg_list in per_sensor.items():
        # wczytaj i oczyść segmenty
        seg_dfs = []
        seg_lengths = []
        for seg_id, path in seg_list:
            df = load_csv(path, drop_artificial=args.drop_artificial)
            # policz cechy (na czystym)
            for c in BASE_COLS:
                df[f"d_{c}"] = df[c].diff().fillna(0.0)
                df[f"std{args.std_win}_{c}"] = df[c].rolling(
                    args.std_win, min_periods=max(2, args.std_win//2)
                ).std().fillna(0.0)
            df["__seg_id__"] = (seg_id if seg_id is not None else -1)
            seg_dfs.append(df)
            seg_lengths.append(len(df))

        if not seg_dfs:
            continue

        # sklej w jedną „oś” + znajdź indeksy granic (tuż przed nowym segmentem)
        timeline = pd.DataFrame([], columns=seg_dfs[0].columns)
        boundaries = []  # indeksy (0..N-1) w timeline, gdzie KOŃCZY się segment
        cursor = 0
        for i, df in enumerate(seg_dfs):
            n = len(df)
            if n == 0:
                # pusty segment – granica i tak istnieje, ale nic nie dodajemy
                continue
            timeline = pd.concat([timeline, df], ignore_index=True)
            cursor += n
            if i < len(seg_dfs) - 1:
                boundaries.append(cursor - 1)  # granica po ostatnim indeksie tego segmentu

        # kanały do modelu
        chan_cols = [
            "value_temp","value_hum","value_acid",
            "d_value_temp","d_value_hum","d_value_acid",
            f"std{args.std_win}_value_temp", f"std{args.std_win}_value_hum", f"std{args.std_win}_value_acid"
        ]
        # brakujące -> 0 (pochodne/std), a wartości – zostaw jak są (NaN okno i tak odfiltrujemy)
        timeline[[c for c in chan_cols if c.startswith(("d_","std"))]] = timeline[[c for c in chan_cols if c.startswith(("d_","std"))]].fillna(0.0)

        n = len(timeline); L = args.window; S = args.stride
        if n < L:
            print(f"[SKIP] sensor {sensor}: za krótka oś ({n} < {L})")
            continue

        # wygeneruj okna po całej osi
        for start in range(0, n - L + 1, S):
            end = start + L - 1
            win = timeline.iloc[start:start+L].reset_index(drop=True)

            # odfiltruj okna kompletnie puste w wartościach
            if np.isnan(win[BASE_COLS].to_numpy(dtype=float)).all():
                continue

            # granice wewnątrz okna: b in (start .. end]
            b_offsets = [int(b - start) for b in boundaries if start < b <= end]

            # etykiety
            labels = label_window(win, args, b_offsets)
            labels["sensor"] = sensor
            labels["start_idx"] = start
            labels["boundaries_in_window"] = ";".join(map(str, b_offsets)) if b_offsets else ""
            labels["OK"]=int(labels["OK"])
            labels["GAP"]=int(labels["GAP"])
            labels["STUCK"]=int(labels["STUCK"])
            labels["SPIKE"]=int(labels["SPIKE"])
            labels["RANGE"] = int(labels["RANGE"])

            # tensor okna
            X_win = win[chan_cols].to_numpy(dtype=float)   # [L, C]
            X_list.append(X_win)
            Y_rows.append(labels)
            total_windows += 1

        print(f"[OK] sensor {sensor}: okien={total_windows}")

    if not X_list:
        print("[WARN] Brak wygenerowanych okien."); return

    # 3) Złóż i zapisz dataset
    X = np.stack(X_list, axis=0)  # [N,L,C]
    y_df = pd.DataFrame(Y_rows)

    label_cols = ["OK","GAP","STUCK","SPIKE","RANGE"]
    for c in label_cols:
        if c not in y_df.columns: y_df[c] = 0

    # zapis tensora
    if args.order in ["torch","both"]:
        np.save(Path(out_dir) / "X_torch.npy", np.transpose(X, (0,2,1)))  # [N,C,L]
    if args.order in ["keras","both"]:
        np.save(Path(out_dir) / "X_keras.npy", X)                         # [N,L,C]
    y_df.to_csv(Path(out_dir) / "y.csv", index=False)

    schema = {
        "channels": [
            "value_temp","value_hum","value_acid",
            "d_value_temp","d_value_hum","d_value_acid",
            f"std{args.std_win}_value_temp", f"std{args.std_win}_value_hum", f"std{args.std_win}_value_acid"
        ],
        "labels": label_cols,
        "window": args.window, "stride": args.stride, "std_win": args.std_win,
        "notes": "GAP=okno przecina granice segmentów; bez CROSS_VIOLATION; SPIKE robust"
    }
    (Path(out_dir)/"label_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"[DONE] dataset -> {out_dir}")
    print(f"[STATS] okien: {len(X)}, etykiety: {label_cols}")

    # 4) (Opcjonalnie) rysowanie PNG dla wybranej etykiety
    if args.plot_label != "NONE":
        sel = y_df.index if args.plot_label=="ALL" else y_df.index[y_df[args.plot_label]==1]
        sel = sel.tolist()
        if args.plot_limit and args.plot_limit > 0:
            sel = sel[:args.plot_limit]
        plots_dir = Path(out_dir) / f"plots_{args.plot_label.lower()}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        chan_cols_plot = schema["channels"][:3]  # tylko 3 wartości (czytelniej)
        for i in sel:
            win_df = pd.DataFrame(X[i], columns=schema["channels"])
            b_offsets = []
            if "boundaries_in_window" in y_df.columns and isinstance(y_df.loc[i,"boundaries_in_window"], str):
                b_offsets = [int(v) for v in y_df.loc[i,"boundaries_in_window"].split(";") if v!=""]
            # zbuduj listę aktywnych etykiet dla okna i ładny napis
            label_cols = ["GAP", "STUCK", "SPIKE", "RANGE", "OK"]
            labels_here = [lbl for lbl in label_cols
                           if (lbl in y_df.columns and int(y_df.loc[i, lbl]) == 1)]
            # jeśli jedyna etykieta to OK — zostaw OK; w przeciwnym razie usuń OK
            if len(labels_here) > 1 and "OK" in labels_here:
                labels_here.remove("OK")

            labels_str = ",".join(labels_here) if labels_here else "NONE"

            title = f"win#{i} | labels={labels_str} | boundaries={b_offsets}"
            out_png = plots_dir / f"window_{i:06d}__{labels_str}.png"

            plot_window(win_df, chan_cols_plot, b_offsets, out_png, title)
            plots_written += 1
        print(f"[PLOTS] zapisano {plots_written} PNG -> {plots_dir}")

if __name__ == "__main__":
    main()
