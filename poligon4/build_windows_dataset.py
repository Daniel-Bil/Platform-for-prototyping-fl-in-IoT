from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd

BASE_COLS = ["value_temp", "value_hum", "value_acid"]

# ---------- pomocnicze: metryki / reguły ----------
def nanstd(x: pd.Series) -> float:
    v = x.to_numpy(dtype=float)
    return float(np.nanstd(v)) if v.size else np.nan

def unique_ratio(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.nunique()/len(x)) if len(x) else 0.0

def lin_slope(y: pd.Series) -> float:
    y = y.astype(float)
    n = len(y)
    if n < 2 or y.isna().all():
        return 0.0
    mask = ~y.isna()
    if mask.sum() < 2:
        return 0.0
    x = np.arange(n, dtype=float)[mask]
    yy = y[mask].to_numpy(dtype=float)
    x = x - x.mean()
    denom = (x**2).sum()
    if denom == 0:
        return 0.0
    return float((x*yy).sum()/denom)

def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if a.isna().all() or b.isna().all():
        return 0.0
    return float(a.astype(float).corr(b.astype(float)))

# ---------- etykietowanie okna ----------
def label_window(win: pd.DataFrame, args) -> dict:
    out = {}

    # metryki bazowe
    for c in BASE_COLS:
        out[f"std_{c}"] = nanstd(win[c])
        out[f"min_{c}"] = float(win[c].min(skipna=True))
        out[f"max_{c}"] = float(win[c].max(skipna=True))
        out[f"uniq_ratio_{c}"] = unique_ratio(win[c])

    # jeżeli mamy policzone d_*, std*_*
    for c in BASE_COLS:
        dcol = f"d_{c}"
        sdcol = f"std{args.std_win}_{c}"
        if dcol in win.columns:
            out[f"max_abs_{dcol}"] = float(win[dcol].abs().max(skipna=True))
        else:
            out[f"max_abs_{dcol}"] = float(win[c].diff().abs().max(skipna=True))
        out[f"mean_{sdcol}"] = float(win[sdcol].mean(skipna=True)) if sdcol in win.columns else np.nan

    # slope
    out["slope_temp"] = lin_slope(win["value_temp"])
    out["slope_hum"]  = lin_slope(win["value_hum"])
    out["slope_acid"] = lin_slope(win["value_acid"])

    # korelacje krzyżowe (heurystyka – opcjonalne)
    out["corr_temp_hum"]  = safe_corr(win["value_temp"], win["value_hum"])
    out["corr_temp_acid"] = safe_corr(win["value_temp"], win["value_acid"])

    # --- Reguły (multi-label) ---
    # GAP = duży skok (po usunięciu sztucznych próbek)
    out["GAP"] = int(
        (out["max_abs_d_value_temp"] >= args.thr_gap_temp) or
        (out["max_abs_d_value_hum"]  >= args.thr_gap_hum)  or
        (out["max_abs_d_value_acid"] >= args.thr_gap_acid)
    )

    # STUCK = niska zmienność
    out["STUCK"] = int(
        (out["std_value_temp"] <= args.thr_stuck_std) or
        (out["std_value_hum"]  <= args.thr_stuck_std) or
        (out["std_value_acid"] <= args.thr_stuck_std)
    )

    # SPIKE = bardzo duża pojedyncza różnica
    out["SPIKE"] = int(
        (out["max_abs_d_value_temp"] >= args.thr_spike_temp) or
        (out["max_abs_d_value_hum"]  >= args.thr_spike_hum)  or
        (out["max_abs_d_value_acid"] >= args.thr_spike_acid)
    )

    # RESET = duża zmiana poziomu (różnica średnich połówek okna)
    def reset_col(col: str, thr: float) -> bool:
        n = len(win)
        if n < 8: return False
        m1 = float(win[col].iloc[: n//2].mean())
        m2 = float(win[col].iloc[n//2 :].mean())
        return abs(m2 - m1) >= thr
    out["RESET"] = int(
        reset_col("value_temp", args.thr_reset_temp) or
        reset_col("value_hum",  args.thr_reset_hum)  or
        reset_col("value_acid", args.thr_reset_acid)
    )

    # DRIFT = wyraźny trend liniowy
    out["DRIFT"] = int(
        (abs(out["slope_temp"]) >= args.thr_drift_slope_temp) or
        (abs(out["slope_hum"])  >= args.thr_drift_slope_hum)  or
        (abs(out["slope_acid"]) >= args.thr_drift_slope_acid)
    )

    # QUANT = mało unikalnych wartości
    out["QUANT"] = int(
        (out["uniq_ratio_value_temp"] <= args.thr_quant_uniq_ratio) or
        (out["uniq_ratio_value_hum"]  <= args.thr_quant_uniq_ratio) or
        (out["uniq_ratio_value_acid"] <= args.thr_quant_uniq_ratio)
    )

    # RANGE = poza zakresem fizycznym
    out["RANGE"] = int(
        (win["value_hum"].lt(0).any() or win["value_hum"].gt(100).any()) or
        (win["value_acid"].lt(0).any() or win["value_acid"].gt(14).any()) or
        (win["value_temp"].lt(args.temp_min).any() or win["value_temp"].gt(args.temp_max).any())
    )

    # CROSS_VIOLATION (opcjonalne, heurystyka: dodatnia korelacja powyżej progu)
    out["CROSS_VIOLATION"] = int(
        (out["corr_temp_hum"]  > args.thr_cross_corr_max) or
        (out["corr_temp_acid"] > args.thr_cross_corr_max)
    )

    # OK = brak błędów
    any_err = any(out[k]==1 for k in ["GAP","STUCK","SPIKE","RESET","DRIFT","QUANT","RANGE","CROSS_VIOLATION"])
    out["OK"] = int(not any_err)
    return out

# ---------- główna procedura ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Katalog wejściowy (rekurencyjnie)")
    ap.add_argument("--pattern", default="*.csv", help="Wzorzec plików (np. *_segment_*.csv)")
    ap.add_argument("--out-dir", required=True, help="Katalog wyjściowy")
    ap.add_argument("--window", type=int, default=40, help="Długość okna (L)")
    ap.add_argument("--stride", type=int, default=10, help="Krok przesuwu okna")
    ap.add_argument("--std-win", type=int, default=11, help="Okno do rolling std")
    ap.add_argument("--drop-artificial", action="store_true", help="Usuń wiersze is_artificial==True jeśli istnieją")
    # progi
    ap.add_argument("--thr-gap-temp",  type=float, default=6.0)
    ap.add_argument("--thr-gap-hum",   type=float, default=20.0)
    ap.add_argument("--thr-gap-acid",  type=float, default=0.5)
    ap.add_argument("--thr-stuck-std", type=float, default=0.02)
    ap.add_argument("--thr-spike-temp", type=float, default=4.0)
    ap.add_argument("--thr-spike-hum",  type=float, default=12.0)
    ap.add_argument("--thr-spike-acid", type=float, default=0.25)
    ap.add_argument("--thr-reset-temp", type=float, default=3.0)
    ap.add_argument("--thr-reset-hum",  type=float, default=10.0)
    ap.add_argument("--thr-reset-acid", type=float, default=0.3)
    ap.add_argument("--thr-drift-slope-temp", type=float, default=0.01)
    ap.add_argument("--thr-drift-slope-hum",  type=float, default=0.05)
    ap.add_argument("--thr-drift-slope-acid", type=float, default=0.001)
    ap.add_argument("--thr-quant-uniq-ratio", type=float, default=0.05)
    ap.add_argument("--thr-cross-corr-max",   type=float, default=0.3)
    ap.add_argument("--temp-min", type=float, default=-20.0)
    ap.add_argument("--temp-max", type=float, default=50.0)
    # kolejność tensorów
    ap.add_argument("--order", choices=["torch","keras","both"], default="both",
                    help="Wyjściowy kształt: torch=[N,C,L], keras=[N,L,C]")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    X_list = []   # do złożenia na końcu
    Y_rows = []

    files = sorted(data_dir.rglob(args.pattern))
    if not files:
        print(f"[WARN] Brak plików w {data_dir} dla wzorca {args.pattern}")
        return

    for p in files:
        try:
            df = pd.read_csv(p).reset_index(drop=True)

            # usuwanie sztucznych próbek
            had_artificial = False
            if args.drop_artificial and "is_artificial" in df.columns:
                had_artificial = bool(df["is_artificial"].any())
                df = df[df["is_artificial"] == False].reset_index(drop=True)

            # uzupełnij brakujące kolumny
            for c in BASE_COLS:
                if c not in df.columns:
                    df[c] = np.nan

            # policz cechy NA OCZYSZCZONYM SYGNALE
            for c in BASE_COLS:
                df[c] = df[c].astype(float)
                df[f"d_{c}"] = df[c].diff()
                df[f"std{args.std_win}_{c}"] = df[c].rolling(
                    args.std_win, min_periods=max(2, args.std_win//2)
                ).std()

            # przygotuj kanały do X
            chan_cols = [
                "value_temp","value_hum","value_acid",
                "d_value_temp","d_value_hum","d_value_acid",
                f"std{args.std_win}_value_temp", f"std{args.std_win}_value_hum", f"std{args.std_win}_value_acid"
            ]
            # fill NaN w cechach pochodnych (początkowe NaN po diff / std)
            df[chan_cols] = df[chan_cols].fillna(0.0)

            n = len(df); L = args.window; S = args.stride
            if n < L:
                continue

            # iteruj okna
            for start in range(0, n - L + 1, S):
                win = df.iloc[start:start+L].reset_index(drop=True)

                # okno nie może być całe NaN w kanałach bazowych
                if np.isnan(win[BASE_COLS].to_numpy(dtype=float)).all():
                    continue

                # etykiety (rule-based)
                labels = label_window(win, args)
                labels["file"] = str(p)
                labels["start_idx"] = start
                labels["had_artificial_in_source"] = int(had_artificial)

                # zbuduj tensor okna: [L, C]
                X_win = win[chan_cols].to_numpy(dtype=float)   # [L, 9]
                X_list.append(X_win)
                Y_rows.append(labels)

            print(f"[OK] {p.relative_to(data_dir)} -> okien: {len(range(0, max(0,n-L+1), S))}")

        except Exception as e:
            print(f"[ERR] {p}: {e}")

    if not X_list:
        print("[WARN] Brak wygenerowanych okien.")
        return

    # złożenie w tablice
    X = np.stack(X_list, axis=0)  # [N, L, C]
    y_df = pd.DataFrame(Y_rows)

    # zapis
    label_cols = ["OK","GAP","STUCK","SPIKE","RESET","DRIFT","QUANT","RANGE","CROSS_VIOLATION"]
    # upewnij się, że istnieją
    for c in label_cols:
        if c not in y_df.columns: y_df[c] = 0

    # kolejność kanałów (opis)
    channel_names = chan_cols

    if args.order in ["torch","both"]:
        X_torch = np.transpose(X, (0, 2, 1))  # [N, C, L]
        np.save(out_dir / "X_torch.npy", X_torch)
    if args.order in ["keras","both"]:
        np.save(out_dir / "X_keras.npy", X)   # [N, L, C]

    y_df.to_csv(out_dir / "y.csv", index=False)

    schema = {
        "channels": channel_names,
        "labels": label_cols,
        "window": args.window,
        "stride": args.stride,
        "std_win": args.std_win,
        "order_saved": args.order
    }
    (out_dir / "label_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"[DONE] Zapisano: {out_dir}")

if __name__ == "__main__":
    main()
