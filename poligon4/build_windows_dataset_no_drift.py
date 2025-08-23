from pathlib import Path
import argparse, json, re
import numpy as np, pandas as pd

BASE_COLS = ["value_temp","value_hum","value_acid"]

def nanstd(x): return float(np.nanstd(x.to_numpy(dtype=float))) if len(x) else np.nan
def unique_ratio(x):
    x = x.dropna()
    return float(x.nunique()/len(x)) if len(x) else 0.0
def safe_corr(a,b):
    if a.isna().all() or b.isna().all(): return 0.0
    return float(a.astype(float).corr(b.astype(float)))

def label_window(win: pd.DataFrame, args) -> dict:
    out = {}
    # metryki podstawowe
    for c in BASE_COLS:
        out[f"std_{c}"] = nanstd(win[c])
        out[f"min_{c}"] = float(win[c].min(skipna=True))
        out[f"max_{c}"] = float(win[c].max(skipna=True))
        out[f"uniq_ratio_{c}"] = unique_ratio(win[c])
        dcol = f"d_{c}"
        if dcol in win.columns:
            out[f"max_abs_{dcol}"] = float(win[dcol].abs().max(skipna=True))
        else:
            out[f"max_abs_{dcol}"] = float(win[c].diff().abs().max(skipna=True))

    out["corr_temp_hum"]  = safe_corr(win["value_temp"], win["value_hum"])
    out["corr_temp_acid"] = safe_corr(win["value_temp"], win["value_acid"])

    # etykiety (BEZ DRIFT)
    out["GAP"] = int(
        (out["max_abs_d_value_temp"] >= args.thr_gap_temp) or
        (out["max_abs_d_value_hum"]  >= args.thr_gap_hum)  or
        (out["max_abs_d_value_acid"] >= args.thr_gap_acid)
    )
    out["STUCK"] = int(
        (out["std_value_temp"] <= args.thr_stuck_std) or
        (out["std_value_hum"]  <= args.thr_stuck_std) or
        (out["std_value_acid"] <= args.thr_stuck_std)
    )
    out["SPIKE"] = int(
        (out["max_abs_d_value_temp"] >= args.thr_spike_temp) or
        (out["max_abs_d_value_hum"]  >= args.thr_spike_hum)  or
        (out["max_abs_d_value_acid"] >= args.thr_spike_acid)
    )
    def reset_col(col, thr):
        n=len(win);
        if n<8: return False
        m1=float(win[col].iloc[:n//2].mean()); m2=float(win[col].iloc[n//2:].mean())
        return abs(m2-m1)>=thr
    out["RESET"] = int(
        reset_col("value_temp", args.thr_reset_temp) or
        reset_col("value_hum",  args.thr_reset_hum)  or
        reset_col("value_acid", args.thr_reset_acid)
    )
    out["QUANT"] = int(
        (out["uniq_ratio_value_temp"] <= args.thr_quant_uniq_ratio) or
        (out["uniq_ratio_value_hum"]  <= args.thr_quant_uniq_ratio) or
        (out["uniq_ratio_value_acid"] <= args.thr_quant_uniq_ratio)
    )
    out["RANGE"] = int(
        (win["value_hum"].lt(0).any() or win["value_hum"].gt(100).any()) or
        (win["value_acid"].lt(0).any() or win["value_acid"].gt(14).any()) or
        (win["value_temp"].lt(args.temp_min).any() or win["value_temp"].gt(args.temp_max).any())
    )
    out["CROSS_VIOLATION"] = int(
        (out["corr_temp_hum"]  > args.thr_cross_corr_max) or
        (out["corr_temp_acid"] > args.thr_cross_corr_max)
    )

    any_err = any(out[k]==1 for k in ["GAP","STUCK","SPIKE","RESET","QUANT","RANGE","CROSS_VIOLATION"])
    out["OK"] = int(not any_err)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--std-win", type=int, default=11)
    ap.add_argument("--drop-artificial", action="store_true")

    # progi (bez drift)
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
    ap.add_argument("--thr-quant-uniq-ratio", type=float, default=0.05)
    ap.add_argument("--thr-cross-corr-max",   type=float, default=0.3)
    ap.add_argument("--temp-min", type=float, default=-20.0)
    ap.add_argument("--temp-max", type=float, default=50.0)
    ap.add_argument("--order", choices=["torch","keras","both"], default="both")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_list, Y_rows = [], []
    files = sorted(data_dir.rglob(args.pattern))
    if not files:
        print(f"[WARN] Brak plików w {data_dir}");
        return

    for p in files:
        try:
            df = pd.read_csv(p).reset_index(drop=True)
            if args.drop_artificial and "is_artificial" in df.columns:
                df = df[df["is_artificial"] == False].reset_index(drop=True)
            for c in BASE_COLS:
                if c not in df.columns: df[c] = np.nan
                df[c] = df[c].astype(float)
                df[f"d_{c}"] = df[c].diff().fillna(0.0)
                df[f"std{args.std_win}_{c}"] = df[c].rolling(
                    args.std_win, min_periods=max(2, args.std_win // 2)
                ).std().fillna(0.0)

            chan_cols = [
                "value_temp", "value_hum", "value_acid",
                "d_value_temp", "d_value_hum", "d_value_acid",
                f"std{args.std_win}_value_temp", f"std{args.std_win}_value_hum", f"std{args.std_win}_value_acid"
            ]
            n, L, S = len(df), args.window, args.stride
            if n < L: continue
            for start in range(0, n - L + 1, S):
                win = df.iloc[start:start + L].reset_index(drop=True)
                if np.isnan(win[BASE_COLS].to_numpy(dtype=float)).all(): continue
                labels = label_window(win, args)
                labels["file"] = str(p);
                labels["start_idx"] = start
                X_win = win[chan_cols].to_numpy(dtype=float)  # [L, C]
                X_list.append(X_win);
                Y_rows.append(labels)
            print(f"[OK] {p.relative_to(data_dir)}")
        except Exception as e:
            print(f"[ERR] {p}: {e}")

    if not X_list:
        print("[WARN] Brak okien.");
        return

    X = np.stack(X_list, axis=0)  # [N,L,C]
    y_df = pd.DataFrame(Y_rows)
    label_cols = ["OK", "GAP", "STUCK", "SPIKE", "RESET", "QUANT", "RANGE", "CROSS_VIOLATION"]
    for c in label_cols:
        if c not in y_df.columns: y_df[c] = 0

    if args.order in ["torch", "both"]:
        np.save(out_dir / "X_torch.npy", np.transpose(X, (0, 2, 1)))
    if args.order in ["keras", "both"]:
        np.save(out_dir / "X_keras.npy", X)
    y_df.to_csv(out_dir / "y.csv", index=False)

    schema = {
        "channels": chan_cols,
        "labels": label_cols,
        "window": args.window, "stride": args.stride, "std_win": args.std_win
    }
    (out_dir / "label_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"[DONE] -> {out_dir}")


if __name__ == "__main__":
    main()
