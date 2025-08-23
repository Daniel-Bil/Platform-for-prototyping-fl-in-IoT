from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import re

VALUE_COLS = ["value_temp", "value_hum", "value_acid"]

def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p).reset_index(drop=True)
    for c in VALUE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def nan_safe_std(x: pd.Series) -> float:
    return float(np.nanstd(x.values)) if len(x) else np.nan

def corr_safe(a: pd.Series, b: pd.Series) -> float:
    aa, bb = a.astype(float), b.astype(float)
    if aa.isna().all() or bb.isna().all(): return np.nan
    return float(aa.corr(bb))

def ratio_unique(y: pd.Series) -> float:
    y = y.dropna()
    return float(y.nunique()/len(y)) if len(y) else 0.0

def pct_artificial(df: pd.DataFrame) -> float:
    return float((df["is_artificial"]==True).mean()) if "is_artificial" in df.columns else 0.0

def first_std_col(df: pd.DataFrame, col_base: str) -> str|None:
    pat = re.compile(rf"^std\d+_{re.escape(col_base)}$")
    for c in df.columns:
        if pat.match(c): return c
    return None

def classify_one(df: pd.DataFrame, args) -> dict:
    out = {}
    out["rows"] = len(df)
    out["pct_artificial"] = pct_artificial(df)

    # statystyki bazowe
    for c in VALUE_COLS:
        out[f"std_{c}"] = nan_safe_std(df[c])
        out[f"min_{c}"] = float(np.nanmin(df[c].values)) if not df[c].isna().all() else np.nan
        out[f"max_{c}"] = float(np.nanmax(df[c].values)) if not df[c].isna().all() else np.nan
        out[f"uniq_ratio_{c}"] = ratio_unique(df[c])

    # std okienne (jeśli dostępne)
    std_cols = {c: first_std_col(df, c) for c in VALUE_COLS}

    # różnice (dla spike/reset/gap)
    for c in VALUE_COLS:
        dcol = f"d_{c}"
        if dcol in df.columns:
            out[f"max_abs_{dcol}"] = float(df[dcol].abs().max(skipna=True))
        else:
            out[f"max_abs_{dcol}"] = float(df[c].astype(float).diff().abs().max(skipna=True))

    # korelacje krzyżowe (heurystyka do CROSS_VIOLATION)
    out["corr_temp_hum"]  = corr_safe(df["value_temp"], df["value_hum"])
    out["corr_temp_acid"] = corr_safe(df["value_temp"], df["value_acid"])

    # -------- etykiety (BEZ DRIFT) --------
    out["MISSING_GAP"] = int(out["pct_artificial"] >= args.thr_missing_pct)

    def stuck_from_std(col: str) -> bool:
        sc = std_cols.get(col)
        if sc is None:
            return (out[f"std_{col}"] <= args.thr_stuck_std*0.5) or (out[f"uniq_ratio_{col}"] <= args.thr_quant_uniq_ratio)
        s = df[sc]
        thr, L = args.thr_stuck_std, args.thr_stuck_len
        below = (s < thr).astype(int).fillna(0)
        run = max_run = 0
        for v in below:
            run = run+1 if v==1 else 0
            if run > max_run: max_run = run
        return max_run >= L

    out["STUCK_TEMP"] = int(stuck_from_std("value_temp"))
    out["STUCK_HUM"]  = int(stuck_from_std("value_hum"))
    out["STUCK_ACID"] = int(stuck_from_std("value_acid"))
    out["STUCK"]      = int(out["STUCK_TEMP"] or out["STUCK_HUM"] or out["STUCK_ACID"])

    def spike_flag(col: str, thr: float) -> int:
        return int(out[f"max_abs_d_{col}"] >= thr)

    out["SPIKE_TEMP"] = spike_flag("value_temp", args.thr_spike_temp)
    out["SPIKE_HUM"]  = spike_flag("value_hum",  args.thr_spike_hum)
    out["SPIKE_ACID"] = spike_flag("value_acid", args.thr_spike_acid)
    out["SPIKE"]      = int(out["SPIKE_TEMP"] or out["SPIKE_HUM"] or out["SPIKE_ACID"])

    def reset_flag(col: str, thr: float) -> int:
        n = len(df)
        if out[f"max_abs_d_{col}"] < thr or n < 8: return 0
        m1 = float(df[col].iloc[:n//2].mean()); m2 = float(df[col].iloc[n//2:].mean())
        return int(abs(m2 - m1) >= thr)

    out["RESET_TEMP"] = reset_flag("value_temp", args.thr_reset_temp)
    out["RESET_HUM"]  = reset_flag("value_hum",  args.thr_reset_hum)
    out["RESET_ACID"] = reset_flag("value_acid", args.thr_reset_acid)
    out["RESET"]      = int(out["RESET_TEMP"] or out["RESET_HUM"] or out["RESET_ACID"])

    out["QUANT_TEMP"] = int(out["uniq_ratio_value_temp"] <= args.thr_quant_uniq_ratio)
    out["QUANT_HUM"]  = int(out["uniq_ratio_value_hum"]  <= args.thr_quant_uniq_ratio)
    out["QUANT_ACID"] = int(out["uniq_ratio_value_acid"] <= args.thr_quant_uniq_ratio)
    out["QUANT"]      = int(out["QUANT_TEMP"] or out["QUANT_HUM"] or out["QUANT_ACID"])

    out["RANGE_TEMP"] = int((df["value_temp"] < args.temp_min).any() or (df["value_temp"] > args.temp_max).any())
    out["RANGE_HUM"]  = int((df["value_hum"]  < 0).any() or (df["value_hum"] > 100).any())
    out["RANGE_ACID"] = int((df["value_acid"] < 0).any() or (df["value_acid"] > 14).any())
    out["RANGE"]      = int(out["RANGE_TEMP"] or out["RANGE_HUM"] or out["RANGE_ACID"])

    cv1 = (not np.isnan(out["corr_temp_hum"]))  and (out["corr_temp_hum"]  > args.thr_cross_corr_max)
    cv2 = (not np.isnan(out["corr_temp_acid"])) and (out["corr_temp_acid"] > args.thr_cross_corr_max)
    out["CROSS_VIOLATION"] = int(cv1 or cv2)

    labels = ["MISSING_GAP","STUCK","SPIKE","RESET","QUANT","RANGE","CROSS_VIOLATION"]
    out["pred_any_error"] = int(any(out[l]==1 for l in labels))
    out["OK"] = int(not out["pred_any_error"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--out", default="manifest_rules_no_drift.csv")

    ap.add_argument("--thr-missing-pct", type=float, default=0.05)
    ap.add_argument("--thr-stuck-std", type=float, default=0.02)
    ap.add_argument("--thr-stuck-len", type=int, default=30)
    ap.add_argument("--thr-spike-temp", type=float, default=4.0)
    ap.add_argument("--thr-spike-hum",  type=float, default=12.0)
    ap.add_argument("--thr-spike-acid", type=float, default=0.25)
    ap.add_argument("--thr-reset-temp", type=float, default=3.0)
    ap.add_argument("--thr-reset-hum",  type=float, default=10.0)
    ap.add_argument("--thr-reset-acid", type=float, default=0.3)
    ap.add_argument("--thr-quant-uniq-ratio", type=float, default=0.05)
    ap.add_argument("--thr-cross-corr-max", type=float, default=0.3)
    ap.add_argument("--temp-min", type=float, default=-20.0)
    ap.add_argument("--temp-max", type=float, default=50.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = []
    for p in sorted(data_dir.rglob(args.pattern)):
        try:
            df = load_csv(p)
            res = classify_one(df, args)
            res["file"] = str(p)
            m = re.search(r"df_RuralIoT_([^_]+)", p.name)
            if m: res["sensor"] = m.group(1)
            m2 = re.search(r"segment_(\d+)", p.name)
            if m2: res["segment_id"] = int(m2.group(1))
            rows.append(res)
            print(f"[OK] {p.relative_to(data_dir)}")
        except Exception as e:
            print(f"[ERR] {p}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[DONE] -> {args.out}")

if __name__ == "__main__":
    main()
