from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import re

VALUE_COLS = ["value_temp", "value_hum", "value_acid"]

def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # wymagane kolumny
    for c in VALUE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df.reset_index(drop=True)

def nan_safe_std(x: pd.Series) -> float:
    return float(np.nanstd(x.values)) if len(x) else np.nan

def linreg_slope(y: pd.Series) -> float:
    n = len(y)
    if n < 2 or y.isna().all():
        return np.nan
    x = np.arange(n, dtype=float)
    mask = ~y.isna()
    if mask.sum() < 2:
        return np.nan
    x = x[mask]; yy = y[mask].astype(float).values
    x = x - x.mean()
    denom = (x**2).sum()
    if denom == 0:
        return 0.0
    return float((x*yy).sum()/denom)

def corr_safe(a: pd.Series, b: pd.Series) -> float:
    aa = a.astype(float); bb = b.astype(float)
    if aa.isna().all() or bb.isna().all():
        return np.nan
    return float(aa.corr(bb))

def ratio_unique(y: pd.Series) -> float:
    y = y.dropna()
    if len(y) == 0: return 0.0
    return float(y.nunique()/len(y))

def pct_artificial(df: pd.DataFrame) -> float:
    if "is_artificial" not in df.columns:
        return 0.0
    return float((df["is_artificial"]==True).mean())

def first_valid_std_col(df: pd.DataFrame, col_base: str) -> str | None:
    # znajdź pierwszą kolumnę std*_<col_base>, jeśli istnieje
    pat = re.compile(rf"^std\d+_{re.escape(col_base)}$")
    for c in df.columns:
        if pat.match(c): return c
    return None

def first_deriv_col(df: pd.DataFrame, col_base: str) -> str | None:
    c = f"d_{col_base}"
    return c if c in df.columns else None

def classify_one(df: pd.DataFrame, args) -> dict:
    out = {}

    # --- pomocnicze metryki ---
    n = len(df)
    out["rows"] = n
    out["pct_artificial"] = pct_artificial(df)

    # STD okienne (jeśli policzone wcześniej) – przydadzą się do STUCK
    std_cols = {}
    for c in VALUE_COLS:
        std_col = first_valid_std_col(df, c)
        std_cols[c] = std_col
        if std_col is not None:
            out[f"mean_{std_col}"] = float(df[std_col].mean(skipna=True))

    # różnice – przydadzą się do SPIKE/RESET
    diff_cols = {}
    for c in VALUE_COLS:
        dcol = first_deriv_col(df, c)
        diff_cols[c] = dcol
        if dcol is not None:
            out[f"max_abs_{dcol}"] = float(df[dcol].abs().max(skipna=True))

    # podstawowe statystyki
    for c in VALUE_COLS:
        out[f"std_{c}"] = nan_safe_std(df[c])
        out[f"min_{c}"] = float(np.nanmin(df[c].values)) if not df[c].isna().all() else np.nan
        out[f"max_{c}"] = float(np.nanmax(df[c].values)) if not df[c].isna().all() else np.nan
        out[f"uniq_ratio_{c}"] = ratio_unique(df[c])

    # slope (drift)
    out["slope_temp"] = linreg_slope(df["value_temp"])
    out["slope_hum"]  = linreg_slope(df["value_hum"])
    out["slope_acid"] = linreg_slope(df["value_acid"])

    # korelacje krzyżowe (heurystyki)
    out["corr_temp_hum"]  = corr_safe(df["value_temp"], df["value_hum"])
    out["corr_temp_acid"] = corr_safe(df["value_temp"], df["value_acid"])

    # --- Reguły (etykiety) ---

    # MISSING_GAP: jeśli mamy kolumnę is_artificial i jej odsetek > próg
    out["MISSING_GAP"] = int(out["pct_artificial"] >= args.thr_missing_pct)

    # STUCK: rolling std bardzo niskie przez długi czas
    def stuck_from_std(col: str) -> bool:
        std_col = std_cols.get(col)
        if std_col is None:
            # fallback: użyj globalnej std oraz liczby unikalnych wartości
            return (out[f"std_{col}"] <= args.thr_stuck_std*0.5) or (out[f"uniq_ratio_{col}"] <= args.thr_quant_uniq_ratio)
        s = df[std_col]
        # sekwencja >= L z wartościami < thr
        thr = args.thr_stuck_std
        L   = args.thr_stuck_len
        below = (s < thr).astype(int).fillna(0)
        # maksymalna długość ciągu
        max_run = 0; run = 0
        for v in below:
            if v == 1: run += 1; max_run = max(max_run, run)
            else: run = 0
        return max_run >= L

    out["STUCK_TEMP"] = int(stuck_from_std("value_temp"))
    out["STUCK_HUM"]  = int(stuck_from_std("value_hum"))
    out["STUCK_ACID"] = int(stuck_from_std("value_acid"))
    out["STUCK"]      = int(out["STUCK_TEMP"] or out["STUCK_HUM"] or out["STUCK_ACID"])

    # SPIKE: maks |d_*| powyżej progu
    def spike_flag(col: str, thr: float) -> int:
        dcol = diff_cols.get(col)
        if dcol is None:
            # fallback: użyj różnicy sąsiednich po surowych wartościach
            d = df[col].astype(float).diff().abs().max(skipna=True)
            return int(d >= thr)
        return int(df[dcol].abs().max(skipna=True) >= thr)

    out["SPIKE_TEMP"] = spike_flag("value_temp", args.thr_spike_temp)
    out["SPIKE_HUM"]  = spike_flag("value_hum",  args.thr_spike_hum)
    out["SPIKE_ACID"] = spike_flag("value_acid", args.thr_spike_acid)
    out["SPIKE"]      = int(out["SPIKE_TEMP"] or out["SPIKE_HUM"] or out["SPIKE_ACID"])

    # RESET: pojedynczy duży skok (jak spike), ale interpretujemy go jako zmianę poziomu;
    # prosty przybliżacz: max|Δ| > próg ORAZ wyraźna różnica średnich 1. i 2. połowy
    def reset_flag(col: str, thr: float) -> int:
        dcol = diff_cols.get(col)
        maxjump = (df[dcol].abs().max(skipna=True) if dcol is not None
                   else df[col].astype(float).diff().abs().max(skipna=True))
        if maxjump < thr:
            return 0
        n = len(df);
        if n < 8:
            return 0
        m1 = float(df[col].iloc[:n//2].mean())
        m2 = float(df[col].iloc[n//2:].mean())
        return int(abs(m2 - m1) >= thr)
    out["RESET_TEMP"] = reset_flag("value_temp", args.thr_reset_temp)
    out["RESET_HUM"]  = reset_flag("value_hum",  args.thr_reset_hum)
    out["RESET_ACID"] = reset_flag("value_acid", args.thr_reset_acid)
    out["RESET"]      = int(out["RESET_TEMP"] or out["RESET_HUM"] or out["RESET_ACID"])

    # DRIFT: długi, spójny trend (slope o znaku ≠ 0 i |slope| >= próg)
    out["DRIFT_TEMP"] = int(abs(out["slope_temp"]) >= args.thr_drift_slope_temp)
    out["DRIFT_HUM"]  = int(abs(out["slope_hum"])  >= args.thr_drift_slope_hum)
    # dla acid możesz chcieć wykrywać głównie spadek (ujemny slope)
    out["DRIFT_ACID"] = int(abs(out["slope_acid"]) >= abs(args.thr_drift_slope_acid))
    out["DRIFT"]      = int(out["DRIFT_TEMP"] or out["DRIFT_HUM"] or out["DRIFT_ACID"])

    # QUANT: mało unikalnych wartości (schodki)
    out["QUANT_TEMP"] = int(out["uniq_ratio_value_temp"] <= args.thr_quant_uniq_ratio)
    out["QUANT_HUM"]  = int(out["uniq_ratio_value_hum"]  <= args.thr_quant_uniq_ratio)
    out["QUANT_ACID"] = int(out["uniq_ratio_value_acid"] <= args.thr_quant_uniq_ratio)
    out["QUANT"]      = int(out["QUANT_TEMP"] or out["QUANT_HUM"] or out["QUANT_ACID"])

    # RANGE: naruszenie zakresów fizycznych
    out["RANGE_TEMP"] = int((df["value_temp"] < args.temp_min).any() or (df["value_temp"] > args.temp_max).any())
    out["RANGE_HUM"]  = int((df["value_hum"]  < 0).any() or (df["value_hum"] > 100).any())
    out["RANGE_ACID"] = int((df["value_acid"] < 0).any() or (df["value_acid"] > 14).any())
    out["RANGE"]      = int(out["RANGE_TEMP"] or out["RANGE_HUM"] or out["RANGE_ACID"])

    # CROSS_VIOLATION: heurystyka – oczekujemy raczej ujemnej korelacji temp-hum i temp-acid
    cv1 = (not np.isnan(out["corr_temp_hum"]))  and (out["corr_temp_hum"]  > args.thr_cross_corr_max)
    cv2 = (not np.isnan(out["corr_temp_acid"])) and (out["corr_temp_acid"] > args.thr_cross_corr_max)
    out["CROSS_VIOLATION"] = int(cv1 or cv2)

    # OK i ANY
    labels = ["MISSING_GAP","STUCK","SPIKE","RESET","DRIFT","QUANT","RANGE","CROSS_VIOLATION"]
    out["pred_any_error"] = int(any(out[l]==1 for l in labels))
    out["OK"] = int(not out["pred_any_error"])

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Katalog wejściowy (rekurencyjnie)")
    ap.add_argument("--pattern", default="*.csv", help="Wzorzec plików CSV (np. *_segment_*.csv)")
    ap.add_argument("--out", default="manifest_rules.csv", help="Ścieżka wyniku (CSV)")

    # Progi
    ap.add_argument("--thr-missing-pct", type=float, default=0.05, help="Udział is_artificial dla MISSING_GAP")
    ap.add_argument("--thr-stuck-std", type=float, default=0.02, help="Prog rolling std dla STUCK")
    ap.add_argument("--thr-stuck-len", type=int, default=30, help="Min. długość ciągu niskiego std")
    ap.add_argument("--thr-spike-temp", type=float, default=3.5)
    ap.add_argument("--thr-spike-hum",  type=float, default=10.0)
    ap.add_argument("--thr-spike-acid", type=float, default=0.2)
    ap.add_argument("--thr-reset-temp", type=float, default=3.0)
    ap.add_argument("--thr-reset-hum",  type=float, default=10.0)
    ap.add_argument("--thr-reset-acid", type=float, default=0.3)
    ap.add_argument("--thr-drift-slope-temp", type=float, default=0.01)   # zmiana / próbkę
    ap.add_argument("--thr-drift-slope-hum",  type=float, default=0.05)
    ap.add_argument("--thr-drift-slope-acid", type=float, default=0.001)
    ap.add_argument("--thr-quant-uniq-ratio", type=float, default=0.05)
    ap.add_argument("--thr-cross-corr-max", type=float, default=0.3, help="korelacja dodatnia > tego progu -> naruszenie")
    ap.add_argument("--temp-min", type=float, default=-20.0)
    ap.add_argument("--temp-max", type=float, default=50.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = []
    files = sorted(data_dir.rglob(args.pattern))
    if not files:
        print(f"[WARN] Brak plików w {data_dir} dla wzorca {args.pattern}")
    for p in files:
        try:
            df = load_csv(p)
            res = classify_one(df, args)
            res["file"] = str(p)
            # spróbuj wydobyć sensor i segment z nazwy
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
        print(f"[DONE] Zapisano -> {args.out}")

if __name__ == "__main__":
    main()
