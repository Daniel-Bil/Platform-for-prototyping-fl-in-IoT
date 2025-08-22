from pathlib import Path
import argparse
import pandas as pd
import numpy as np

VALUE_COLS = ["value_temp","value_hum","value_acid"]

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.reset_index(drop=True)

def add_features(df: pd.DataFrame, features: list[str], win: int) -> pd.DataFrame:
    out = df.copy()
    # pochodne po indeksie
    for c in features:
        out[f"d_{c}"] = out[c].astype(float).diff()
    # rolling std
    for c in features:
        out[f"std{win}_{c}"] = out[c].astype(float).rolling(
            win, min_periods=max(2, win//2)
        ).std()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Katalog z plikami CSV (np. segmented_methods_pchip_3H/sensor_001)")
    ap.add_argument("--out-dir", required=True,
                    help="Katalog do zapisu wyników")
    ap.add_argument("--features", nargs="+", default=VALUE_COLS,
                    help="Kolumny do obliczeń")
    ap.add_argument("--rolling", type=int, default=11,
                    help="Długość okna rolling std (w próbkach)")
    ap.add_argument("--pattern", default="*.csv",
                    help="Wzorzec plików CSV (np. *_segment_*.csv)")
    args = ap.parse_args()

    in_dir  = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob(args.pattern))
    if not files:
        print(f"[WARN] brak plików dla wzorca {args.pattern} w {in_dir}")
        return

    for p in files:
        try:
            df = load_csv(p)
            df_aug = add_features(df, args.features, args.rolling)
            out_path = out_dir / p.name
            df_aug.to_csv(out_path, index=False)
            print(f"[OK] {p.name} -> {out_path}")
        except Exception as e:
            print(f"[ERR] {p}: {e}")

if __name__ == "__main__":
    main()
