from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

VALUE_COLS_DEFAULT = ["value_temp","value_hum","value_acid"]

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.reset_index(drop=True)

def add_features(df: pd.DataFrame, features: list[str], win: int) -> pd.DataFrame:
    out = df.copy()
    for c in features:
        out[c] = out[c].astype(float)
    # pochodne po indeksie
    for c in features:
        out[f"d_{c}"] = out[c].diff()
    # rolling std
    for c in features:
        out[f"std{win}_{c}"] = out[c].rolling(win, min_periods=max(2, win//2)).std()
    return out

def sensor_from_path(p: Path) -> str:
    # spróbuj z nazwy pliku df_RuralIoT_<ID>*.csv
    m = re.search(r"df_RuralIoT_([^_/]+)", p.name)
    if m:
        return m.group(1)
    # spróbuj z katalogu sensor_<ID>
    m = re.search(r"sensor_([^/]+)", str(p.parent))
    return m.group(1) if m else "unknown"

def ensure_dir(base_out: Path, in_path: Path) -> Path:
    # odtwórz relatywną strukturę katalogów względem data-dir
    return base_out

def plot_signal_and_diff(df_aug: pd.DataFrame, features: list[str], out_png: Path, title: str):
    seg = df_aug.reset_index(drop=True)
    x = seg.index
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    rows = list(enumerate(features))
    for i, col in rows:
        # sygnał + (opcjonalnie) punkty artificial
        ax = axes[i, 0]
        ax.plot(x, seg[col], lw=1, label=col)
        if "is_artificial" in seg.columns:
            art = seg[seg["is_artificial"] == True]
            if not art.empty:
                ax.scatter(art.index, art[col], marker="x", s=18, label="artificial")
        ax.set_ylabel(col)
        ax.legend(loc="best")

        # pochodna
        ax2 = axes[i, 1]
        ax2.plot(x, seg[f"d_{col}"], lw=1, label=f"d_{col}")
        ax2.axhline(0.0, ls="--", lw=0.8)
        ax2.set_ylabel(f"d_{col}")
        ax2.legend(loc="best")

    axes[-1, 0].set_xlabel("sample index")
    axes[-1, 1].set_xlabel("sample index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_std(df_aug: pd.DataFrame, features: list[str], win: int, out_png: Path, title: str):
    seg = df_aug.reset_index(drop=True)
    x = seg.index
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for i, col in enumerate(features):
        std_col = f"std{win}_{col}"
        axes[i].plot(x, seg[std_col], lw=1, label=std_col)
        axes[i].set_ylabel(std_col)
        axes[i].legend(loc="best")
    axes[-1].set_xlabel("sample index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Katalog z wejściowymi CSV (rekurencyjnie). Np. segmented_methods_pchip_3H")
    ap.add_argument("--out-dir", required=True,
                    help="Katalog wyjściowy (CSV + wykresy)")
    ap.add_argument("--features", nargs="+", default=VALUE_COLS_DEFAULT,
                    help="Które kolumny przetwarzać (domyślnie: temp, hum, acid)")
    ap.add_argument("--rolling", type=int, default=11,
                    help="Długość okna rolling std (w próbkach)")
    ap.add_argument("--pattern", default="*.csv",
                    help="Wzorzec plików (np. *_segment_*.csv)")
    args = ap.parse_args()

    in_dir  = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest
    manifest_rows = []

    files = sorted(in_dir.rglob(args.pattern))
    if not files:
        print(f"[WARN] Brak plików dla wzorca {args.pattern} w {in_dir}")
        return

    for p in files:
        try:
            df = load_csv(p)
            df_aug = add_features(df, args.features, args.rolling)

            # zapisz CSV
            # zachowujemy drzewo: <out_dir>/<opcjonalnie sensor_XXX>/<basename>.csv
            sensor = sensor_from_path(p)
            out_dir_sensor = out_dir / f"sensor_{sensor}"
            out_dir_sensor.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir_sensor / p.name
            df_aug.to_csv(out_csv, index=False)

            # wykresy
            plots_dir = out_dir_sensor / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            title = f"{p.name} — std window={args.rolling}"
            plot_signal_and_diff(df_aug, args.features, plots_dir / f"{p.stem}_signal_diff.png", title)
            plot_std(df_aug, args.features, args.rolling, plots_dir / f"{p.stem}_std.png", title)

            manifest_rows.append({
                "input": str(p),
                "output_csv": str(out_csv),
                "plots_dir": str(plots_dir),
                "rows": len(df_aug),
                "features": ",".join(args.features),
                "rolling": args.rolling
            })

            print(f"[OK] {p.relative_to(in_dir)} -> {out_csv.relative_to(out_dir)}")

        except Exception as e:
            print(f"[ERR] {p}: {e}")

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(out_dir / "_manifest.csv", index=False)
        print(f"[OK] Manifest -> {out_dir / '_manifest.csv'}")

if __name__ == "__main__":
    main()
