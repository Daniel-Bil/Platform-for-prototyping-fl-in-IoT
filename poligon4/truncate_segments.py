# truncate_segments.py  — X=sample index (nie czas)
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("../dane/humidity_ofset_fixed")
OUT_DIR = Path("truncated_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUT_MAP = {
    "001": (100, 100),
    "002": (100, 100),
    "003": (100, 100),
    "010": (200, 200),
    "21":  (450, 100),
    "22":  (450, 100),
    "23":  (450, 800),
}

TIME_COL = "time"
COL_TEMP = "value_temp"
COL_HUM  = "value_hum"
COL_ACID = "value_acid"
COL_PV   = "value_PV"

# Stałe zakresy osi Y (spójne porównania); zmień na None, jeśli nie chcesz limitów
YLIMS = {
    COL_HUM:  (0, None),
    COL_ACID: (0, None),
    COL_TEMP: (-5, None),
    COL_PV:   (0, None),
}

def truncate_df(df: pd.DataFrame, cut_start: int, cut_end: int) -> pd.DataFrame:
    n = len(df)
    start = min(max(cut_start, 0), n)
    end = min(max(cut_end, 0), n - start)
    return df.iloc[start:n - end].reset_index(drop=True).copy()

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parsujemy i sortujemy po czasie (żeby kolejność próbek była chronologiczna),
    # ale rysujemy później po INDEKSIE, nie po czasie.
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
        df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    # Upewnij się, że kluczowe kolumny istnieją
    for c in [COL_HUM, COL_ACID, COL_TEMP, COL_PV]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _apply_ylim(ax, col):
    lo, hi = YLIMS.get(col, (None, None))
    if lo is not None or hi is not None:
        ax.set_ylim(lo, hi)

def save_overview_index(df: pd.DataFrame, sensor: str, out_png: Path):
    x = df.index  # << klucz: oś X = indeks próbki
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    axes[0].plot(x, df[COL_HUM])
    axes[0].set_title(f"Sensor {sensor} — {COL_HUM}")
    axes[0].set_ylabel("Humidity [%]"); _apply_ylim(axes[0], COL_HUM)

    axes[1].plot(x, df[COL_ACID])
    axes[1].set_title(f"Sensor {sensor} — {COL_ACID}")
    axes[1].set_ylabel("pH"); _apply_ylim(axes[1], COL_ACID)

    axes[2].plot(x, df[COL_TEMP])
    axes[2].set_title(f"Sensor {sensor} — {COL_TEMP}")
    axes[2].set_ylabel("Temp [°C]"); _apply_ylim(axes[2], COL_TEMP)

    axes[3].plot(x, df[COL_PV])
    axes[3].set_title(f"Sensor {sensor} — {COL_PV}")
    axes[3].set_ylabel("PV"); _apply_ylim(axes[3], COL_PV)
    axes[3].set_xlabel("sample index")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def save_compare_panel(df_orig: pd.DataFrame, df_tr: pd.DataFrame, sensor: str, out_png: Path):
    # Porównanie: dwie kolumny (oryginalny vs. przycięty), X = indeks
    rows = [(COL_HUM, "Humidity [%]"), (COL_ACID, "pH"), (COL_TEMP, "Temp [°C]"), (COL_PV, "PV")]
    fig, axes = plt.subplots(len(rows), 2, figsize=(16, 10), sharex=False)
    if len(rows) == 1:
        axes = axes.reshape(1, 2)

    for i, (col, ylabel) in enumerate(rows):
        # oryginalny
        ax = axes[i, 0]
        ax.plot(df_orig.index, df_orig[col])
        ax.set_title(f"{col} — original")
        ax.set_ylabel(ylabel); _apply_ylim(ax, col)
        ax.set_xlabel("sample index")

        # przycięty
        ax = axes[i, 1]
        ax.plot(df_tr.index, df_tr[col])
        ax.set_title(f"{col} — truncated")
        ax.set_ylabel(ylabel); _apply_ylim(ax, col)
        ax.set_xlabel("sample index")

    fig.suptitle(f"Sensor {sensor} — comparison (index-based X)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    print(f"[INFO] Input dir:  {DATA_DIR.resolve()}")
    print(f"[INFO] Output dir: {OUT_DIR.resolve()}")

    for sensor, (cut_start, cut_end) in CUT_MAP.items():
        in_path = DATA_DIR / f"df_RuralIoT_{sensor}.csv"
        if not in_path.exists():
            alt_path = DATA_DIR / f"{sensor}.csv"
            path_to_use = alt_path if alt_path.exists() else None
            if path_to_use is None:
                print(f"[WARN] Missing file: {in_path.name} (and {alt_path.name}) — skipping")
                continue
        else:
            path_to_use = in_path

        print(f"[OK] Processing: {path_to_use.name} | cut_start={cut_start}, cut_end={cut_end}")

        try:
            df = load_csv(path_to_use)
        except Exception as e:
            print(f"[ERR] Load failed for {path_to_use.name}: {e}")
            continue

        # Zapis wykresu ORYGINALNEGO (index-based)
        png_orig = OUT_DIR / f"{sensor}_original_index_overview.png"
        save_overview_index(df, sensor, png_orig)

        # Truncation + zapis CSV
        df_tr = truncate_df(df, cut_start, cut_end)
        if df_tr.empty:
            print(f"[WARN] After truncation {path_to_use.name} is empty — skipping saves.")
            continue

        out_csv = OUT_DIR / f"df_RuralIoT_{sensor}_truncated.csv"
        df_tr.to_csv(out_csv, index=False)
        print(f"[OK] Saved CSV -> {out_csv}")

        # Zapis wykresu PRZYCIĘTEGO (index-based)
        png_tr = OUT_DIR / f"{sensor}_truncated_index_overview.png"
        save_overview_index(df_tr, sensor, png_tr)

        # Plansza porównawcza (oryginalny vs przycięty)
        png_cmp = OUT_DIR / f"{sensor}_compare_index.png"
        save_compare_panel(df, df_tr, sensor, png_cmp)
        print(f"[OK] Plots -> {png_orig.name}, {png_tr.name}, {png_cmp.name}")

    print("[DONE] Finished.")

if __name__ == "__main__":
    main()
