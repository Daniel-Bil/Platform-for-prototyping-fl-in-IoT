import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

LABELS = ["OK","GAP","STUCK","SPIKE","RESET","QUANT","RANGE","CROSS_VIOLATION"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="katalog z X_keras.npy i y.csv")
    ap.add_argument("--label", required=True, help=f"etykieta do filtrowania, np. {','.join(LABELS)}")
    ap.add_argument("--out-dir", default=None, help="katalog wyjściowy na PNG (domyślnie: <data-dir>/plots_<label>)")
    args = ap.parse_args()

    d = Path(args.data_dir)
    X = np.load(d / "X_keras.npy")   # [N, L, C]
    y = pd.read_csv(d / "y.csv")

    if args.label not in y.columns:
        raise ValueError(f"Etykieta {args.label} nie istnieje w y.csv. Dostępne: {list(y.columns)}")

    # wybór okien z daną etykietą
    idxs = y.index[y[args.label] == 1].tolist()
    if not idxs:
        print(f"[WARN] Brak okien z etykietą {args.label}")
        return

    # nazwy kanałów z label_schema.json
    schema_path = d / "label_schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text())
        chan_names = schema["channels"]
    else:
        chan_names = [f"ch{i}" for i in range(X.shape[2])]

    # katalog wyjściowy
    out_dir = Path(args.out_dir) if args.out_dir else (d / f"plots_{args.label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Zapisuję {len(idxs)} okien z etykietą {args.label} do {out_dir}")

    for i in idxs:
        win = X[i]  # [L, C]
        labels_here = [c for c in LABELS if c in y.columns and y.loc[i,c]==1]

        fig, axes = plt.subplots(X.shape[2], 1, figsize=(12, 2*X.shape[2]), sharex=True)
        if X.shape[2] == 1: axes = [axes]

        for j, ax in enumerate(axes):
            ax.plot(win[:, j], lw=1)
            ax.set_ylabel(chan_names[j])
        axes[-1].set_xlabel("sample index")
        fig.suptitle(f"okno #{i}, labels={labels_here}")
        plt.tight_layout()

        out_path = out_dir / f"window_{i:05d}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    print(f"[DONE] Zapisano {len(idxs)} plików PNG do {out_dir}")

if __name__ == "__main__":
    main()
