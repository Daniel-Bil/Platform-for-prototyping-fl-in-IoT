
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

def main():
    ap = argparse.ArgumentParser(description="Train a global TimeGAN model on preprocessed windows.")
    ap.add_argument("--data-dir", required=True, help="Directory with X_train.npy, scaler.pkl, feature_names.json")
    ap.add_argument("--outdir", required=True, help="Output directory for model and samples")
    ap.add_argument("--seq-len", type=int, default=None, help="Sequence length (auto if None; inferred from X_train)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--noise-dim", type=int, default=32)
    ap.add_argument("--layers-dim", type=int, default=128)
    ap.add_argument("--sample-n", type=int, default=16, help="How many sequences to sample after training")
    ap.add_argument("--save-model-name", default="timegan_global.pkl")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data artifacts
    X_train = np.load(data_dir / "X_train.npy")  # (N, T, F), normalized [0,1]
    feature_names = json.loads((data_dir / "feature_names.json").read_text())
    scaler = load(data_dir / "scaler.pkl")

    if X_train.ndim != 3:
        raise ValueError(f"Expected X_train to be 3D (N, T, F), got shape {X_train.shape}")
    N, T, F = X_train.shape
    if args.seq_len is None:
        args.seq_len = T
    elif args.seq_len != T:
        print(f"[WARN] Provided seq_len={args.seq_len} != data T={T}. Using data T.")
        args.seq_len = T

    assert F == len(feature_names), "Mismatch between features in X_train and feature_names.json"

    # Flatten windows into a single DataFrame as ydata likes to take a DataFrame
    # with 'num_cols' specifying which columns are numeric features.
    flat = X_train.reshape(-1, F)  # (N*T, F)
    df_train = pd.DataFrame(flat, columns=feature_names)

    # Define model and training parameters
    gan_args = ModelParameters(
        batch_size=args.batch_size,
        lr=args.lr,
        noise_dim=args.noise_dim,
        layers_dim=args.layers_dim
    )
    train_args = TrainParameters(
        epochs=args.epochs,
        sequence_length=args.seq_len,
        number_sequences=F
    )

    # Train synthesizer
    synth = TimeSeriesSynthesizer(modelname="timegan", model_parameters=gan_args)
    print(f"Starting training: N={N}, T={T}, F={F}, epochs={args.epochs}, batch={args.batch_size}")
    synth.fit(df_train, train_args, num_cols=feature_names)

    # Save model
    model_path = outdir / args.save_model_name
    synth.save(str(model_path))
    print(f"Saved model to: {model_path}")

    # Sample synthetic sequences (normalized space)
    sample_n = max(1, int(args.sample_n))
    synthetic = synth.sample(sample_n)  # shape (sample_n, T, F) in [0,1]
    # ydata/TimeGAN zwraca listę [sample_n] elementów, każdy (T, F)
    if isinstance(synthetic, (list, tuple)):
        synthetic = np.asarray(synthetic)  # -> (N, T, F)

    np.save(outdir / "synthetic_norm.npy", synthetic)

    # Inverse transform to real units
    syn_real = scaler.inverse_transform(synthetic.reshape(-1, F)).reshape(sample_n, T, F)
    np.save(outdir / "synthetic_real.npy", syn_real)

    # Quick sanity plots for the first few sequences (per feature)
    # One figure per feature, each sequence as a line.
    for j, feat in enumerate(feature_names):
        plt.figure()
        for i in range(min(sample_n, 6)):  # plot up to 6 sequences to keep it readable
            plt.plot(syn_real[i, :, j])
        plt.title(f"Synthetic (real units): {feat}")
        plt.xlabel("t (samples)")
        plt.ylabel(feat)
        plt.tight_layout()
        fig_path = outdir / f"plot_synthetic_{feat}.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()

    # Also save a small CSV preview of the first sample (real units)
    preview = pd.DataFrame(syn_real[0], columns=feature_names)
    preview.to_csv(outdir / "synthetic_preview_first_sample.csv", index=False)

    print("Done. Outputs:")
    print(f"  - Model: {model_path}")
    print(f"  - synthetic_norm.npy (shape {synthetic.shape})")
    print(f"  - synthetic_real.npy (shape {syn_real.shape})")
    print(f"  - plot_synthetic_*.png")
    print(f"  - synthetic_preview_first_sample.csv")

if __name__ == "__main__":
    main()
