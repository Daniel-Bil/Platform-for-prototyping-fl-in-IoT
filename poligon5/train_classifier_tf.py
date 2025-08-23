# -*- coding: utf-8 -*-
"""
Trener multilabel dla okien [N, L, C] w TensorFlow/Keras:
- ładuje X_keras.npy + y.csv z katalogu danych,
- normalizuje cechy per-kanał (mean/std z train),
- model: 1D-CNN -> BiGRU -> Dense(sigmoid),
- metryki: F1 macro + per-klasa, precision/recall,
- zapis: SavedModel, scaler, predykcje testu i raport CSV.

Użycie (przykład na Twoich danych):
python train_classifier_tf.py \
  --data-dir windows_L40_S10_timeline_test_final \
  --out-dir poligon5_runs/baseline_L40C9 \
  --epochs 40 --batch-size 64 --seed 42
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ----------------------------
# Utils
# ----------------------------
def load_dataset(data_dir: Path):
    X = np.load(data_dir / "X_keras.npy")   # [N, L, C]
    y = pd.read_csv(data_dir / "y.csv")
    # domyślne etykiety do uczenia:
    candidate_labels = ["OK","GAP","STUCK","SPIKE","RANGE"]
    labels = [c for c in candidate_labels if c in y.columns]
    if not labels:
        raise ValueError(f"Nie znalazłem żadnych z {candidate_labels} w y.csv")
    Y = y[labels].astype(int).to_numpy()    # [N, K]
    # kanały (meta)
    schema_path = data_dir / "label_schema.json"
    channels = None
    if schema_path.exists():
        schema = json.loads(schema_path.read_text())
        channels = schema.get("channels", None)
    return X, Y, labels, channels, y  # y (DataFrame) zachowujemy do metadanych

def train_val_test_split(X, Y, test_size=0.15, val_size=0.15, seed=42):
    # najpierw train+temp / test, potem train / val
    N = X.shape[0]
    X_tr, X_te, Y_tr, Y_te, idx_tr, idx_te = train_test_split(
        X, Y, np.arange(N), test_size=test_size, random_state=seed, shuffle=True
    )
    val_rel = val_size / (1.0 - test_size)
    X_tr, X_va, Y_tr, Y_va, idx_tr2, idx_va = train_test_split(
        X_tr, Y_tr, idx_tr, test_size=val_rel, random_state=seed, shuffle=True
    )
    return (X_tr, Y_tr, idx_tr2), (X_va, Y_va, idx_va), (X_te, Y_te, idx_te)

def fit_standardizer_per_channel(X_train):
    # X_train: [N,L,C] -> mean/std po wszystkich próbkach i oknach, dla każdego kanału C
    mu = np.nanmean(X_train, axis=(0,1), keepdims=True)  # [1,1,C]
    sd = np.nanstd (X_train, axis=(0,1), keepdims=True)  # [1,1,C]
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd

def apply_standardizer(X, mu, sd):
    return (X - mu) / sd

def build_model(input_shape, n_labels: int):
    L, C = input_shape
    inputs = tf.keras.Input(shape=(L, C))
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_labels, activation="sigmoid")(x)  # multilabel
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", multi_label=True, name="AUC_PR")]
    )
    return model

def f1_report(y_true, y_prob, labels, thr=0.5):
    y_hat = (y_prob >= thr).astype(int)
    # per-class
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average=None, zero_division=0
    )
    # macro
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_hat, average="macro", zero_division=0
    )
    rep = pd.DataFrame({"label": labels, "precision": p, "recall": r, "f1": f1})
    summary = {"precision_macro": p_m, "recall_macro": r_m, "f1_macro": f1_m}
    return rep, summary, y_hat

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="katalog z X_keras.npy i y.csv")
    ap.add_argument("--out-dir", required=True, help="katalog wynikowy")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5, help="próg dla multilabel F1")
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Wczytaj dane
    X, Y, labels, channels, y_meta = load_dataset(data_dir)
    N, L, C = X.shape
    print(f"[INFO] X: {X.shape}, Y: {Y.shape}, labels={labels}")

    # 2) Split (na razie zwykły losowy — bez specjalnego balansowania)
    (X_tr, Y_tr, idx_tr), (X_va, Y_va, idx_va), (X_te, Y_te, idx_te) = train_val_test_split(
        X, Y, test_size=0.15, val_size=0.15, seed=args.seed
    )

    # 3) Normalizacja per-kanał (na train), zapis parametrów
    mu, sd = fit_standardizer_per_channel(X_tr)
    X_tr_n = apply_standardizer(X_tr, mu, sd)
    X_va_n = apply_standardizer(X_va, mu, sd)
    X_te_n = apply_standardizer(X_te, mu, sd)
    np.save(out_dir / "scaler_mu.npy", mu)
    np.save(out_dir / "scaler_sd.npy", sd)

    # 4) Model
    model = build_model(input_shape=(L, C), n_labels=len(labels))
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "best_model.keras"),
        monitor="val_loss", save_best_only=True, verbose=1
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    # 5) Trening
    history = model.fit(
        X_tr_n, Y_tr,
        validation_data=(X_va_n, Y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ckpt, es],
        verbose=2
    )

    # Zapis historii
    pd.DataFrame(history.history).to_csv(out_dir / "train_history.csv", index=False)

    # 6) Ewaluacja na teście (F1@threshold)
    Yp_te = model.predict(X_te_n, batch_size=args.batch_size)
    rep_te, summary_te, Yhat_te = f1_report(Y_te, Yp_te, labels, thr=args.threshold)
    rep_te.to_csv(out_dir / "test_report_per_class.csv", index=False)
    pd.Series(summary_te).to_csv(out_dir / "test_report_summary.csv")
    print(f"[TEST] F1_macro={summary_te['f1_macro']:.3f}  Precision_macro={summary_te['precision_macro']:.3f}  Recall_macro={summary_te['recall_macro']:.3f}")

    # 7) Zapis predykcji testu (przyda się do przeglądu)
    cols_prob = [f"prob_{lbl}" for lbl in labels]
    df_pred = pd.DataFrame({
        "idx": idx_te
    })
    for j, lbl in enumerate(labels):
        df_pred[cols_prob[j]] = Yp_te[:, j]
        df_pred[f"pred_{lbl}"] = Yhat_te[:, j]
        df_pred[f"true_{lbl}"] = Y_te[:, j]
    df_pred.sort_values("idx").to_csv(out_dir / "test_predictions.csv", index=False)

    # 8) Zapis modelu w formacie SavedModel (oprócz best_model.keras)
    model.save(out_dir / "savedmodel")

    # 9) Zapis metadanych
    meta = {
        "labels": labels,
        "channels": channels,
        "L": int(L), "C": int(C),
        "seed": args.seed,
        "threshold": args.threshold
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[DONE] Wyniki -> {out_dir}")

if __name__ == "__main__":
    main()
