"""
========================================================================================
BASELINE EXPERIMENT: LOCAL-ONLY ML (LOWER BOUND - DOLNA GRANICA MOŻLIWOŚCI)
========================================================================================
Temat pracy: Analiza i porównanie algorytmów uczenia federacyjnego w systemach IoT
             z uwzględnieniem kosztów telekomunikacyjnych i wydajności detekcji anomalii.

OPIS MODUŁU:
    Skrypt implementuje dolny punkt odniesienia (tzw. Lower Bound / Podłogę) badający
    skuteczność modeli w warunkach pełnej izolacji węzłów. Każdy z 7 czujników IoT
    trenuje własną, niezależną instancję modelu wyeksportowanego z GUI wyłącznie na
    własnym lokalnym zbiorze danych (train.csv). Nie zachodzi tu żadna komunikacja
    z serwerem, wymiana wag ani udostępnianie próbek (Total Data MB = 0.00).

CEL NAUKOWY W PRACY MAGISTERSKIEJ:
    1. Udowodnienie konieczności współpracy węzłów w sieciach rozproszonych IoT.
    2. Wykazanie negatywnego wpływu heterogeniczności danych (Non-IID) na modele
       izolowane – pojedynczy czujnik widzi tylko wycinek warunków środowiskowych,
       przez co jego zdolność do uogólniania i wykrywania globalnych anomalii na
       zbiorze testowym jest zauważalnie niższa niż w federacji (np. FedAvg/FedPAQ).
========================================================================================
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from json_model_parser import build_model_from_json

# ==========================================
# KONFIGURACJA EKSPERYMENTU LOKALNEGO
# ==========================================
FL_DATASET_DIR = os.environ.get("FL_DATASET_DIR", os.path.join("data", "fl_dataset_real"))
JSON_MODEL_PATH = "fl_model.json"
TOTAL_EPOCHS = 15  # Taki sam budżet epok dla sprawiedliwego porównania
BATCH_SIZE = 32
SEQ_LEN = 6
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def load_client_data(client_dir):
    train_df = pd.read_csv(
        os.path.join(client_dir, "train.csv"),
        index_col="time",
        parse_dates=True,
    )
    val_df = pd.read_csv(
        os.path.join(client_dir, "val.csv"), index_col="time", parse_dates=True
    )
    test_df = pd.read_csv(
        os.path.join(client_dir, "test.csv"),
        index_col="time",
        parse_dates=True,
    )

    features = ["value_temp", "value_hum", "value_acid", "value_PV"]
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(train_df[features])
    y_train = train_df["label"].values

    X_val_scaled = scaler.transform(val_df[features])
    y_val = val_df["label"].values

    X_test_scaled = scaler.transform(test_df[features])
    y_test = test_df["label"].values

    X_train, y_train = create_sequences(X_train_scaled, y_train, SEQ_LEN)
    X_val, y_val = create_sequences(X_val_scaled, y_val, SEQ_LEN)
    X_test, y_test = create_sequences(X_test_scaled, y_test, SEQ_LEN)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    client_dirs = sorted(glob.glob(os.path.join(FL_DATASET_DIR, "client_*")))

    if not client_dirs:
        print("❌ Nie znaleziono folderów klientów w data/fl_dataset!")
        return

    print("============================================================")
    print("🚀 START EKSPERYMENTU: Local-Only ML (Lower Bound - Podłoga)")
    print("============================================================")

    all_y_true, all_y_pred = [], []
    total_client_train_time = 0.0

    for client_dir in client_dirs:
        client_id = os.path.basename(client_dir)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_client_data(
            client_dir
        )

        model = build_model_from_json(JSON_MODEL_PATH, SEQ_LEN, 4)

        print(
            f"📡 [{client_id}] Trening lokalny w izolacji ({TOTAL_EPOCHS} epok)..."
        )
        t0 = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            epochs=TOTAL_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=0,
        )
        total_client_train_time += time.perf_counter() - t0

        if len(X_test) > 0:
            preds = model.predict(X_test, verbose=0)
            y_pred = (preds > 0.5).astype(int)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

    f1_w = f1_score(
        all_y_true, all_y_pred, average="weighted", zero_division=0
    )
    f1_anom = f1_score(
        all_y_true, all_y_pred, pos_label=1, zero_division=0
    )

    print("\n" + "=" * 80)
    print("📈 SZCZEGÓŁOWY RAPORT KLASYFIKACJI (LOCAL-ONLY ML)")
    print("=" * 80)
    print(
        classification_report(
            all_y_true,
            all_y_pred,
            target_names=["0 - Zdrowe", "1 - Anomalia"],
        )
    )

    summary = {
        "Method": "Local-Only ML (Lower Bound)",
        "Architecture": os.path.basename(JSON_MODEL_PATH),
        "Weighted_F1": round(f1_w, 4),
        "Anomaly_F1": round(f1_anom, 4),
        "Total_Data_MB": 0.00,  # Zero transferu przez sieć
        "Client_Train_Time_s": round(total_client_train_time, 2),
        "Server_Train_Time_s": 0.00,  # Serwer w ogóle nie bierze udziału
    }

    results_df = pd.DataFrame([summary])
    print("\n" + "=" * 80)
    print("📊 PODSUMOWANIE DO TABELI W PRACY MAGISTERSKIEJ:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    results_df.to_csv("local_only_results.csv", index=False)
    print("✅ Zapisano podsumowanie jako 'local_only_results.csv'.")


if __name__ == "__main__":
    main()