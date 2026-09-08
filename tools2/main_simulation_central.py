"""
========================================================================================
BASELINE EXPERIMENT: CENTRALIZED ML (UPPER BOUND - GÓRNA GRANICA MOŻLIWOŚCI)
========================================================================================
Temat pracy: Analiza i porównanie algorytmów uczenia federacyjnego w systemach IoT
             z uwzględnieniem kosztów telekomunikacyjnych i wydajności detekcji anomalii.

OPIS MODUŁU:
    Skrypt implementuje referencyjny punkt odniesienia (tzw. Upper Bound / Sufit) dla
    eksperymentów uczenia federacyjnego. Symuluje tradycyjne, scentralizowane podejście
    do uczenia maszynowego, w którym surowe dane pomiarowe ze WSZYSTKICH czujników IoT
    są przesyłane przez sieć i komasowane w jednej relacyjnej bazie na serwerze w chmurze.

CEL NAUKOWY W PRACY MAGISTERSKIEJ:
    1. Wyznaczenie maksymalnej osiągalnej jakości detekcji anomalii (F1-Score), jaką
       model może uzyskać mając jednoczesny dostęp do globalnego rozkładu danych.
    2. Pomiar kosztu telekomunikacyjnego (Total Data MB) przesyłu surowych plików CSV.
    3. Obliczenie tzw. "Federated Gap" (różnicy między F1 modelu centralnego a modelami
       FL), co pozwala empirycznie udowodnić, że Federated Learning pozwala zaoszczędzić
       >99% pasma sieciowego przy minimalnym (rzędu ~2%) spadku jakości predykcji.

RYGOR METODOLOGICZNY:
    * Sprawiedliwy budżet epok (Fair Epoch Budget): Model centralny trenowany jest przez
      dokładnie 15 epok (odpowiada to 5 rundom po 3 epoki lokalne w symulacji FL).
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

# ==========================================
# IMPORT PARSERA ARCHITEKTURY Z GUI
# ==========================================
from json_model_parser import build_model_from_json

# ==========================================
# KONFIGURACJA EKSPERYMENTU CENTRALNEGO
# ==========================================
FL_DATASET_DIR = os.environ.get("FL_DATASET_DIR", os.path.join("data", "fl_dataset_real"))
JSON_MODEL_PATH = "fl_model.json"  # <-- Ścieżka do Twojego pliku z GUI

# Budżet epok = ROUNDS * LOCAL_EPOCHS z eksperymentu FL (np. 5 rund * 3 epoki = 15)
TOTAL_EPOCHS = 15
BATCH_SIZE = 32
SEQ_LEN = 6        # Okno czasowe: 6 próbek (1 godzina)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def create_sequences(X, y, seq_len):
    """Tworzy okna przesuwne dla szeregów czasowych."""
    Xs, ys = [], []
    for end in range(seq_len - 1, len(X)):
        Xs.append(X[end - seq_len + 1 : end + 1])
        ys.append(y[end])
    return np.array(Xs), np.array(ys)


def load_and_pool_all_data(client_dirs):
    """
    Ładuje dane ze WSZYSTKICH czujników i łączy je w jeden centralny zbiór.
    Mierzy również fizyczny rozmiar surowych danych w MB (koszt transferu do chmury).
    """
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []

    features = ["value_temp", "value_hum", "value_acid", "value_PV"]
    total_raw_bytes = 0

    print("📥 Ładowanie i łączenie danych z węzłów IoT do zbioru centralnego...")

    for client_dir in client_dirs:
        client_id = os.path.basename(client_dir)

        # Pomiar rozmiaru plików na dysku (ile bajtów trzeba by wysłać przez sieć)
        for fn in ["train.csv", "val.csv", "test.csv"]:
            file_path = os.path.join(client_dir, fn)
            if os.path.exists(file_path):
                total_raw_bytes += os.path.getsize(file_path)

        # Wczytanie danych
        train_df = pd.read_csv(os.path.join(client_dir, "train.csv"), index_col="time", parse_dates=True)
        val_df = pd.read_csv(os.path.join(client_dir, "val.csv"), index_col="time", parse_dates=True)
        test_df = pd.read_csv(os.path.join(client_dir, "test.csv"), index_col="time", parse_dates=True)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[features])
        y_train = train_df["label"].values

        X_val_scaled = scaler.transform(val_df[features])
        y_val = val_df["label"].values

        X_test_scaled = scaler.transform(test_df[features])
        y_test = test_df["label"].values

        X_tr_seq, y_tr_seq = create_sequences(X_train_scaled, y_train, SEQ_LEN)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, SEQ_LEN)
        X_te_seq, y_te_seq = create_sequences(X_test_scaled, y_test, SEQ_LEN)

        all_X_train.append(X_tr_seq)
        all_y_train.append(y_tr_seq)
        all_X_val.append(X_val_seq)
        all_y_val.append(y_val_seq)
        all_X_test.append(X_te_seq)
        all_y_test.append(y_te_seq)

    # Komasowanie (pooling) macierzy do pojedynczego zbioru centralnego
    X_train_global = np.concatenate(all_X_train, axis=0)
    y_train_global = np.concatenate(all_y_train, axis=0)
    X_val_global = np.concatenate(all_X_val, axis=0)
    y_val_global = np.concatenate(all_y_val, axis=0)
    X_test_global = np.concatenate(all_X_test, axis=0)
    y_test_global = np.concatenate(all_y_test, axis=0)

    total_raw_mb = total_raw_bytes / (1024 * 1024)

    print(f"  -> Zbiór Treningowy : {X_train_global.shape[0]} sekwencji")
    print(f"  -> Zbiór Walidacyjny: {X_val_global.shape[0]} sekwencji")
    print(f"  -> Zbiór Testowy    : {X_test_global.shape[0]} sekwencji")
    print(f"  -> Wolumen surowych danych (transfer do chmury): {total_raw_mb:.2f} MB")

    return (
        (X_train_global, y_train_global),
        (X_val_global, y_val_global),
        (X_test_global, y_test_global),
        total_raw_mb,
    )


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    client_dirs = sorted(glob.glob(os.path.join(FL_DATASET_DIR, "client_*")))

    if not client_dirs:
        print("❌ Nie znaleziono folderów klientów w data/fl_dataset!")
        return

    print("============================================================")
    print("🚀 START EKSPERYMENTU: Centralized ML (Upper Bound - Sufit)")
    print("============================================================")

    # 1. Przygotowanie skomasowanego zbioru danych
    train_data, val_data, test_data, total_raw_mb = load_and_pool_all_data(client_dirs)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # 2. Budowa modelu z pliku JSON (dla spójności architektury z FL)
    num_features = 4
    model = build_model_from_json(JSON_MODEL_PATH, SEQ_LEN, num_features)
    model.summary(print_fn=lambda x: print(f"    [Architektura] {x}"))

    # 3. Trening na jednym serwerze centralnym
    print(f"\n📡 Rozpoczynam trening centralny ({TOTAL_EPOCHS} epok)...")
    t0 = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    train_time_s = time.perf_counter() - t0
    print(f"✅ Trening zakończony w czasie: {train_time_s:.2f} s")

    # 4. Ewaluacja na skomasowanym zbiorze testowym
    print("\n📊 Ewaluacja globalna na połączonym zbiorze testowym...")
    predictions = model.predict(X_test, verbose=0)
    y_pred = (predictions > 0.5).astype(int)

    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_anom = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print("\n" + "=" * 80)
    print("📈 SZCZEGÓŁOWY RAPORT KLASYFIKACJI (CENTRALIZED ML)")
    print("=" * 80)
    print(classification_report(y_test, y_pred, target_names=["0 - Zdrowe", "1 - Anomalia"]))

    # 5. Podsumowanie w formacie kompatybilnym z Twoją tabelą magisterską
    summary = {
        "Method": "Centralized ML (Upper Bound)",
        "Architecture": os.path.basename(JSON_MODEL_PATH),
        "Weighted_F1": round(f1_w, 4),
        "Anomaly_F1": round(f1_anom, 4),
        "Total_Data_MB": round(total_raw_mb, 2),  # Przesłanie surowych plików .csv
        "Client_Train_Time_s": 0.00,              # Klienci nie liczą nic (tylko wysyłają dane)
        "Server_Train_Time_s": round(train_time_s, 2), # Czas obliczeń serwera w chmurze
    }

    results_df = pd.DataFrame([summary])
    print("\n" + "=" * 80)
    print("📊 PODSUMOWANIE DO TABELI W PRACY MAGISTERSKIEJ:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Zapis macierzy pomyłek
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Zdrowe", "Anomalia"],
        yticklabels=["Zdrowe", "Anomalia"],
    )
    plt.title("Centralized ML Confusion Matrix (Upper Bound)")
    plt.ylabel("Prawdziwa Etykieta (Ground Truth)")
    plt.xlabel("Predykcja Modelu Centralnego")
    plt.tight_layout()
    plt.savefig("centralized_confusion_matrix.png", dpi=300)
    plt.close()
    print("✅ Zapisano macierz pomyłek jako 'centralized_confusion_matrix.png'.")

    # Zapis wyniku do CSV
    results_df.to_csv("centralized_results.csv", index=False)
    print("✅ Zapisano podsumowanie jako 'centralized_results.csv'.")


if __name__ == "__main__":
    main()