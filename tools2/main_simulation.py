import os
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# ==========================================
# IMPORTY MODUŁÓW I PARSERA JSON
# ==========================================
from method_fedavg import fedavg_aggregate
from method_hierfavg import hierfavg_edge_aggregate, hierfavg_cloud_aggregate
from method_fedma import fedma_aggregate
from method_fedpaq import fedpaq_aggregate
from method_fedprox import train_fedprox_client
from json_model_parser import build_model_from_json

# ==========================================
# KONFIGURACJA SYMULACJI BADAWCZEJ
# ==========================================
FL_DATASET_DIR = os.path.join("data", "fl_dataset")
JSON_MODEL_PATH = "fl_model.json"  # <-- Ścieżka do Twojego pliku z GUI
ROUNDS = 5            # Liczba rund komunikacyjnych
LOCAL_EPOCHS = 3      # Epoki lokalne u klienta
BATCH_SIZE = 32
SEQ_LEN = 6           # Okno czasowe: 6 próbek (1 godzina)
MU_PROX = 0.01        # Parametr mu dla FedProx
EDGE_NODES = 2        # Węzły brzegowe dla HierFedAvg
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
    train_df = pd.read_csv(os.path.join(client_dir, "train.csv"), index_col="time", parse_dates=True)
    val_df = pd.read_csv(os.path.join(client_dir, "val.csv"), index_col="time", parse_dates=True)
    test_df = pd.read_csv(os.path.join(client_dir, "test.csv"), index_col="time", parse_dates=True)

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


def calculate_model_size_mb(model, bits=32):
    total_params = model.count_params()
    bytes_per_param = bits / 8.0
    return (total_params * bytes_per_param) / (1024 * 1024)


def evaluate_global_model(model, weights, clients_data):
    model.set_weights(weights)
    all_y_true, all_y_pred = [], []

    for data in clients_data:
        _, _, (X_test, y_test) = data
        if len(X_test) == 0:
            continue
        preds = model.predict(X_test, verbose=0)
        y_pred = (preds > 0.5).astype(int)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
    f1_anomaly = f1_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)
    return f1, f1_anomaly, np.array(all_y_true), np.array(all_y_pred)


def run_experiment(method, json_path, clients_data, client_dirs):
    num_features = 4
    print(f"\n============================================================")
    print(f"🚀 START EKSPERYMENTU: Metoda = {method} | Z pliku JSON = {json_path}")
    print(f"============================================================")

    # Budowanie modelu z pliku JSON
    dummy_model = build_model_from_json(json_path, SEQ_LEN, num_features)
    dummy_model.summary(print_fn=lambda x: print(f"    [Architektura] {x}"))

    bits = 8 if method == "FedPAQ" else 32
    single_model_mb = calculate_model_size_mb(dummy_model, bits=bits)
    global_weights = dummy_model.get_weights()

    total_data_transferred_mb = 0.0
    total_client_train_time = 0.0
    total_server_agg_time = 0.0
    history = []

    for round_num in range(1, ROUNDS + 1):
        print(f"\n--- RUNDA {round_num}/{ROUNDS} [{method}] ---")
        local_weights = []
        round_client_time = 0.0

        for i, (client_dir, data) in enumerate(zip(client_dirs, clients_data)):
            client_id = os.path.basename(client_dir)
            (X_train, y_train), (X_val, y_val), _ = data

            local_model = build_model_from_json(json_path, SEQ_LEN, num_features)
            local_model.set_weights(global_weights)

            # Uplink: Klient odbiera model globalny (32-bit)
            total_data_transferred_mb += calculate_model_size_mb(dummy_model, bits=32)

            t0 = time.perf_counter()
            if method == "FedProx":
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
                w_client = train_fedprox_client(
                    local_model, global_weights, train_dataset, LOCAL_EPOCHS, mu=MU_PROX, client_id=client_id
                )
                local_weights.append(w_client)
            else:
                print(f"    📡 [Fit | {client_id}] Trenowanie lokalne ({LOCAL_EPOCHS} epok)...")
                local_model.fit(X_train, y_train, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                local_weights.append(local_model.get_weights())

            round_client_time += time.perf_counter() - t0

            # Downlink: Klient wysyła wagi
            total_data_transferred_mb += single_model_mb

        total_client_train_time += round_client_time

        # Agregacja
        print(f"  ☁️ [Serwer] Agregowanie wag algorytmem: {method}...")
        t_agg_start = time.perf_counter()
        if method == "FedAvg":
            global_weights = fedavg_aggregate(local_weights)
        elif method == "FedMA":
            global_weights = fedma_aggregate(global_weights, local_weights)
        elif method == "FedPAQ":
            global_weights = fedpaq_aggregate(local_weights)
        elif method == "HierFedAvg":
            # 1. Dzielimy INDEKSY klientów (np. [0,1,2,3,4,5,6] na 2 grupy), a nie macierze wag
            edge_indices = np.array_split(
                range(len(local_weights)), EDGE_NODES
            )

            # 2. Bezpieczne grupowanie wag w czystym Pythonie (unikamy błędu NumPy inhomogeneous shape)
            edge_groups = [
                [local_weights[i] for i in idx_group]
                for idx_group in edge_indices
                if len(idx_group) > 0
            ]

            # 3. Agregacja brzegowa (Edge), a następnie chmurowa (Cloud)
            edge_weights = [
                hierfavg_edge_aggregate(group) for group in edge_groups
            ]
            global_weights = hierfavg_cloud_aggregate(edge_weights)
        t_agg_end = time.perf_counter()
        total_server_agg_time += t_agg_end - t_agg_start

        # Ewaluacja po rundzie
        f1_w, f1_anom, _, _ = evaluate_global_model(dummy_model, global_weights, clients_data)
        history.append({
            "Round": round_num,
            "Transferred_MB": total_data_transferred_mb,
            "Weighted_F1": f1_w,
            "Anomaly_F1": f1_anom
        })
        print(f"  ✅ Runda {round_num}/{ROUNDS} Zakończona | F1-Weighted: {f1_w:.4f} | F1-Anomaly: {f1_anom:.4f} | Suma danych: {total_data_transferred_mb:.2f} MB")

    f1_w, f1_anom, y_true, y_pred = evaluate_global_model(dummy_model, global_weights, clients_data)

    metrics_summary = {
        "Method": method,
        "Architecture": os.path.basename(json_path),
        "Weighted_F1": round(f1_w, 4),
        "Anomaly_F1": round(f1_anom, 4),
        "Total_Data_MB": round(total_data_transferred_mb, 2),
        "Client_Train_Time_s": round(total_client_train_time, 2),
        "Server_Agg_Time_s": round(total_server_agg_time, 4)
    }

    return metrics_summary, pd.DataFrame(history), (y_true, y_pred)


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    client_dirs = sorted(glob.glob(os.path.join(FL_DATASET_DIR, "client_*")))

    if not client_dirs:
        print("❌ Nie znaleziono folderów klientów w data/fl_dataset!")
        return

    print(f"Znaleziono {len(client_dirs)} klientów IoT. Ładowanie danych...")
    clients_data = [load_client_data(cd) for cd in client_dirs]

    # Testowane metody
    methods = ["FedAvg", "FedProx", "FedMA", "FedPAQ", "HierFedAvg"]  # Możesz tu dodać: "FedMA", "FedPAQ", "HierFedAvg"
    # methods = ["FedMA", "FedPAQ", "HierFedAvg"]  # Możesz tu dodać: "FedMA", "FedPAQ", "HierFedAvg"
    summary_results = []

    for method in methods:
        summary, _, _ = run_experiment(method, JSON_MODEL_PATH, clients_data, client_dirs)
        summary_results.append(summary)

    print("\n" + "=" * 80)
    print("📊 PODSUMOWANIE WYNIKÓW BADAŃ (ARCHITEKTURA Z PLIKU JSON)")
    print("=" * 80)
    results_df = pd.DataFrame(summary_results)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()