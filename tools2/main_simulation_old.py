import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# IMPORTY TWOICH METOD FEDERACYJNYCH
# ==========================================
from method_fedavg import fedavg_aggregate
from method_hierfavg import hierfavg_edge_aggregate, hierfavg_cloud_aggregate
from method_fedma import fedma_aggregate
from method_fedpaq import fedpaq_aggregate
from method_fedprox import train_fedprox_client

# ==========================================
# KONFIGURACJA SYMULACJI
# ==========================================
FL_DATASET_DIR = os.path.join("data", "fl_dataset")
ROUNDS = 5  # Liczba rund komunikacyjnych z serwerem
LOCAL_EPOCHS = 3  # Liczba epok trenowanych lokalnie u klienta w jednej rundzie
BATCH_SIZE = 32
SEQ_LEN = 6  # Okno czasowe = 6 próbek (1 godzina historii na wykrycie anomalii)

# WYBÓR METODY BADANEJ W SYMULACJI
# Dostępne opcje: "FedAvg", "FedMA", "FedProx", "FedPAQ", "HierFedAvg"
CURRENT_METHOD = "FedAvg"
MU_PROX = 0.01  # Parametr mu dla FedProx
EDGE_NODES = 2  # Liczba węzłów brzegowych dla HierFedAvg


def create_sequences(X, y, seq_len):
    """Tworzy okna przesuwne. Sieć musi widzieć historię, by wykryć Flatline lub Drift."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def load_client_data(client_dir):
    """Ładuje i skaluje dane dla pojedynczego klienta."""
    train_df = pd.read_csv(os.path.join(client_dir, "train.csv"), index_col='time', parse_dates=True)
    val_df = pd.read_csv(os.path.join(client_dir, "val.csv"), index_col='time', parse_dates=True)
    test_df = pd.read_csv(os.path.join(client_dir, "test.csv"), index_col='time', parse_dates=True)

    features = ['value_temp', 'value_hum', 'value_acid', 'value_PV']

    scaler = StandardScaler()

    # Skalujemy na podstawie zbioru treningowego, aby nie oszukiwać na testowym
    X_train_scaled = scaler.fit_transform(train_df[features])
    y_train = train_df['label'].values

    X_val_scaled = scaler.transform(val_df[features])
    y_val = val_df['label'].values

    X_test_scaled = scaler.transform(test_df[features])
    y_test = test_df['label'].values

    # Tworzenie sekwencji
    X_train, y_train = create_sequences(X_train_scaled, y_train, SEQ_LEN)
    X_val, y_val = create_sequences(X_val_scaled, y_val, SEQ_LEN)
    X_test, y_test = create_sequences(X_test_scaled, y_test, SEQ_LEN)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_local_model(seq_len, num_features):
    """Model klasyfikacji binarnej na węźle IoT (Klient)."""
    model = models.Sequential([
        layers.Input(shape=(seq_len, num_features)),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Zwraca prawdopodobieństwo anomalii [0-1]
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ==========================================
# GŁÓWNA PĘTLA SYMULACJI (SERWER FL)
# ==========================================
def run_simulation():
    client_dirs = glob.glob(os.path.join(FL_DATASET_DIR, "client_*"))
    if not client_dirs:
        print("❌ Nie znaleziono folderów klientów. Odpal najpierw 05_prepare_fl_dataset.py.")
        return

    print(f"🌍 Serwer FL uruchomiony. Aktywna metoda: {CURRENT_METHOD}")
    print(f"Liczba klientów biorących udział w federacji: {len(client_dirs)}")

    # 1. Załadowanie danych wszystkich klientów do pamięci
    clients_data = []
    for c_dir in client_dirs:
        clients_data.append(load_client_data(c_dir))

    # 2. Inicjalizacja globalnego modelu na serwerze
    num_features = 4
    global_model = build_local_model(SEQ_LEN, num_features)
    global_weights = global_model.get_weights()

    # 3. Trening Federacyjny (Rundy komunikacyjne)
    for round_num in range(1, ROUNDS + 1):
        print(f"\n{'=' * 15} RUNDA KOMUNIKACYJNA {round_num}/{ROUNDS} [{CURRENT_METHOD}] {'=' * 15}")

        local_weights = []

        for i, (client_dir, data) in enumerate(zip(client_dirs, clients_data)):
            client_id = os.path.basename(client_dir)
            (X_train, y_train), (X_val, y_val), _ = data

            # Krok 3a: Klient pobiera wagi z serwera
            local_model = build_local_model(SEQ_LEN, num_features)
            local_model.set_weights(global_weights)

            # Krok 3b: Trening lokalny w zależności od wybranej metody
            print(f"📡 Trenowanie lokalne u {client_id}...")

            if CURRENT_METHOD == "FedProx":
                # Trening z regularyzacją proksymalną z method_fedprox.py
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
                w_client = train_fedprox_client(local_model, global_weights, train_dataset, LOCAL_EPOCHS, mu=MU_PROX)
                local_weights.append(w_client)
            else:
                # Klasyczny trening dla FedAvg, FedMA, FedPAQ oraz HierFedAvg
                local_model.fit(X_train, y_train,
                                epochs=LOCAL_EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=(X_val, y_val),
                                verbose=0)
                local_weights.append(local_model.get_weights())

        # Krok 3c: Agregacja na serwerze (Twoje metody ze źródeł)
        print(f"☁️ Serwer: Agregacja wag ze wszystkich węzłów za pomocą {CURRENT_METHOD}...")

        if CURRENT_METHOD == "FedAvg":
            global_weights = fedavg_aggregate(local_weights)

        elif CURRENT_METHOD == "FedMA":
            global_weights = fedma_aggregate(global_weights, local_weights)

        elif CURRENT_METHOD == "FedPAQ":
            global_weights = fedpaq_aggregate(local_weights)

        elif CURRENT_METHOD == "HierFedAvg":
            # Hierarchiczna agregacja w 2 etapach:
            # 1. Agregacja na serwerach brzegowych (Edge)
            edge_groups = np.array_split(local_weights, EDGE_NODES)
            edge_weights = [hierfavg_edge_aggregate(group) for group in edge_groups if len(group) > 0]
            # 2. Agregacja końcowa w chmurze (Cloud)
            global_weights = hierfavg_cloud_aggregate(edge_weights)

    # ==========================================
    # 4. EWALUACJA KOŃCOWA NA ZBIORZE TESTOWYM
    # ==========================================
    print("\n✅ Trening federacyjny zakończony. Rozpoczynam ostateczną ewaluację...")

    global_model.set_weights(global_weights)

    all_y_true = []
    all_y_pred = []

    for data in clients_data:
        _, _, (X_test, y_test) = data
        predictions = global_model.predict(X_test, verbose=0)
        y_pred = (predictions > 0.5).astype(int)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    print(f"\n📊 GLOBALNY RAPORT KLASYFIKACJI ANOMALII ({CURRENT_METHOD}):")
    print(classification_report(all_y_true, all_y_pred, target_names=["0 - Zdrowe", "1 - Anomalia"]))

    # Rysowanie Macierzy Pomyłek (Confusion Matrix)
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Zdrowe", "Anomalia"],
                yticklabels=["Zdrowe", "Anomalia"])
    plt.title(f'Global Model Confusion Matrix ({CURRENT_METHOD})')
    plt.ylabel('Prawdziwa Etykieta (Ground Truth)')
    plt.xlabel('Predykcja Modelu FL')
    plt.tight_layout()
    plt.savefig(f"global_confusion_matrix_{CURRENT_METHOD}.png", dpi=300)
    plt.close()

    print(f"✅ Zapisano macierz pomyłek jako 'global_confusion_matrix_{CURRENT_METHOD}.png'.")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    run_simulation()