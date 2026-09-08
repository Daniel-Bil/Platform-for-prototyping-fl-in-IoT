import os
import glob
import numpy as np
import pandas as pd

# Konfiguracja ścieżek
SYNTHETIC_DIR = os.path.join("data", "synthetic_files")
CLIENTS_DIR = os.path.join("data", "federated_clients")
FL_DATASET_DIR = os.path.join("data", "fl_dataset")

# Proporcje podziału chronologicznego
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15


# TEST_SPLIT to naturalnie pozostałe 15%

def prepare_fl_data():
    client_files = glob.glob(os.path.join(CLIENTS_DIR, "*_CLIENT_DATA.csv"))

    if not client_files:
        print(f"❌ Brak plików w {CLIENTS_DIR}.")
        return

    print("⚙️ Generowanie etykiet i podział na zbiory FL (Train/Val/Test)...")

    for corrupted_path in client_files:
        filename = os.path.basename(corrupted_path)
        sensor_id = filename.replace('_CLIENT_DATA.csv', '')

        # Szukamy odpowiadającego mu czystego pliku
        clean_path = os.path.join(SYNTHETIC_DIR, f"{sensor_id}_SYNTHETIC.csv")

        if not os.path.exists(clean_path):
            print(f"⚠️ Brak czystego pliku dla {sensor_id}, pomijam.")
            continue

        # Wczytanie danych
        df_clean = pd.read_csv(clean_path, index_col='time', parse_dates=True)
        df_corr = pd.read_csv(corrupted_path, index_col='time', parse_dates=True)

        features = ['value_temp', 'value_hum', 'value_acid', 'value_PV']

        # 1. GENEROWANIE ETYKIET (Ground Truth)
        # Zaczynamy od samych zer (wszystko zdrowe)
        df_corr['label'] = 0

        # Szukamy anomalii:
        # A) Wartości NaN (Dropouts)
        is_nan = df_corr[features].isna().any(axis=1)

        # B) Różnice względem czystego pliku (Drift, Flatline).
        # Z tolerancją 0.01 na ewentualne zaokrąglenia przy zapisie do CSV.
        is_modified = (df_clean[features] - df_corr[features].fillna(df_clean[features])).abs().max(axis=1) > 0.01

        # Oznaczamy jedynką (1) każdy wiersz, który jest NaN lub został zmodyfikowany
        df_corr.loc[is_nan | is_modified, 'label'] = 1

        # 2. IMPUTACJA DANYCH DLA SIECI NEURONOWEJ
        # Sieci nie potrafią mnożyć przez NaN. Luki (dropouts) musimy wypełnić
        # (np. powielając ostatnią znaną wartość), ale sieć i tak musi się nauczyć,
        # że taki nienaturalny "płaski" odcinek to błąd na podstawie etykiety.
        df_corr[features] = df_corr[features].ffill().bfill()

        # 3. CHRONOLOGICZNY PODZIAŁ DANYCH
        total_rows = len(df_corr)
        train_end = int(total_rows * TRAIN_SPLIT)
        val_end = int(total_rows * (TRAIN_SPLIT + VAL_SPLIT))

        df_train = df_corr.iloc[:train_end]
        df_val = df_corr.iloc[train_end:val_end]
        df_test = df_corr.iloc[val_end:]

        # 4. ZAPIS DO FOLDERÓW KLIENTA
        client_out_dir = os.path.join(FL_DATASET_DIR, f"client_{sensor_id}")
        os.makedirs(client_out_dir, exist_ok=True)

        df_train.to_csv(os.path.join(client_out_dir, "train.csv"))
        df_val.to_csv(os.path.join(client_out_dir, "val.csv"))
        df_test.to_csv(os.path.join(client_out_dir, "test.csv"))

        print(f"✅ Klient {sensor_id}: Train ({len(df_train)}), Val ({len(df_val)}), Test ({len(df_test)}). "
              f"Anomalie: {df_corr['label'].sum()} prób.")


if __name__ == "__main__":
    prepare_fl_data()
    print("\n🎉 Dane dla symulatora Federated Learning są gotowe do użycia w 'main_simulation_old.py'!")