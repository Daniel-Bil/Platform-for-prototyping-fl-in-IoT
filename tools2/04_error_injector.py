"""LEGACY synthetic-data fault injector.

Kept for optional VAE/synthetic demonstrations.  The final thesis benchmark is
prepared by ``05_prepare_fl_dataset.py`` directly from cleaned real RuralIoT
measurements and does not call this script.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

SYNTHETIC_DIR = os.path.join("data", "synthetic_files")
OUTPUT_DIR = os.path.join("data", "federated_clients")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# BIBLIOTEKA WIRUSÓW (METODY NISZCZĄCE)
# ==========================================
def inject_dropouts(series, drop_rate=0.3):
    """Losowo usuwa N% próbek (wstawia NaN)."""
    corrupted = series.copy()
    mask = np.random.rand(len(corrupted)) < drop_rate
    corrupted[mask] = np.nan
    return corrupted


def inject_drift(series, max_drift_at_end=15.0):
    """Liniowo dodaje błąd pomiarowy rosnący wraz z upływem czasu."""
    drift_array = np.linspace(0, max_drift_at_end, len(series))
    return series + drift_array


def inject_flatline(series, flat_value=100.0, start_fraction=0.6):
    """Od pewnego momentu czasu czujnik zawiesza się na jednej wartości."""
    corrupted = series.copy()
    start_idx = int(len(corrupted) * start_fraction)
    corrupted.iloc[start_idx:] = flat_value
    # Dodajemy delikatny szum, żeby nie była to sztuczna linia prosta (np. szum ADC)
    noise = np.random.normal(0, 0.5, len(corrupted) - start_idx)
    corrupted.iloc[start_idx:] += noise
    return corrupted


# ==========================================
# GŁÓWNY PROCES NISZCZENIA
# ==========================================
def create_corrupted_client(csv_path):
    filename = os.path.basename(csv_path)
    sensor_id = filename.replace('_SYNTHETIC.csv', '')
    print(f"🦠 Wstrzykiwanie błędów do: {sensor_id}...")

    df = pd.read_csv(csv_path, index_col='time', parse_dates=True)
    df_corrupted = df.copy()

    # --- PROFILOWANIE KLIENTÓW (NON-IID) ---
    # Definiujemy różne awarie dla różnych czujników, żeby utrudnić zadanie algorytmom FL

    if "010" in sensor_id:
        # Klient 010: Ogromne problemy z zasięgiem (gubienie 40% pakietów w całym sprzęcie)
        print("   -> Profil: Massive Dropouts (Utrata 40% pakietów)")
        for col in df_corrupted.columns:
            df_corrupted[col] = inject_dropouts(df_corrupted[col], drop_rate=0.4)

    elif "002" in sensor_id or "001" in sensor_id:
        # Klienci 001/002: Rozkalibrowana temperatura (powoli rośnie o 15 stopni do końca roku)
        print("   -> Profil: Temperature Drift (Rozkalibrowanie)")
        df_corrupted['value_temp'] = inject_drift(df_corrupted['value_temp'], max_drift_at_end=15.0)

    elif "22" in sensor_id or "23" in sensor_id:
        # Klienci 22/23: Zalany czujnik wilgotności po 60% czasu działania
        print("   -> Profil: Humidity Flatline (Zwarcie na 100%)")
        df_corrupted['value_hum'] = inject_flatline(df_corrupted['value_hum'], flat_value=100.0, start_fraction=0.6)

    else:
        # Reszta klientów dostaje lekki szum i 10% utraty pakietów (Standardowy klient)
        print("   -> Profil: Minor connection issues (10% Dropouts)")
        for col in df_corrupted.columns:
            df_corrupted[col] = inject_dropouts(df_corrupted[col], drop_rate=0.1)

    # Zapis zepsutego pliku klienta
    out_csv = os.path.join(OUTPUT_DIR, f"{sensor_id}_CLIENT_DATA.csv")
    df_corrupted.to_csv(out_csv)

    # --- WIZUALIZACJA PORÓWNAWCZA (Czyste vs Zepsute) ---
    # Agregacja makro (dzienna), by pokazać długoterminowe uszkodzenia
    df_clean_mean = df.resample('1D').mean()
    df_corr_mean = df_corrupted.resample('1D').mean()

    fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Federated Client Profile: {sensor_id}', fontsize=16, weight='bold')

    colors = ['#2ca02c', '#d62728']  # Zielony (Zdrowy), Czerwony (Uszkodzony)

    for i, col in enumerate(df.columns):
        ax = axes[i]
        # Linia zdrowa (Oryginalny Syntetyk)
        ax.plot(df_clean_mean.index, df_clean_mean[col], color=colors[0], linewidth=2, alpha=0.5,
                label='Clean Synthetic')
        # Linia zepsuta (Dane Klienta)
        ax.plot(df_corr_mean.index, df_corr_mean[col], color=colors[1], linewidth=1.5, label='Corrupted Client Data')

        ax.set_ylabel(col.replace('value_', '').upper(), weight='bold')
        ax.legend(loc='upper left')

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"{sensor_id}_client_profile.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✅ Utworzono klienta: {out_png}")


if __name__ == "__main__":
    synthetic_files = glob.glob(os.path.join(SYNTHETIC_DIR, "*_SYNTHETIC.csv"))
    if not synthetic_files:
        print(f"❌ Brak plików w {SYNTHETIC_DIR}. Odpal najpierw generator syntetyczny.")
    else:
        print("⚠️ ROZPOCZĘTO INFEKCJĘ DANYCH...")
        for file in synthetic_files:
            create_corrupted_client(file)
        print("\n🎉 Zakończono! Klienci do Federated Learning są gotowi w folderze federated_clients.")