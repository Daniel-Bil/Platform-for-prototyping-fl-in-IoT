#!/usr/bin/env python3
"""
01_clean_and_profile.py
Etap 1: Analiza eksploracyjna, filtracja medianowa, resampling i profilowanie
danych sensorowych RuralIoT.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Konfiguracja stylu wykresów pod publikację naukową
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.autolayout': True
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "eda_profiles")

os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Granice fizyczne dla czujników środowiskowych
LIMITS = {
    'value_temp': (-30.0, 60.0),
    'value_hum': (0.0, 100.0),
    'value_acid': (0.0, 14.0),
    'value_PV': (0.0, 10.0)
}

SPIKE_THRESHOLDS = {
    'value_temp': 5.0,
    'value_hum': 20.0,
    'value_acid': 1.5,
    'value_PV': 3.0
}

FEATURE_LABELS = {
    'value_temp': 'Temperature [°C]',
    'value_hum': 'Relative Humidity [%]',
    'value_acid': 'Acidity [pH]',
    'value_PV': 'Solar Potential [V]'
}

COLORS = {
    'value_temp': '#d62728',
    'value_hum': '#1f77b4',
    'value_acid': '#9467bd',
    'value_PV': '#ff7f0e'
}


def remove_spikes_rolling_median(series: pd.Series, threshold: float, window: int = 5) -> pd.Series:
    """Usuwa impulsowe piki przy użyciu kroczącej mediany."""
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    is_spike = (series - rolling_median).abs() > threshold
    clean_series = series.copy()
    clean_series.loc[is_spike] = np.nan
    return clean_series


def process_sensor_file(file_path: str):
    filename = os.path.basename(file_path)
    sensor_id = filename.replace("df_RuralIoT_", "").replace(".csv", "")
    print(f"\n📡 Przetwarzanie stacji: {filename} (ID: {sensor_id})...")

    df = pd.read_csv(file_path)
    if 'time' not in df.columns:
        print(f"⚠️ Brak kolumny 'time' w {filename}. Pomijam.")
        return

    # 1. Parsowanie czasu i sortowanie
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time')

    # Odcięcie niestabilnej fazy rozruchu czujnika (pierwsze 200 odczytów)
    if len(df) > 500:
        df = df.iloc[200:].copy()

    df.set_index('time', inplace=True)
    active_cols = [c for c in LIMITS.keys() if c in df.columns]

    # =========================================================================
    # WYKRES 1: Surowy profil z zaznaczeniem anomalii zerowych i nasyceń
    # =========================================================================
    fig, axes = plt.subplots(nrows=len(active_cols), ncols=1, figsize=(14, 9), sharex=True)
    if len(active_cols) == 1:
        axes = [axes]

    fig.suptitle(f"Raw Empirical Telemetry Profile: Sensor {sensor_id}", fontsize=15, weight='bold')

    for idx, col in enumerate(active_cols):
        ax = axes[idx]
        ax.plot(df.index, df[col], color=COLORS[col], linewidth=1.2, alpha=0.85, label='Raw Telemetry')
        ax.set_ylabel(FEATURE_LABELS[col], weight='bold')

        # Wykrywanie patologii sprzętowych (zera w temp/hum, wartości poza skalą)
        anomalies = df[
            (df[col] <= 0.0) |
            (df[col] > LIMITS[col][1])
        ]
        if not anomalies.empty:
            ax.scatter(anomalies.index, anomalies[col], color='black', edgecolor='red',
                       s=25, zorder=5, label='Zero/Out-of-bound Anomaly')

        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Time (UTC)", weight='bold')
    plt.tight_layout()
    raw_plot_path = os.path.join(RESULTS_DIR, f"{sensor_id}_01_raw_profile.png")
    plt.savefig(raw_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 2. Czyszczenie, siatka 10-minutowa i interpolacja
    # =========================================================================
    # Resampling do sztywnej siatki 10-minutowej
    df_resampled = df[active_cols].resample('10min').mean()
    df_cleaned = df_resampled.copy()

    for col in active_cols:
        # A. Zamiana sztucznych zer (np. w temperaturze i wilgotności) na NaN
        if col in ['value_temp', 'value_hum']:
            df_cleaned.loc[df_cleaned[col] <= 0.0, col] = np.nan

        # B. Filtracja igieł medianą kroczącą
        df_cleaned[col] = remove_spikes_rolling_median(
            df_cleaned[col],
            threshold=SPIKE_THRESHOLDS[col],
            window=5
        )

        # C. Obcięcie do fizycznych klamer (clamping)
        min_v, max_v = LIMITS[col]
        df_cleaned[col] = df_cleaned[col].clip(lower=min_v, upper=max_v)

        # D. Liniowa interpolacja przerw do maksymalnie 4h (24 próbki x 10 min)
        df_cleaned[col] = df_cleaned[col].interpolate(method='linear', limit=24)

    # =========================================================================
    # WYKRES 2: Wizualizacja porównawcza (Raw Resampled vs Cleaned & Interpolated)
    # =========================================================================
    fig, axes = plt.subplots(nrows=len(active_cols), ncols=2, figsize=(16, 10), sharex=True)
    fig.suptitle(f"Data Cleansing & Regularization: Sensor {sensor_id}", fontsize=15, weight='bold')

    for idx, col in enumerate(active_cols):
        color = COLORS[col]

        # Lewa kolumna: Resampled surowy
        ax_left = axes[idx, 0]
        ax_left.plot(df_resampled.index, df_resampled[col], color=color, alpha=0.5, label='Raw (Resampled 10-min)')
        ax_left.set_ylabel(FEATURE_LABELS[col], weight='bold')
        ax_left.set_title(f"Unfiltered: {col}")
        ax_left.legend(loc='upper right')

        # Prawa kolumna: Wyczyszczony i zinterpolowany
        ax_right = axes[idx, 1]
        ax_right.plot(df_cleaned.index, df_cleaned[col], color=color, linewidth=1.4, label='Filtered & Interpolated (max 4h)')
        ax_right.set_title(f"Cleaned: {col}")
        ax_right.legend(loc='upper right')

    for ax in axes.flatten():
        ax.tick_params(axis='x', rotation=30)

    axes[-1, 0].set_xlabel("Time (UTC)", weight='bold')
    axes[-1, 1].set_xlabel("Time (UTC)", weight='bold')

    plt.tight_layout()
    cmp_plot_path = os.path.join(RESULTS_DIR, f"{sensor_id}_02_cleaning_comparison.png")
    plt.savefig(cmp_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Zapis wyczyszczonego pliku CSV
    clean_csv_path = os.path.join(CLEANED_DATA_DIR, f"{sensor_id}_CLEANED.csv")
    df_cleaned.to_csv(clean_csv_path)

    # Statystyki do terminala
    valid_samples = df_cleaned.dropna().shape[0]
    total_samples = len(df_cleaned)
    print(f"   -> Zapisano wykresy w: {RESULTS_DIR}")
    print(f"   -> Zapisano oczyszczony zbiór: {clean_csv_path}")
    print(f"   -> Ciągłych próbek bez NaN: {valid_samples} / {total_samples} ({valid_samples/total_samples*100:.1f}%)")


def main():
    raw_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
    if not raw_files:
        print(f"❌ Nie znaleziono żadnych plików CSV w folderze {RAW_DATA_DIR}!")
        print("Upewnij się, że pliki df_RuralIoT_*.csv znajdują się w '0synthetic_data_poligon/data/raw/'.")
        return

    print(f"🚀 Znaleziono {len(raw_files)} plików do przetworzenia.")
    for f in raw_files:
        process_sensor_file(f)

    print("\n✅ Wszystkie stacje zostały przefiltrowane i sprofilowane.")


if __name__ == "__main__":
    main()