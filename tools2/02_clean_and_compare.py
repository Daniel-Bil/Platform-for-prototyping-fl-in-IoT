import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

DATA_DIR = os.path.join("data", "original_files")
OUTPUT_DIR = os.path.join("data", "clean_visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIMITS = {
    'value_temp': (-30, 60),
    'value_hum': (0, 100),
    'value_acid': (0, 14),
    'value_PV': (0, 10)
}


def remove_spikes(series, threshold, window=5):
    """
    Usuwa igły używając kroczącej mediany.
    Odporniejsza na 'szerokie' igły (trwające 2-3 pomiary z rzędu)
    oraz spadki na samym końcu pliku (dzięki min_periods=1).
    """
    # Obliczamy medianę z lokalnego sąsiedztwa
    rolling_med = series.rolling(window=window, center=True, min_periods=1).median()

    # Wyłapujemy momenty, które drastycznie odjeżdżają od mediany
    spike_mask = (series - rolling_med).abs() > threshold

    clean_series = series.copy()
    clean_series.loc[spike_mask] = np.nan
    return clean_series


def process_and_compare(csv_path):
    filename = os.path.basename(csv_path)
    sensor_id = filename.split('.')[0]
    print(f"🛠️ Przetwarzanie: {sensor_id}...")

    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time')

    # Odcięcie fazy stabilizacji (pierwsze 200 próbek)
    if len(df) > 500:
        df = df.iloc[200:]

    df.set_index('time', inplace=True)

    cols_to_plot = [c for c in LIMITS.keys() if c in df.columns]

    # Resampling do sztywnej 10-minutowej siatki
    df_resampled = df[cols_to_plot].resample('10min').mean()
    df_clean = df_resampled.copy()

    for col in cols_to_plot:
        # A. Ucinanie absurdów fizycznych
        min_val, max_val = LIMITS[col]
        df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)

        # B. Usuwanie igieł za pomocą kroczącej mediany
        if col == 'value_temp':
            df_clean.loc[df_clean[col] == 0.0, col] = float('nan')
            df_clean[col] = remove_spikes(df_clean[col], threshold=5.0)

        elif col == 'value_acid':
            # Bardzo ostry rygor dla pH: odchył od lokalnej normy > 1.5 wylatuje
            df_clean[col] = remove_spikes(df_clean[col], threshold=1.5, window=5)

        elif col == 'value_hum':
            df_clean[col] = remove_spikes(df_clean[col], threshold=20.0, window=5)

        # C. Interpolacja - łatamy dziury do 4 godzin (limit=24)
        df_clean[col] = df_clean[col].interpolate(method='linear', limit=24)

    # =======================================================
    # WIZUALIZACJA PORÓWNAWCZA (Czysta, bez czerwonych kropek)
    # =======================================================
    fig, axes = plt.subplots(nrows=len(cols_to_plot), ncols=2, figsize=(18, 12), sharex=True)
    fig.suptitle(f'Data Pipeline: {sensor_id} (Raw vs Cleaned)', fontsize=18, weight='bold')

    colors = ['#d62728', '#1f77b4', '#9467bd', '#17becf']

    for i, col in enumerate(cols_to_plot):
        color = colors[i % len(colors)]

        # Lewa strona - surowe zresamplowane (teraz czyste od kropek!)
        ax_raw = axes[i, 0]
        ax_raw.plot(df_resampled.index, df_resampled[col], color=color, alpha=0.6, label='Raw (Resampled)')
        ax_raw.set_ylabel(col.replace('value_', '').upper(), weight='bold')
        ax_raw.set_title(f"Przed czyszczeniem ({col})")
        ax_raw.legend(loc='upper right')

        # Prawa strona - po czyszczeniu i interpolacji
        ax_clean = axes[i, 1]
        ax_clean.plot(df_clean.index, df_clean[col], color=color, linewidth=1.5, label='Cleaned & Interpolated')
        ax_clean.set_title(f"Po czyszczeniu ({col})")
        ax_clean.legend(loc='upper right')

    for ax in axes.flatten():
        ax.tick_params(axis='x', rotation=45)

    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{sensor_id}_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    clean_csv_path = os.path.join("data", f"{sensor_id}_CLEANED.csv")
    df_clean.to_csv(clean_csv_path)
    print(f"✅ Zapisano pliki dla {sensor_id}.")


if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"❌ Nie znaleziono plików CSV w {DATA_DIR}")
    else:
        for file in csv_files:
            process_and_compare(file)
        print("\n🎉 Czyszczenie zakończone.")