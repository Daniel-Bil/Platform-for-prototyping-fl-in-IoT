import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfiguracja stylu wykresów (akademicki, czysty)
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

DATA_DIR = os.path.join("data", "original_files")
OUTPUT_DIR = os.path.join("data", "clean_visualizations")

# Tworzymy folder na nowe wykresy, jeśli nie istnieje
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_sensor_data(csv_path):
    filename = os.path.basename(csv_path)
    sensor_id = filename.split('.')[0]

    print(f"📊 Przetwarzanie: {filename}...")

    # 1. Wczytanie danych
    df = pd.read_csv(csv_path)

    # 2. Konwersja czasu do obiektu datetime (bardzo ważne dla osi X!)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Wybieramy tylko kolumny numeryczne (omijamy ewentualne metadane)
    cols_to_plot = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
    available_cols = [c for c in cols_to_plot if c in df.columns]

    if not available_cols:
        print(f"⚠️ Brak odpowiednich kolumn w {filename}")
        return

    # 3. Generowanie wykresu (Subplots ze wspólną osią X)
    fig, axes = plt.subplots(nrows=len(available_cols), ncols=1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Raw Data Profile: {sensor_id}', fontsize=16, weight='bold')

    # Kolory dla poszczególnych miar, aby praca była czytelna
    colors = ['#d62728', '#1f77b4', '#9467bd', '#17becf']

    for i, col in enumerate(available_cols):
        ax = axes[i]
        # Rysujemy linię
        ax.plot(df.index, df[col], color=colors[i % len(colors)], linewidth=1.5, alpha=0.9)

        # Upiększanie
        ax.set_ylabel(col.replace('value_', '').upper(), weight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Zaznaczanie obszarów zerowych (anomalii) na czerwono dla wizualnego podkreślenia błędów
        anomalies = df[df[col] <= 0]
        if not anomalies.empty:
            ax.scatter(anomalies.index, anomalies[col], color='red', s=20, zorder=5, label='Anomalies/Zeros')
            ax.legend(loc='upper right', fontsize=10)

    plt.xlabel('Time')
    plt.tight_layout()

    # Zapis w wysokiej rozdzielczości (DPI 300 - standard dla publikacji)
    out_path = os.path.join(OUTPUT_DIR, f"{sensor_id}_raw_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Zapisano: {out_path}")


if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if not csv_files:
        print(f"❌ Nie znaleziono plików CSV w {DATA_DIR}")
    else:
        for file in csv_files:
            plot_sensor_data(file)
        print("\n🎉 Zakończono generowanie nowych wykresów.")