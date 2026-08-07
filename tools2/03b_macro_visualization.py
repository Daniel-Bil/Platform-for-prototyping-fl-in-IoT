import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

SYNTHETIC_DIR = os.path.join("data", "synthetic_files")
OUTPUT_DIR = os.path.join("data", "synthetic_files", "macro_views")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_macro_view(csv_path):
    filename = os.path.basename(csv_path)
    sensor_id = filename.replace('_SYNTHETIC.csv', '')
    print(f"📊 Generowanie widoku Makro dla: {sensor_id} ({filename})")

    # Wczytanie całego, ogromnego pliku (np. 43k próbek)
    df = pd.read_csv(csv_path, index_col='time', parse_dates=True)
    features = ['value_temp', 'value_hum', 'value_acid', 'value_PV']

    # AGREGACJA: Grupujemy dane w bloki 1-dniowe ('1D')
    # Dla każdego dnia liczymy średnią, minimum i maksimum
    df_mean = df.resample('1D').mean()
    df_min = df.resample('1D').min()
    df_max = df.resample('1D').max()

    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Macro View: {sensor_id} (Full Synthetic Dataset Overview)', fontsize=18, weight='bold')

    colors = ['#d62728', '#1f77b4', '#9467bd', '#17becf']

    for i, col in enumerate(features):
        ax = axes[i]
        c = colors[i % len(colors)]

        # Rysujemy zacieniowany obszar między MIN a MAX każdego dnia
        ax.fill_between(df_mean.index, df_min[col], df_max[col], color=c, alpha=0.2, label='Daily Min-Max Range')

        # Rysujemy mocną linię dla średniej dziennej
        ax.plot(df_mean.index, df_mean[col], color=c, linewidth=2, label='Daily Mean')

        ax.set_ylabel(col.replace('value_', '').upper(), weight='bold')
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time (Aggregated by Days)')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{sensor_id}_macro_view.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Zapisano widok makro: {out_path}")


if __name__ == "__main__":
    synthetic_files = glob.glob(os.path.join(SYNTHETIC_DIR, "*_SYNTHETIC.csv"))
    if not synthetic_files:
        print(f"❌ Nie znaleziono plików *_SYNTHETIC.csv w {SYNTHETIC_DIR}. Wygeneruj je najpierw!")
    else:
        for file in synthetic_files:
            plot_macro_view(file)
        print("\n🎉 Wizualizacja Makro zakończona pomyślnie.")