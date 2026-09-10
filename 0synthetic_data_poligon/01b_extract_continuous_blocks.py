#!/usr/bin/env python3
"""
01b_extract_continuous_blocks.py
Etap 1b: Selekcja ciągłych, w pełni kompletnych bloków czasowych bez braków (NaN)
i bez sztucznych linii nasycenia pod modele statystyczne (ARIMA) i generatywne (TimeGAN/VAE).
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "continuous_blocks")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "continuous_segments")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
FEATURE_LABELS = {
    'value_temp': 'Temperature [°C]',
    'value_hum': 'Humidity [%]',
    'value_acid': 'Acidity [pH]',
    'value_PV': 'Solar Potential [V]'
}
COLORS = {
    'value_temp': '#d62728',
    'value_hum': '#1f77b4',
    'value_acid': '#9467bd',
    'value_PV': '#ff7f0e'
}

# Minimalna długość ciągłego bloku: 7 dni x 144 próbki/dobę = 1008 kroków
MIN_SAMPLES = 144 * 7


def find_continuous_valid_segments(df: pd.DataFrame, min_length: int = MIN_SAMPLES):
    """
    Znajduje ciągłe indeksy bez NaN oraz bez zablokowanej wariancji (np. stałe 100% wilgotności).
    """
    # 1. Maska wierszy bez żadnego NaN
    valid_mask = ~df[FEATURES].isna().any(axis=1)

    # 2. Identyfikacja spójnych bloków boolowskich
    block_ids = (~valid_mask).cumsum()
    valid_blocks = block_ids[valid_mask]

    valid_segments = []
    for b_id, group in valid_blocks.groupby(valid_blocks):
        if len(group) >= min_length:
            segment = df.loc[group.index]

            # 3. Weryfikacja biologiczno-fizyczna: odrzucenie zaciętych sensorów
            # Standardowe odchylenie wilgotności w cyklu dobowym musi być > 1.5%
            # Nie dopuszczamy sztucznych linii na 100% (jak w stacji 003)
            hum_std = segment['value_hum'].std()
            temp_std = segment['value_temp'].std()

            if hum_std >= 2.0 and temp_std >= 1.0:
                valid_segments.append(segment)
            else:
                print(
                    f"      [Odrzucono blok {len(segment)} próbek z powodu zablokowanej wariancji (std_hum={hum_std:.2f}, std_temp={temp_std:.2f})]")

    return valid_segments


def process_file(csv_path: str):
    filename = os.path.basename(csv_path)
    sensor_id = filename.replace("_CLEANED.csv", "")
    print(f"\n🔍 Skanowanie stacji: {sensor_id} ({filename})...")

    df = pd.read_csv(csv_path, parse_dates=['time'])
    df.set_index('time', inplace=True)

    # Upewniamy się, że są wszystkie 4 cechy
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        print(f"   ⚠️ Brak kolumn {missing_cols}. Pomijam.")
        return

    segments = find_continuous_valid_segments(df, min_length=MIN_SAMPLES)
    if not segments:
        print(
            f"   ❌ Brak w 100% ciągłego segmentu >= {MIN_SAMPLES} próbek ({MIN_SAMPLES / 144:.1f} dni) dla stacji {sensor_id}.")
        return

    # Wybieramy najdłuższy dostępny ciągły segment
    longest_segment = max(segments, key=len)
    num_days = len(longest_segment) / 144
    start_t = longest_segment.index[0].strftime("%Y-%m-%d %H:%M")
    end_t = longest_segment.index[-1].strftime("%Y-%m-%d %H:%M")

    print(f"   ✅ Znaleziono najdłuższy ciągły blok:")
    print(f"      Zakres: {start_t} -> {end_t}")
    print(f"      Liczba próbek: {len(longest_segment)} ({num_days:.1f} pełnych dni bez ani jednego NaN)")

    # 1. Zapis do CSV
    out_csv = os.path.join(OUTPUT_DIR, f"{sensor_id}_CONTINUOUS_BLOCK.csv")
    longest_segment.to_csv(out_csv)
    print(f"      Zapisano: {out_csv}")

    # 2. Generowanie wykresu weryfikacyjnego 300 DPI
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Continuous Valid Training Block: Sensor {sensor_id} ({num_days:.1f} Days)", fontsize=14,
                 weight='bold')

    for idx, col in enumerate(FEATURES):
        ax = axes[idx]
        ax.plot(longest_segment.index, longest_segment[col], color=COLORS[col], linewidth=1.3)
        ax.set_ylabel(FEATURE_LABELS[col], weight='bold')

    axes[-1].set_xlabel("Time (UTC)", weight='bold')
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, f"{sensor_id}_continuous_block.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    cleaned_files = sorted(glob.glob(os.path.join(CLEANED_DIR, "*_CLEANED.csv")))
    if not cleaned_files:
        print(f"❌ Brak plików w {CLEANED_DIR}. Uruchom najpierw 01_clean_and_profile.py!")
        return

    print("🚀 Rozpoczynam ekstrakcję ciągłych bloków danych...")
    for f in cleaned_files:
        process_file(f)
    print("\n🎉 Zakończono ekstrakcję. Wyselekcjonowane bloki znajdują się w data/continuous_blocks/.")


if __name__ == "__main__":
    main()