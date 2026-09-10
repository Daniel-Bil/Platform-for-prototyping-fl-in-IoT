#!/usr/bin/env python3
"""
03_evaluate_synthetic.py
Etap 3: Kompleksowa ewaluacja metryk i weryfikacja empiryczna:
Ground Truth vs ARIMA vs Conv1D-VAE vs TimeGAN
Metryki: Trajektorie 5-dniowe, Macierze korelacji (Pearson), PDF/CDF, Autokorelacja (ACF)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_FILE = os.path.join(BASE_DIR, "data", "continuous_blocks", "002_CONTINUOUS_BLOCK.csv")
ARIMA_FILE = os.path.join(BASE_DIR, "data", "synthetic_output", "002_SYNTHETIC_ARIMA.csv")
VAE_FILE = os.path.join(BASE_DIR, "data", "synthetic_output", "002_SYNTHETIC_VAE.csv")
TIMEGAN_FILE = os.path.join(BASE_DIR, "data", "synthetic_output", "002_SYNTHETIC_TIMEGAN.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "evaluation_metrics")

os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
LABELS = ['Temp', 'Humidity', 'Acidity', 'Solar PV']


def evaluate_trajectories(df_real, df_arima, df_vae, df_timegan, days=5):
    """Generuje 5-dniowe porownanie trajektorii czasowych 4 modeli."""
    plot_len = 144 * days
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 11), sharex=True)
    fig.suptitle(f"{days}-Day Trajectory Comparison: Ground Truth vs ARIMA vs VAE vs TimeGAN", fontsize=13, weight='bold')

    for i, col in enumerate(FEATURES):
        ax = axes[i]
        ax.plot(df_real.index[:plot_len], df_real[col].iloc[:plot_len], color='black', linewidth=1.6, label='Real (Ground Truth)')
        ax.plot(df_real.index[:plot_len], df_arima[col].iloc[:plot_len], color='#1f77b4', linestyle='--', linewidth=1.2, label='ARIMA')
        ax.plot(df_real.index[:plot_len], df_vae[col].iloc[:plot_len], color='#d62728', alpha=0.8, linewidth=1.2, label='Conv1D-VAE')
        ax.plot(df_real.index[:plot_len], df_timegan[col].iloc[:plot_len], color='#2ca02c', alpha=0.85, linewidth=1.3, label='TimeGAN (SoTA)')
        ax.set_ylabel(col, weight='bold')
        if i == 0:
            ax.legend(loc='upper right', ncol=4)

    axes[-1].set_xlabel("Time (UTC)", weight='bold')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "002_trajectories_all_models.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres trajektorii: {out_path}")


def evaluate_cross_correlation(df_real, df_arima, df_vae, df_timegan):
    """Porownawcze mapy ciepla korelacji miedzykanalowych."""
    corr_real = df_real[FEATURES].corr()
    corr_arima = df_arima[FEATURES].corr()
    corr_vae = df_vae[FEATURES].corr()
    corr_timegan = df_timegan[FEATURES].corr()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle("Cross-Feature Correlation Matrices across Synthesis Paradigms", fontsize=13, weight='bold')

    datasets = [
        (corr_real, "Real Ground Truth", axes[0]),
        (corr_arima, "ARIMA Baseline", axes[1]),
        (corr_vae, "Conv1D-VAE", axes[2]),
        (corr_timegan, "TimeGAN (SoTA)", axes[3])
    ]

    for corr, title, ax in datasets:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0,
                    xticklabels=LABELS, yticklabels=LABELS, cbar=(ax == axes[-1]), ax=ax)
        ax.set_title(title, weight='bold')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "002_cross_correlation_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano macierze korelacji: {out_path}")

    # Wyswietlenie wartosci w terminalu do tabeli LaTeX
    print("\n--- Podsumowanie korelacji Pearsona ---")
    pairs = [('value_temp', 'value_hum'), ('value_temp', 'value_acid'), ('value_temp', 'value_PV'),
             ('value_hum', 'value_acid'), ('value_hum', 'value_PV'), ('value_acid', 'value_PV')]
    for c1, c2 in pairs:
        print(f"{c1} <-> {c2} | Real: {corr_real.loc[c1, c2]:.2f} | ARIMA: {corr_arima.loc[c1, c2]:.2f} | VAE: {corr_vae.loc[c1, c2]:.2f} | TimeGAN: {corr_timegan.loc[c1, c2]:.2f}")


def evaluate_distributions(df_real, df_arima, df_vae, df_timegan):
    """Wykresy gestosci (PDF) oraz dystrybuanty (CDF) dla 4 modeli."""
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))
    fig.suptitle("Marginal Distributions (PDF) and Cumulative Probability (CDF)", fontsize=14, weight='bold')

    models_data = [
        (df_real, 'black', '-', 'Real', 1.6),
        (df_arima, '#1f77b4', '--', 'ARIMA', 1.2),
        (df_vae, '#d62728', '-', 'VAE', 1.2),
        (df_timegan, '#2ca02c', '-', 'TimeGAN', 1.3)
    ]

    for i, col in enumerate(FEATURES):
        ax_pdf = axes[i, 0]
        ax_cdf = axes[i, 1]

        for df_m, color, style, name, lw in models_data:
            data = df_m[col].dropna().values

            # PDF
            sns.kdeplot(data, color=color, linestyle=style, linewidth=lw, label=name, ax=ax_pdf)

            # CDF
            sorted_data = np.sort(data)
            y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax_cdf.plot(sorted_data, y_vals, color=color, linestyle=style, linewidth=lw, label=name)

        ax_pdf.set_title(f"PDF: {LABELS[i]}", weight='bold')
        ax_cdf.set_title(f"CDF: {LABELS[i]}", weight='bold')
        if i == 0:
            ax_pdf.legend(loc='upper right')
            ax_cdf.legend(loc='lower right')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "002_pdf_cdf_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykresy PDF/CDF: {out_path}")


def evaluate_autocorrelation(df_real, df_arima, df_vae, df_timegan, max_lags=24):
    """Porownanie autokorelacji ACF dla opoznien dobowych."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle(f"Autocorrelation Structure (ACF Lags 0 to {max_lags})", fontsize=14, weight='bold')

    models_data = [
        (df_real, 'black', '-', 'o', 'Real', 1.4),
        (df_arima, '#1f77b4', '--', 's', 'ARIMA', 1.2),
        (df_vae, '#d62728', '-', '^', 'VAE', 1.2),
        (df_timegan, '#2ca02c', '-', 'd', 'TimeGAN', 1.3)
    ]

    for i, col in enumerate(FEATURES):
        ax = axes[i]
        for df_m, color, style, marker, name, lw in models_data:
            acf_vals = acf(df_m[col].dropna(), nlags=max_lags)
            lags = np.arange(len(acf_vals))
            ax.plot(lags, acf_vals, marker=marker, markersize=4, linestyle=style, color=color, linewidth=lw, label=name)

        ax.set_title(f"ACF: {LABELS[i]}", weight='bold')
        ax.set_xlabel("Lags (10-min steps)")
        ax.set_ylabel("Autocorrelation")
        if i == 0:
            ax.legend(loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "002_autocorrelation_acf.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano autokorelacje ACF: {out_path}")


def main():
    for p in [REAL_FILE, ARIMA_FILE, VAE_FILE, TIMEGAN_FILE]:
        if not os.path.exists(p):
            print(f"Blad: Brak pliku {p}!")
            return

    print("Wczytywanie szeregow 4 modeli...")
    df_real = pd.read_csv(REAL_FILE, parse_dates=['time']).set_index('time')
    df_arima = pd.read_csv(ARIMA_FILE, parse_dates=['time']).set_index('time')
    df_vae = pd.read_csv(VAE_FILE, parse_dates=['time']).set_index('time')
    df_timegan = pd.read_csv(TIMEGAN_FILE, parse_dates=['time']).set_index('time')

    evaluate_trajectories(df_real, df_arima, df_vae, df_timegan)
    evaluate_cross_correlation(df_real, df_arima, df_vae, df_timegan)
    evaluate_distributions(df_real, df_arima, df_vae, df_timegan)
    evaluate_autocorrelation(df_real, df_arima, df_vae, df_timegan)
    print("\nKompletna ewaluacja zakonczona.")


if __name__ == "__main__":
    main()