#!/usr/bin/env python3
"""
Análise de Afinamento Espectral dos Gaps
Complementa o estudo geométrico, revelando zonas de ressonância harmônica
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks

ARQUIVO = "results.csv"
WINDOW = 10000      # Tamanho da janela (ajuste conforme o dataset)
STEP = 5000         # Passo entre janelas
MAX_PRIMOS = 500000 # Amostra total

print("=" * 80)
print("ANÁLISE DE AFINAÇÃO ESPECTRAL DOS GAPS")
print("=" * 80)

df = pd.read_csv(ARQUIVO, nrows=MAX_PRIMOS, on_bad_lines='skip')
primos = df.iloc[:, 0].values.astype(np.float64)
gaps = np.diff(primos)

n_janelas = (len(gaps) - WINDOW) // STEP
fft_map = np.zeros((n_janelas, WINDOW // 2))
freqs = np.fft.fftfreq(WINDOW, d=1)[:WINDOW // 2]

for i in range(n_janelas):
    start = i * STEP
    end = start + WINDOW
    sub_gaps = gaps[start:end]
    sub_gaps -= np.mean(sub_gaps)
    
    spec = np.abs(fft(sub_gaps))[:WINDOW // 2]
    spec /= np.max(spec)
    fft_map[i, :] = spec

print(f"[OK] FFT por janelas concluída ({n_janelas} janelas)")

# --------------------------------------------------------------------------
# Identificar picos médios (frequências harmônicas dominantes)
mean_spec = np.mean(fft_map, axis=0)
peaks, props = find_peaks(mean_spec, height=0.1)
harmonics = freqs[peaks]

print(f"[OK] Frequências dominantes detectadas: {len(harmonics)}")
for i, f in enumerate(harmonics[:10], 1):
    print(f"  {i:2d}. f = {f:.4e}")

# --------------------------------------------------------------------------
# PLOT: mapa de afinação (espectrograma)
plt.figure(figsize=(14, 8))
plt.imshow(np.log10(fft_map.T + 1e-4), aspect='auto', origin='lower',
           extent=[0, n_janelas, freqs[0], freqs[-1]],
           cmap='viridis')
plt.colorbar(label='log10(Potência normalizada)')
plt.xlabel("Janela (posição sequencial)")
plt.ylabel("Frequência espacial (1/gap)")
plt.title("Mapa de Afinação Espectral dos Gaps (Twin Primes)")
for f in harmonics:
    plt.axhline(f, color='r', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("afinacao_espectral_gaps.png", dpi=200)
print("[OK] Gráfico salvo: afinacao_espectral_gaps.png")

# --------------------------------------------------------------------------
# Correlação entre entropia local e potência harmônica
entropias = []
potencias = []

for i in range(n_janelas):
    sub = gaps[i*STEP:i*STEP+WINDOW]
    p, _ = np.histogram(sub, bins=50, density=True)
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    entropias.append(H)
    potencias.append(np.max(fft_map[i, :]))

corr = np.corrcoef(entropias, potencias)[0,1]
print(f"Correlação entropia-potência: {corr:.4f}")
