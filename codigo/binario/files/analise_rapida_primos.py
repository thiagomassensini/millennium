#!/usr/bin/env python3
"""
Análise Rápida: Densidade de Primos vs f_cosmos
Versão simplificada para execução rápida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import sys

print("=" * 80)
print("ANÁLISE RÁPIDA: PERIODICIDADE EM PRIMOS GÊMEOS")
print("=" * 80)
print()

# ==============================================================================
# PARÂMETROS
# ==============================================================================

if len(sys.argv) > 1:
    ARQUIVO = sys.argv[1]
else:
    ARQUIVO = "results.csv"

# Número de linhas para análise (None = todas)
MAX_LINHAS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000  # 1M por padrão

# Tamanho da janela para densidade local
WINDOW_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

print(f"Arquivo: {ARQUIVO}")
print(f"Linhas: {MAX_LINHAS if MAX_LINHAS else 'todas'}")
print(f"Janela: {WINDOW_SIZE}")
print()

# ==============================================================================
# CARREGAR DADOS
# ==============================================================================

print("Carregando dados...")
try:
    df = pd.read_csv(ARQUIVO, header=None, nrows=MAX_LINHAS)
    primos = df.iloc[:, 0].values.astype(np.float64)
    print(f"✓ {len(primos)} primos carregados")
    print(f"  Range: {primos.min():.6e} → {primos.max():.6e}")
    print(f"  Span: {primos.max() - primos.min():.6e}")
    print()
except Exception as e:
    print(f"✗ Erro: {e}")
    sys.exit(1)

# ==============================================================================
# DENSIDADE LOCAL
# ==============================================================================

print("Calculando densidade local...")

n_windows = (len(primos) - WINDOW_SIZE) // (WINDOW_SIZE // 10)
posicoes = np.zeros(n_windows)
densidades = np.zeros(n_windows)
gaps_medios = np.zeros(n_windows)

idx = 0
for i in range(0, len(primos) - WINDOW_SIZE, WINDOW_SIZE // 10):
    if idx >= n_windows:
        break
    
    janela = primos[i:i+WINDOW_SIZE]
    posicoes[idx] = np.mean(janela)
    
    # Densidade = N / (max - min)
    span = janela.max() - janela.min()
    if span > 0:
        densidades[idx] = WINDOW_SIZE / span
    
    # Gap médio
    gaps = np.diff(janela)
    gaps_medios[idx] = np.mean(gaps)
    
    idx += 1

posicoes = posicoes[:idx]
densidades = densidades[:idx]
gaps_medios = gaps_medios[:idx]

print(f"✓ {len(posicoes)} janelas analisadas")
print(f"  Densidade média: {np.mean(densidades):.6e}")
print(f"  Desvio padrão: {np.std(densidades):.6e}")
print(f"  Coef. variação: {np.std(densidades)/np.mean(densidades):.4f}")
print(f"  Gap médio: {np.mean(gaps_medios):.2f}")
print()

# ==============================================================================
# ANÁLISE ESPECTRAL
# ==============================================================================

print("Análise espectral...")

# Normalizar
dens_norm = (densidades - np.mean(densidades)) / np.std(densidades)

# FFT
N = len(dens_norm)
yf = fft(dens_norm)
xf = fftfreq(N, d=1.0)  # Frequência em "unidades de janela"

# Só frequências positivas
mask = xf > 0
freqs = xf[mask]
power = np.abs(yf[mask])**2

# Detectar picos
threshold = np.mean(power) + 3 * np.std(power)
picos, props = signal.find_peaks(power, height=threshold, distance=5)

print(f"✓ {len(picos)} picos detectados")

if len(picos) > 0:
    # Ordenar por potência
    idx_sorted = np.argsort(power[picos])[::-1]
    
    print("\nTop 10 frequências:")
    for i, idx in enumerate(idx_sorted[:10], 1):
        f = freqs[picos[idx]]
        p = power[picos[idx]]
        periodo = 1.0 / f if f > 0 else np.inf
        print(f"  {i:2d}. freq={f:.6f} | período={periodo:.2f} janelas | potência={p:.2e}")

print()

# ==============================================================================
# ANÁLISE DE GAPS
# ==============================================================================

print("Análise de gaps entre primos...")

all_gaps = np.diff(primos)
print(f"  Total de gaps: {len(all_gaps)}")
print(f"  Gap médio: {np.mean(all_gaps):.4f}")
print(f"  Gap mediano: {np.median(all_gaps):.4f}")
print(f"  Desvio padrão: {np.std(all_gaps):.4f}")
print(f"  Mínimo: {np.min(all_gaps):.4f}")
print(f"  Máximo: {np.max(all_gaps):.4f}")

# Distribuição de gaps
gap_counts = np.bincount(all_gaps.astype(int))
print(f"\nDistribuição (gaps ≤ 30):")
for gap in range(2, min(31, len(gap_counts))):
    if gap_counts[gap] > 0:
        pct = 100 * gap_counts[gap] / len(all_gaps)
        print(f"    gap={gap:2d}: {gap_counts[gap]:8d} ({pct:5.2f}%)")

print()

# ==============================================================================
# VISUALIZAÇÃO
# ==============================================================================

fig = plt.figure(figsize=(16, 10))

# Plot 1: Distribuição de primos
ax1 = plt.subplot(3, 3, 1)
counts, bins, _ = ax1.hist(primos, bins=100, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Valor do primo')
ax1.set_ylabel('Frequência')
ax1.set_title('Distribuição de Primos')
ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax1.grid(True, alpha=0.3)

# Plot 2: Densidade local
ax2 = plt.subplot(3, 3, 2)
ax2.plot(posicoes, densidades, 'b-', alpha=0.5, linewidth=0.5)
ax2.set_xlabel('Posição (valor do primo)')
ax2.set_ylabel('Densidade local')
ax2.set_title('Densidade Local de Primos')
ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax2.grid(True, alpha=0.3)

# Plot 3: Densidade normalizada
ax3 = plt.subplot(3, 3, 3)
ax3.plot(dens_norm, 'g-', alpha=0.7, linewidth=0.5)
ax3.set_xlabel('Índice da janela')
ax3.set_ylabel('Densidade normalizada (σ)')
ax3.set_title('Densidade Normalizada')
ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
ax3.grid(True, alpha=0.3)

# Plot 4: Espectro de potência (linear)
ax4 = plt.subplot(3, 3, 4)
ax4.plot(freqs, power, 'b-', alpha=0.5, linewidth=0.5)
if len(picos) > 0:
    ax4.plot(freqs[picos], power[picos], 'ro', markersize=6, label='Picos')
ax4.set_xlabel('Frequência (ciclos/janela)')
ax4.set_ylabel('Potência')
ax4.set_title('Espectro de Potência (linear)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Espectro de potência (log)
ax5 = plt.subplot(3, 3, 5)
ax5.semilogy(freqs, power, 'b-', alpha=0.5, linewidth=0.5)
if len(picos) > 0:
    ax5.semilogy(freqs[picos], power[picos], 'ro', markersize=6, label='Picos')
ax5.set_xlabel('Frequência (ciclos/janela)')
ax5.set_ylabel('Potência (log)')
ax5.set_title('Espectro de Potência (log)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Distribuição de gaps
ax6 = plt.subplot(3, 3, 6)
ax6.hist(all_gaps[all_gaps <= 50], bins=50, alpha=0.7, edgecolor='black')
ax6.set_xlabel('Gap entre primos consecutivos')
ax6.set_ylabel('Frequência')
ax6.set_title('Distribuição de Gaps (≤50)')
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3)

# Plot 7: Gap médio ao longo do range
ax7 = plt.subplot(3, 3, 7)
ax7.plot(posicoes, gaps_medios, 'm-', alpha=0.7, linewidth=0.5)
ax7.set_xlabel('Posição')
ax7.set_ylabel('Gap médio')
ax7.set_title('Evolução do Gap Médio')
ax7.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax7.grid(True, alpha=0.3)

# Plot 8: Autocorrelação
ax8 = plt.subplot(3, 3, 8)
autocorr = np.correlate(dens_norm, dens_norm, mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr = autocorr / autocorr[0]
lags = np.arange(len(autocorr))
ax8.plot(lags[:min(200, len(lags))], autocorr[:min(200, len(lags))], 'c-', linewidth=1)
ax8.set_xlabel('Lag')
ax8.set_ylabel('Autocorrelação')
ax8.set_title('Autocorrelação de Densidade')
ax8.axhline(0, color='k', linestyle='--', alpha=0.3)
ax8.grid(True, alpha=0.3)

# Plot 9: Scatter densidade vs posição
ax9 = plt.subplot(3, 3, 9)
ax9.scatter(posicoes, densidades, alpha=0.3, s=1)
ax9.set_xlabel('Posição')
ax9.set_ylabel('Densidade')
ax9.set_title('Densidade vs Posição')
ax9.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analise_rapida_primos.png', dpi=150, bbox_inches='tight')
print("✓ Gráfico salvo: analise_rapida_primos.png")

# ==============================================================================
# RESUMO
# ==============================================================================

print("\n" + "=" * 80)
print("RESUMO DA ANÁLISE")
print("=" * 80)
print()
print(f"Primos analisados: {len(primos):,}")
print(f"Range: {primos.min():.6e} → {primos.max():.6e}")
print(f"Gap médio: {np.mean(all_gaps):.4f} ± {np.std(all_gaps):.4f}")
print(f"Densidade média: {np.mean(densidades):.6e}")
print(f"Variação de densidade: {100*np.std(densidades)/np.mean(densidades):.2f}%")
print()

if len(picos) > 0:
    print(f"Periodicidades detectadas: {len(picos)}")
    print("Principal:")
    idx_max = np.argmax(power[picos])
    f_principal = freqs[picos[idx_max]]
    periodo_principal = 1.0 / f_principal
    print(f"  Frequência: {f_principal:.6f} ciclos/janela")
    print(f"  Período: {periodo_principal:.2f} janelas")
    print(f"  Período em números: ~{periodo_principal * WINDOW_SIZE / 10:.0f} primos")
else:
    print("Nenhuma periodicidade significativa detectada")

print()
print("=" * 80)
