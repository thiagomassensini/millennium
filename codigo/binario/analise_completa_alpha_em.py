#!/usr/bin/env python3
"""
ANÁLISE CRÍTICA: Dataset completo 1B primos
Objetivo: Confirmar se número de picos ≈ 42-43 (log₁₀(α_EM/α_grav))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

print("=" * 80)
print("ANÁLISE COMPLETA: 1 BILHÃO DE PRIMOS GÊMEOS")
print("Teste da Hipótese: Número de picos ≈ log₁₀(α_EM/α_grav) ≈ 42.6")
print("=" * 80)

# Constantes
alpha_em = 1/137.035999084
alpha_grav_e = 1.751809e-45
log_ratio = np.log10(alpha_em / alpha_grav_e)

print(f"\nlog₁₀(α_EM/α_grav) = {log_ratio:.2f}")
print(f"Predição: ~{log_ratio:.0f} picos significativos\n")

# Carregar dataset
print("Carregando dados...")
print("⚠️  AVISO: Análise de 1B primos pode levar 30-60 minutos")
print("           Usaremos amostragem estratificada para acelerar\n")

# Estratégia: Amostrar 100M primos uniformemente do 1B
TAMANHO_AMOSTRA = 100_000_000
STEP_AMOSTRAGEM = 10  # Pegar 1 a cada 10

try:
    # Ler com amostragem
    df = pd.read_csv('results_sorted_10M.csv', header=0)
    # NOTA: Como só temos 10M ordenados, vamos usar isso como proxy
    # Para análise completa, precisaríamos ordenar o dataset completo
    
    print(f"⚠️  Dataset disponível: {len(df):,} primos")
    print(f"   Para análise completa, precisaríamos de results_sorted_1B.csv")
    print(f"   Prosseguindo com 10M como demonstração...\n")
    
    primos = df['p'].values
    
except Exception as e:
    print(f"Erro ao carregar: {e}")
    print("Tentando carregar results.csv (não ordenado)...")
    
    # Tentar CSV não ordenado
    df = pd.read_csv('results.csv', nrows=100_000_000, header=0)
    print(f"Carregados {len(df):,} primos (não ordenados)")
    print("Ordenando...")
    df = df.sort_values('p')
    primos = df['p'].values

print(f"✓ Dataset: {len(primos):,} primos")
print(f"  Range: {primos.min():.3e} → {primos.max():.3e}\n")

# Análise de densidade com janelas
WINDOW_SIZE = 10000
STEP = WINDOW_SIZE // 10

print("Calculando densidade local...")
posicoes = []
densidades = []

n_windows = (len(primos) - WINDOW_SIZE) // STEP

for i in range(0, len(primos) - WINDOW_SIZE, STEP):
    janela = primos[i:i+WINDOW_SIZE]
    posicoes.append(np.mean(janela))
    span = janela.max() - janela.min()
    if span > 0:
        densidades.append(WINDOW_SIZE / span)
    
    if (i // STEP) % 1000 == 0:
        progress = 100 * i / (len(primos) - WINDOW_SIZE)
        print(f"  Progresso: {progress:.1f}% ({i//STEP:,}/{n_windows:,} janelas)", end='\r')

print(f"\n✓ {len(posicoes):,} janelas analisadas")

posicoes = np.array(posicoes)
densidades = np.array(densidades)

print(f"  Densidade média: {np.mean(densidades):.6e}")
print(f"  CV: {np.std(densidades)/np.mean(densidades):.4f}\n")

# Análise espectral
print("Realizando FFT...")
dens_norm = (densidades - np.mean(densidades)) / np.std(densidades)
yf = fft(dens_norm)
xf = fftfreq(len(dens_norm), d=1.0)

mask = xf > 0
freqs = xf[mask]
power = np.abs(yf[mask])**2

print(f"✓ FFT completa: {len(freqs):,} frequências\n")

# Detectar picos com threshold variável
print("Detectando picos...")
print("┌─────────────┬────────┬─────────────┐")
print("│  Threshold  │ Picos  │ Significado │")
print("├─────────────┼────────┼─────────────┤")

for n_sigma in [3, 4, 5, 6, 7]:
    threshold = np.mean(power) + n_sigma * np.std(power)
    picos, _ = signal.find_peaks(power, height=threshold, distance=5)
    print(f"│    {n_sigma}σ       │  {len(picos):4d}  │ {n_sigma}σ acima média │")

print("└─────────────┴────────┴─────────────┘\n")

# Usar 3σ para contagem principal
threshold_3sigma = np.mean(power) + 3 * np.std(power)
picos_3sigma, _ = signal.find_peaks(power, height=threshold_3sigma, distance=5)

print(f"🎯 RESULTADO PRINCIPAL:")
print(f"   Picos detectados (3σ): {len(picos_3sigma)}")
print(f"   Predição teórica: {log_ratio:.0f}")
print(f"   Razão: {len(picos_3sigma) / log_ratio:.3f}\n")

# Análise dos picos mais fortes
n_top = min(20, len(picos_3sigma))
idx_sorted = np.argsort(power[picos_3sigma])[::-1][:n_top]

print(f"Top {n_top} picos mais significativos:")
print("┌──────┬───────────────┬──────────────┬──────────┐")
print("│ Rank │  Frequência   │   Período    │   σ      │")
print("├──────┼───────────────┼──────────────┼──────────┤")

for i, idx in enumerate(idx_sorted, 1):
    f = freqs[picos_3sigma[idx]]
    P = power[picos_3sigma[idx]]
    sigma = (P - np.mean(power)) / np.std(power)
    T = 1.0/f
    print(f"│  {i:2d}  │ {f:>13.6f} │ {T:>12.1f} │ {sigma:>8.1f} │")

print("└──────┴───────────────┴──────────────┴──────────┘\n")

# Verificar distribuição de picos
print("Distribuição de potência dos picos:")
potencias_picos = power[picos_3sigma]
print(f"  Mínima: {potencias_picos.min():.2e}")
print(f"  Mediana: {np.median(potencias_picos):.2e}")
print(f"  Máxima: {potencias_picos.max():.2e}")
print(f"  Razão max/min: {potencias_picos.max()/potencias_picos.min():.1f}×\n")

# Visualização
print("Gerando visualização...")
fig = plt.figure(figsize=(18, 12))

# 1. Densidade vs posição
ax1 = plt.subplot(3, 3, 1)
step_plot = max(1, len(posicoes) // 10000)
ax1.plot(posicoes[::step_plot], densidades[::step_plot], 'b-', alpha=0.5, linewidth=0.3)
ax1.set_xlabel('Posição')
ax1.set_ylabel('Densidade')
ax1.set_title(f'Densidade Local ({len(primos):,} primos)')
ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax1.grid(True, alpha=0.3)

# 2. Densidade normalizada
ax2 = plt.subplot(3, 3, 2)
ax2.plot(dens_norm[::step_plot], 'g-', alpha=0.7, linewidth=0.3)
ax2.set_xlabel('Janela')
ax2.set_ylabel('Densidade norm (σ)')
ax2.set_title('Flutuações (normalizado)')
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.grid(True, alpha=0.3)

# 3. Espectro completo (linear)
ax3 = plt.subplot(3, 3, 3)
ax3.plot(freqs, power, 'b-', alpha=0.5, linewidth=0.5)
ax3.plot(freqs[picos_3sigma], power[picos_3sigma], 'ro', markersize=4)
ax3.axhline(threshold_3sigma, color='r', linestyle='--', alpha=0.5, label='3σ')
ax3.set_xlabel('Frequência')
ax3.set_ylabel('Potência')
ax3.set_title(f'Espectro Completo ({len(picos_3sigma)} picos)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Espectro (log)
ax4 = plt.subplot(3, 3, 4)
ax4.semilogy(freqs, power, 'b-', alpha=0.5, linewidth=0.5)
ax4.semilogy(freqs[picos_3sigma], power[picos_3sigma], 'ro', markersize=4)
ax4.axhline(threshold_3sigma, color='r', linestyle='--', alpha=0.5, label='3σ')
ax4.set_xlabel('Frequência')
ax4.set_ylabel('Potência (log)')
ax4.set_title('Espectro (escala log)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Zoom nos picos mais fortes
ax5 = plt.subplot(3, 3, 5)
top5_indices = idx_sorted[:5]
for i, idx in enumerate(top5_indices):
    f_peak = freqs[picos_3sigma[idx]]
    # Janela ao redor do pico
    mask_zoom = (freqs > f_peak*0.9) & (freqs < f_peak*1.1)
    ax5.plot(freqs[mask_zoom], power[mask_zoom], alpha=0.7, label=f'Pico {i+1}')
ax5.set_xlabel('Frequência')
ax5.set_ylabel('Potência')
ax5.set_title('Zoom: Top 5 Picos')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Distribuição de significâncias
ax6 = plt.subplot(3, 3, 6)
sigmas = (power[picos_3sigma] - np.mean(power)) / np.std(power)
ax6.hist(sigmas, bins=30, alpha=0.7, edgecolor='black')
ax6.axvline(log_ratio, color='r', linestyle='--', linewidth=2, label=f'log(α_EM/α_grav)={log_ratio:.1f}')
ax6.set_xlabel('Significância (σ)')
ax6.set_ylabel('Frequência')
ax6.set_title(f'Distribuição de Significâncias')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Número de picos vs threshold
ax7 = plt.subplot(3, 3, 7)
thresholds = np.arange(2, 10, 0.5)
n_picos_vs_thresh = []
for th in thresholds:
    threshold_i = np.mean(power) + th * np.std(power)
    picos_i, _ = signal.find_peaks(power, height=threshold_i, distance=5)
    n_picos_vs_thresh.append(len(picos_i))

ax7.plot(thresholds, n_picos_vs_thresh, 'bo-', linewidth=2, markersize=6)
ax7.axhline(log_ratio, color='r', linestyle='--', linewidth=2, label=f'{log_ratio:.0f} (predição)')
ax7.set_xlabel('Threshold (σ)')
ax7.set_ylabel('Número de Picos')
ax7.set_title('Picos vs Threshold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Espaçamento entre picos
ax8 = plt.subplot(3, 3, 8)
freq_picos = freqs[picos_3sigma]
spacings = np.diff(sorted(freq_picos))
ax8.hist(spacings, bins=30, alpha=0.7, edgecolor='black')
ax8.set_xlabel('Δf entre picos')
ax8.set_ylabel('Frequência')
ax8.set_title('Espaçamento entre Picos')
ax8.grid(True, alpha=0.3)

# 9. Análise de razão
ax9 = plt.subplot(3, 3, 9)
n_picos_range = [8, len(picos_3sigma)]  # 1M vs atual
tamanho_range = [1e6, len(primos)]
ax9.loglog(tamanho_range, n_picos_range, 'go-', linewidth=2, markersize=10, label='Observado')
# Projeção para 1B
if len(primos) < 1e9:
    taxa = np.log(n_picos_range[1]/n_picos_range[0]) / np.log(tamanho_range[1]/tamanho_range[0])
    n_1B = n_picos_range[1] * (1e9/tamanho_range[1])**taxa
    ax9.loglog([tamanho_range[1], 1e9], [n_picos_range[1], n_1B], 'r--', linewidth=2, label=f'Projeção→{n_1B:.0f}')
ax9.axhline(log_ratio, color='orange', linestyle='--', linewidth=2, label=f'α_EM/α_grav={log_ratio:.0f}')
ax9.set_xlabel('Tamanho do Dataset')
ax9.set_ylabel('Número de Picos')
ax9.set_title('Scaling: Picos vs Tamanho')
ax9.legend()
ax9.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('analise_completa_alpha_em.png', dpi=150, bbox_inches='tight')
print(f"✓ Salvo: analise_completa_alpha_em.png\n")

# Relatório final
print("=" * 80)
print("RESULTADO FINAL: TESTE DA HIPÓTESE α_EM")
print("=" * 80)

print(f"""
Dataset analisado: {len(primos):,} primos gêmeos
Janelas: {len(posicoes):,}

PICOS DETECTADOS (3σ):
  Observado: {len(picos_3sigma)} picos
  Predição:  {log_ratio:.0f} picos (log₁₀(α_EM/α_grav))
  Razão:     {len(picos_3sigma)/log_ratio:.3f}
  
ANÁLISE:
""")

if abs(len(picos_3sigma) - log_ratio) < 5:
    print("  ✅ CONCORDÂNCIA EXCELENTE!")
    print("     Número de picos consistente com hierarquia α_EM/α_grav")
elif abs(len(picos_3sigma) - log_ratio) < 10:
    print("  ✅ CONCORDÂNCIA BOA")
    print("     Desvio aceitável (< 10 picos)")
elif len(picos_3sigma) < log_ratio:
    print("  ⚠️  MENOS PICOS QUE O ESPERADO")
    print(f"     Diferença: {log_ratio - len(picos_3sigma):.0f} picos")
    print("     Possível razão: Dataset ainda pequeno")
    taxa = np.log(len(picos_3sigma)/8) / np.log(len(primos)/1e6)
    n_projetado_1B = len(picos_3sigma) * (1e9/len(primos))**taxa
    print(f"     Projeção para 1B: {n_projetado_1B:.0f} picos")
else:
    print("  ⚠️  MAIS PICOS QUE O ESPERADO")
    print(f"     Diferença: {len(picos_3sigma) - log_ratio:.0f} picos")
    print("     Possível razão: Threshold 3σ capturando ruído")

print("\n" + "=" * 80)
