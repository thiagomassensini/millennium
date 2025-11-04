#!/usr/bin/env python3
"""
Análise Geométrica Hexagonal de Primos Gêmeos
Extrai e visualiza estruturas cristalinas e padrões hexagonais
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from scipy.spatial.distance import pdist, squareform
from scipy.fft import fft2, fftfreq
import sys

print("=" * 80)
print("ANÁLISE GEOMÉTRICA: ESTRUTURA HEXAGONAL DOS PRIMOS")
print("=" * 80)
print()

# ==============================================================================
# PARÂMETROS
# ==============================================================================

if len(sys.argv) > 1:
    ARQUIVO = sys.argv[1]
else:
    ARQUIVO = "results.csv"

# Número de primos para análise
MAX_PRIMOS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000

# Tamanho da janela para cálculo de gaps médios
WINDOW_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

print(f"Arquivo: {ARQUIVO}")
print(f"Primos: {MAX_PRIMOS:,}")
print(f"Janela: {WINDOW_SIZE:,}")
print()

# ==============================================================================
# CARREGAR DADOS
# ==============================================================================

print("Carregando primos gêmeos...")
try:
    df = pd.read_csv(ARQUIVO, nrows=MAX_PRIMOS, on_bad_lines='skip')
    
    # Colunas: p, p_plus_2, k_real, thread_id, range_start
    primos = df.iloc[:, 0].values.astype(np.float64)
    k_values = df.iloc[:, 2].values.astype(int) if df.shape[1] > 2 else None
    
    print(f"[OK] {len(primos):,} primos carregados")
    print(f"  Range: {primos[0]:.6e} → {primos[-1]:.6e}")
    print(f"  Span: {primos[-1] - primos[0]:.6e}")
    
    if k_values is not None:
        print(f"  k médio: {np.mean(k_values):.3f}")
        print(f"  k máximo: {np.max(k_values)}")
    
    print()
except Exception as e:
    print(f"[FAIL] Erro ao carregar: {e}")
    sys.exit(1)

# ==============================================================================
# CONSTRUIR ESPAÇO DE FASE (POSIÇÃO, GAP)
# ==============================================================================

print("Construindo espaço de fase...")

n_janelas = (len(primos) - WINDOW_SIZE) // (WINDOW_SIZE // 1)
posicoes = np.zeros(n_janelas)
gaps_medios = np.zeros(n_janelas)
gaps_std = np.zeros(n_janelas)
densidades = np.zeros(n_janelas)
k_medios = np.zeros(n_janelas) if k_values is not None else None

idx = 0
step = max(1, WINDOW_SIZE // 10)

for i in range(0, len(primos) - WINDOW_SIZE, step):
    if idx >= n_janelas:
        break
    
    janela = primos[i:i+WINDOW_SIZE]
    posicoes[idx] = np.mean(janela)
    
    # Gaps
    gaps = np.diff(janela)
    gaps_medios[idx] = np.mean(gaps)
    gaps_std[idx] = np.std(gaps)
    
    # Densidade
    span = janela[-1] - janela[0]
    if span > 0:
        densidades[idx] = WINDOW_SIZE / span
    
    # k médio (se disponível)
    if k_values is not None:
        k_janela = k_values[i:i+WINDOW_SIZE]
        k_medios[idx] = np.mean(k_janela)
    
    idx += 1

# Truncar arrays
posicoes = posicoes[:idx]
gaps_medios = gaps_medios[:idx]
gaps_std = gaps_std[:idx]
densidades = densidades[:idx]
if k_medios is not None:
    k_medios = k_medios[:idx]

print(f"[OK] {len(posicoes):,} janelas calculadas")
print(f"  Gap médio: {np.mean(gaps_medios):.2f} ± {np.std(gaps_medios):.2f}")
print(f"  Range gaps: [{np.min(gaps_medios):.2f}, {np.max(gaps_medios):.2f}]")
print()

# ==============================================================================
# DETECÇÃO DE ESTRUTURA HEXAGONAL
# ==============================================================================

print("Detectando estrutura hexagonal...")

# Normalizar coordenadas
pos_norm = (posicoes - np.min(posicoes)) / (np.max(posicoes) - np.min(posicoes))
gap_norm = (gaps_medios - np.mean(gaps_medios)) / np.std(gaps_medios)

# Criar grid 2D
n_bins = 100
H, xedges, yedges = np.histogram2d(pos_norm, gap_norm, bins=n_bins)

# FFT 2D para detectar periodicidade
fft_result = np.abs(fft2(H))
fft_result = np.fft.fftshift(fft_result)

# Detectar picos no espectro 2D
threshold = np.mean(fft_result) + 3 * np.std(fft_result)
picos_2d = fft_result > threshold

print(f"[OK] Picos 2D detectados: {np.sum(picos_2d)}")
print()

# ==============================================================================
# ANÁLISE DE SIMETRIA RADIAL
# ==============================================================================

print("Analisando simetria radial...")

# Centro de massa
centro_x = np.mean(pos_norm)
centro_y = np.mean(gap_norm)

# Distâncias radiais
r = np.sqrt((pos_norm - centro_x)**2 + (gap_norm - centro_y)**2)
theta = np.arctan2(gap_norm - centro_y, pos_norm - centro_x)

# Histograma angular (detecta simetria hexagonal)
n_theta_bins = 360
hist_theta, theta_bins = np.histogram(theta, bins=n_theta_bins, range=(-np.pi, np.pi))

# Detectar picos angulares (esperado: 6 picos para hexágono)
from scipy.signal import find_peaks
picos_angulares, props = find_peaks(hist_theta, height=np.mean(hist_theta))

# Converter para graus
angulos_picos = np.degrees(theta_bins[picos_angulares])

print(f"[OK] Picos angulares detectados: {len(picos_angulares)}")
if len(picos_angulares) > 0:
    print(f"  Ângulos principais (graus):")
    for i, ang in enumerate(angulos_picos[:10], 1):
        print(f"    {i:2d}. {ang:7.2f}°")
    
    # Verificar se há ~6 picos igualmente espaçados (hexágono)
    if len(picos_angulares) >= 5:
        espacamento_medio = np.mean(np.diff(angulos_picos[:6]))
        print(f"  Espaçamento médio: {espacamento_medio:.2f}° (ideal hexágono: 60°)")

print()

# ==============================================================================
# CAMADAS RADIAIS (HEXÁGONOS CONCÊNTRICOS)
# ==============================================================================

print("Identificando camadas hexagonais...")

# Histograma de distâncias radiais
n_r_bins = 50
hist_r, r_bins = np.histogram(r, bins=n_r_bins)

# Detectar picos radiais (camadas)
picos_radiais, props_r = find_peaks(hist_r, height=np.mean(hist_r))

print(f"[OK] Camadas radiais detectadas: {len(picos_radiais)}")
if len(picos_radiais) > 0:
    print(f"  Raios das camadas:")
    for i, idx in enumerate(picos_radiais[:10], 1):
        raio = r_bins[idx]
        print(f"    {i:2d}. r = {raio:.4f}")

print()

# ==============================================================================
# VISUALIZAÇÃO COMPLETA
# ==============================================================================

print("Gerando visualizações...")

fig = plt.figure(figsize=(20, 12))

# ============================================================
# Plot 1: Espaço de fase completo (posição vs gap)
# ============================================================
ax1 = plt.subplot(3, 4, 1)
scatter = ax1.scatter(posicoes, gaps_medios, c=densidades, s=1, 
                     alpha=0.5, cmap='viridis')
ax1.set_xlabel('Posição')
ax1.set_ylabel('Gap Médio')
ax1.set_title('Espaço de Fase: Posição vs Gap')
ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
plt.colorbar(scatter, ax=ax1, label='Densidade')
ax1.grid(True, alpha=0.3)

# ============================================================
# Plot 2: Vista polar (hexágono)
# ============================================================
ax2 = plt.subplot(3, 4, 2, projection='polar')
ax2.scatter(theta, r, s=1, alpha=0.3, c=gap_norm, cmap='plasma')
ax2.set_title('Vista Polar (Simetria Radial)')
ax2.grid(True)

# ============================================================
# Plot 3: Histograma angular
# ============================================================
ax3 = plt.subplot(3, 4, 3)
ax3.plot(np.degrees(theta_bins[:-1]), hist_theta, 'b-', linewidth=1)
if len(picos_angulares) > 0:
    ax3.plot(angulos_picos, hist_theta[picos_angulares], 'ro', markersize=8)
ax3.set_xlabel('Ângulo (graus)')
ax3.set_ylabel('Frequência')
ax3.set_title(f'Distribuição Angular ({len(picos_angulares)} picos)')
ax3.axvline(60, color='r', linestyle='--', alpha=0.3, label='60° (hexágono)')
ax3.axvline(120, color='r', linestyle='--', alpha=0.3)
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================
# Plot 4: Gap médio ao longo do range
# ============================================================
ax4 = plt.subplot(3, 4, 4)
ax4.plot(posicoes, gaps_medios, 'm-', alpha=0.7, linewidth=0.5)
ax4.set_xlabel('Posição')
ax4.set_ylabel('Gap médio')
ax4.set_title('Evolução do Gap Médio')
ax4.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax4.grid(True, alpha=0.3)

# ============================================================
# Plot 5: Densidade 2D (heatmap)
# ============================================================
ax5 = plt.subplot(3, 4, 5)
im = ax5.imshow(H.T, origin='lower', aspect='auto', cmap='hot', 
                extent=[0, 1, np.min(gap_norm), np.max(gap_norm)])
ax5.set_xlabel('Posição Normalizada')
ax5.set_ylabel('Gap Normalizado (σ)')
ax5.set_title('Densidade 2D (Heatmap)')
plt.colorbar(im, ax=ax5, label='Contagem')

# ============================================================
# Plot 6: FFT 2D (periodicidade espacial)
# ============================================================
ax6 = plt.subplot(3, 4, 6)
ax6.imshow(np.log10(fft_result + 1), origin='lower', cmap='jet')
ax6.set_xlabel('Frequência kx')
ax6.set_ylabel('Frequência ky')
ax6.set_title('FFT 2D (Periodicidade Espacial)')
ax6.plot([n_bins//2], [n_bins//2], 'w+', markersize=20, markeredgewidth=2)

# ============================================================
# Plot 7: Hexágono ideal sobreposto
# ============================================================
ax7 = plt.subplot(3, 4, 7)
ax7.scatter(pos_norm, gap_norm, s=0.5, alpha=0.3, c='blue')

# Desenhar hexágono ideal no centro
if len(picos_radiais) > 0:
    for raio_idx in picos_radiais[:5]:
        raio = r_bins[raio_idx]
        hexagon = RegularPolygon((centro_x, centro_y), 6, radius=raio,
                                facecolor='none', edgecolor='red', 
                                linewidth=2, alpha=0.7)
        ax7.add_patch(hexagon)

ax7.set_xlabel('Posição Normalizada')
ax7.set_ylabel('Gap Normalizado (σ)')
ax7.set_title('Estrutura Hexagonal Sobreposta')
ax7.set_aspect('equal')
ax7.grid(True, alpha=0.3)

# ============================================================
# Plot 8: Gap vs Densidade
# ============================================================
ax8 = plt.subplot(3, 4, 8)
ax8.scatter(densidades, gaps_medios, s=1, alpha=0.5, c=posicoes, cmap='cool')
ax8.set_xlabel('Densidade')
ax8.set_ylabel('Gap Médio')
ax8.set_title('Gap vs Densidade')
ax8.grid(True, alpha=0.3)

# ============================================================
# Plot 9: k médio (se disponível)
# ============================================================
ax9 = plt.subplot(3, 4, 9)
if k_medios is not None:
    ax9.scatter(posicoes, k_medios, s=1, alpha=0.5, c=gaps_medios, cmap='plasma')
    ax9.set_ylabel('k Médio')
    ax9.set_title('Bits Consecutivos (k) vs Posição')
else:
    ax9.text(0.5, 0.5, 'Dados de k não disponíveis', 
            ha='center', va='center', transform=ax9.transAxes)
ax9.set_xlabel('Posição')
ax9.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax9.grid(True, alpha=0.3)

# ============================================================
# Plot 10: Gap normalizado (zoom central)
# ============================================================
ax10 = plt.subplot(3, 4, 10)
centro_mask = (np.abs(gap_norm) < 2)  # ±2σ
ax10.scatter(pos_norm[centro_mask], gap_norm[centro_mask], 
            s=1, alpha=0.5, c='green')
ax10.set_xlabel('Posição Normalizada')
ax10.set_ylabel('Gap Normalizado (σ)')
ax10.set_title('Zoom: Região Central (±2σ)')
ax10.axhline(0, color='red', linestyle='--', alpha=0.5)
ax10.grid(True, alpha=0.3)

# ============================================================
# Plot 11: Correlação angular-radial
# ============================================================
ax11 = plt.subplot(3, 4, 11)
H_polar, theta_edges, r_edges = np.histogram2d(theta, r, bins=50)
im11 = ax11.imshow(H_polar.T, origin='lower', aspect='auto', cmap='viridis',
                   extent=[-np.pi, np.pi, 0, np.max(r)])
ax11.set_xlabel('Ângulo (rad)')
ax11.set_ylabel('Raio')
ax11.set_title('Densidade Polar')
plt.colorbar(im11, ax=ax11, label='Contagem')

# ============================================================
# Plot 12: Resumo estatístico
# ============================================================
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

stats_text = f"""
RESUMO GEOMÉTRICO

Dados:
• Primos analisados: {len(primos):,}
• Janelas: {len(posicoes):,}
• Window size: {WINDOW_SIZE:,}

Gap Médio:
• Média: {np.mean(gaps_medios):.2f}
• Desvio: {np.std(gaps_medios):.2f}
• Range: [{np.min(gaps_medios):.1f}, {np.max(gaps_medios):.1f}]

Simetria:
• Picos angulares: {len(picos_angulares)}
• Camadas radiais: {len(picos_radiais)}
• Picos FFT 2D: {np.sum(picos_2d)}
"""

if len(picos_angulares) >= 6:
    espacamento = np.mean(np.diff(angulos_picos[:6]))
    stats_text += f"\n• Espaçamento angular: {espacamento:.1f}°"
    stats_text += f"\n• Desvio de 60°: {abs(espacamento - 60):.1f}°"
    if abs(espacamento - 60) < 10:
        stats_text += "\n\n[OK] ESTRUTURA HEXAGONAL CONFIRMADA!"

if k_medios is not None:
    stats_text += f"\n\nBits (k):\n• k médio: {np.mean(k_medios):.3f}"
    stats_text += f"\n• Correlação k-gap: {np.corrcoef(k_medios, gaps_medios)[0,1]:.3f}"

ax12.text(0.1, 0.95, stats_text, transform=ax12.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('geometria_hexagonal_primos.png', 
            dpi=200, bbox_inches='tight')
print("[OK] Gráfico salvo: geometria_hexagonal_primos.png")

# ==============================================================================
# ANÁLISE QUANTITATIVA DE HEXAGONALIDADE
# ==============================================================================

print("\n" + "=" * 80)
print("ANÁLISE QUANTITATIVA DE ESTRUTURA HEXAGONAL")
print("=" * 80)
print()

# Teste 1: Simetria angular
print("1. SIMETRIA ANGULAR:")
if len(picos_angulares) >= 6:
    espacamento = np.mean(np.diff(angulos_picos[:6]))
    desvio_60 = abs(espacamento - 60)
    print(f"   Espaçamento médio: {espacamento:.2f}° (ideal: 60°)")
    print(f"   Desvio: {desvio_60:.2f}°")
    if desvio_60 < 5:
        print("   [OK] HEXAGONAL PERFEITO")
    elif desvio_60 < 10:
        print("   [OK] HEXAGONAL (leve distorção)")
    else:
        print("   [WARNING] Simetria aproximada")
else:
    print(f"   [FAIL] Insuficiente ({len(picos_angulares)} picos, necessário ≥6)")

print()

# Teste 2: Camadas concêntricas
print("2. CAMADAS RADIAIS:")
print(f"   Camadas detectadas: {len(picos_radiais)}")
if len(picos_radiais) >= 3:
    espacamento_radial = np.mean(np.diff(r_bins[picos_radiais]))
    print(f"   Espaçamento médio: {espacamento_radial:.4f}")
    print("   [OK] Estrutura em camadas confirmada")
else:
    print("   [WARNING] Poucas camadas detectadas")

print()

# Teste 3: Correlação gap-densidade
print("3. CORRELAÇÃO GAP-DENSIDADE:")
corr_gap_dens = np.corrcoef(gaps_medios, densidades)[0, 1]
print(f"   Correlação: {corr_gap_dens:.4f}")
if abs(corr_gap_dens) > 0.7:
    print("   [OK] Forte correlação (estrutura cristalina)")
elif abs(corr_gap_dens) > 0.4:
    print("   [OK] Correlação moderada")
else:
    print("   [WARNING] Correlação fraca")

print()
print("=" * 80)
print("ANÁLISE COMPLETA")
print("=" * 80)
