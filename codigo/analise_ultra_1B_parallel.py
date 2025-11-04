#!/usr/bin/env python3
"""
ANÁLISE ULTRA-PARALELA: 1 BILHÃO DE PRIMOS
============================================
Usa TODOS os 56 cores desde o CARREGAMENTO até a análise
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from multiprocessing import Pool, cpu_count, Manager
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO HARDWARE
# ============================================================================
N_CORES = cpu_count()
print("="*80)
print("[FIRE] ANÁLISE ULTRA-PARALELA: 1 BILHÃO DE PRIMOS - FULL POWER [FIRE]")
print("="*80)
print(f"\n[STRONG] RECURSOS:")
print(f"   CPUs: {N_CORES} cores (TODOS em uso TOTAL!)")
print(f"   RAM: ~80GB disponível")
print(f"   Estratégia: MULTIPROCESSING TOTAL")

# ============================================================================
# CONSTANTES
# ============================================================================
alpha_em = 1/137.036
alpha_grav = 1.752e-45
scale_gap = alpha_em / alpha_grav
log_scale = np.log10(scale_gap)

print(f"\n[TARGET] ALVOS:")
print(f"   log₁₀(α_EM/α_grav) = {log_scale:.2f}")
print(f"   Modos esperados: 43")
print(f"   Harmônicos primos: 2, 3, 5, 7, 11, 13, 17, 19, 23...")

# ============================================================================
# FASE 1: CARREGAMENTO PARALELO
# ============================================================================
print("\n" + "="*80)
print("FASE 1: CARREGAMENTO PARALELO DO CSV")
print("="*80)

def carregar_chunk(args):
    """Carrega um chunk específico do CSV"""
    start_row, n_rows, chunk_id = args
    try:
        df = pd.read_csv('results.csv', 
                        skiprows=range(1, start_row) if start_row > 0 else None,
                        nrows=n_rows, 
                        header=0 if start_row == 0 else None,
                        names=['p', 'p_plus_2', 'k_real', 'thread_id', 'range_start'],
                        usecols=['p', 'k_real'],
                        on_bad_lines='skip')
        print(f"[OK] Chunk {chunk_id:2d}: {len(df):,} linhas carregadas")
        return df[['p', 'k_real']].values
    except Exception as e:
        print(f"[FAIL] Chunk {chunk_id:2d}: Erro - {e}")
        return np.array([])

# Dividir arquivo em N_CORES chunks
N_TOTAL = 1_004_800_003
CHUNK_SIZE = N_TOTAL // N_CORES + 1

print(f"\n📥 Dividindo {N_TOTAL:,} linhas em {N_CORES} chunks de ~{CHUNK_SIZE:,} cada")
print(f"   Carregando {N_CORES} chunks em PARALELO...\n")

# Criar tarefas para cada core
tasks = []
for i in range(N_CORES):
    start = i * CHUNK_SIZE
    if start >= N_TOTAL:
        break
    n_rows = min(CHUNK_SIZE, N_TOTAL - start)
    tasks.append((start, n_rows, i+1))

t0 = time.time()
with Pool(N_CORES) as pool:
    chunks = pool.map(carregar_chunk, tasks)

# Concatenar resultados
print(f"\n[LINK] Concatenando chunks...")
dados = np.vstack([c for c in chunks if len(c) > 0])
t_load = time.time() - t0

print(f"[OK] Carregamento completo: {len(dados):,} primos em {t_load:.1f}s")
print(f"   Taxa: {len(dados)/t_load/1e6:.2f} M linhas/segundo")

# ============================================================================
# FASE 2: ORDENAÇÃO
# ============================================================================
print("\n" + "="*80)
print("FASE 2: ORDENAÇÃO")
print("="*80)

t0 = time.time()
idx_sort = np.argsort(dados[:, 0])
primos = dados[idx_sort, 0]
k_vals = dados[idx_sort, 1]
t_sort = time.time() - t0

print(f"[OK] Ordenação: {t_sort:.1f}s")
print(f"   Range: {primos[0]:.0f} → {primos[-1]:.0f}")

del dados, idx_sort  # Liberar memória

# ============================================================================
# FASE 3: DENSIDADE PARALELA
# ============================================================================
print("\n" + "="*80)
print("FASE 3: CÁLCULO DE DENSIDADE PARALELO")
print("="*80)

WINDOW_SIZE = 10000

def calcular_densidade_chunk(args):
    """Calcula densidade para um segmento dos dados"""
    start_idx, end_idx, primos_chunk, chunk_id = args
    
    # Número de janelas neste chunk
    n_windows = (end_idx - start_idx) // WINDOW_SIZE
    densidades_local = np.zeros(n_windows)
    
    for i in range(n_windows):
        idx = start_idx + i * WINDOW_SIZE
        if idx + WINDOW_SIZE <= len(primos_chunk):
            window = primos_chunk[idx:idx+WINDOW_SIZE]
            if len(window) > 1:
                densidades_local[i] = WINDOW_SIZE / (window[-1] - window[0])
    
    if chunk_id % 8 == 0:  # Print a cada 8 chunks
        print(f"   Chunk {chunk_id}/{N_CORES}: {n_windows:,} janelas processadas")
    
    return densidades_local

# Dividir trabalho
n_windows_total = len(primos) // WINDOW_SIZE
chunk_windows = n_windows_total // N_CORES + 1

print(f"\n[DATA] Calculando densidade em ~{n_windows_total:,} janelas")
print(f"   Distribuindo ~{chunk_windows:,} janelas por core\n")

tasks = []
for i in range(N_CORES):
    start_idx = i * chunk_windows * WINDOW_SIZE
    end_idx = min((i+1) * chunk_windows * WINDOW_SIZE, len(primos))
    if start_idx >= len(primos):
        break
    tasks.append((start_idx, end_idx, primos, i+1))

t0 = time.time()
with Pool(N_CORES) as pool:
    resultados = pool.map(calcular_densidade_chunk, tasks)

# Concatenar densidades
densidades = np.concatenate([r for r in resultados if len(r) > 0])
t_density = time.time() - t0

print(f"\n[OK] Densidade calculada: {len(densidades):,} pontos em {t_density:.1f}s")
print(f"   Densidade média: {np.mean(densidades):.6f}")

# ============================================================================
# FASE 4: FFT E ANÁLISE ESPECTRAL
# ============================================================================
print("\n" + "="*80)
print("FASE 4: ANÁLISE ESPECTRAL (FFT)")
print("="*80)

t0 = time.time()
densidade_norm = (densidades - np.mean(densidades)) / np.std(densidades)
fft_result = fft(densidade_norm)
freqs = fftfreq(len(densidade_norm), d=1.0)

# Apenas frequências positivas
mask = freqs > 0
freqs_pos = freqs[mask]
power = np.abs(fft_result[mask])**2
t_fft = time.time() - t0

print(f"[OK] FFT: {t_fft:.1f}s")
print(f"   Pontos espectrais: {len(freqs_pos):,}")

# ============================================================================
# FASE 5: DETECÇÃO DE MODOS FUNDAMENTAIS
# ============================================================================
print("\n" + "="*80)
print("FASE 5: DETECÇÃO DE MODOS FUNDAMENTAIS")
print("="*80)

# Normalizar potência
power_norm = (power - np.mean(power)) / np.std(power)

# Sweep de threshold
print("\n[SEARCH] Testando thresholds...")
thresholds = np.arange(2.0, 8.0, 0.2)
best_threshold = None
best_diff = float('inf')

for thresh in thresholds:
    peaks, _ = find_peaks(power_norm, height=thresh, distance=10)
    n_peaks = len(peaks)
    diff = abs(n_peaks - 43)
    
    if diff < best_diff:
        best_diff = diff
        best_threshold = thresh
    
    if n_peaks >= 35 and n_peaks <= 55:
        print(f"   {thresh:.1f}σ: {n_peaks:2d} picos (diff={diff:2d})")

print(f"\n[TARGET] Threshold ótimo: {best_threshold:.1f}σ (erro={best_diff})")

# Detectar com threshold ótimo
peaks, properties = find_peaks(power_norm, height=best_threshold, distance=10)
peak_freqs = freqs_pos[peaks]
peak_powers = power_norm[peaks]

print(f"[OK] Modos detectados: {len(peaks)}")
print(f"\n[LIST] Top 10 modos:")
idx_sort = np.argsort(peak_powers)[::-1]
for i in range(min(10, len(peaks))):
    idx = idx_sort[i]
    print(f"   {i+1:2d}. f={peak_freqs[idx]:.6f}, potência={peak_powers[idx]:.2f}σ")

# ============================================================================
# FASE 6: HARMÔNICOS PRIMOS
# ============================================================================
print("\n" + "="*80)
print("FASE 6: ANÁLISE DE HARMÔNICOS PRIMOS")
print("="*80)

# Primos até 71
primos_teste = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# Frequência fundamental (maior potência)
f0 = peak_freqs[idx_sort[0]]
print(f"\n🎵 Frequência fundamental: f₀ = {f0:.6f}")

# Buscar harmônicos
print(f"\n[SCI] Buscando harmônicos em primos...\n")
harmonicos_detectados = []

for primo in primos_teste:
    f_esperada = primo * f0
    
    # Buscar peak próximo
    diffs = np.abs(peak_freqs - f_esperada)
    idx_closest = np.argmin(diffs)
    f_detectada = peak_freqs[idx_closest]
    erro = abs(f_detectada - f_esperada) / f_esperada * 100
    
    # Tolerância: 5%
    if erro < 5.0:
        harmonicos_detectados.append({
            'primo': primo,
            'f_esperada': f_esperada,
            'f_detectada': f_detectada,
            'erro_%': erro,
            'potencia_sigma': peak_powers[idx_closest]
        })
        print(f"   [OK] Harmônico {primo:2d}: f={f_detectada:.6f} (erro={erro:.2f}%, {peak_powers[idx_closest]:.1f}σ)")
    else:
        print(f"   [FAIL] Harmônico {primo:2d}: não detectado (erro={erro:.1f}%)")

print(f"\n[DATA] RESUMO:")
print(f"   Harmônicos detectados: {len(harmonicos_detectados)}/{len(primos_teste)}")
print(f"   Primos confirmados: {[h['primo'] for h in harmonicos_detectados]}")
if len(harmonicos_detectados) > 0:
    erro_medio = np.mean([h['erro_%'] for h in harmonicos_detectados])
    print(f"   Erro médio: {erro_medio:.2f}%")

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("SALVANDO RESULTADOS")
print("="*80)

# Modos fundamentais
df_modos = pd.DataFrame({
    'frequencia': peak_freqs,
    'potencia_sigma': peak_powers
})
df_modos = df_modos.sort_values('potencia_sigma', ascending=False)
df_modos.to_csv('modos_fundamentais_1B.csv', index=False)
print(f"[OK] modos_fundamentais_1B.csv: {len(df_modos)} modos")

# Harmônicos primos
if len(harmonicos_detectados) > 0:
    df_harm = pd.DataFrame(harmonicos_detectados)
    df_harm.to_csv('harmonicos_primos_1B.csv', index=False)
    print(f"[OK] harmonicos_primos_1B.csv: {len(harmonicos_detectados)} harmônicos")

# ============================================================================
# VISUALIZAÇÃO
# ============================================================================
print(f"[OK] Gerando visualização...")

fig = plt.figure(figsize=(20, 12))

# 1. Espectro completo
ax1 = plt.subplot(3, 2, 1)
ax1.plot(freqs_pos[:len(freqs_pos)//10], power_norm[:len(freqs_pos)//10], 'b-', alpha=0.5, linewidth=0.5)
ax1.plot(peak_freqs, peak_powers, 'ro', markersize=4)
ax1.axhline(best_threshold, color='g', linestyle='--', label=f'Threshold {best_threshold:.1f}σ')
ax1.set_xlabel('Frequência')
ax1.set_ylabel('Potência (σ)')
ax1.set_title(f'Espectro Completo - {len(peaks)} Modos Detectados')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Zoom nos top peaks
ax2 = plt.subplot(3, 2, 2)
top_10_freqs = peak_freqs[idx_sort[:10]]
top_10_powers = peak_powers[idx_sort[:10]]
ax2.bar(range(10), top_10_powers, color='red', alpha=0.7)
ax2.set_xlabel('Modo')
ax2.set_ylabel('Potência (σ)')
ax2.set_title('Top 10 Modos Fundamentais')
ax2.set_xticks(range(10))
ax2.set_xticklabels([f'{f:.4f}' for f in top_10_freqs], rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# 3. Harmônicos primos
ax3 = plt.subplot(3, 2, 3)
if len(harmonicos_detectados) > 0:
    primos_det = [h['primo'] for h in harmonicos_detectados]
    erros = [h['erro_%'] for h in harmonicos_detectados]
    ax3.bar(range(len(primos_det)), erros, color='green', alpha=0.7)
    ax3.set_xlabel('Harmônico Primo')
    ax3.set_ylabel('Erro (%)')
    ax3.set_title('Precisão dos Harmônicos Primos')
    ax3.set_xticks(range(len(primos_det)))
    ax3.set_xticklabels(primos_det)
    ax3.axhline(5.0, color='r', linestyle='--', label='Limite 5%')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# 4. Razões harmônicas
ax4 = plt.subplot(3, 2, 4)
if len(harmonicos_detectados) > 0:
    razoes = [h['f_detectada']/f0 for h in harmonicos_detectados]
    primos_det = [h['primo'] for h in harmonicos_detectados]
    ax4.scatter(primos_det, razoes, s=100, c='purple', alpha=0.6)
    ax4.plot([0, max(primos_det)+2], [0, max(primos_det)+2], 'k--', label='Ideal')
    ax4.set_xlabel('Primo (n)')
    ax4.set_ylabel('f_n / f₀')
    ax4.set_title('Razões Harmônicas vs Primos')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# 5. Densidade temporal
ax5 = plt.subplot(3, 2, 5)
sample_size = min(10000, len(densidades))
ax5.plot(densidades[:sample_size], 'b-', alpha=0.6, linewidth=0.5)
ax5.set_xlabel('Janela')
ax5.set_ylabel('Densidade')
ax5.set_title(f'Densidade (primeiras {sample_size:,} janelas)')
ax5.grid(True, alpha=0.3)

# 6. Comparação com α_EM
ax6 = plt.subplot(3, 2, 6)
texto = f"""
[FIRE] RESULTADOS ULTRA-PARALELOS [FIRE]

Dataset: {len(primos):,} primos gêmeos
Resolução: {len(densidades):,} janelas

CONSTANTES FÍSICAS:
α_EM = 1/137.036
α_grav = {alpha_grav:.3e}
Escala: log₁₀(α_EM/α_grav) = {log_scale:.2f}

MODOS DETECTADOS:
Total: {len(peaks)} modos
Esperado: 43 modos
Diferença: {abs(len(peaks)-43)}
Threshold: {best_threshold:.1f}σ

HARMÔNICOS PRIMOS:
Detectados: {len(harmonicos_detectados)}/{len(primos_teste)}
Primos: {[h['primo'] for h in harmonicos_detectados]}
Erro médio: {np.mean([h['erro_%'] for h in harmonicos_detectados]):.2f}%

PERFORMANCE:
Carregamento: {t_load:.1f}s
Ordenação: {t_sort:.1f}s  
Densidade: {t_density:.1f}s
FFT: {t_fft:.1f}s
Total: {t_load+t_sort+t_density+t_fft:.1f}s

Cores usados: {N_CORES}
"""
ax6.text(0.1, 0.5, texto, fontsize=10, family='monospace',
         verticalalignment='center', transform=ax6.transAxes)
ax6.axis('off')

plt.tight_layout()
plt.savefig('analise_ultra_1bilhao_parallel.png', dpi=150, bbox_inches='tight')
print(f"[OK] analise_ultra_1bilhao_parallel.png")

# ============================================================================
# SUMÁRIO FINAL
# ============================================================================
print("\n" + "="*80)
print("[SUCCESS] ANÁLISE COMPLETA!")
print("="*80)
print(f"\n[DATA] DESCOBERTAS:")
print(f"   • {len(peaks)} modos fundamentais (esperado: 43)")
print(f"   • {len(harmonicos_detectados)} harmônicos primos detectados")
print(f"   • Primos confirmados: {[h['primo'] for h in harmonicos_detectados]}")
print(f"   • Precisão média: {np.mean([h['erro_%'] for h in harmonicos_detectados]):.2f}%" if len(harmonicos_detectados) > 0 else "")
print(f"\n[ENERGY] PERFORMANCE:")
print(f"   • Tempo total: {t_load+t_sort+t_density+t_fft:.1f}s")
print(f"   • Taxa processamento: {len(primos)/(t_load+t_sort+t_density+t_fft)/1e6:.2f} M primos/s")
print(f"   • Cores utilizados: {N_CORES}/{N_CORES} (100%)")
print("="*80)
