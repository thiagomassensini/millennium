#!/usr/bin/env python3
"""
ANÁLISE 1B - QUANTUM OPTIMIZED (56 cores, 60GB RAM)
====================================================
Versão ultra-otimizada para detectar harmônicos primos
Foco: Harmônico 137 (α_EM^-1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch
from multiprocessing import Pool, Manager
import time
import gc
import warnings
import psutil
import sys
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO QUANTUM
# ============================================================================
N_CORES = 56
MEMORIA_DISPONIVEL_GB = 55  # Deixar margem para SO
CHUNK_SIZE = 100_000_000  # 100M linhas por chunk (otimizado)

print("=" * 80)
print("[FIRE] ANÁLISE 1 BILHÃO - QUANTUM OPTIMIZED [FIRE]")
print("=" * 80)
print(f"\n[SETTINGS]  CONFIGURAÇÃO:")
print(f"   Cores: {N_CORES}")
print(f"   RAM disponível: {MEMORIA_DISPONIVEL_GB} GB")
print(f"   Chunk size: {CHUNK_SIZE:,}")

# Constantes físicas
ALPHA_EM = 1.0 / 137.035999084
ALPHA_GRAV_ELECTRON = 1.7518e-45
SCALE_GAP = ALPHA_EM / ALPHA_GRAV_ELECTRON
LOG_SCALE = np.log10(SCALE_GAP)

print(f"\n[TARGET] ALVOS TEÓRICOS:")
print(f"   α_EM = 1/{1/ALPHA_EM:.6f}")
print(f"   log₁₀(α_EM/α_grav) = {LOG_SCALE:.2f}")
print(f"   Modos esperados: ~43")
print(f"   Harmônico crítico: 137")

# Primos para harmônicos
PRIMOS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137]

# ============================================================================
# FASE 1: CARREGAMENTO INTELIGENTE
# ============================================================================
print("\n" + "=" * 80)
print("FASE 1: CARREGAMENTO INTELIGENTE")
print("=" * 80)

def carregar_primos_otimizado(arquivo='results.csv', max_linhas=1_004_800_003):
    """Carrega primos com gestão eficiente de memória"""
    
    print(f"\n📥 Iniciando carregamento de {max_linhas:,} linhas...")
    print(f"   Arquivo: {arquivo}")
    
    t0 = time.time()
    chunks = []
    total_carregado = 0
    
    try:
        # Ler em chunks grandes (otimizado para 60GB RAM)
        for i, chunk in enumerate(pd.read_csv(
            arquivo,
            chunksize=CHUNK_SIZE,
            usecols=[0],  # Apenas primeira coluna (primos)
            names=['p'],
            header=None,
            dtype=np.float64,
            engine='c',  # Engine C (mais rápido)
            low_memory=False
        )):
            chunk_primos = chunk['p'].values
            chunks.append(chunk_primos)
            total_carregado += len(chunk_primos)
            
            # Progress report a cada 100M
            if (i + 1) % (100_000_000 // CHUNK_SIZE) == 0:
                mem_uso = psutil.virtual_memory().percent
                print(f"   [OK] {total_carregado:,} linhas ({mem_uso:.1f}% RAM)")
            
            # Parar se atingir limite
            if total_carregado >= max_linhas:
                break
                
    except Exception as e:
        print(f"   [WARNING]  Erro ao carregar: {e}")
        print(f"   Carregado até agora: {total_carregado:,}")
    
    # Concatenar todos os chunks
    print("\n📦 Concatenando chunks...")
    primos = np.concatenate(chunks)
    del chunks
    gc.collect()
    
    t_load = time.time() - t0
    
    print(f"\n[OK] Carregamento completo:")
    print(f"   Total: {len(primos):,} primos")
    print(f"   Tempo: {t_load:.1f}s")
    print(f"   Taxa: {len(primos)/t_load:,.0f} linhas/s")
    print(f"   Range: {primos[0]:.0f} → {primos[-1]:.0f}")
    
    return primos

# ============================================================================
# FASE 2: DENSIDADE COM SLIDING WINDOWS PARALELO
# ============================================================================

def calcular_janela_densidade(args):
    """Calcula densidade para um bloco de janelas (worker paralelo)"""
    primos_slice, start_idx, end_idx, window_size, step, worker_id = args
    
    densidades_local = []
    posicoes_local = []
    
    for i in range(0, len(primos_slice) - window_size, step):
        window = primos_slice[i:i+window_size]
        span = window[-1] - window[0]
        
        if span > 0:
            dens = window_size / span
            densidades_local.append(dens)
            posicoes_local.append(np.mean(window))
    
    if worker_id % 10 == 0:
        print(f"      Worker {worker_id:2d}: {len(densidades_local):,} janelas")
    
    return np.array(densidades_local), np.array(posicoes_local)

def calcular_densidade_sliding_paralelo(primos, window_size=10000, step=1000, n_cores=56):
    """Calcula densidade com sliding windows em paralelo"""
    
    print("\n" + "=" * 80)
    print("FASE 2: DENSIDADE COM SLIDING WINDOWS")
    print("=" * 80)
    
    print(f"\n[DATA] Configuração:")
    print(f"   Window size: {window_size:,}")
    print(f"   Step: {step:,}")
    print(f"   Overlap: {100*(1-step/window_size):.1f}%")
    print(f"   Janelas esperadas: ~{(len(primos)-window_size)//step:,}")
    
    t0 = time.time()
    
    # Dividir trabalho entre workers
    n_primos_por_worker = len(primos) // n_cores
    overlap_worker = window_size  # Overlap entre workers para continuidade
    
    tasks = []
    for i in range(n_cores):
        start = i * n_primos_por_worker
        end = min((i + 1) * n_primos_por_worker + overlap_worker, len(primos))
        
        if start >= len(primos):
            break
            
        primos_slice = primos[start:end]
        tasks.append((primos_slice, start, end, window_size, step, i + 1))
    
    print(f"\n[STRONG] Processando com {len(tasks)} workers em paralelo...\n")
    
    # Processar em paralelo
    with Pool(n_cores) as pool:
        resultados = pool.map(calcular_janela_densidade, tasks)
    
    # Concatenar resultados
    print("\n📦 Concatenando resultados dos workers...")
    densidades = np.concatenate([r[0] for r in resultados if len(r[0]) > 0])
    posicoes = np.concatenate([r[1] for r in resultados if len(r[1]) > 0])
    
    t_density = time.time() - t0
    
    print(f"\n[OK] Densidade calculada:")
    print(f"   Janelas: {len(densidades):,}")
    print(f"   Tempo: {t_density:.1f}s")
    print(f"   Taxa: {len(densidades)/t_density:,.0f} janelas/s")
    print(f"   Densidade média: {np.mean(densidades):.8f}")
    print(f"   CV: {np.std(densidades)/np.mean(densidades):.4f}")
    
    return densidades, posicoes, t_density

# ============================================================================
# FASE 3: FFT E DETECÇÃO DE MODOS
# ============================================================================

def analisar_espectro(densidades, target_modes=43):
    """Análise espectral completa com busca de threshold ótimo"""
    
    print("\n" + "=" * 80)
    print("FASE 3: ANÁLISE ESPECTRAL")
    print("=" * 80)
    
    t0 = time.time()
    
    # Normalizar
    print("\n[GEOM] Normalizando densidade...")
    dens_norm = (densidades - np.mean(densidades)) / np.std(densidades)
    
    # FFT
    print("🔄 Calculando FFT...")
    yf = fft(dens_norm)
    xf = fftfreq(len(dens_norm), d=1.0)
    
    # Apenas frequências positivas
    mask = xf > 0
    freqs = xf[mask]
    power = np.abs(yf[mask])**2
    
    # Normalizar potência
    power_norm = (power - np.mean(power)) / np.std(power)
    
    t_fft = time.time() - t0
    
    print(f"\n[OK] FFT completa:")
    print(f"   Tempo: {t_fft:.1f}s")
    print(f"   Pontos espectrais: {len(freqs):,}")
    print(f"   Resolução: Δf = {1/len(densidades):.8f}")
    
    # Buscar threshold ótimo para ~43 modos
    print(f"\n[SEARCH] Buscando threshold ótimo (alvo: {target_modes} modos)...")
    
    best_threshold = None
    best_diff = float('inf')
    
    for thresh in np.arange(2.0, 8.0, 0.1):
        peaks, _ = find_peaks(power_norm, height=thresh, distance=10)
        n_peaks = len(peaks)
        diff = abs(n_peaks - target_modes)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = thresh
        
        # Mostrar alguns candidatos
        if target_modes - 10 <= n_peaks <= target_modes + 10:
            print(f"   {thresh:.1f}σ: {n_peaks:2d} picos (diff={diff:2d})")
    
    # Detectar picos com threshold ótimo
    peaks, _ = find_peaks(power_norm, height=best_threshold, distance=10)
    
    print(f"\n[TARGET] Threshold ótimo: {best_threshold:.1f}σ")
    print(f"   Modos detectados: {len(peaks)}")
    print(f"   Diferença do alvo: {abs(len(peaks) - target_modes)}")
    
    return freqs, power_norm, peaks, best_threshold, t_fft

# ============================================================================
# FASE 4: BUSCA DE HARMÔNICOS PRIMOS
# ============================================================================

def buscar_harmonicos_primos(freqs, power_norm, peaks, primos_lista, tolerancia=0.15):
    """Busca harmônicos primos (especialmente 137)"""
    
    print("\n" + "=" * 80)
    print("FASE 4: BUSCA DE HARMÔNICOS PRIMOS")
    print("=" * 80)
    
    peak_freqs = freqs[peaks]
    peak_powers = power_norm[peaks]
    
    # Frequência fundamental
    idx_fundamental = np.argmax(peak_powers)
    f0 = peak_freqs[idx_fundamental]
    
    print(f"\n🎵 Frequência fundamental:")
    print(f"   f₀ = {f0:.8f} ciclos/janela")
    print(f"   Potência: {peak_powers[idx_fundamental]:.1f}σ")
    
    print(f"\n[SCI] Buscando harmônicos PRIMOS (tolerância {tolerancia*100:.0f}%)...\n")
    
    harmonicos = []
    
    for primo in primos_lista:
        f_esperada = primo * f0
        
        # Buscar pico mais próximo
        diffs = np.abs(peak_freqs - f_esperada)
        idx_closest = np.argmin(diffs)
        f_detectada = peak_freqs[idx_closest]
        
        erro_rel = abs(f_detectada - f_esperada) / f_esperada
        
        if erro_rel < tolerancia:
            harmonicos.append({
                'primo': primo,
                'f_esperada': f_esperada,
                'f_detectada': f_detectada,
                'erro': erro_rel * 100,
                'potencia_sigma': peak_powers[idx_closest]
            })
            
            simbolo = "[FIRE]" if primo == 137 else "[OK]"
            print(f"   {simbolo} n={primo:3d}: f={f_detectada:.6f} (erro={erro_rel*100:.2f}%, {peak_powers[idx_closest]:.1f}σ)")
        else:
            if primo in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 137]:
                print(f"   [FAIL] n={primo:3d}: não detectado (erro={erro_rel*100:.1f}%)")
    
    print(f"\n[DATA] RESUMO HARMÔNICOS:")
    print(f"   Detectados: {len(harmonicos)}/{len(primos_lista)}")
    
    if len(harmonicos) > 0:
        primos_det = [h['primo'] for h in harmonicos]
        print(f"   Primos confirmados: {primos_det}")
        print(f"   Erro médio: {np.mean([h['erro'] for h in harmonicos]):.2f}%")
        
        if 137 in primos_det:
            print(f"\n   [FIRE][FIRE][FIRE] HARMÔNICO 137 (α_EM⁻¹) DETECTADO! [FIRE][FIRE][FIRE]")
            idx_137 = next(i for i, h in enumerate(harmonicos) if h['primo'] == 137)
            h137 = harmonicos[idx_137]
            print(f"   Detalhes:")
            print(f"      f₁₃₇ = {h137['f_detectada']:.6f}")
            print(f"      Erro: {h137['erro']:.2f}%")
            print(f"      Potência: {h137['potencia_sigma']:.1f}σ")
    
    return f0, harmonicos

# ============================================================================
# VISUALIZAÇÃO COMPLETA
# ============================================================================

def gerar_visualizacao(freqs, power_norm, peaks, best_threshold, 
                       harmonicos, primos_lista, densidades, output='analise_1B_quantum.png'):
    """Gera visualização completa dos resultados"""
    
    print("\n" + "=" * 80)
    print("GERANDO VISUALIZAÇÃO")
    print("=" * 80)
    
    fig = plt.figure(figsize=(24, 14))
    
    peak_freqs = freqs[peaks]
    peak_powers = power_norm[peaks]
    
    # 1. Espectro completo
    ax1 = plt.subplot(3, 3, 1)
    max_freq_plot = 0.1  # Plotar apenas até 0.1 para visualizar melhor
    mask_plot = freqs <= max_freq_plot
    ax1.plot(freqs[mask_plot], power_norm[mask_plot], 'b-', alpha=0.3, linewidth=0.5)
    
    mask_peaks = peak_freqs <= max_freq_plot
    ax1.plot(peak_freqs[mask_peaks], peak_powers[mask_peaks], 'ro', markersize=4, label='Picos')
    
    ax1.axhline(best_threshold, color='g', linestyle='--', linewidth=2, label=f'{best_threshold:.1f}σ')
    ax1.set_xlabel('Frequência (ciclos/janela)', fontsize=11)
    ax1.set_ylabel('Potência (σ)', fontsize=11)
    ax1.set_title(f'Espectro de Potência - {len(peaks)} Modos', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Top modos
    ax2 = plt.subplot(3, 3, 2)
    idx_sorted = np.argsort(peak_powers)[::-1]
    n_top = min(30, len(peaks))
    colors = ['red' if i == 0 else 'blue' for i in range(n_top)]
    ax2.bar(range(n_top), peak_powers[idx_sorted[:n_top]], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Rank', fontsize=11)
    ax2.set_ylabel('Potência (σ)', fontsize=11)
    ax2.set_title(f'Top {n_top} Modos Mais Fortes', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Harmônicos primos detectados
    ax3 = plt.subplot(3, 3, 3)
    if len(harmonicos) > 0:
        primos_det = [h['primo'] for h in harmonicos]
        erros = [h['erro'] for h in harmonicos]
        colors = ['red' if p == 137 else 'green' for p in primos_det]
        
        bars = ax3.bar(range(len(primos_det)), erros, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(15, color='r', linestyle='--', linewidth=2, label='Limite 15%')
        ax3.set_xlabel('Harmônico', fontsize=11)
        ax3.set_ylabel('Erro (%)', fontsize=11)
        ax3.set_title(f'Harmônicos Primos - {len(primos_det)} Detectados', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(primos_det)))
        ax3.set_xticklabels(primos_det, rotation=90, fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Destacar 137
        idx_137 = next((i for i, p in enumerate(primos_det) if p == 137), None)
        if idx_137 is not None:
            bars[idx_137].set_height(bars[idx_137].get_height())
            bars[idx_137].set_edgecolor('darkred')
            bars[idx_137].set_linewidth(3)
    
    # 4. Razões harmônicas vs primos
    ax4 = plt.subplot(3, 3, 4)
    if len(harmonicos) > 0:
        f0 = peak_freqs[np.argmax(peak_powers)]
        primos_det = [h['primo'] for h in harmonicos]
        razoes = [h['f_detectada'] / f0 for h in harmonicos]
        colors_scatter = ['red' if p == 137 else 'purple' for p in primos_det]
        
        ax4.scatter(primos_det, razoes, s=120, c=colors_scatter, alpha=0.8, edgecolors='black', linewidths=2)
        ax4.plot([0, max(primos_det) + 10], [0, max(primos_det) + 10], 'k--', linewidth=2, label='Ideal f/f₀ = n')
        ax4.set_xlabel('Primo (n)', fontsize=11)
        ax4.set_ylabel('f_n / f₀', fontsize=11)
        ax4.set_title('Razões Harmônicas vs Números Primos', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Densidade temporal (sample)
    ax5 = plt.subplot(3, 3, 5)
    sample_size = min(100000, len(densidades))
    ax5.plot(densidades[:sample_size], 'b-', alpha=0.4, linewidth=0.3)
    ax5.set_xlabel('Janela', fontsize=11)
    ax5.set_ylabel('Densidade', fontsize=11)
    ax5.set_title(f'Densidade Local (primeiras {sample_size:,} janelas)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Distribuição de densidade
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(densidades, bins=100, alpha=0.7, color='cyan', edgecolor='black')
    ax6.axvline(np.mean(densidades), color='red', linestyle='--', linewidth=2, label=f'Média: {np.mean(densidades):.6f}')
    ax6.set_xlabel('Densidade', fontsize=11)
    ax6.set_ylabel('Frequência', fontsize=11)
    ax6.set_title('Distribuição de Densidade', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Frequências dos harmônicos
    ax7 = plt.subplot(3, 3, 7)
    if len(harmonicos) > 0:
        primos_det = [h['primo'] for h in harmonicos]
        freqs_det = [h['f_detectada'] for h in harmonicos]
        colors_freq = ['red' if p == 137 else 'orange' for p in primos_det]
        
        ax7.scatter(primos_det, freqs_det, s=100, c=colors_freq, alpha=0.8, edgecolors='black', linewidths=1.5)
        ax7.set_xlabel('Primo (n)', fontsize=11)
        ax7.set_ylabel('Frequência (ciclos/janela)', fontsize=11)
        ax7.set_title('Frequências dos Harmônicos', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
    
    # 8. Potências dos harmônicos
    ax8 = plt.subplot(3, 3, 8)
    if len(harmonicos) > 0:
        primos_det = [h['primo'] for h in harmonicos]
        potencias = [h['potencia_sigma'] for h in harmonicos]
        colors_pot = ['red' if p == 137 else 'blue' for p in primos_det]
        
        ax8.bar(range(len(primos_det)), potencias, color=colors_pot, alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Harmônico', fontsize=11)
        ax8.set_ylabel('Potência (σ)', fontsize=11)
        ax8.set_title('Significância dos Harmônicos', fontsize=12, fontweight='bold')
        ax8.set_xticks(range(len(primos_det)))
        ax8.set_xticklabels(primos_det, rotation=90, fontsize=8)
        ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Resumo textual
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    resumo = f"""
[FIRE] ANÁLISE 1 BILHÃO - QUANTUM OPTIMIZED [FIRE]

DATASET:
  • Primos analisados: {len(densidades)*10:,}
  • Janelas: {len(densidades):,}
  • Overlap: 90%
  
MODOS FUNDAMENTAIS:
  • Detectados: {len(peaks)} modos
  • Esperado: 43 modos (log₁₀(α_EM/α_grav))
  • Diferença: {abs(len(peaks)-43)}
  • Threshold: {best_threshold:.1f}σ
  
HARMÔNICOS PRIMOS:
  • Detectados: {len(harmonicos)}/{len(primos_lista)}
  • Primos: {[h['primo'] for h in harmonicos][:10]}...
  • Erro médio: {np.mean([h['erro'] for h in harmonicos]):.2f}%
  
{"[FIRE] HARMÔNICO 137 DETECTADO! [FIRE]" if 137 in [h['primo'] for h in harmonicos] else "[FAIL] Harmônico 137 não detectado"}

CONSTANTES FÍSICAS:
  • α_EM = {ALPHA_EM:.8f}
  • α_EM⁻¹ = {1/ALPHA_EM:.6f}
  • α_grav(e⁻) = {ALPHA_GRAV_ELECTRON:.4e}
  • Scale gap: {LOG_SCALE:.2f} ordens
  
SISTEMA:
  • Cores: {N_CORES}
  • RAM: {MEMORIA_DISPONIVEL_GB} GB
"""
    
    ax9.text(0.05, 0.95, resumo, transform=ax9.transAxes,
            fontsize=10, family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"\n[OK] Salvo: {output}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execução principal"""
    
    print(f"\n{'='*80}")
    print("INICIANDO ANÁLISE COMPLETA")
    print(f"{'='*80}\n")
    
    tempo_total_inicio = time.time()
    
    # FASE 1: Carregar primos
    primos = carregar_primos_otimizado()
    
    # Ordenar se necessário
    if not np.all(np.diff(primos) >= 0):
        print("\n[DATA] Ordenando dataset...")
        t0_sort = time.time()
        primos = np.sort(primos)
        t_sort = time.time() - t0_sort
        print(f"[OK] Ordenação: {t_sort:.1f}s")
    
    # FASE 2: Calcular densidade
    densidades, posicoes, t_density = calcular_densidade_sliding_paralelo(
        primos, 
        window_size=10000, 
        step=1000, 
        n_cores=N_CORES
    )
    
    # FASE 3: Análise espectral
    freqs, power_norm, peaks, best_threshold, t_fft = analisar_espectro(densidades, target_modes=43)
    
    # FASE 4: Buscar harmônicos
    f0, harmonicos = buscar_harmonicos_primos(freqs, power_norm, peaks, PRIMOS, tolerancia=0.15)
    
    # FASE 5: Visualização
    gerar_visualizacao(freqs, power_norm, peaks, best_threshold, 
                      harmonicos, PRIMOS, densidades)
    
    # Salvar CSVs
    print("\n" + "=" * 80)
    print("SALVANDO DADOS")
    print("=" * 80)
    
    # Modos fundamentais
    peak_freqs = freqs[peaks]
    peak_powers = power_norm[peaks]
    df_modos = pd.DataFrame({
        'frequencia': peak_freqs,
        'potencia_sigma': peak_powers,
        'periodo_janelas': 1.0 / peak_freqs
    })
    df_modos = df_modos.sort_values('potencia_sigma', ascending=False)
    df_modos.to_csv('modos_fundamentais_1B.csv', index=False)
    print(f"[OK] modos_fundamentais_1B.csv: {len(df_modos)} modos")
    
    # Harmônicos
    if len(harmonicos) > 0:
        df_harm = pd.DataFrame(harmonicos)
        df_harm.to_csv('harmonicos_primos_1B.csv', index=False)
        print(f"[OK] harmonicos_primos_1B.csv: {len(harmonicos)} harmônicos")
    
    tempo_total = time.time() - tempo_total_inicio
    
    # RESUMO FINAL
    print("\n" + "=" * 80)
    print("[SUCCESS] ANÁLISE COMPLETA!")
    print("=" * 80)
    
    print(f"\n[TIME]  TEMPO TOTAL: {tempo_total/60:.1f} minutos")
    print(f"\n[DATA] RESULTADOS:")
    print(f"   • Primos analisados: {len(primos):,}")
    print(f"   • Janelas: {len(densidades):,}")
    print(f"   • Modos detectados: {len(peaks)} (esperado: 43)")
    print(f"   • Harmônicos primos: {len(harmonicos)}/{len(PRIMOS)}")
    
    if len(harmonicos) > 0:
        print(f"   • Primos confirmados: {[h['primo'] for h in harmonicos]}")
        print(f"   • Precisão média: {np.mean([h['erro'] for h in harmonicos]):.2f}%")
        
        if 137 in [h['primo'] for h in harmonicos]:
            print(f"\n   [FIRE][FIRE][FIRE] HARMÔNICO 137 (α_EM⁻¹) CONFIRMADO! [FIRE][FIRE][FIRE]")
            h137 = next(h for h in harmonicos if h['primo'] == 137)
            print(f"   Erro: {h137['erro']:.2f}%")
            print(f"   Potência: {h137['potencia_sigma']:.1f}σ")
        else:
            print(f"\n   [FAIL] Harmônico 137 não detectado")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()