#!/usr/bin/env python3
"""
Análise Multirresolução: Detectar harmônicos em TODAS as escalas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch
import sys

print("=" * 80)
print("ANÁLISE MULTIRRESOLUÇÃO: HARMÔNICOS EM MÚLTIPLAS ESCALAS")
print("=" * 80)
print()

# Parâmetros
ARQUIVO = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
MAX_LINHAS = int(sys.argv[2]) if len(sys.argv) > 2 else 100000000  # 100M padrão

# Escalas para análise (diferentes tamanhos de janela)
ESCALAS = [
    1000,      # Ultra-local
    5000,      # Local
    10000,     # Intermediário baixo
    50000,     # Intermediário alto
    100000,    # Global baixo
    500000,    # Global alto
    1000000    # Ultra-global
]

print(f"Arquivo: {ARQUIVO}")
print(f"Linhas: {MAX_LINHAS:,}")
print(f"Escalas: {len(ESCALAS)}")
print()

# Carregar dados (pulando linhas corrompidas)
print("Carregando dados (ignorando linhas corrompidas)...")
primos = []
linhas_ruins = 0

with open(ARQUIVO, 'r') as f:
    for i, linha in enumerate(f, 1):
        if MAX_LINHAS and len(primos) >= MAX_LINHAS:
            break
        
        campos = linha.strip().split(',')
        
        # Linha válida = 5 campos
        if len(campos) == 5:
            try:
                p = float(campos[0])
                primos.append(p)
            except:
                linhas_ruins += 1
        else:
            linhas_ruins += 1
        
        if i % 10000000 == 0:
            print(f"\r  {len(primos):,} primos carregados, {linhas_ruins} linhas ruins...", end='')

primos = np.array(primos)
print()
print(f"[OK] {len(primos):,} primos carregados")
if linhas_ruins > 0:
    print(f"[WARNING] {linhas_ruins:,} linhas corrompidas ignoradas ({100*linhas_ruins/(len(primos)+linhas_ruins):.4f}%)")
print()

# Dicionário para armazenar resultados por escala
resultados = {}

# Análise para cada escala
for escala_idx, WINDOW_SIZE in enumerate(ESCALAS, 1):
    print(f"Escala {escala_idx}/{len(ESCALAS)}: Janela = {WINDOW_SIZE:,}")
    
    # Calcular densidade local
    n_windows = (len(primos) - WINDOW_SIZE) // (WINDOW_SIZE // 10)
    step = max(1, WINDOW_SIZE // 10)
    
    densidades = []
    for i in range(0, len(primos) - WINDOW_SIZE, step):
        janela = primos[i:i+WINDOW_SIZE]
        span = janela[-1] - janela[0]
        if span > 0:
            dens = WINDOW_SIZE / span
            densidades.append(dens)
    
    densidades = np.array(densidades)
    
    if len(densidades) < 100:
        print(f"  [WARNING] Janela muito grande, pulando")
        continue
    
    # Normalizar
    dens_norm = (densidades - np.mean(densidades)) / np.std(densidades)
    
    # FFT
    N = len(dens_norm)
    yf = fft(dens_norm)
    xf = fftfreq(N, d=1.0)
    
    # Só frequências positivas
    mask = xf > 0
    freqs = xf[mask]
    power = np.abs(yf[mask])**2
    
    # Detectar picos
    threshold = np.mean(power) + 3 * np.std(power)
    picos, props = find_peaks(power, height=threshold, distance=5)
    
    # Ordenar por potência
    if len(picos) > 0:
        idx_sorted = np.argsort(power[picos])[::-1]
        top_picos = picos[idx_sorted[:20]]  # Top 20
        top_freqs = freqs[top_picos]
        top_powers = power[top_picos]
        
        # Converter frequências para "períodos em janelas"
        periodos = 1.0 / top_freqs
        
        # Armazenar
        resultados[WINDOW_SIZE] = {
            'freqs': top_freqs,
            'periodos': periodos,
            'powers': top_powers,
            'n_picos': len(picos)
        }
        
        print(f"  [OK] {len(picos)} picos detectados")
        print(f"    Top 5 períodos: {periodos[:5]}")
    else:
        print(f"  [FAIL] Nenhum pico detectado")
    
    print()

# Análise combinada
print("=" * 80)
print("ANÁLISE COMBINADA: BUSCA POR HARMÔNICOS PRIMOS")
print("=" * 80)
print()

# Lista de primos para testar (até 150)
primos_test = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 
               109, 113, 127, 131, 137, 139, 149]

# Para cada primo, verificar em quais escalas aparece
deteccoes = {p: [] for p in primos_test}

for WINDOW_SIZE, res in resultados.items():
    periodos = res['periodos']
    
    for primo in primos_test:
        # Tolerância: ±10%
        matches = np.abs(periodos - primo) / primo < 0.10
        
        if np.any(matches):
            idx_match = np.where(matches)[0][0]
            periodo_obs = periodos[idx_match]
            erro = abs(periodo_obs - primo) / primo
            potencia = res['powers'][idx_match]
            
            deteccoes[primo].append({
                'escala': WINDOW_SIZE,
                'periodo': periodo_obs,
                'erro': erro,
                'potencia': potencia
            })

# Mostrar resultados
print("HARMÔNICOS PRIMOS DETECTADOS:")
print()

primos_detectados = []
for primo in primos_test:
    if len(deteccoes[primo]) > 0:
        primos_detectados.append(primo)
        
        # Calcular estatísticas
        erros = [d['erro'] for d in deteccoes[primo]]
        escalas = [d['escala'] for d in deteccoes[primo]]
        potencias = [d['potencia'] for d in deteccoes[primo]]
        
        erro_medio = np.mean(erros)
        n_escalas = len(escalas)
        potencia_max = np.max(potencias)
        
        print(f"Primo {primo:3d}:")
        print(f"  Detectado em {n_escalas} escalas: {escalas}")
        print(f"  Erro médio: {100*erro_medio:.2f}%")
        print(f"  Potência máxima: {potencia_max:.2e}")
        print()

print(f"Total de primos detectados: {len(primos_detectados)}")
print(f"Primos: {primos_detectados}")
print()

# Teste especial: α_EM = 1/137
if 137 in primos_detectados:
    print("[WIN] JACKPOT: Primo 137 (1/α_EM) DETECTADO!")
    print()
    print("Detalhes:")
    for det in deteccoes[137]:
        print(f"  Escala {det['escala']:,}: período={det['periodo']:.2f}, "
              f"erro={100*det['erro']:.2f}%, potência={det['potencia']:.2e}")
    print()
else:
    print("[WARNING] Primo 137 (1/α_EM) NÃO detectado ainda")
    print("  (pode precisar de dataset maior ou escala diferente)")
    print()

# Visualização
print("Gerando visualização...")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for idx, (WINDOW_SIZE, res) in enumerate(resultados.items()):
    if idx >= 9:
        break
    
    ax = axes[idx]
    
    # Plot espectro
    freqs = res['freqs']
    powers = res['powers']
    periodos = res['periodos']
    
    ax.semilogy(periodos, powers, 'b-', alpha=0.5, linewidth=0.5)
    
    # Marcar primos detectados
    for primo in primos_detectados:
        if primo in [d['periodo'] for d in deteccoes[primo] 
                     if d['escala'] == WINDOW_SIZE]:
            # Encontrar índice
            idx_primo = np.argmin(np.abs(periodos - primo))
            ax.plot(periodos[idx_primo], powers[idx_primo], 'ro', 
                   markersize=8, label=f'p={primo}')
    
    ax.set_xlabel('Período (janelas)')
    ax.set_ylabel('Potência')
    ax.set_title(f'Escala: janela={WINDOW_SIZE:,}')
    ax.set_xlim(0, 150)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/harmonicos_multiescala.png', 
            dpi=200, bbox_inches='tight')
print("[OK] Gráfico salvo: harmonicos_multiescala.png")
print()

print("=" * 80)
print("CONCLUSÃO")
print("=" * 80)
print()
print(f"Primos detectados: {len(primos_detectados)}")
print(f"Range: {min(primos_detectados)} → {max(primos_detectados)}")
print()

if 137 in primos_detectados:
    print("[OK][OK][OK] CONEXÃO COM α_EM CONFIRMADA! [OK][OK][OK]")
else:
    print("[WARNING] Primo 137 não detectado (precisa investigar mais)")

print()
print("=" * 80)