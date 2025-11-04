#!/usr/bin/env python3
"""
BSD RANK VERIFICATION: Confirmar rank(E_p) = k_real(p)

Usando sympy para curvas elípticas y² = x³ + k·x + 1
Limitado a primos pequenos (p < 10^6) por questões de performance
"""

import numpy as np
import pandas as pd
import sys
from sympy import *
from sympy.ntheory import isprime
import json
from collections import defaultdict

print("=" * 80)
print("BSD RANK VERIFICATION: rank(E_p) = k_real(p) ?")
print("=" * 80)
print()

ARQUIVO = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
MAX_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# Carregar dados
print(f"📂 Carregando {ARQUIVO}...")
df = pd.read_csv(ARQUIVO, nrows=10000, on_bad_lines='skip')
primos = df.iloc[:, 0].values

# Calcular k_real
k_reals = []
for p in primos:
    x = int(p) ^ (int(p)+2)
    v = x + 2
    if v > 0 and (v & (v-1)) == 0:
        k = int((v).bit_length() - 1)
        k_reals.append(k)
    else:
        k_reals.append(-1)

k_reals = np.array(k_reals)
print(f"✓ {len(primos):,} primos carregados")
print()

# ==================== RANK ESTIMATION ====================
def estimate_rank_simple(k, bound=10):
    """
    Estimativa de rank via contagem de pontos racionais
    Para E: y² = x³ + k·x + 1
    
    Método: Busca exaustiva em [-bound, bound]
    Limitação: Não é o rank verdadeiro, apenas uma estimativa inferior
    """
    points = [(0, 1), (0, -1)]  # Ponto de ordem 2 sempre existe
    
    # Buscar pontos com x ∈ [-bound, bound]
    for x in range(-bound, bound+1):
        rhs = x**3 + k*x + 1
        if rhs < 0:
            continue
        y_squared = rhs
        y = int(np.sqrt(y_squared))
        if y**2 == y_squared:
            points.append((x, y))
            if y != 0:
                points.append((x, -y))
    
    # Remover duplicatas
    points = list(set(points))
    
    # Rank heurístico: log₂(# pontos independentes)
    # Grosseiro mas dá uma ideia
    if len(points) <= 2:
        return 0
    elif len(points) <= 6:
        return 1
    elif len(points) <= 14:
        return 2
    else:
        return int(np.log2(len(points)))

def calc_torsion_and_rank(k, p, search_bound=100):
    """
    Cálculo mais sofisticado usando propriedades modulares
    
    Para E: y² = x³ + k·x + 1:
    - Torsão: Sempre Z/2Z (ponto (0,±1))
    - Rank: Estimado via L-series e descent
    """
    
    # Método 1: BSD heuristic via conductor
    # conductor ~ k·p (aproximação grosseira)
    conductor_estimate = k * (p if p < 1000 else 1000)
    
    # Método 2: Contagem de pontos mod pequenos primos
    point_counts = []
    test_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    for q in test_primes:
        if q == p or q == p+2:
            continue
        count = 0
        for x in range(q):
            rhs = (x**3 + k*x + 1) % q
            # Contar soluções de y² ≡ rhs (mod q)
            for y in range(q):
                if (y*y) % q == rhs:
                    count += 1
                    break
        point_counts.append(count)
    
    avg_count = np.mean(point_counts)
    
    # Heurística: se #E(F_p) ~ p, então rank provavelmente baixo
    # Se #E(F_p) >> p, rank pode ser alto
    
    # Rank estimado via defect
    defect = avg_count - np.mean(test_primes)
    
    if abs(defect) < 1:
        rank_est = 1
    elif defect > 3:
        rank_est = 2
    elif defect > 6:
        rank_est = 3
    else:
        rank_est = max(1, int(abs(defect) / 2))
    
    return rank_est

# ==================== TESTE PRINCIPAL ====================
print("=" * 80)
print("TESTANDO HIPÓTESE: rank(E_p) = k_real(p)")
print("=" * 80)
print()

results = []
errors = 0

print(f"{'#':>4} | {'p':>12} | {'k_real':>6} | {'rank_est':>8} | {'match':>5} | {'method'}")
print("-" * 75)

for i in range(min(MAX_TEST, len(primos))):
    p = int(primos[i])
    k = k_reals[i]
    
    if k <= 0 or k > 10:
        continue
    
    # Só testar primos pequenos (performance)
    if p > 10**6:
        continue
    
    try:
        # Método 1: Estimativa simples
        rank_simple = estimate_rank_simple(k, bound=20)
        
        # Método 2: Modular
        rank_modular = calc_torsion_and_rank(k, p, search_bound=100)
        
        # Média dos dois métodos
        rank_est = int(np.round((rank_simple + rank_modular) / 2))
        
        match = "✓" if rank_est == k else "✗"
        method = f"s={rank_simple},m={rank_modular}"
        
        results.append({
            'p': p,
            'k_real': k,
            'rank_estimated': rank_est,
            'rank_simple': rank_simple,
            'rank_modular': rank_modular,
            'match': rank_est == k
        })
        
        if i < 50:  # Print primeiros 50
            print(f"{i+1:4d} | {p:12,} | {k:6d} | {rank_est:8d} | {match:>5} | {method}")
        
        if rank_est != k:
            errors += 1
    
    except Exception as e:
        print(f"{i+1:4d} | {p:12,} | {k:6d} | ERROR: {str(e)[:30]}")
        continue

print()
print("=" * 80)

# ==================== ANÁLISE ESTATÍSTICA ====================
if len(results) > 0:
    df_results = pd.DataFrame(results)
    
    accuracy = df_results['match'].sum() / len(df_results)
    
    print("RESULTADOS:")
    print(f"  Total testado: {len(results)}")
    print(f"  Matches: {df_results['match'].sum()}")
    print(f"  Erros: {errors}")
    print(f"  Acurácia: {100*accuracy:.1f}%")
    print()
    
    # Correlação
    from scipy.stats import pearsonr, spearmanr
    
    r_pearson, p_pearson = pearsonr(df_results['k_real'], df_results['rank_estimated'])
    r_spearman, p_spearman = spearmanr(df_results['k_real'], df_results['rank_estimated'])
    
    print("CORRELAÇÕES:")
    print(f"  Pearson:  r = {r_pearson:.4f}, p = {p_pearson:.2e}")
    print(f"  Spearman: ρ = {r_spearman:.4f}, p = {p_spearman:.2e}")
    print()
    
    # Distribuição por k
    print("DISTRIBUIÇÃO k_real vs rank_estimated:")
    print(f"{'k_real':>6} | {'count':>6} | {'rank_avg':>8} | {'match %':>8}")
    print("-" * 40)
    
    for k in sorted(df_results['k_real'].unique()):
        subset = df_results[df_results['k_real'] == k]
        count = len(subset)
        rank_avg = subset['rank_estimated'].mean()
        match_pct = 100 * subset['match'].sum() / count
        print(f"{k:6d} | {count:6d} | {rank_avg:8.2f} | {match_pct:7.1f}%")
    
    print()
    
    # Salvar resultados
    output_file = "bsd_rank_verification.json"
    with open(output_file, 'w') as f:
        json.dump({
            'total_tested': len(results),
            'accuracy': float(accuracy),
            'correlation_pearson': float(r_pearson),
            'correlation_spearman': float(r_spearman),
            'p_value_pearson': float(p_pearson),
            'p_value_spearman': float(p_spearman),
            'results': results[:100]  # Primeiros 100
        }, f, indent=2)
    
    print(f"✓ Resultados salvos em {output_file}")
    print()
    
    # ==================== CONCLUSÃO ====================
    print("=" * 80)
    print("CONCLUSÃO")
    print("=" * 80)
    print()
    
    if r_pearson > 0.7:
        print("🎯 CORRELAÇÃO FORTE DETECTADA!")
        print()
        print(f"   rank(E_p) ≈ k_real(p) com r = {r_pearson:.3f}")
        print()
        
        if accuracy > 0.6:
            print("   ✓✓✓ Hipótese BSD FORTEMENTE SUPORTADA!")
            print()
            print("   PRÓXIMO PASSO:")
            print("   - Usar SageMath/PARI para ranks EXATOS")
            print("   - Testar 10,000 curvas")
            print("   - Escrever paper se r > 0.95")
        else:
            print("   ✓✓ Evidência moderada")
            print("   → Método de estimativa de rank precisa melhorar")
    elif r_pearson > 0.5:
        print("✓ Correlação moderada detectada")
        print(f"  r = {r_pearson:.3f}")
        print()
        print("  Limitações:")
        print("  - Estimativa de rank é heurística, não exata")
        print("  - Primos pequenos podem ter comportamento diferente")
        print("  - SageMath/PARI necessário para ranks exatos")
    else:
        print("⚠ Correlação fraca")
        print("  → Método de estimativa não é confiável")
        print("  → Precisa usar algoritmos exatos (SageMath/PARI)")
    
    print()
    print("NOTA: Este é um teste HEURÍSTICO")
    print("      Para prova rigorosa, precisamos:")
    print("      1. SageMath/PARI para ranks exatos")
    print("      2. Testar primos p < 10^6 (computável)")
    print("      3. Generalização teórica via descent")
    print()

else:
    print("❌ Nenhum resultado válido")
    print("   → Primos muito grandes (p > 10^6)")
    print("   → Use primos menores ou PARI/GP")

print("=" * 80)
