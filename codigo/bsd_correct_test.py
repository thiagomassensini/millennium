#!/usr/bin/env python3
"""
BSD CORRETO: Curva deve depender de P, não só de k!

Nova família: E_p: y² = x³ + p·x + (p+2)
Ou: E_p: y² = x³ + k_real·p·x + 1
"""

import numpy as np
from sympy import isprime
from cypari2 import Pari
import json

pari = Pari()

print("=" * 80)
print("BSD RANK TEST: CURVAS DEPENDENTES DE P")
print("=" * 80)
print()

# Gerar primos gêmeos
def generate_twin_primes(max_n):
    twins = []
    for p in range(3, max_n, 2):
        if isprime(p) and isprime(p+2):
            twins.append(p)
    return twins

print("[SEARCH] Gerando primos gêmeos < 10,000...")
twin_primes = generate_twin_primes(10000)
print(f"[OK] {len(twin_primes)} primos gêmeos encontrados")
print()

# Calcular k_real
def calc_k_real(p):
    x = p ^ (p+2)
    v = x + 2
    if v > 0 and (v & (v-1)) == 0:
        return v.bit_length() - 1
    return -1

# Testar MÚLTIPLAS famílias de curvas
def test_curve_family(name, get_curve_params):
    """
    get_curve_params(p, k) -> (a, b) para y² = x³ + a·x + b
    """
    print(f"\n{'='*80}")
    print(f"TESTANDO: {name}")
    print(f"{'='*80}\n")
    
    results = []
    k_to_rank = {}
    
    print(f"{'#':>4} | {'p':>8} | {'k_real':>6} | {'rank':>6} | {'conductor':>12}")
    print("-" * 65)
    
    for i, p in enumerate(twin_primes[:100]):
        k = calc_k_real(p)
        if k <= 0 or k > 10:
            continue
        
        try:
            a, b = get_curve_params(p, k)
            E = pari.ellinit([0, 0, 0, a, b])
            
            # Rank e conductor
            rank_data = pari.ellanalyticrank(E)
            rank = int(rank_data[0])
            
            conductor = int(pari.ellglobalred(E)[0])
            
            results.append({
                'p': int(p),
                'k_real': k,
                'rank': rank,
                'conductor': conductor,
                'a': int(a),
                'b': int(b)
            })
            
            if k not in k_to_rank:
                k_to_rank[k] = []
            k_to_rank[k].append(rank)
            
            if i < 30:
                print(f"{i+1:4d} | {p:8d} | {k:6d} | {rank:6d} | {conductor:12d}")
        
        except Exception as e:
            if i < 5:
                print(f"{i+1:4d} | {p:8d} | ERROR: {str(e)[:30]}")
            continue
    
    if len(results) == 0:
        print("\n[FAIL] Nenhum resultado")
        return None
    
    # Análise
    print(f"\n{'='*80}")
    print("ANÁLISE:")
    print(f"{'='*80}\n")
    
    from scipy.stats import pearsonr
    k_list = [r['k_real'] for r in results]
    rank_list = [r['rank'] for r in results]
    
    r_corr, p_val = pearsonr(k_list, rank_list)
    
    print(f"Correlação: r = {r_corr:.4f}, p = {p_val:.2e}")
    print()
    
    print("Distribuição por k:")
    print(f"{'k':>3} | {'count':>6} | {'rank_avg':>9} | {'rank_std':>9}")
    print("-" * 45)
    
    for k in sorted(k_to_rank.keys()):
        ranks = k_to_rank[k]
        print(f"{k:3d} | {len(ranks):6d} | {np.mean(ranks):9.3f} | {np.std(ranks):9.3f}")
    
    print()
    
    if r_corr > 0.7:
        print(f"[OK][OK][OK] CORRELAÇÃO FORTE! Esta família é promissora!")
    elif r_corr > 0.5:
        print(f"[OK][OK] Correlação moderada")
    else:
        print(f"[FAIL] Correlação fraca")
    
    return {
        'name': name,
        'correlation': r_corr,
        'p_value': p_val,
        'results': results[:50]
    }

# ==================== FAMÍLIAS DE CURVAS ====================

families = []

# Família 1: y² = x³ + p·x + (p+2)
families.append(
    test_curve_family(
        "E_p: y² = x³ + p·x + (p+2)",
        lambda p, k: (p, p+2)
    )
)

# Família 2: y² = x³ + (k·p)·x + 1
families.append(
    test_curve_family(
        "E_p: y² = x³ + (k·p)·x + 1",
        lambda p, k: (k*p, 1)
    )
)

# Família 3: y² = x³ + k·x + p
families.append(
    test_curve_family(
        "E_p: y² = x³ + k·x + p",
        lambda p, k: (k, p)
    )
)

# Família 4: y² = x³ + p·x + k
families.append(
    test_curve_family(
        "E_p: y² = x³ + p·x + k",
        lambda p, k: (p, k)
    )
)

# Família 5: y² = x³ + (p mod k²)·x + k
families.append(
    test_curve_family(
        "E_p: y² = x³ + (p mod k²)·x + k",
        lambda p, k: (p % (k*k) if k > 1 else p, k)
    )
)

# ==================== MELHOR FAMÍLIA ====================
print("\n" + "=" * 80)
print("RESUMO: MELHORES FAMÍLIAS")
print("=" * 80 + "\n")

families = [f for f in families if f is not None]
families.sort(key=lambda x: x['correlation'], reverse=True)

for i, f in enumerate(families):
    print(f"{i+1}. {f['name']}")
    print(f"   Correlação: r = {f['correlation']:.4f} (p = {f['p_value']:.2e})")
    print()

if len(families) > 0 and families[0]['correlation'] > 0.7:
    print(f"[WIN] MELHOR: {families[0]['name']}")
    print(f"   r = {families[0]['correlation']:.4f}")
    print()
    print("[OK] Esta família mostra conexão forte entre k_real e rank!")
    print("[OK] Investigar mais para confirmar BSD!")
else:
    print("[WARNING] Nenhuma família mostrou correlação forte > 0.7")
    print("  → Pode precisar de família diferente")
    print("  → Ou k_real não mapeia diretamente para rank")

print("\n" + "=" * 80)

# Salvar
with open('bsd_families_comparison.json', 'w') as f:
    json.dump(families, f, indent=2)

print("[OK] Resultados salvos em bsd_families_comparison.json")
print("=" * 80)
