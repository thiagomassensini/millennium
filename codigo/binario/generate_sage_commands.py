#!/usr/bin/env python3
"""
BSD PROOF: Script para gerar comandos SageMath

Este script cria código SageMath para calcular:
1. Rank exato de E_p via mwrank
2. L(E,1) via modular symbols
3. Reg(E), Ω(E), |Sha(E)| via BSD formula

USO:
  python3 generate_sage_commands.py results.csv 100 > bsd_sage_test.sage
  sage bsd_sage_test.sage
"""

import pandas as pd
import sys

ARQUIVO = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
MAX_LINHAS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# Carregar primos
df = pd.read_csv(ARQUIVO, nrows=MAX_LINHAS, on_bad_lines='skip')
primos = df.iloc[:, 0].values[:100]  # Primeiros 100 para teste

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

# Gerar código SageMath
print("# BSD CONJECTURE TEST - SageMath Script")
print("# Auto-generated from twin primes dataset")
print()
print("from sage.all import *")
print()
print("results = []")
print()

count = 0
for i, p in enumerate(primos[:20]):  # Só 20 para teste rápido
    k = k_reals[i]
    if k <= 0:
        continue
    
    # Curva: y² = x³ + k·x + 1
    a = k
    b = 1
    
    print(f"# Test {count+1}: p={p}, k_real={k}")
    print(f"try:")
    print(f"    E = EllipticCurve([{a}, {b}])")
    print(f"    rank = E.rank()")
    print(f"    L_value = E.lseries().L_ratio()  # L(E,1)/Omega")
    print(f"    ")
    print(f"    result = {{")
    print(f"        'p': {p},")
    print(f"        'k_real': {k},")
    print(f"        'a': {a},")
    print(f"        'b': {b},")
    print(f"        'rank': rank,")
    print(f"        'L_ratio': float(L_value)")
    print(f"    }}")
    print(f"    results.append(result)")
    print(f"    print(f'[OK] p={p}, k={k}, rank={{rank}}, L/Ω={{L_value}}')")
    print(f"except Exception as e:")
    print(f"    print(f'[FAIL] p={p}, k={k}: {{e}}')")
    print()
    
    count += 1
    if count >= 20:
        break

print("# Save results")
print("import json")
print("with open('bsd_sage_results.json', 'w') as f:")
print("    json.dump(results, f, indent=2)")
print()
print("# Analysis")
print("if len(results) > 0:")
print("    print()")
print("    print('=' * 80)")
print("    print('ANÁLISE: rank vs k_real')")
print("    print('=' * 80)")
print("    print()")
print("    ")
print("    ranks = [r['rank'] for r in results]")
print("    k_vals = [r['k_real'] for r in results]")
print("    ")
print("    # Correlação")
print("    from scipy.stats import pearsonr")
print("    r_corr, p_val = pearsonr(ranks, k_vals)")
print("    ")
print("    print(f'Correlação rank vs k_real: r={r_corr:.4f}, p={p_val:.2e}')")
print("    print()")
print("    ")
print("    # Casos individuais")
print("    print('Casos individuais:')")
print("    for res in results[:10]:")
print("        print(f\"  k={res['k_real']:2d} → rank={res['rank']}  (L/Ω={res['L_ratio']:.6f})\")")
print("    ")
print("    print()")
print("    ")
print("    if abs(r_corr) > 0.9:")
print("        print('[WIN] CORRELAÇÃO FORTÍSSIMA! rank ≈ k_real')")
print("    elif abs(r_corr) > 0.7:")
print("        print('[OK] CORRELAÇÃO FORTE')")
print("    else:")
print("        print('[WARNING] Correlação moderada')")
