"""
BSD INVESTIGATION NOTEBOOK

Objetivo: Entender TEORICAMENTE por que rank(E_p) = (log₂(k)+1)//2

HIPÓTESES A TESTAR:
1. Torção é sempre trivial para k=2^n?
2. dim(Selmer²) tem relação com bits do XOR?
3. Discriminante tem fatoração especial?
4. Função L tem estrutura que força ordem de zero = (n+1)//2?

DADOS CONHECIDOS:
- k=2 (2^1, n=1) → rank=1 = (1+1)//2 ✓
- k=4 (2^2, n=2) → rank=1 = (2+1)//2 ✓
- k=8 (2^3, n=3) → rank=2 = (3+1)//2 ✓
- k=16 (2^4, n=4) → rank=2 = (4+1)//2 ✓

PADRÃO: rank aumenta 1 a cada 2 dobramentos de k
"""

# ==================== SETUP ====================
from bsd_theoretical_workspace import *

# Selecionar casos de teste
# Para cada k=2^n, pegar alguns primos p pequenos

TEST_CASES = []

# k=2: pegar primeiros 5 primos gêmeos com k=2
# k=4: pegar primeiros 5 primos gêmeos com k=4
# etc.

print("Gerando casos de teste...")

import cypari2
pari = cypari2.Pari()

def is_prime(n):
    return bool(pari.isprime(n))

def calc_k_real(p):
    if p % 2 == 0:
        return None
    xor = p ^ (p + 2)
    val = xor + 2
    if val & (val - 1) != 0:
        return None
    k = val.bit_length() - 2
    return k if 0 <= k < 25 else None

# Gerar casos
target_ks = [2, 4, 8, 16]
cases_per_k = 10

for target_k in target_ks:
    print(f"\nBuscando primos para k={target_k}...")
    found = []
    p = 3
    while len(found) < cases_per_k and p < 10_000_000:
        if is_prime(p) and is_prime(p+2):
            k = calc_k_real(p)
            if k == target_k:
                found.append(p)
                print(f"  {len(found)}/{cases_per_k}: p={p}")
        p += 2
    
    for p in found:
        TEST_CASES.append((p, target_k))

print(f"\n✓ Total de casos: {len(TEST_CASES)}")

# ==================== ANÁLISE COMPLETA ====================

print("\n" + "="*80)
print("INICIANDO ANÁLISE COMPLETA")
print("="*80)

results = comprehensive_analysis(TEST_CASES, 'bsd_theoretical_analysis.json')

# ==================== DETECTAR PADRÕES ====================

detect_patterns(results)

# ==================== INVESTIGAÇÃO ESPECÍFICA ====================

print("\n" + "="*80)
print("INVESTIGAÇÃO: TORÇÃO")
print("="*80)

torsion_by_k = {}
for r in results:
    k = r['k']
    tors = r['basic_info'].get('torsion_order')
    if k not in torsion_by_k:
        torsion_by_k[k] = []
    if tors is not None:
        torsion_by_k[k].append(tors)

for k in sorted(torsion_by_k.keys()):
    torsions = torsion_by_k[k]
    unique = set(torsions)
    print(f"k={k}: {unique}")
    if len(unique) == 1 and 1 in unique:
        print(f"  ✓ Torção SEMPRE trivial para k={k}")

# ==================== INVESTIGAÇÃO: DISCRIMINANTE ====================

print("\n" + "="*80)
print("INVESTIGAÇÃO: DISCRIMINANTE")
print("="*80)

for k in [2, 4, 8, 16]:
    print(f"\nk={k}:")
    k_results = [r for r in results if r['k'] == k][:3]
    
    for r in k_results:
        p = r['p']
        disc = r['basic_info'].get('discriminant')
        if disc:
            factors = factorize_discriminant(disc)
            print(f"  p={p}: Δ = {disc}")
            print(f"    Fatores: {factors}")

# ==================== INVESTIGAÇÃO: SELMER ====================

print("\n" + "="*80)
print("INVESTIGAÇÃO: SELMER GROUPS (2-descent)")
print("="*80)

for k in [2, 4, 8, 16]:
    print(f"\nk={k}:")
    k_results = [r for r in results if r['k'] == k][:5]
    
    selmer_dims = []
    for r in k_results:
        descent = r.get('descent', {})
        lower = descent.get('rank_lower')
        upper = descent.get('rank_upper')
        
        if lower is not None and upper is not None:
            print(f"  p={r['p']}: rank ∈ [{lower}, {upper}]")
            if lower == upper:
                selmer_dims.append(lower)
    
    if selmer_dims:
        print(f"  Selmer dimensions (when exact): {set(selmer_dims)}")

# ==================== CONCLUSÃO ====================

print("\n" + "="*80)
print("PRÓXIMOS PASSOS")
print("="*80)
print()
print("Baseado nos resultados acima, investigar:")
print()
print("1. Se torção é SEMPRE trivial → rank = dim(Sel²) diretamente")
print("2. Padrão na fatoração de Δ → estrutura especial?")
print("3. Relação dim(Sel²) com n → dim = f(n)?")
print("4. Estudar extensões quadráticas geradas por XOR")
print()
print("Resultados salvos em: bsd_theoretical_analysis.json")
print()
