# BSD CONJECTURE TEST - SageMath Script
# Auto-generated from twin primes dataset

from sage.all import *

results = []

# Test 1: p=1000010400000761, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400000761,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400000761, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400000761, k=2: {e}')

# Test 2: p=1000010400000791, k_real=4
try:
    E = EllipticCurve([4, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400000791,
        'k_real': 4,
        'a': 4,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400000791, k=4, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400000791, k=4: {e}')

# Test 3: p=1000010400000881, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400000881,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400000881, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400000881, k=2: {e}')

# Test 4: p=1000010400001607, k_real=4
try:
    E = EllipticCurve([4, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400001607,
        'k_real': 4,
        'a': 4,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400001607, k=4, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400001607, k=4: {e}')

# Test 5: p=1000010400005759, k_real=8
try:
    E = EllipticCurve([8, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400005759,
        'k_real': 8,
        'a': 8,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400005759, k=8, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400005759, k=8: {e}')

# Test 6: p=1000010400006029, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400006029,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400006029, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400006029, k=2: {e}')

# Test 7: p=1000010400009029, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400009029,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400009029, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400009029, k=2: {e}')

# Test 8: p=1000010400009737, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400009737,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400009737, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400009737, k=2: {e}')

# Test 9: p=1000010400010157, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400010157,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400010157, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400010157, k=2: {e}')

# Test 10: p=1000010400012137, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400012137,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400012137, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400012137, k=2: {e}')

# Test 11: p=1000010400012641, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400012641,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400012641, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400012641, k=2: {e}')

# Test 12: p=1000010400012971, k_real=3
try:
    E = EllipticCurve([3, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400012971,
        'k_real': 3,
        'a': 3,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400012971, k=3, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400012971, k=3: {e}')

# Test 13: p=1000010400013577, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400013577,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400013577, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400013577, k=2: {e}')

# Test 14: p=1000010400014411, k_real=3
try:
    E = EllipticCurve([3, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400014411,
        'k_real': 3,
        'a': 3,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400014411, k=3, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400014411, k=3: {e}')

# Test 15: p=1000010400014867, k_real=3
try:
    E = EllipticCurve([3, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400014867,
        'k_real': 3,
        'a': 3,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400014867, k=3, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400014867, k=3: {e}')

# Test 16: p=1000010400015269, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400015269,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400015269, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400015269, k=2: {e}')

# Test 17: p=1000010400015461, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400015461,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400015461, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400015461, k=2: {e}')

# Test 18: p=1000010400016247, k_real=4
try:
    E = EllipticCurve([4, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400016247,
        'k_real': 4,
        'a': 4,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400016247, k=4, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400016247, k=4: {e}')

# Test 19: p=1000010400016799, k_real=6
try:
    E = EllipticCurve([6, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400016799,
        'k_real': 6,
        'a': 6,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400016799, k=6, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400016799, k=6: {e}')

# Test 20: p=1000010400018521, k_real=2
try:
    E = EllipticCurve([2, 1])
    rank = E.rank()
    L_value = E.lseries().L_ratio()  # L(E,1)/Omega
    
    result = {
        'p': 1000010400018521,
        'k_real': 2,
        'a': 2,
        'b': 1,
        'rank': rank,
        'L_ratio': float(L_value)
    }
    results.append(result)
    print(f'✓ p=1000010400018521, k=2, rank={rank}, L/Ω={L_value}')
except Exception as e:
    print(f'✗ p=1000010400018521, k=2: {e}')

# Save results
import json
with open('bsd_sage_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Analysis
if len(results) > 0:
    print()
    print('=' * 80)
    print('ANÁLISE: rank vs k_real')
    print('=' * 80)
    print()
    
    ranks = [r['rank'] for r in results]
    k_vals = [r['k_real'] for r in results]
    
    # Correlação
    from scipy.stats import pearsonr
    r_corr, p_val = pearsonr(ranks, k_vals)
    
    print(f'Correlação rank vs k_real: r={r_corr:.4f}, p={p_val:.2e}')
    print()
    
    # Casos individuais
    print('Casos individuais:')
    for res in results[:10]:
        print(f"  k={res['k_real']:2d} → rank={res['rank']}  (L/Ω={res['L_ratio']:.6f})")
    
    print()
    
    if abs(r_corr) > 0.9:
        print('🏆 CORRELAÇÃO FORTÍSSIMA! rank ≈ k_real')
    elif abs(r_corr) > 0.7:
        print('✓ CORRELAÇÃO FORTE')
    else:
        print('⚠ Correlação moderada')
