#!/usr/bin/env python3
"""
BSD TEST - Powers of 2 Hypothesis
Test if rank(k=2^n) = n - 1 for k=2,4,8,16,32,64
"""

import cypari2
import json
import time
from collections import defaultdict

pari = cypari2.Pari()

def is_prime(n):
    return bool(pari.isprime(n))

def calc_k_real(p):
    """Calculate k_real(p) = log2((p XOR (p+2)) + 2) - 1"""
    if p % 2 == 0:
        return None
    xor = p ^ (p + 2)
    val = xor + 2
    if val & (val - 1) != 0:
        return None
    k = val.bit_length() - 2
    return k if 0 <= k < 25 else None

def get_rank(p, k):
    """Get rank of E_p: y^2 = x^3 + (p mod k^2)*x + k"""
    try:
        a = int(p % (k * k)) if k > 0 else 0
        b = k
        E = pari.ellinit([0, 0, 0, a, b])
        rank_data = pari.ellanalyticrank(E)
        return int(rank_data[0])
    except Exception as e:
        return None

def generate_twin_primes_for_k(target_k, max_search=100_000_000):
    """Generate twin primes with specific k_real value"""
    twins = []
    p = 3
    checked = 0
    
    print(f"  Searching for k={target_k} primes (target: 100 curves)...")
    
    while len(twins) < 100 and checked < max_search:
        if is_prime(p) and is_prime(p + 2):
            k = calc_k_real(p)
            if k == target_k:
                twins.append(p)
                if len(twins) % 10 == 0:
                    print(f"    Found {len(twins)}/100 (p={p:,})")
        p += 2
        checked += 1
        
        if checked % 1_000_000 == 0:
            print(f"    Checked {checked:,} candidates, found {len(twins)} twins")
    
    return twins

def main():
    print("="*80)
    print("BSD TEST - POWERS OF 2 HYPOTHESIS")
    print("="*80)
    print()
    print("Testing: rank(k=2^n) = n - 1")
    print()
    
    # Target k values (powers of 2)
    target_ks = [2, 4, 8, 16, 32]  # 2^1, 2^2, 2^3, 2^4, 2^5
    
    results = []
    
    for target_k in target_ks:
        n = target_k.bit_length() - 1  # log2(k)
        expected_rank = n - 1 if n > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"Testing k={target_k} (2^{n})")
        print(f"Expected rank: {expected_rank} (if hypothesis holds)")
        print(f"{'='*80}")
        
        t0 = time.time()
        twins = generate_twin_primes_for_k(target_k)
        t1 = time.time()
        
        if len(twins) == 0:
            print(f"  ❌ No twin primes found with k={target_k}")
            results.append({
                'k': target_k,
                'n': n,
                'expected_rank': expected_rank,
                'found_primes': 0,
                'ranks': []
            })
            continue
        
        print(f"  ✓ Found {len(twins)} twin primes in {t1-t0:.1f}s")
        print(f"  Computing ranks...")
        
        ranks = []
        for i, p in enumerate(twins):
            rank = get_rank(p, target_k)
            if rank is not None:
                ranks.append(rank)
            
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(twins)}")
        
        t2 = time.time()
        
        # Analyze
        if len(ranks) == 0:
            print(f"  ❌ No ranks computed")
            continue
        
        rank_avg = sum(ranks) / len(ranks)
        rank_std = (sum((r - rank_avg)**2 for r in ranks) / len(ranks))**0.5
        
        from collections import Counter
        rank_dist = Counter(ranks)
        
        print()
        print(f"  RESULTS:")
        print(f"    Curves tested: {len(ranks)}")
        print(f"    Rank average: {rank_avg:.3f}")
        print(f"    Rank std: {rank_std:.3f}")
        print(f"    Distribution: {dict(rank_dist)}")
        
        # Check hypothesis
        if rank_std == 0.0:
            print(f"    ✓✓✓ DETERMINISTIC: All curves have rank={int(rank_avg)}")
            
            if int(rank_avg) == expected_rank:
                print(f"    🎯 HYPOTHESIS CONFIRMED: rank = {n} - 1 = {expected_rank}")
            else:
                print(f"    ⚠️  HYPOTHESIS FAILED: expected {expected_rank}, got {int(rank_avg)}")
        else:
            print(f"    ⚠️  NOT deterministic (std={rank_std:.3f})")
        
        print(f"  Time: {t2-t1:.1f}s")
        
        results.append({
            'k': target_k,
            'n': n,
            'n_curves': len(ranks),
            'expected_rank': expected_rank,
            'rank_avg': rank_avg,
            'rank_std': rank_std,
            'rank_distribution': dict(rank_dist),
            'sample_primes': twins[:10],
            'all_ranks': ranks
        })
    
    # Final summary
    print()
    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print()
    print(f"{'k':>4} | {'2^n':>4} | {'n':>3} | {'Expected':>8} | {'Observed':>8} | {'std':>6} | Match?")
    print("-"*70)
    
    hypothesis_confirmed = 0
    hypothesis_tested = 0
    
    for r in results:
        if r.get('n_curves', 0) == 0:
            continue
        
        hypothesis_tested += 1
        k = r['k']
        n = r['n']
        exp = r['expected_rank']
        obs = r['rank_avg']
        std = r['rank_std']
        
        match = ""
        if std < 0.01 and abs(obs - exp) < 0.01:
            match = "✓✓✓"
            hypothesis_confirmed += 1
        elif std < 0.01:
            match = f"✗ (got {int(obs)})"
        else:
            match = f"~ (std={std:.2f})"
        
        print(f"{k:4d} | 2^{n}  | {n:3d} | {exp:8d} | {obs:8.3f} | {std:6.3f} | {match}")
    
    print()
    print(f"Hypothesis confirmed: {hypothesis_confirmed}/{hypothesis_tested}")
    print()
    
    if hypothesis_confirmed == hypothesis_tested and hypothesis_tested >= 3:
        print("🎯 HYPOTHESIS CONFIRMED!")
        print()
        print("   rank(E_p) = log₂(k_real(p)) - 1")
        print()
        print("   for k = 2^n (powers of 2)")
        print()
    elif hypothesis_confirmed >= hypothesis_tested * 0.8:
        print("📊 HYPOTHESIS MOSTLY CONFIRMED")
        print("   (some edge cases need investigation)")
        print()
    else:
        print("❌ HYPOTHESIS NOT CONFIRMED")
        print("   Pattern more complex than n-1")
        print()
    
    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hypothesis': 'rank(k=2^n) = n - 1',
        'results': results,
        'confirmed': hypothesis_confirmed,
        'tested': hypothesis_tested
    }
    
    with open('bsd_powers_of_2_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Results saved to bsd_powers_of_2_test.json")
    print()

if __name__ == '__main__':
    main()
