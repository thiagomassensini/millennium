#!/usr/bin/env python3
"""
BSD MASSIVE TEST - 10,000 curves
Test Fibonacci hypothesis: rank(E_p) = F(k_real(p) - 2)
"""

import cypari2
import json
import time
from collections import defaultdict
import sys

pari = cypari2.Pari()

def is_prime(n):
    """Simple primality test using PARI"""
    return bool(pari.isprime(n))

def calc_k_real(p):
    """Calculate k_real(p) = log2((p XOR (p+2)) + 2) - 1"""
    if p % 2 == 0:
        return None
    xor = p ^ (p + 2)
    val = xor + 2
    # Check if power of 2
    if val & (val - 1) != 0:
        return None
    k = val.bit_length() - 2
    return k if 0 <= k < 25 else None

def get_rank(p, k):
    """
    Get rank of E_p: y^2 = x^3 + (p mod k^2)*x + k
    Using PARI's ellanalyticrank
    """
    try:
        a = int(p % (k * k)) if k > 0 else 0
        b = k
        E = pari.ellinit([0, 0, 0, a, b])
        rank_data = pari.ellanalyticrank(E)
        return int(rank_data[0])
    except Exception as e:
        return None

def generate_twin_primes(limit):
    """Generate twin primes < limit"""
    twins = []
    p = 3
    while p < limit:
        if is_prime(p) and is_prime(p + 2):
            twins.append(p)
        p += 2
        if len(twins) >= 10000:  # Stop at 10k
            break
    return twins

def fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

def main():
    print("="*80)
    print("BSD MASSIVE TEST - 10,000 CURVES")
    print("="*80)
    print()
    print("Generating twin primes < 1,000,000...")
    
    t0 = time.time()
    twins = generate_twin_primes(1_000_000)
    t1 = time.time()
    
    print(f"[OK] Generated {len(twins)} twin primes in {t1-t0:.1f}s")
    print()
    print("Computing k_real and ranks...")
    print()
    
    # Group by k_real
    k_data = defaultdict(list)
    
    processed = 0
    skipped = 0
    
    for i, p in enumerate(twins):
        k = calc_k_real(p)
        if k is None or k < 2 or k > 10:
            skipped += 1
            continue
        
        rank = get_rank(p, k)
        if rank is not None:
            k_data[k].append(rank)
            processed += 1
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(twins) - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{len(twins)} ({100*(i+1)/len(twins):.1f}%) | "
                  f"Processed: {processed} | Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min")
    
    t2 = time.time()
    print()
    print(f"[OK] Processed {processed} curves in {t2-t1:.1f}s")
    print(f"  Skipped: {skipped}")
    print()
    
    # Analyze results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    
    print(f"{'k':>3} | {'n':>6} | {'rank_avg':>9} | {'rank_std':>9} | {'F(k-2)':>7} | {'k-2':>5} | Match")
    print("-"*85)
    
    results = []
    for k in sorted(k_data.keys()):
        ranks = k_data[k]
        avg = sum(ranks) / len(ranks)
        std = (sum((r - avg)**2 for r in ranks) / len(ranks))**0.5
        
        fib_k_minus_2 = fibonacci(k - 2)
        k_minus_2 = k - 2
        
        # Check matches
        match = ""
        if abs(avg - fib_k_minus_2) < 0.3:
            match = "F(k-2) [OK][OK][OK]"
        elif abs(avg - k_minus_2) < 0.3:
            match = "k-2 [OK]"
        elif abs(avg - fib_k_minus_2) < 0.5:
            match = "F(k-2) ~"
        
        print(f"{k:3d} | {len(ranks):6d} | {avg:9.3f} | {std:9.3f} | {fib_k_minus_2:7d} | {k_minus_2:5d} | {match}")
        
        results.append({
            'k': k,
            'n_curves': len(ranks),
            'rank_avg': avg,
            'rank_std': std,
            'F_k_minus_2': fib_k_minus_2,
            'k_minus_2': k_minus_2,
            'all_ranks': ranks[:100]  # Save first 100 for verification
        })
    
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    
    # Test Fibonacci hypothesis
    fib_matches = sum(1 for r in results if abs(r['rank_avg'] - r['F_k_minus_2']) < 0.3)
    linear_matches = sum(1 for r in results if abs(r['rank_avg'] - r['k_minus_2']) < 0.3)
    
    print(f"Fibonacci F(k-2) matches: {fib_matches}/{len(results)}")
    print(f"Linear (k-2) matches: {linear_matches}/{len(results)}")
    print()
    
    if fib_matches >= len(results) * 0.7:
        print("[TARGET] FIBONACCI PATTERN CONFIRMED!")
        print("   rank(E_p) ≈ F(k_real(p) - 2)")
        print()
        print("   This is a MAJOR result connecting:")
        print("   - Twin prime distribution")
        print("   - Fibonacci sequence")
        print("   - BSD conjecture ranks")
        print()
        print("   [WARNING]  PAPER READY FOR SUBMISSION")
    elif linear_matches >= len(results) * 0.7:
        print("[DATA] LINEAR PATTERN CONFIRMED!")
        print("   rank(E_p) ≈ k_real(p) - 2")
        print()
        print("   Still publishable - linear relationship found")
    else:
        print("🤔 PATTERN UNCLEAR")
        print("   More investigation needed")
    
    print()
    print(f"Total time: {time.time() - t0:.1f}s")
    print()
    
    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_curves': processed,
        'total_time_s': time.time() - t0,
        'results': results,
        'fibonacci_matches': fib_matches,
        'linear_matches': linear_matches
    }
    
    with open('bsd_massive_test_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("[OK] Results saved to bsd_massive_test_results.json")
    print()

if __name__ == '__main__':
    main()
