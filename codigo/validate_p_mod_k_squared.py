#!/usr/bin/env python3
"""
Validação da Propriedade p mod k²
==================================

OBJETIVO CRÍTICO: Validar empiricamente que para twin primes com k_real = k = 2^n,
a propriedade p == k² - 1 (mod k²) é VERDADEIRA.

Esta é a BASE matemática da conexão XOR → BSD!
Se falhar, toda a teoria precisa revisão! [ALERT]

Dataset: 1,004,800,004 twin primes do results.csv
Estratégia: Processar em chunks para economizar memória
"""

import sys
import csv
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

def calc_k_real(p: int) -> int:
    """Calcula k_real de um twin prime"""
    if p % 2 == 0:
        return -1
    xor_val = p ^ (p + 2)
    target = xor_val + 2
    if target & (target - 1) != 0:
        return -1
    k = target.bit_length() - 2
    return k if 0 <= k < 64 else -1

def is_power_of_2(n: int) -> bool:
    """Verifica se n é potência de 2"""
    return n > 0 and (n & (n - 1)) == 0

def validate_chunk(chunk: List[Tuple[int, int]], verbose: bool = False) -> Dict:
    """
    Valida um chunk de twin primes
    
    Retorna:
    - counts: dict com contagem por k
    - residues: dict[k] -> set de resíduos (p mod k²)
    - exceptions: lista de (p, k, expected, observed)
    """
    counts = defaultdict(int)
    residues = defaultdict(set)
    exceptions = []
    
    for p, k_real in chunk:
        counts[k_real] += 1
        
        # Para k = 2^n, verifica propriedade p mod k²
        k = k_real
        if is_power_of_2(k) and k > 0:
            k_squared = k * k
            expected_residue = k_squared - 1
            observed_residue = p % k_squared
            
            residues[k].add(observed_residue)
            
            if observed_residue != expected_residue:
                exceptions.append({
                    'p': p,
                    'k': k,
                    'k_squared': k_squared,
                    'expected': expected_residue,
                    'observed': observed_residue,
                    'diff': abs(observed_residue - expected_residue)
                })
                
                if verbose and len(exceptions) <= 10:
                    print(f"  [WARNING]  EXCEÇÃO: p={p}, k={k}, esperado={expected_residue}, observado={observed_residue}")
    
    return {
        'counts': counts,
        'residues': residues,
        'exceptions': exceptions
    }

def main():
    parser = argparse.ArgumentParser(
        description='Valida propriedade p == k²-1 (mod k²) em twin primes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--file', type=str, 
                        default='/home/thlinux/relacionalidadegeral/codigo/binario/results.csv',
                        help='Arquivo CSV de twin primes')
    parser.add_argument('--max-lines', type=int, default=1_000_000,
                        help='Máximo de linhas a processar (0 = todas)')
    parser.add_argument('--chunk-size', type=int, default=100_000,
                        help='Tamanho do chunk para processamento')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostra primeiras exceções')
    parser.add_argument('--k-values', type=int, nargs='+',
                        help='Valores específicos de k para focar (ex: 2 4 8 16)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VALIDAÇÃO: p == k² - 1 (mod k²) para k = 2^n")
    print("=" * 80)
    print(f"\n[FOLDER] Arquivo: {args.file}")
    print(f"[DATA] Limite: {args.max_lines:,} linhas" if args.max_lines > 0 else "[DATA] Processando TUDO")
    print(f"[CFG] Chunk size: {args.chunk_size:,}")
    if args.k_values:
        print(f"[TARGET] Focando em k = {args.k_values}")
    
    # Acumuladores globais
    global_counts = defaultdict(int)
    global_residues = defaultdict(set)
    global_exceptions = []
    
    # Processamento
    chunk = []
    lines_processed = 0
    start_time = time.time()
    
    print("\n[START] Iniciando processamento...")
    
    try:
        with open(args.file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if args.max_lines > 0 and lines_processed >= args.max_lines:
                    break
                
                try:
                    p = int(row['p'])
                    k_real = int(row['k_real'])
                    
                    # Filtro por k específico (se solicitado)
                    if args.k_values and k_real not in args.k_values:
                        continue
                    
                    chunk.append((p, k_real))
                    
                    # Processa chunk quando atingir tamanho
                    if len(chunk) >= args.chunk_size:
                        result = validate_chunk(chunk, verbose=args.verbose)
                        
                        # Acumula resultados
                        for k, count in result['counts'].items():
                            global_counts[k] += count
                        for k, residue_set in result['residues'].items():
                            global_residues[k].update(residue_set)
                        global_exceptions.extend(result['exceptions'])
                        
                        lines_processed += len(chunk)
                        chunk = []
                        
                        # Progress
                        if lines_processed % 1_000_000 == 0:
                            elapsed = time.time() - start_time
                            rate = lines_processed / elapsed
                            print(f"   Processadas: {lines_processed:,} linhas ({rate:.0f} lines/s)")
                
                except (ValueError, KeyError) as e:
                    if args.verbose:
                        print(f"  [WARNING]  Linha inválida: {e}")
                    continue
            
            # Processa último chunk
            if chunk:
                result = validate_chunk(chunk, verbose=args.verbose)
                for k, count in result['counts'].items():
                    global_counts[k] += count
                for k, residue_set in result['residues'].items():
                    global_residues[k].update(residue_set)
                global_exceptions.extend(result['exceptions'])
                lines_processed += len(chunk)
    
    except FileNotFoundError:
        print(f"\n[FAIL] ERRO: Arquivo não encontrado: {args.file}")
        return 1
    except KeyboardInterrupt:
        print(f"\n\n[WARNING]  Interrompido pelo usuário após {lines_processed:,} linhas")
    
    elapsed = time.time() - start_time
    
    # ==================== RELATÓRIO ====================
    
    print("\n" + "=" * 80)
    print("RESULTADOS DA VALIDAÇÃO")
    print("=" * 80)
    
    print(f"\n[TIME]  Tempo: {elapsed:.2f}s ({lines_processed/elapsed:.0f} lines/s)")
    print(f"[DATA] Linhas processadas: {lines_processed:,}")
    print(f"[NUM] Valores de k encontrados: {len(global_counts)}")
    
    # Distribuição por k
    print("\n" + "-" * 80)
    print("DISTRIBUIÇÃO POR k:")
    print("-" * 80)
    print(f"{'k':<6} {'Count':<12} {'% Total':<10} {'Power of 2?':<12} {'Resíduos únicos':<20}")
    print("-" * 80)
    
    total = sum(global_counts.values())
    for k in sorted(global_counts.keys()):
        count = global_counts[k]
        pct = 100 * count / total
        is_pow2 = "[OK] SIM" if is_power_of_2(k) else "[FAIL] NÃO"
        n_residues = len(global_residues.get(k, set()))
        
        print(f"{k:<6} {count:<12,} {pct:<10.2f} {is_pow2:<12} {n_residues:<20}")
    
    # Análise de k = 2^n
    print("\n" + "=" * 80)
    print("ANÁLISE DE k = 2^n (POTÊNCIAS DE 2)")
    print("=" * 80)
    
    powers_of_2 = [k for k in sorted(global_counts.keys()) if is_power_of_2(k)]
    
    if not powers_of_2:
        print("\n[WARNING]  Nenhum k = 2^n encontrado no dataset!")
    else:
        print(f"\n[OK] Encontrados k = 2^n: {powers_of_2}")
        print("\n" + "-" * 80)
        print("VALIDAÇÃO DA PROPRIEDADE p == k² - 1 (mod k²):")
        print("-" * 80)
        print(f"{'k':<6} {'n':<6} {'k²':<12} {'Esperado (k²-1)':<18} {'Resíduos observados':<25} {'Status':<10}")
        print("-" * 80)
        
        all_valid = True
        
        for k in powers_of_2:
            n = k.bit_length() - 1
            k_sq = k * k
            expected = k_sq - 1
            observed_residues = global_residues.get(k, set())
            
            if len(observed_residues) == 1 and expected in observed_residues:
                status = "[OK] VÁLIDO"
            else:
                status = "[FAIL] FALHOU"
                all_valid = False
            
            residues_str = str(sorted(list(observed_residues))[:5])
            if len(observed_residues) > 5:
                residues_str += "..."
            
            print(f"{k:<6} {n:<6} {k_sq:<12,} {expected:<18,} {residues_str:<25} {status:<10}")
        
        # VEREDITO FINAL
        print("\n" + "=" * 80)
        print("[TARGET] VEREDITO FINAL")
        print("=" * 80)
        
        if all_valid and not global_exceptions:
            print("\n[OK][OK][OK] PROPRIEDADE CONFIRMADA! [OK][OK][OK]")
            print(f"\nPara TODOS os {total:,} twin primes com k = 2^n:")
            print(f"   p == k² - 1 (mod k²)")
            print(f"\nZERO exceções encontradas!")
            print(f"\nA BASE MATEMÁTICA da conexão XOR → BSD está VALIDADA! [SUCCESS]")
            return 0
        else:
            print("\n[FAIL][FAIL][FAIL] PROPRIEDADE FALHOU! [FAIL][FAIL][FAIL]")
            print(f"\nEncontradas {len(global_exceptions):,} exceções!")
            print(f"\nA teoria PRECISA REVISÃO! [ALERT]")
            
            if global_exceptions and args.verbose:
                print("\n" + "-" * 80)
                print("PRIMEIRAS 20 EXCEÇÕES:")
                print("-" * 80)
                for exc in global_exceptions[:20]:
                    print(f"  p={exc['p']:,}, k={exc['k']}, esperado={exc['expected']:,}, observado={exc['observed']:,}, diff={exc['diff']:,}")
            
            return 1

if __name__ == '__main__':
    sys.exit(main())
