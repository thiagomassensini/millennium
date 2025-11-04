#!/usr/bin/env python3
"""
RIEMANN HYPOTHESIS + TWIN PRIME XOR STRUCTURE
==============================================

Testa se a "memória sistêmica" do XOR aparece nos zeros da função zeta.

Hipóteses a testar:
1. Espaçamento entre zeros correlaciona com P(k) = 2^(-k)
2. k_real(p) prediz altura de zeros
3. Zeros aparecem em harmônicos de potências de 2
4. Estrutura binária do XOR está codificada na linha crítica Re(s)=1/2

Método:
- Calcular 10.000 primeiros zeros não-triviais de ζ(s)
- Usar mpmath com precisão de 100 decimais
- Comparar distribuição de gaps com distribuição de k_real
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    import mpmath
    mpmath.mp.dps = 50  # 50 casas decimais (suficiente para análise)
except ImportError:
    print("ERRO: Instale mpmath primeiro: pip install mpmath")
    exit(1)

import numpy as np
from scipy import stats


def calculate_zeta_zeros(n_zeros=10000, start_t=0):
    """
    Calcula os primeiros n_zeros zeros não-triviais de ζ(s).
    
    Zeros estão na linha crítica: ζ(1/2 + it) = 0
    
    Retorna lista de valores t (parte imaginária).
    """
    print(f"Calculando {n_zeros} zeros de ζ(s)...")
    print(f"Precisão: {mpmath.mp.dps} decimais")
    print()
    
    zeros = []
    t = start_t
    
    start_time = time.time()
    
    for i in range(n_zeros):
        # Encontrar próximo zero
        t = mpmath.zetazero(i + 1)
        zeros.append(float(t.imag))
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_zeros - i - 1) / rate
            
            print(f"  [{i+1:5d}/{n_zeros}] t = {zeros[-1]:15.6f} | "
                  f"Taxa: {rate:.1f} zeros/s | ETA: {eta:.0f}s")
    
    elapsed = time.time() - start_time
    print()
    print(f"✓ {n_zeros} zeros calculados em {elapsed:.1f}s")
    print(f"  Primeiro zero: t = {zeros[0]:.6f}")
    print(f"  Último zero:   t = {zeros[-1]:.6f}")
    
    return zeros


def analyze_zero_gaps(zeros):
    """
    Analisa espaçamento entre zeros consecutivos.
    
    Compara com P(k) = 2^(-k) dos twin primes.
    """
    print()
    print("="*80)
    print("ANÁLISE DE GAPS ENTRE ZEROS")
    print("="*80)
    print()
    
    gaps = [zeros[i+1] - zeros[i] for i in range(len(zeros)-1)]
    
    print(f"Total de gaps: {len(gaps)}")
    print(f"Gap mínimo:    {min(gaps):.6f}")
    print(f"Gap máximo:    {max(gaps):.6f}")
    print(f"Gap médio:     {np.mean(gaps):.6f}")
    print(f"Gap mediano:   {np.median(gaps):.6f}")
    print(f"Desvio padrão: {np.std(gaps):.6f}")
    print()
    
    # Distribuição de gaps
    print("Distribuição de gaps:")
    print()
    
    # Dividir em bins (como k_real)
    # k_real varia de 2 a ~20 para twin primes
    # Vamos mapear gaps para "níveis de energia"
    
    # Gap médio teórico entre zeros: 2π/log(t/2π) (fórmula de Riemann-von Mangoldt)
    t_mean = np.mean(zeros)
    gap_theory = 2 * np.pi / np.log(t_mean / (2 * np.pi))
    
    print(f"Gap teórico médio (RvM): {gap_theory:.6f}")
    print()
    
    # Normalizar gaps pelo valor teórico
    normalized_gaps = [g / gap_theory for g in gaps]
    
    # Mapear para "níveis" (análogo a k_real)
    # Nível k se gap está próximo de 2^k * gap_base
    levels = []
    for g in normalized_gaps:
        if g <= 0:
            continue
        level = np.log2(g)
        levels.append(level)
    
    # Distribuição de níveis
    level_counter = Counter([round(l) for l in levels])
    
    print("Distribuição de níveis (análogo a k_real):")
    print()
    print("Nível | Contagem | Proporção | Teórico 2^(-k)")
    print("------|----------|-----------|----------------")
    
    total = len(levels)
    for level in sorted(level_counter.keys()):
        count = level_counter[level]
        prop = count / total
        
        # Para níveis negativos, usar 2^|k|
        if level >= 0:
            theory = 2**(-level) if level <= 10 else 0
        else:
            theory = 2**(abs(level))
        
        print(f"{level:5d} | {count:8d} | {prop:8.4f}  | {theory:8.4f}")
    
    print()
    
    # Estatísticas
    corr_theory = []
    corr_obs = []
    
    for level in range(-2, 6):  # Níveis -2 a 5
        if level in level_counter:
            count = level_counter[level]
            prop = count / total
            
            if level >= 0:
                theory = 2**(-level)
            else:
                theory = 2**(abs(level))
            
            corr_obs.append(prop)
            corr_theory.append(theory)
    
    if corr_obs and corr_theory:
        correlation = np.corrcoef(corr_obs, corr_theory)[0, 1]
        print(f"Correlação com 2^(-k): r = {correlation:.4f}")
    
    return {
        'gaps': gaps,
        'normalized_gaps': normalized_gaps,
        'levels': levels,
        'level_distribution': dict(level_counter),
        'statistics': {
            'min': min(gaps),
            'max': max(gaps),
            'mean': float(np.mean(gaps)),
            'median': float(np.median(gaps)),
            'std': float(np.std(gaps)),
            'theoretical_mean': gap_theory
        }
    }


def test_power_of_2_harmonics(zeros):
    """
    Testa se zeros aparecem em harmônicos de potências de 2.
    
    Se XOR tem "memória sistêmica", zeros deveriam aparecer em:
    t ≈ 2^k * constante
    """
    print()
    print("="*80)
    print("TESTE DE HARMÔNICOS (POTÊNCIAS DE 2)")
    print("="*80)
    print()
    
    # Para cada zero, calcular log2(t)
    log2_zeros = [np.log2(t) for t in zeros if t > 0]
    
    # Ver se há picos em inteiros (potências de 2)
    fractional_parts = [x - int(x) for x in log2_zeros]
    
    # Histograma
    hist, bins = np.histogram(fractional_parts, bins=20, range=(0, 1))
    
    print("Distribuição de log₂(t) mod 1:")
    print()
    print("Bin   | Contagem | Esperado | Diferença")
    print("------|----------|----------|----------")
    
    expected = len(fractional_parts) / 20
    
    for i, count in enumerate(hist):
        bin_center = (bins[i] + bins[i+1]) / 2
        diff = count - expected
        
        print(f"{bin_center:.2f} | {count:8d} | {expected:8.1f} | {diff:+8.1f}")
    
    # Teste qui-quadrado
    chi2, p_value = stats.chisquare(hist)
    
    print()
    print(f"Teste χ²: {chi2:.2f}, p-value = {p_value:.6f}")
    
    if p_value > 0.05:
        print("  → Distribuição é UNIFORME (não há picos em potências de 2)")
    else:
        print("  → Distribuição NÃO é uniforme (possível estrutura!)")
    
    return {
        'log2_zeros': log2_zeros,
        'fractional_parts': fractional_parts,
        'histogram': hist.tolist(),
        'bins': bins.tolist(),
        'chi2_test': {
            'statistic': float(chi2),
            'p_value': float(p_value)
        }
    }


def correlate_with_twin_primes(zeros, twin_primes_file='results.csv'):
    """
    Correlaciona zeros com k_real dos twin primes.
    
    Hipótese: k_real(p) prediz altura de zeros?
    """
    print()
    print("="*80)
    print("CORRELAÇÃO COM TWIN PRIMES")
    print("="*80)
    print()
    
    # Carregar distribuição P(k) dos twin primes
    # (já sabemos: P(k) = 2^(-k))
    
    P_k = {
        2: 0.5080,
        3: 0.2440,
        4: 0.1248,
        5: 0.0620,
        6: 0.0310,
        7: 0.0155,
        8: 0.0077,
        9: 0.0039,
        10: 0.0019
    }
    
    print("Distribuição P(k) dos twin primes:")
    for k, prob in P_k.items():
        print(f"  k={k:2d}: P(k) = {prob:.4f} (teórico: {2**(-k):.4f})")
    
    print()
    
    # Ideia: se há conexão, a "energia" dos zeros deveria seguir P(k)
    
    # Mapear zeros para "níveis de energia" k
    # Usar escala logarítmica: k ~ log2(t)
    
    zero_levels = {}
    for t in zeros:
        if t <= 0:
            continue
        
        # Nível baseado em altura
        level = int(np.log2(t)) if t >= 1 else 0
        
        if level not in zero_levels:
            zero_levels[level] = 0
        zero_levels[level] += 1
    
    print("Distribuição de zeros por nível:")
    print()
    print("Nível log₂(t) | Zeros | Proporção")
    print("--------------|-------|----------")
    
    total_zeros = len([z for z in zeros if z > 0])
    
    for level in sorted(zero_levels.keys()):
        count = zero_levels[level]
        prop = count / total_zeros
        print(f"{level:13d} | {count:5d} | {prop:8.4f}")
    
    print()
    print("Comparação: P(k) decresce exponencialmente (2^(-k))")
    print("             Zeros por nível aumentam com altura (crescimento esperado)")
    print()
    print("Conexão direta é IMPROVÁVEL (escalas diferentes)")
    print("Mas: a estrutura BINÁRIA pode estar codificada de outra forma!")
    
    return {
        'P_k_twin_primes': P_k,
        'zero_levels': zero_levels
    }


def main():
    print("="*80)
    print("RIEMANN HYPOTHESIS + TWIN PRIME XOR STRUCTURE")
    print("="*80)
    print()
    print("Testando se 'memória sistêmica' do XOR aparece nos zeros de ζ(s)")
    print()
    
    # Calcular zeros
    n_zeros = 1000  # Reduzido para teste inicial
    zeros = calculate_zeta_zeros(n_zeros)
    
    # Salvar zeros
    results = {
        'n_zeros': n_zeros,
        'zeros': zeros,
        'first_zero': zeros[0],
        'last_zero': zeros[-1]
    }
    
    # Análise de gaps
    gap_analysis = analyze_zero_gaps(zeros)
    results['gap_analysis'] = gap_analysis
    
    # Teste de harmônicos
    harmonic_analysis = test_power_of_2_harmonics(zeros)
    results['harmonic_analysis'] = harmonic_analysis
    
    # Correlação com twin primes
    twin_prime_corr = correlate_with_twin_primes(zeros)
    results['twin_prime_correlation'] = twin_prime_corr
    
    # Salvar resultados
    output_file = Path(__file__).parent / 'riemann_zeros_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("="*80)
    print("RESULTADOS SALVOS")
    print("="*80)
    print(f"Arquivo: {output_file}")
    print()
    
    print("="*80)
    print("PRÓXIMOS PASSOS")
    print("="*80)
    print()
    print("1. Analisar espectro de Fourier dos gaps")
    print("2. Testar transformada de Mellin")
    print("3. Verificar teoria de números p-ádicos")
    print("4. Calcular pair correlation de Montgomery")
    print("5. Comparar com GUE (Gaussian Unitary Ensemble)")
    print()


if __name__ == '__main__':
    main()
