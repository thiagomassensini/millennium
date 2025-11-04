#!/usr/bin/env python3
"""
ANÁLISE PROFUNDA: RIEMANN + XOR
================================

Analisar espectro de Fourier dos gaps entre zeros.
Ver se há frequências relacionadas a potências de 2.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import matplotlib.pyplot as plt
from scipy import fft, stats
from pathlib import Path


def load_results():
    """Carregar resultados anteriores."""
    file_path = Path(__file__).parent / 'riemann_zeros_analysis.json'
    with open(file_path) as f:
        return json.load(f)


def fourier_analysis(gaps):
    """
    Análise de Fourier dos gaps.
    
    Se há estrutura de potências de 2, deve aparecer no espectro.
    """
    print("="*80)
    print("ANÁLISE DE FOURIER DOS GAPS")
    print("="*80)
    print()
    
    # FFT
    n = len(gaps)
    gaps_array = np.array(gaps)
    
    # Remover média (DC component)
    gaps_centered = gaps_array - np.mean(gaps_array)
    
    # FFT
    fft_result = fft.fft(gaps_centered)
    frequencies = fft.fftfreq(n)
    
    # Power spectrum (magnitude ao quadrado)
    power = np.abs(fft_result)**2
    
    # Apenas frequências positivas
    positive_freq_idx = frequencies > 0
    freq_positive = frequencies[positive_freq_idx]
    power_positive = power[positive_freq_idx]
    
    # Ordenar por potência
    sorted_idx = np.argsort(power_positive)[::-1]
    
    print("Top 20 frequências com maior potência:")
    print()
    print("Rank | Frequência  | Período   | Potência    | log₂(período)")
    print("-----|-------------|-----------|-------------|---------------")
    
    for i, idx in enumerate(sorted_idx[:20]):
        freq = freq_positive[idx]
        period = 1/freq if freq != 0 else np.inf
        pow = power_positive[idx]
        log2_period = np.log2(period) if period != np.inf else np.inf
        
        print(f"{i+1:4d} | {freq:11.6f} | {period:9.4f} | {pow:11.2e} | {log2_period:13.6f}")
    
    print()
    
    # Verificar se períodos são potências de 2
    print("Verificando períodos próximos a potências de 2:")
    print()
    
    for k in range(1, 11):
        target_period = 2**k
        
        # Encontrar frequências próximas
        target_freq = 1/target_period
        close_idx = np.abs(freq_positive - target_freq) < 0.01
        
        if np.any(close_idx):
            close_freqs = freq_positive[close_idx]
            close_powers = power_positive[close_idx]
            
            for f, p in zip(close_freqs, close_powers):
                actual_period = 1/f
                print(f"  k={k} (2^{k}={target_period:4d}): "
                      f"período={actual_period:7.2f} | potência={p:10.2e}")
    
    # Plotar espectro
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Espectro completo
    ax1.semilogy(freq_positive, power_positive, 'b-', alpha=0.7)
    ax1.set_xlabel('Frequência')
    ax1.set_ylabel('Potência (log scale)')
    ax1.set_title('Espectro de Fourier dos Gaps entre Zeros de Riemann')
    ax1.grid(True, alpha=0.3)
    
    # Zoom nas baixas frequências
    low_freq_idx = freq_positive < 0.1
    ax2.semilogy(freq_positive[low_freq_idx], power_positive[low_freq_idx], 'r-', alpha=0.7)
    ax2.set_xlabel('Frequência')
    ax2.set_ylabel('Potência (log scale)')
    ax2.set_title('Zoom: Baixas Frequências')
    ax2.grid(True, alpha=0.3)
    
    # Marcar potências de 2
    for k in range(1, 8):
        freq_mark = 1/(2**k)
        if freq_mark < 0.1:
            ax2.axvline(freq_mark, color='green', linestyle='--', alpha=0.5, 
                       label=f'2^{k}' if k <= 3 else '')
    
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('riemann_fourier_spectrum.png', dpi=150)
    print()
    print("✓ Gráfico salvo: riemann_fourier_spectrum.png")
    
    return {
        'frequencies': freq_positive.tolist()[:100],
        'power': power_positive.tolist()[:100],
        'top_20_indices': sorted_idx[:20].tolist()
    }


def pair_correlation(zeros):
    """
    Pair correlation de Montgomery.
    
    Compara espaçamento normalizado entre zeros com distribuição esperada (GUE).
    """
    print()
    print("="*80)
    print("PAIR CORRELATION (MONTGOMERY)")
    print("="*80)
    print()
    
    # Normalizar gaps pelo espaçamento médio local
    zeros_array = np.array(zeros)
    n = len(zeros_array)
    
    # Espaçamento médio local: d(t) = 2π/log(t/2π)
    mean_spacing = [2*np.pi / np.log(t/(2*np.pi)) for t in zeros_array]
    
    # Calcular R(s) = pair correlation function
    # R(s) mede probabilidade de ter dois zeros separados por distância s (normalizada)
    
    max_s = 10.0  # Até s=10
    s_bins = np.linspace(0, max_s, 100)
    R_observed = np.zeros(len(s_bins) - 1)
    
    # Calcular gaps normalizados
    for i in range(n-1):
        gap = zeros_array[i+1] - zeros_array[i]
        normalized_gap = gap / mean_spacing[i]
        
        # Adicionar à histogram
        bin_idx = np.digitize([normalized_gap], s_bins) - 1
        if 0 <= bin_idx[0] < len(R_observed):
            R_observed[bin_idx[0]] += 1
    
    # Normalizar
    R_observed = R_observed / (np.sum(R_observed) * (s_bins[1] - s_bins[0]))
    
    # GUE prediction: R(s) = 1 - (sin(πs)/(πs))²
    s_centers = (s_bins[:-1] + s_bins[1:]) / 2
    R_GUE = 1 - (np.sin(np.pi * s_centers) / (np.pi * s_centers))**2
    R_GUE[s_centers == 0] = 0  # Lidar com s=0
    
    # Comparar
    print("Comparação com GUE (Random Matrix Theory):")
    print()
    print("   s   | R(s) obs | R(s) GUE | Diferença")
    print("-------|----------|----------|----------")
    
    for i in range(0, len(s_centers), 10):
        s = s_centers[i]
        r_obs = R_observed[i]
        r_gue = R_GUE[i]
        diff = r_obs - r_gue
        
        print(f"{s:6.2f} | {r_obs:8.4f} | {r_gue:8.4f} | {diff:+8.4f}")
    
    # Correlação
    valid_idx = ~np.isnan(R_observed) & ~np.isnan(R_GUE)
    if np.any(valid_idx):
        correlation = np.corrcoef(R_observed[valid_idx], R_GUE[valid_idx])[0,1]
        print()
        print(f"Correlação com GUE: r = {correlation:.6f}")
        
        if correlation > 0.95:
            print("  → Fortíssima correlação! Zeros seguem distribuição GUE!")
        elif correlation > 0.8:
            print("  → Forte correlação com RMT!")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(s_centers, R_observed, 'b-', label='Observado (Zeros de Riemann)', linewidth=2)
    plt.plot(s_centers, R_GUE, 'r--', label='GUE (Random Matrix Theory)', linewidth=2)
    plt.xlabel('s (gap normalizado)')
    plt.ylabel('R(s) (pair correlation)')
    plt.title('Pair Correlation: Riemann vs GUE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_s)
    plt.ylim(0, 1.2)
    plt.savefig('riemann_pair_correlation.png', dpi=150)
    print()
    print("✓ Gráfico salvo: riemann_pair_correlation.png")
    
    return {
        's_centers': s_centers.tolist(),
        'R_observed': R_observed.tolist(),
        'R_GUE': R_GUE.tolist(),
        'correlation': float(correlation) if np.any(valid_idx) else None
    }


def binary_structure_test(zeros):
    """
    Teste específico: ver se zeros evitam potências de 2.
    """
    print()
    print("="*80)
    print("TESTE: ZEROS EVITAM POTÊNCIAS DE 2?")
    print("="*80)
    print()
    
    zeros_array = np.array(zeros)
    
    # Para cada potência de 2, contar quantos zeros estão próximos
    print("Distância mínima de zeros até potências de 2:")
    print()
    print("2^k  | Valor    | Zeros a ±1% | Zeros a ±5% | Min dist")
    print("-----|----------|-------------|-------------|----------")
    
    for k in range(1, 11):
        power = 2**k
        
        if power > zeros_array[-1]:
            break
        
        # Contar zeros próximos
        within_1pct = np.sum(np.abs(zeros_array - power) < 0.01 * power)
        within_5pct = np.sum(np.abs(zeros_array - power) < 0.05 * power)
        
        # Distância mínima
        min_dist = np.min(np.abs(zeros_array - power))
        min_dist_pct = 100 * min_dist / power
        
        print(f"2^{k:2d} | {power:8.1f} | {within_1pct:11d} | {within_5pct:11d} | "
              f"{min_dist_pct:7.3f}%")
    
    print()
    
    # Testar estatisticamente
    # Expectativa: se zeros são uniformes, deveria haver ~N/100 zeros a ±1% de cada potência
    n_zeros_total = len(zeros_array)
    expected_1pct = n_zeros_total * 0.02  # ±1% = 2% do range
    
    print(f"Expectativa (distribuição uniforme): {expected_1pct:.1f} zeros a ±1%")
    print()
    
    # Contar observados
    observed_counts = []
    for k in range(4, 10):  # 2^4=16 até 2^9=512
        power = 2**k
        if power > zeros_array[-1]:
            break
        within_1pct = np.sum(np.abs(zeros_array - power) < 0.01 * power)
        observed_counts.append(within_1pct)
    
    if observed_counts:
        mean_observed = np.mean(observed_counts)
        print(f"Média observada: {mean_observed:.1f} zeros a ±1%")
        
        # Análise estatística simples
        print()
        ratio = mean_observed / expected_1pct
        print(f"Razão obs/esperado: {ratio:.4f}")
        
        if ratio < 0.5:
            print("  → Zeros EVITAM FORTEMENTE potências de 2!")
            print("     (menos de 50% do esperado)")
        elif ratio < 0.8:
            print("  → Zeros EVITAM potências de 2!")
        elif ratio > 1.5:
            print("  → Zeros PREFEREM potências de 2!")
        else:
            print("  → Distribuição aproximadamente uniforme")


def main():
    print("="*80)
    print("ANÁLISE PROFUNDA: RIEMANN + XOR")
    print("="*80)
    print()
    
    # Carregar dados
    results = load_results()
    zeros = results['zeros']
    gaps = results['gap_analysis']['gaps']
    
    print(f"Dados carregados: {len(zeros)} zeros, {len(gaps)} gaps")
    print()
    
    # Análise de Fourier
    fourier_results = fourier_analysis(gaps)
    
    # Pair correlation
    pair_corr_results = pair_correlation(zeros)
    
    # Teste de estrutura binária
    binary_structure_test(zeros)
    
    # Salvar resultados estendidos
    extended_results = {
        **results,
        'fourier_analysis': fourier_results,
        'pair_correlation': pair_corr_results
    }
    
    output_file = Path(__file__).parent / 'riemann_extended_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(extended_results, f, indent=2)
    
    print()
    print("="*80)
    print("CONCLUSÕES PRELIMINARES")
    print("="*80)
    print()
    print("1. Teste χ² anterior: p=0.000043 → estrutura NÃO uniforme em log₂(t)")
    print("2. Análise de Fourier: verificar se há picos em períodos 2^k")
    print("3. Pair correlation: comparar com GUE (Random Matrix Theory)")
    print("4. Teste direto: zeros evitam/preferem potências de 2?")
    print()
    print(f"Resultados salvos em: {output_file}")


if __name__ == '__main__':
    main()
