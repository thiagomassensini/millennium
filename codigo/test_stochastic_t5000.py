#!/usr/bin/env python3
"""
Teste rápido com t=5000 para ter 4º ponto de dados
Confirmar padrão de convergência
"""

import numpy as np
import json

print("=" * 70)
print("TESTE RÁPIDO - t=5000 trials")
print("=" * 70)

# Carregar dados reais
with open('/home/thlinux/relacionalidadegeral/validacao/riemann_extended_analysis.json', 'r') as f:
    data = json.load(f)

zeros = np.array(data['zeros'])
gaps = np.diff(zeros)
gap_mean = np.mean(gaps)
gap_std = np.std(gaps)

# Distribuição real de níveis (target)
gap_analysis = data['gap_analysis']
level_dist_real = gap_analysis['level_distribution']
total_real = sum(level_dist_real.values())
P_real = {int(k): v/total_real for k, v in level_dist_real.items()}

# Parâmetros
theta = 1.0
mu = gap_mean
sigma_ou = gap_std * 0.5
n_trials = 5000
n_steps = 1000
dt = 0.01
snr_coef = 0.05

print(f"\n[PARÂMETROS]")
print(f"  n_trials = {n_trials}")
print(f"  SNR(t) = {snr_coef} × √t")

results = {
    'trials': [],
    'accuracy': [],
    'chi2': []
}

checkpoint_points = [1, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]

print(f"\n{'=' * 70}")
print("SIMULAÇÃO...")
print(f"{'=' * 70}")

# Simulação
for t in range(1, n_trials + 1):
    if t in checkpoint_points:
        print(f"  Trial {t}/{n_trials}...")

    snr = snr_coef * np.sqrt(t)

    # Simular processo OU
    X = np.zeros(n_steps)
    X[0] = mu

    for i in range(1, n_steps):
        dX_ou = theta * (mu - X[i-1]) * dt
        dW_ou = sigma_ou * np.sqrt(dt) * np.random.randn()
        sigma_noise = gap_std * (1.0 / snr) if snr > 0 else gap_std
        dW_noise = sigma_noise * np.sqrt(dt) * np.random.randn()
        X[i] = X[i-1] + dX_ou + dW_ou + dW_noise
        X[i] = max(0.1, X[i])

    # Analisar níveis
    normalized_gaps = X / np.mean(X)
    normalized_gaps = np.clip(normalized_gaps, 1e-10, None)
    levels = np.floor(np.log2(normalized_gaps)).astype(int)
    unique_levels, counts = np.unique(levels, return_counts=True)
    level_dist = {int(lv): cnt/len(levels) for lv, cnt in zip(unique_levels, counts)}

    # Chi-squared
    chi2 = 0
    for level in P_real.keys():
        obs = level_dist.get(level, 0)
        exp = P_real[level]
        if exp > 0:
            chi2 += (obs - exp)**2 / exp

    accuracy = max(0, 1 - chi2/10.0) * 100

    results['trials'].append(t)
    results['accuracy'].append(accuracy)
    results['chi2'].append(chi2)

print(f"\n✓ Simulação completa!")

# Análise
print(f"\n{'=' * 70}")
print("CONVERGÊNCIA")
print(f"{'=' * 70}")

analysis_points = [100, 500, 1000, 2000, 3000, 4000, 5000]

for t_threshold in analysis_points:
    idx = t_threshold - 1
    accuracy_post = np.mean(results['accuracy'][idx:])
    accuracy_std = np.std(results['accuracy'][idx:])
    chi2_post = np.mean(results['chi2'][idx:])
    snr_at_t = snr_coef * np.sqrt(t_threshold)

    print(f"\n[t ≥ {t_threshold}]")
    print(f"  SNR = {snr_at_t:.3f}")
    print(f"  Acurácia: {accuracy_post:.2f}% ± {accuracy_std:.2f}%")
    print(f"  χ²: {chi2_post:.4f}")

# Salvar
output = {
    'n_trials': n_trials,
    'convergence': {}
}

for t_threshold in analysis_points:
    idx = t_threshold - 1
    output['convergence'][str(t_threshold)] = {
        'accuracy_mean': float(np.mean(results['accuracy'][idx:])),
        'accuracy_std': float(np.std(results['accuracy'][idx:])),
        'chi2_mean': float(np.mean(results['chi2'][idx:]))
    }

with open('/home/thlinux/relacionalidadegeral/validacao/test_t5000_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Resultados salvos: validacao/test_t5000_results.json")
print(f"\n{'=' * 70}\n")
