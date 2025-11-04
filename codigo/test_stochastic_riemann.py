#!/usr/bin/env python3
"""
Teste Estocástico para Zeros de Riemann
Processo OU + Ruído Gaussiano Branco com SNR adaptativo

SNR = 0.05 × √t
t começa em 1, incrementa a cada rodada
Quando t ≥ 50, ruído estabiliza e padrão emerge
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Carregar dados reais
print("=" * 60)
print("TESTE ESTOCÁSTICO: Processo OU para Zeros de Riemann")
print("=" * 60)

with open('../validacao/riemann_extended_analysis.json', 'r') as f:
    data = json.load(f)

# Dados reais
zeros = np.array(data['zeros'])
gaps = np.diff(zeros)
gap_mean = np.mean(gaps)
gap_std = np.std(gaps)

print(f"\n[DADOS REAIS]")
print(f"  Zeros: {len(zeros)}")
print(f"  Gap mean: {gap_mean:.6f}")
print(f"  Gap std: {gap_std:.6f}")

# Parâmetros do processo OU
theta = 1.0  # taxa de reversão à média
mu = gap_mean  # média de longo prazo
sigma_ou = gap_std * 0.2  # volatilidade intrínseca (reduzida para deixar espaço pro ruído)

print(f"\n[PROCESSO OU]")
print(f"  θ (reversão): {theta}")
print(f"  μ (média): {mu:.6f}")
print(f"  σ_OU (volatilidade): {sigma_ou:.6f}")

# Função SNR adaptativo
def snr_function(t):
    """SNR = 0.05 × √t""" 
    return 0.5 * np.sqrt(t)

# Teste estocástico
n_trials = 1  # número de rodadas
n_steps = 100  # passos por rodada
dt = 0.01

results = {
    'trials': [],
    'snr_values': [],
    'accuracy': [],
    'level_distributions': [],
    'convergence': []
}

print(f"\n[TESTE ESTOCÁSTICO]")
print(f"  Rodadas: {n_trials}")
print(f"  Passos por rodada: {n_steps}")
print(f"  dt: {dt}")
print(f"\nIniciando simulação...\n")

# Distribuição real de níveis (target)
gap_analysis = data['gap_analysis']
level_dist_real = gap_analysis['level_distribution']
total_real = sum(level_dist_real.values())
P_real = {int(k): v/total_real for k, v in level_dist_real.items()}

print(f"[TARGET] Distribuição real de níveis:")
for level in sorted(P_real.keys()):
    print(f"  Level {level}: {100*P_real[level]:.1f}%")

# Simulação estocástica
for t in range(1, n_trials + 1):
    # SNR adaptativo
    snr = snr_function(t)

    # Simular processo OU
    X = np.zeros(n_steps)
    X[0] = mu  # começar na média

    for i in range(1, n_steps):
        # Processo OU
        dX_ou = theta * (mu - X[i-1]) * dt

        # Ruído OU
        dW_ou = sigma_ou * np.sqrt(dt) * np.random.randn()

        # Ruído gaussiano branco adaptativo
        sigma_noise = gap_std * (1.0 / snr) if snr > 0 else gap_std
        dW_noise = sigma_noise * np.sqrt(dt) * np.random.randn()

        # Atualização
        X[i] = X[i-1] + dX_ou + dW_ou + dW_noise
        X[i] = max(0.1, X[i])  # evitar gaps negativos

    # Analisar níveis binários
    normalized_gaps = X / np.mean(X)
    levels = np.floor(np.log2(normalized_gaps)).astype(int)

    # Distribuição de níveis
    unique_levels, counts = np.unique(levels, return_counts=True)
    level_dist = {int(lv): cnt/len(levels) for lv, cnt in zip(unique_levels, counts)}

    # Calcular "acurácia" = quão próximo está da distribuição real
    # Usar KL divergence ou chi-squared
    chi2 = 0
    for level in P_real.keys():
        obs = level_dist.get(level, 0)
        exp = P_real[level]
        if exp > 0:
            chi2 += (obs - exp)**2 / exp

    # Accuracy = 1 - normalized_chi2
    max_chi2 = 10.0  # normalização
    accuracy = max(0, 1 - chi2/max_chi2) * 100

    results['trials'].append(t)
    results['snr_values'].append(snr)
    results['accuracy'].append(accuracy)
    results['level_distributions'].append(level_dist)
    results['convergence'].append(chi2)

    # Print a cada 10 rodadas
    if t % 10 == 0 or t == 1 or t >= 48:
        print(f"[t={t:3d}] SNR={snr:.4f}, χ²={chi2:.4f}, Accuracy={accuracy:.1f}%")

# Análise final
print(f"\n{'=' * 60}")
print("RESULTADOS")
print(f"{'=' * 60}")

# Convergência
accuracy_final = results['accuracy'][-1]
accuracy_50 = results['accuracy'][49] if len(results['accuracy']) >= 50 else None

print(f"\nAcurácia final (t={n_trials}): {accuracy_final:.2f}%")
if accuracy_50:
    print(f"Acurácia em t=50: {accuracy_50:.2f}%")

# Testar estabilização após t ≥ 50
if len(results['accuracy']) >= 50:
    accuracy_post50 = np.mean(results['accuracy'][49:])
    accuracy_std_post50 = np.std(results['accuracy'][49:])
    print(f"\nPós t≥50:")
    print(f"  Acurácia média: {accuracy_post50:.2f}%")
    print(f"  Desvio padrão: {accuracy_std_post50:.2f}%")

    # Testar se estabilizou
    if accuracy_std_post50 < 5.0:
        print(f"  ✓ ESTABILIZADO (std < 5%)")
    else:
        print(f"  ✗ Ainda oscilando (std = {accuracy_std_post50:.2f}%)")

# Comparação: distribuição final vs real
print(f"\n[COMPARAÇÃO] Distribuição final vs Real:")
final_dist = results['level_distributions'][-1]
print(f"{'Level':<7} {'Real':>8} {'Simulado':>10} {'Erro':>8}")
print("-" * 40)
for level in sorted(P_real.keys()):
    real = 100 * P_real[level]
    sim = 100 * final_dist.get(level, 0)
    error = abs(real - sim)
    print(f"{level:<7} {real:>7.1f}% {sim:>9.1f}% {error:>7.1f}%")

# Gráficos
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. SNR vs t
ax1 = axes[0, 0]
ax1.plot(results['trials'], results['snr_values'], 'b-', linewidth=2)
ax1.axvline(50, color='r', linestyle='--', label='t=50')
ax1.set_xlabel('Trial t', fontsize=12)
ax1.set_ylabel('SNR', fontsize=12)
ax1.set_title('SNR Adaptativo: 0.05 × √t', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Accuracy vs t
ax2 = axes[0, 1]
ax2.plot(results['trials'], results['accuracy'], 'g-', linewidth=2, label='Accuracy')
ax2.axvline(50, color='r', linestyle='--', label='t=50')
ax2.axhline(95, color='orange', linestyle=':', label='95% threshold')
ax2.set_xlabel('Trial t', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Convergência da Acurácia', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim([0, 105])

# 3. Chi-squared vs t
ax3 = axes[1, 0]
ax3.semilogy(results['trials'], results['convergence'], 'purple', linewidth=2)
ax3.axvline(50, color='r', linestyle='--', label='t=50')
ax3.set_xlabel('Trial t', fontsize=12)
ax3.set_ylabel('χ² (log scale)', fontsize=12)
ax3.set_title('Convergência χ²', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Distribuição final vs real
ax4 = axes[1, 1]
levels = sorted(P_real.keys())
real_pcts = [100 * P_real[lv] for lv in levels]
sim_pcts = [100 * final_dist.get(lv, 0) for lv in levels]

x = np.arange(len(levels))
width = 0.35

ax4.bar(x - width/2, real_pcts, width, label='Real', color='blue', alpha=0.7)
ax4.bar(x + width/2, sim_pcts, width, label='Simulado (t=100)', color='green', alpha=0.7)
ax4.set_xlabel('Nível Binário', fontsize=12)
ax4.set_ylabel('Porcentagem (%)', fontsize=12)
ax4.set_title('Distribuição de Níveis: Real vs Simulado', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(levels)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../validacao/stochastic_riemann_test.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/stochastic_riemann_test.png")

# Salvar resultados
output = {
    'parameters': {
        'theta': theta,
        'mu': mu,
        'sigma_ou': sigma_ou,
        'n_trials': n_trials,
        'n_steps': n_steps,
        'dt': dt,
        'snr_formula': '0.05 × √t'
    },
    'results': {
        'final_accuracy': accuracy_final,
        'accuracy_at_t50': accuracy_50,
        'post_t50_mean': accuracy_post50,
        'post_t50_std': accuracy_std_post50,
        'final_chi2': results['convergence'][-1]
    },
    'convergence_data': {
        't_values': results['trials'],
        'snr_values': results['snr_values'],
        'accuracy_values': results['accuracy'],
        'chi2_values': results['convergence']
    }
}

with open('../validacao/stochastic_riemann_test_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/stochastic_riemann_test_results.json")

# Conclusão
print(f"\n{'=' * 60}")
print("CONCLUSÃO")
print(f"{'=' * 60}")

if accuracy_post50 >= 95.0:
    print(f"\n✅ SUCESSO! Padrão emergiu com {accuracy_post50:.1f}% de acurácia após t≥50")
    print(f"   SNR final: {results['snr_values'][-1]:.4f}")
    print(f"   χ² final: {results['convergence'][-1]:.4f}")
elif accuracy_post50 >= 80.0:
    print(f"\n⚠️  PARCIAL: Acurácia {accuracy_post50:.1f}% após t≥50")
    print(f"   Pode precisar de ajustes nos parâmetros")
    print(f"   Sugestões:")
    print(f"   - Aumentar SNR inicial (testar 0.1 × √t)")
    print(f"   - Ajustar θ (taxa de reversão)")
    print(f"   - Aumentar n_steps por rodada")
else:
    print(f"\n❌ Acurácia baixa: {accuracy_post50:.1f}% após t≥50")
    print(f"   Ajustes necessários:")
    print(f"   1. SNR formula (testar α × √t com α > 0.05)")
    print(f"   2. Processo OU (ajustar θ, σ_OU)")
    print(f"   3. Tempo de estabilização (testar t ≥ 100)")

print(f"\n{'=' * 60}\n")
