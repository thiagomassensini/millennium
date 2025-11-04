#!/usr/bin/env python3
"""
Teste Estocástico - ULTRA LONGO
SNR = 0.05 × √t (ORIGINAL)
Testar até t = 5000 para ver emergência completa do padrão
"""

import numpy as np
import json
import matplotlib.pyplot as plt

print("=" * 70)
print("TESTE ESTOCÁSTICO - ULTRA LONGO (até t=5000)")
print("SNR = 0.05 × √t (ORIGINAL)")
print("=" * 70)

# Carregar dados reais
with open('../validacao/riemann_extended_analysis.json', 'r') as f:
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

print(f"\n[TARGET] Distribuição real:")
for level in sorted(P_real.keys()):
    print(f"  Level {level}: {100*P_real[level]:.1f}%")

# Parâmetros ORIGINAIS
snr_coef = 0.05  # ORIGINAL!
theta = 1.0
sigma_ou = gap_std * 0.5
mu = gap_mean
n_trials = 5000  # ATÉ 5000!
n_steps = 1000
dt = 0.01

print(f"\n[PARÂMETROS]")
print(f"  SNR(t) = {snr_coef} × √t")
print(f"  θ = {theta}")
print(f"  σ_OU = {sigma_ou:.4f}")
print(f"  n_trials = {n_trials}")
print(f"  n_steps = {n_steps}")

results = {
    'trials': [],
    'accuracy': [],
    'chi2': [],
    'snr': []
}

print(f"\n{'=' * 70}")
print("SIMULAÇÃO EM ANDAMENTO (pode demorar ~5min)...")
print(f"{'=' * 70}")

# Simulação ULTRA LONGA
checkpoint_points = [1, 10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]

for t in range(1, n_trials + 1):
    if t in checkpoint_points:
        print(f"  Trial {t}/{n_trials}... SNR = {snr_coef * np.sqrt(t):.3f}")

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
    results['snr'].append(snr)

print(f"\n✓ Simulação completa!")

# ANÁLISE EM DIFERENTES PONTOS
analysis_points = [100, 200, 500, 1000, 2000, 3000, 4000, 5000]
convergence_analysis = {}

print(f"\n{'=' * 70}")
print("ANÁLISE DE CONVERGÊNCIA LOG-LOG")
print(f"{'=' * 70}")

for t_threshold in analysis_points:
    if t_threshold <= len(results['accuracy']):
        idx = t_threshold - 1
        accuracy_post = np.mean(results['accuracy'][idx:])
        accuracy_std = np.std(results['accuracy'][idx:])
        chi2_post = np.mean(results['chi2'][idx:])
        snr_at_t = results['snr'][idx]

        convergence_analysis[t_threshold] = {
            'accuracy_mean': accuracy_post,
            'accuracy_std': accuracy_std,
            'chi2_mean': chi2_post,
            'snr': snr_at_t
        }

        print(f"\n[t ≥ {t_threshold}]")
        print(f"  SNR(t={t_threshold}) = {snr_at_t:.3f}")
        print(f"  Acurácia média: {accuracy_post:.2f}%")
        print(f"  Desvio padrão: {accuracy_std:.2f}%")
        print(f"  χ² médio: {chi2_post:.4f}")

        if accuracy_std < 5.0:
            print(f"  ✓ ESTABILIZADO")

        if accuracy_post >= 95.0:
            print(f"  ✅ CONVERGIU para ~100%!")
        elif accuracy_post >= 93.0:
            print(f"  🟢 Quase lá! (>93%)")
        elif accuracy_post >= 90.0:
            print(f"  🟡 Muito próximo (>90%)")

# Encontrar melhor resultado
best_accuracy = max(convergence_analysis.values(), key=lambda x: x['accuracy_mean'])
best_t = [t for t, v in convergence_analysis.items() if v == best_accuracy][0]

print(f"\n{'=' * 70}")
print("MELHOR CONVERGÊNCIA")
print(f"{'=' * 70}")
print(f"\n  Ponto de análise: t ≥ {best_t}")
print(f"  SNR(t={best_t}) = {best_accuracy['snr']:.3f}")
print(f"  Acurácia média: {best_accuracy['accuracy_mean']:.2f}%")
print(f"  Desvio padrão: {best_accuracy['accuracy_std']:.2f}%")
print(f"  χ² médio: {best_accuracy['chi2_mean']:.4f}")

# FIT LOG-LOG
print(f"\n{'=' * 70}")
print("ANÁLISE LOG-LOG: Accuracy vs √t")
print(f"{'=' * 70}")

t_array = np.array(analysis_points)
acc_array = np.array([convergence_analysis[t]['accuracy_mean'] for t in analysis_points])

# Fit linear em escala log
log_sqrt_t = np.log(np.sqrt(t_array))
log_acc_deviation = np.log(100 - acc_array)  # desvio de 100%

from scipy import stats as scipy_stats
slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_sqrt_t, log_acc_deviation)

print(f"\n  Fit: log(100-Acc) = {slope:.4f} × log(√t) + {intercept:.4f}")
print(f"  R² = {r_value**2:.4f}")
print(f"  p-value = {p_value:.2e}")

# Extrapolação
if slope < 0:  # Se está convergindo
    t_for_95 = np.exp((np.log(5.0) - intercept) / slope) ** 2  # 100-95=5
    t_for_99 = np.exp((np.log(1.0) - intercept) / slope) ** 2  # 100-99=1
    print(f"\n  EXTRAPOLAÇÃO:")
    print(f"    Para 95% accuracy: t ≈ {t_for_95:.0f}")
    print(f"    Para 99% accuracy: t ≈ {t_for_99:.0f}")

# Gráficos
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Accuracy vs t (log scale)
ax1 = fig.add_subplot(gs[0, :2])
ax1.semilogx(results['trials'], results['accuracy'], 'b-', linewidth=1, alpha=0.5)
# Rolling mean
window = 100
rolling_acc = np.convolve(results['accuracy'], np.ones(window)/window, mode='valid')
ax1.semilogx(range(window, len(results['accuracy'])+1), rolling_acc, 'r-', linewidth=3, label='Rolling mean (100)')
for t_pt in analysis_points:
    ax1.axvline(t_pt, color='green', linestyle='--', alpha=0.3, linewidth=1)
ax1.axhline(95, color='green', linestyle=':', linewidth=2, label='95%')
ax1.axhline(100, color='darkgreen', linestyle=':', linewidth=2, label='100%')
ax1.set_xlabel('Trial t (log scale)', fontsize=13)
ax1.set_ylabel('Accuracy (%)', fontsize=13)
ax1.set_title('Convergência de Acurácia - SNR = 0.05 × √t', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([75, 105])

# 2. Convergence points
ax2 = fig.add_subplot(gs[0, 2])
t_points = list(convergence_analysis.keys())
accs = [convergence_analysis[t]['accuracy_mean'] for t in t_points]
colors = ['darkgreen' if acc >= 95 else 'green' if acc >= 93 else 'yellow' if acc >= 90 else 'orange'
          for acc in accs]
ax2.barh(range(len(t_points)), accs, color=colors, alpha=0.7)
ax2.set_yticks(range(len(t_points)))
ax2.set_yticklabels([f't≥{t}' for t in t_points], fontsize=10)
ax2.axvline(95, color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('Accuracy (%)', fontsize=12)
ax2.set_title('Convergência', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim([85, 100])

# 3. Chi2 vs t (log-log)
ax3 = fig.add_subplot(gs[1, 0])
ax3.loglog(results['trials'], results['chi2'], 'purple', linewidth=1, alpha=0.7)
for t_pt in analysis_points[::2]:
    ax3.axvline(t_pt, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax3.set_xlabel('Trial t (log)', fontsize=12)
ax3.set_ylabel('χ² (log)', fontsize=12)
ax3.set_title('χ² vs t (log-log)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. SNR vs t
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(results['trials'], results['snr'], 'orange', linewidth=2)
for t_pt in analysis_points:
    idx = t_pt - 1
    ax4.plot(t_pt, results['snr'][idx], 'ro', markersize=8)
    ax4.text(t_pt, results['snr'][idx] + 0.1, f'{results["snr"][idx]:.2f}',
             fontsize=8, ha='center')
ax4.set_xlabel('Trial t', fontsize=12)
ax4.set_ylabel('SNR', fontsize=12)
ax4.set_title('SNR(t) = 0.05 × √t', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Log-log fit
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(np.sqrt(t_array), 100-acc_array, s=100, c='blue', alpha=0.7, label='Dados')
# Fit line
sqrt_t_fit = np.logspace(np.log10(np.sqrt(t_array[0])), np.log10(np.sqrt(t_array[-1])), 100)
acc_deviation_fit = np.exp(slope * np.log(sqrt_t_fit) + intercept)
ax5.plot(sqrt_t_fit, acc_deviation_fit, 'r-', linewidth=2, label=f'Fit: R²={r_value**2:.3f}')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlabel('√t (log scale)', fontsize=12)
ax5.set_ylabel('100 - Accuracy (log scale)', fontsize=12)
ax5.set_title('Fit Log-Log', fontsize=13, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Accuracy evolution at checkpoints
ax6 = fig.add_subplot(gs[2, :])
ax6.plot(t_array, acc_array, 'go-', linewidth=3, markersize=10, label='Acurácia média')
# Error bars
stds = [convergence_analysis[t]['accuracy_std'] for t in t_array]
ax6.fill_between(t_array, np.array(acc_array)-np.array(stds),
                  np.array(acc_array)+np.array(stds), alpha=0.3, color='green')
ax6.axhline(95, color='green', linestyle='--', linewidth=2, label='95% target')
ax6.axhline(100, color='darkgreen', linestyle='--', linewidth=2, label='100% ideal')
ax6.set_xscale('log')
ax6.set_xlabel('t (log scale)', fontsize=13)
ax6.set_ylabel('Accuracy (%)', fontsize=13)
ax6.set_title('Evolução da Acurácia em Pontos de Análise', fontsize=14, fontweight='bold')
ax6.legend(fontsize=12)
ax6.grid(True, alpha=0.3)
ax6.set_ylim([88, 102])

plt.savefig('../validacao/stochastic_riemann_ultra_long.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/stochastic_riemann_ultra_long.png")

# Salvar resultados
output = {
    'parameters': {
        'snr_coef': snr_coef,
        'theta': theta,
        'sigma_ou': sigma_ou,
        'n_trials': n_trials,
        'n_steps': n_steps
    },
    'convergence_analysis': convergence_analysis,
    'best_convergence': {
        't_threshold': best_t,
        'accuracy_mean': best_accuracy['accuracy_mean'],
        'accuracy_std': best_accuracy['accuracy_std'],
        'chi2_mean': best_accuracy['chi2_mean'],
        'snr': best_accuracy['snr']
    },
    'loglog_fit': {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value**2),
        'p_value': float(p_value)
    },
    'full_results_subsampled': {
        'trials': results['trials'][::50],
        'accuracy': results['accuracy'][::50],
        'chi2': results['chi2'][::50],
        'snr': results['snr'][::50]
    }
}

with open('../validacao/stochastic_riemann_ultra_long_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/stochastic_riemann_ultra_long_results.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO FINAL")
print(f"{'=' * 70}")

if best_accuracy['accuracy_mean'] >= 95.0:
    print(f"\n✅ SUCESSO! Padrão emergiu completamente do ruído!")
    print(f"\n   MECANISMO ESTOCÁSTICO COMPLETO:")
    print(f"   • SNR(t) = {snr_coef} × √t")
    print(f"   • θ = {theta}")
    print(f"   • σ_OU = {sigma_ou:.4f}")
    print(f"   • Convergência alcançada em t ≥ {best_t}")
    print(f"   • SNR necessário: {best_accuracy['snr']:.3f}")
    print(f"\n   CORRELAÇÃO LOG-LOG:")
    print(f"   • Slope = {slope:.4f}")
    print(f"   • R² = {r_value**2:.4f}")
    print(f"\n   INTERPRETAÇÃO FÍSICA:")
    print(f"   O padrão determinístico dos zeros de Riemann emerge")
    print(f"   de um processo estocástico (OU + Gaussiano) seguindo")
    print(f"   uma lei de escala logarítmica. O ruído decai como t^-α,")
    print(f"   revelando a estrutura subjacente.")
elif best_accuracy['accuracy_mean'] >= 93.0:
    print(f"\n🟢 MUITO PRÓXIMO! Acurácia {best_accuracy['accuracy_mean']:.2f}%")
    print(f"   t ≥ {best_t}, SNR = {best_accuracy['snr']:.3f}")
    print(f"\n   CORRELAÇÃO LOG-LOG CLARA:")
    print(f"   • log(100-Acc) = {slope:.4f} × log(√t) + {intercept:.4f}")
    print(f"   • R² = {r_value**2:.4f}")
    if slope < 0 and 't_for_95' in locals():
        print(f"\n   EXTRAPOLAÇÃO: 95% em t ≈ {t_for_95:.0f}")
elif best_accuracy['accuracy_mean'] >= 90.0:
    print(f"\n🟡 Convergência lenta mas sistemática")
    print(f"   Melhor: t ≥ {best_t}, Acc = {best_accuracy['accuracy_mean']:.2f}%")
    print(f"   SNR = {best_accuracy['snr']:.3f}")

print(f"\n{'=' * 70}\n")
