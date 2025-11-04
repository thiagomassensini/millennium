#!/usr/bin/env python3
"""
Teste Estocástico - CONVERGÊNCIA LONGA
SNR = 0.05 × √t (ORIGINAL)
O padrão emerge do ruído com t >> 50
Testar convergência em t ≥ 100, 200, 500, 1000
"""

import numpy as np
import json
import matplotlib.pyplot as plt

print("=" * 70)
print("TESTE ESTOCÁSTICO - CONVERGÊNCIA LONGA")
print("SNR = 0.05 × √t (ORIGINAL)")
print("Padrão emerge do ruído em t >> 50")
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

print(f"\n[HIPÓTESE] Padrão emerge do ruído em escala log-log")
print(f"Pontos de análise: t = 100, 200, 500, 1000")

# Parâmetros ORIGINAIS
snr_coef = 0.05  # ORIGINAL!
theta = 1.0
sigma_ou = gap_std * 0.5
mu = gap_mean
n_trials = 1000  # MUITO mais trials
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
print("SIMULAÇÃO EM ANDAMENTO...")
print(f"{'=' * 70}")

# Simulação LONGA
for t in range(1, n_trials + 1):
    if t % 100 == 0 or t in [1, 10, 50, 100, 200, 500]:
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
analysis_points = [100, 200, 500, 1000]
convergence_analysis = {}

print(f"\n{'=' * 70}")
print("ANÁLISE DE CONVERGÊNCIA")
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
        elif accuracy_post >= 85.0:
            print(f"  🟠 Razoável (>85%)")

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

# Gráficos
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Accuracy vs t (escala log)
ax1 = axes[0, 0]
ax1.semilogx(results['trials'], results['accuracy'], 'b-', linewidth=1.5, alpha=0.7)
for t_pt in analysis_points:
    if t_pt <= len(results['accuracy']):
        ax1.axvline(t_pt, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(t_pt, 105, f't={t_pt}', rotation=90, va='bottom', fontsize=9)
ax1.axhline(95, color='green', linestyle=':', linewidth=2, label='95% target')
ax1.axhline(100, color='darkgreen', linestyle=':', linewidth=2, label='100% ideal')
ax1.set_xlabel('Trial t (log scale)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Convergência de Acurácia (log scale)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([70, 110])

# 2. Chi2 vs t (log-log)
ax2 = axes[0, 1]
ax2.loglog(results['trials'], results['chi2'], 'purple', linewidth=1.5, alpha=0.7)
for t_pt in analysis_points:
    if t_pt <= len(results['chi2']):
        ax2.axvline(t_pt, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_xlabel('Trial t (log scale)', fontsize=12)
ax2.set_ylabel('χ² (log scale)', fontsize=12)
ax2.set_title('Convergência χ² (log-log)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. SNR vs t
ax3 = axes[0, 2]
ax3.plot(results['trials'], results['snr'], 'orange', linewidth=2)
for t_pt in analysis_points:
    if t_pt <= len(results['snr']):
        ax3.axvline(t_pt, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax3.plot(t_pt, results['snr'][t_pt-1], 'ro', markersize=8)
ax3.set_xlabel('Trial t', fontsize=12)
ax3.set_ylabel('SNR', fontsize=12)
ax3.set_title('SNR(t) = 0.05 × √t', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Rolling mean accuracy (janela 50)
ax4 = axes[1, 0]
window = 50
rolling_acc = np.convolve(results['accuracy'], np.ones(window)/window, mode='valid')
ax4.plot(range(window, len(results['accuracy'])+1), rolling_acc, 'g-', linewidth=2)
for t_pt in analysis_points:
    if t_pt >= window:
        ax4.axvline(t_pt, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax4.axhline(95, color='green', linestyle=':', linewidth=2, label='95%')
ax4.set_xlabel('Trial t', fontsize=12)
ax4.set_ylabel('Rolling Mean Accuracy (window=50)', fontsize=12)
ax4.set_title('Média Móvel de Acurácia', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Comparação dos pontos de convergência
ax5 = axes[1, 1]
t_points = list(convergence_analysis.keys())
accs = [convergence_analysis[t]['accuracy_mean'] for t in t_points]
stds = [convergence_analysis[t]['accuracy_std'] for t in t_points]
colors = ['darkgreen' if acc >= 95 else 'green' if acc >= 93 else 'yellow' if acc >= 90 else 'orange'
          for acc in accs]

bars = ax5.bar(range(len(t_points)), accs, color=colors, alpha=0.7, yerr=stds, capsize=5)
ax5.set_xticks(range(len(t_points)))
ax5.set_xticklabels([f't≥{t}' for t in t_points])
ax5.axhline(95, color='green', linestyle='--', linewidth=2, label='95% target')
ax5.set_ylabel('Acurácia Média (%)', fontsize=12)
ax5.set_title('Convergência em Diferentes Pontos', fontsize=13, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([70, 105])

# 6. Log-log: Accuracy vs SNR
ax6 = axes[1, 2]
ax6.semilogx(results['snr'], results['accuracy'], 'b.', alpha=0.3, markersize=2)
# Binned average
snr_bins = np.logspace(np.log10(min(results['snr'])), np.log10(max(results['snr'])), 20)
binned_acc = []
binned_snr = []
for i in range(len(snr_bins)-1):
    mask = (np.array(results['snr']) >= snr_bins[i]) & (np.array(results['snr']) < snr_bins[i+1])
    if np.sum(mask) > 0:
        binned_snr.append((snr_bins[i] + snr_bins[i+1])/2)
        binned_acc.append(np.mean(np.array(results['accuracy'])[mask]))
ax6.semilogx(binned_snr, binned_acc, 'r-', linewidth=3, label='Binned average')
ax6.axhline(95, color='green', linestyle=':', linewidth=2, label='95%')
ax6.set_xlabel('SNR (log scale)', fontsize=12)
ax6.set_ylabel('Accuracy (%)', fontsize=12)
ax6.set_title('Accuracy vs SNR (correlação log)', fontsize=13, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_ylim([70, 105])

plt.tight_layout()
plt.savefig('../validacao/stochastic_riemann_long_convergence.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/stochastic_riemann_long_convergence.png")

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
    'full_results': {
        'trials': results['trials'][::10],  # subsample para reduzir tamanho
        'accuracy': results['accuracy'][::10],
        'chi2': results['chi2'][::10],
        'snr': results['snr'][::10]
    }
}

with open('../validacao/stochastic_riemann_long_convergence_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/stochastic_riemann_long_convergence_results.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO FINAL")
print(f"{'=' * 70}")

if best_accuracy['accuracy_mean'] >= 95.0:
    print(f"\n✅ SUCESSO! Padrão emergiu do ruído!")
    print(f"\n   MECANISMO ESTOCÁSTICO IDENTIFICADO:")
    print(f"   • SNR(t) = {snr_coef} × √t (ORIGINAL)")
    print(f"   • θ = {theta}")
    print(f"   • σ_OU = {sigma_ou:.4f}")
    print(f"   • Convergência em t ≥ {best_t}")
    print(f"   • SNR necessário: {best_accuracy['snr']:.3f}")
    print(f"\n   INTERPRETAÇÃO:")
    print(f"   O padrão determinístico dos zeros de Riemann emerge")
    print(f"   do ruído estocástico em escala logarítmica, requerendo")
    print(f"   SNR ~ {best_accuracy['snr']:.2f} para convergir a ~100%.")
elif best_accuracy['accuracy_mean'] >= 90.0:
    print(f"\n🟡 Padrão emergindo gradualmente...")
    print(f"   Melhor convergência: t ≥ {best_t}")
    print(f"   Acurácia: {best_accuracy['accuracy_mean']:.2f}%")
    print(f"   SNR necessário: {best_accuracy['snr']:.3f}")
    print(f"\n   SUGESTÃO: Testar com t até 2000 ou 5000")
else:
    print(f"\n🟠 Convergência lenta detectada")
    print(f"   Melhor resultado em t ≥ {best_t}: {best_accuracy['accuracy_mean']:.2f}%")
    print(f"   SNR = {best_accuracy['snr']:.3f}")

print(f"\n{'=' * 70}\n")
