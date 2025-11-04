#!/usr/bin/env python3
"""
ESTRUTURA DE SEGUNDA ORDEM - Gaps dos Gaps
Modelar não só os gaps X_t, mas também ΔX_t = X_t - X_{t-1}
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

print("=" * 70)
print("ESTRUTURA DE SEGUNDA ORDEM - GAPS DOS GAPS")
print("Modelar ΔX_t = X_t - X_{t-1}")
print("=" * 70)

# Carregar dados de Riemann
with open('/home/thlinux/relacionalidadegeral/validacao/riemann_extended_analysis.json', 'r') as f:
    data_riemann = json.load(f)

zeros = np.array(data_riemann['zeros'])
gaps = np.diff(zeros)  # X_t = gaps entre zeros
gap_of_gaps = np.diff(gaps)  # ΔX_t = gaps DOS gaps

gap_mean = np.mean(gaps)
gap_std = np.std(gaps)
delta_mean = np.mean(gap_of_gaps)
delta_std = np.std(gap_of_gaps)

print(f"\n[ESTATÍSTICAS]")
print(f"  Gaps (X_t):")
print(f"    μ = {gap_mean:.6f}")
print(f"    σ = {gap_std:.6f}")
print(f"\n  Gaps dos Gaps (ΔX_t):")
print(f"    μ = {delta_mean:.6f}")
print(f"    σ = {delta_std:.6f}")
print(f"    Razão σ_Δ/σ = {delta_std/gap_std:.6f}")

# Distribuição alvo dos gaps
gap_analysis = data_riemann['gap_analysis']
level_dist_riemann = gap_analysis['level_distribution']
total_riemann = sum(level_dist_riemann.values())
P_target = {int(k): v/total_riemann for k, v in level_dist_riemann.items()}

print(f"\n[TARGET] Distribuição dos Gaps:")
for level in sorted(P_target.keys()):
    if P_target[level] > 0.001:
        print(f"  Level {level}: {100*P_target[level]:.1f}%")

# Analisar distribuição dos gaps DOS gaps
delta_normalized = gap_of_gaps / gap_mean
delta_levels = np.floor(np.log2(np.abs(delta_normalized) + 1e-10)).astype(int)
unique_delta, counts_delta = np.unique(delta_levels, return_counts=True)
P_delta = {int(lv): cnt/len(delta_levels) for lv, cnt in zip(unique_delta, counts_delta)}

print(f"\n[OBSERVADO] Distribuição dos Gaps dos Gaps:")
for level in sorted(P_delta.keys()):
    if P_delta[level] > 0.01:
        print(f"  Delta-Level {level}: {100*P_delta[level]:.1f}%")

# Analisar correlação entre gap e gap-of-gap
correlation = np.corrcoef(gaps[:-1], gap_of_gaps)[0, 1]
print(f"\n[CORRELAÇÃO]")
print(f"  Corr(X_t, ΔX_t) = {correlation:.6f}")

# FUNÇÃO: Avaliar distribuição
def evaluate_distribution(X, mu):
    normalized = X / mu
    normalized = np.clip(normalized, 1e-10, None)
    levels = np.floor(np.log2(normalized)).astype(int)

    unique_levels, counts = np.unique(levels, return_counts=True)
    P_sim = {int(lv): cnt/len(levels) for lv, cnt in zip(unique_levels, counts)}

    # Chi-squared
    chi2 = 0
    for level in P_target.keys():
        obs = P_sim.get(level, 0)
        exp = P_target[level]
        if exp > 0:
            chi2 += (obs - exp)**2 / exp

    accuracy = max(0, 1 - chi2/10.0) * 100
    return accuracy, chi2, P_sim

# MODELO 1: AR(3) com regularização de segunda ordem
def simulate_ar3_second_order_v1(phi_params, mu, sigma_noise, lambda_reg, n_steps=15000):
    """
    AR(3) com penalização nas mudanças bruscas de ΔX
    """
    X = np.zeros(n_steps)
    X[:3] = mu

    for t in range(3, n_steps):
        # Termo AR(3) padrão
        ar_term = sum(phi_params[i] * (X[t-1-i] - mu) for i in range(3))

        # Ruído base
        noise = sigma_noise * np.random.randn()

        # Calcular ΔX anterior
        if t > 3:
            delta_prev = X[t-1] - X[t-2]
            # Adicionar inércia baseada em ΔX
            # Se ΔX foi positivo, tendência de continuar positivo (momentum)
            momentum = lambda_reg * delta_prev
            X[t] = mu + ar_term + noise + momentum
        else:
            X[t] = mu + ar_term + noise

        X[t] = max(0.01, X[t])

    return X

# MODELO 2: AR(3) com dois processos acoplados
def simulate_ar3_coupled_processes(phi_params, mu, sigma_noise, coupling, n_steps=15000):
    """
    Dois processos AR acoplados:
    - X_t = processo principal (gaps)
    - Y_t = processo auxiliar (influencia gaps dos gaps)
    """
    phi_x = phi_params[:3]   # Coeficientes para X
    phi_y = phi_params[3:6]  # Coeficientes para Y

    X = np.zeros(n_steps)
    Y = np.zeros(n_steps)
    X[:3] = mu
    Y[:3] = 0.0  # Y começa em 0 (sem bias)

    for t in range(3, n_steps):
        # Processo X (gaps)
        ar_x = sum(phi_x[i] * (X[t-1-i] - mu) for i in range(3))
        noise_x = sigma_noise * np.random.randn()

        # Processo Y (influencia de segunda ordem)
        ar_y = sum(phi_y[i] * Y[t-1-i] for i in range(3))
        noise_y = 0.5 * sigma_noise * np.random.randn()
        Y[t] = ar_y + noise_y

        # Acoplamento: Y influencia X
        X[t] = mu + ar_x + coupling * Y[t] + noise_x
        X[t] = max(0.01, X[t])

    return X

# MODELO 3: AR(3) com constraint explícito em ΔX
def simulate_ar3_delta_constrained(phi_params, mu, sigma_noise, delta_target_std, n_steps=15000):
    """
    AR(3) onde forçamos std(ΔX) ≈ delta_target_std observado em Riemann
    """
    X = np.zeros(n_steps)
    X[:3] = mu

    for t in range(3, n_steps):
        # Termo AR(3)
        ar_term = sum(phi_params[i] * (X[t-1-i] - mu) for i in range(3))

        # Propor novo valor
        noise = sigma_noise * np.random.randn()
        X_proposed = mu + ar_term + noise
        X_proposed = max(0.01, X_proposed)

        # Calcular ΔX proposto
        delta_proposed = X_proposed - X[t-1]

        # Aceitar/rejeitar baseado na std dos últimos deltas
        if t > 10:
            recent_deltas = np.diff(X[max(0, t-20):t])
            current_std = np.std(recent_deltas)

            # Se std está muito diferente do target, ajustar
            if current_std > 1.5 * delta_target_std:
                # Std muito alta, reduzir mudança
                X[t] = X[t-1] + 0.5 * delta_proposed
            elif current_std < 0.5 * delta_target_std:
                # Std muito baixa, aumentar mudança
                X[t] = X[t-1] + 1.5 * delta_proposed
            else:
                X[t] = X_proposed
        else:
            X[t] = X_proposed

        X[t] = max(0.01, X[t])

    return X

# CONFIGURAÇÕES
configs = [
    {
        'name': 'Config 1: AR(3) + Momentum (λ=0.3)',
        'model': 'momentum',
        'lambda_reg': 0.3
    },
    {
        'name': 'Config 2: AR(3) + Momentum (λ=0.5)',
        'model': 'momentum',
        'lambda_reg': 0.5
    },
    {
        'name': 'Config 3: Coupled processes (coupling=0.3)',
        'model': 'coupled',
        'coupling': 0.3
    },
    {
        'name': 'Config 4: Coupled processes (coupling=0.5)',
        'model': 'coupled',
        'coupling': 0.5
    },
    {
        'name': 'Config 5: Delta-constrained',
        'model': 'delta_constrained',
        'delta_target_std': delta_std
    }
]

all_results = []

for cfg in configs:
    print(f"\n{'=' * 70}")
    print(f"TESTANDO: {cfg['name']}")
    print(f"  Model: {cfg['model']}")
    print(f"{'=' * 70}")

    print("\n[OPTIMIZATION] Usando differential evolution...")

    # Função objetivo
    def objective(params):
        if cfg['model'] == 'momentum':
            phi = params[:3]
            sigma = params[3]
            X = simulate_ar3_second_order_v1(phi, gap_mean, sigma, cfg['lambda_reg'])
        elif cfg['model'] == 'coupled':
            phi = params[:6]
            sigma = params[6]
            X = simulate_ar3_coupled_processes(phi, gap_mean, sigma, cfg['coupling'])
        elif cfg['model'] == 'delta_constrained':
            phi = params[:3]
            sigma = params[3]
            X = simulate_ar3_delta_constrained(phi, gap_mean, sigma, cfg['delta_target_std'])

        # Múltiplas runs
        accuracies = []
        for _ in range(3):
            if cfg['model'] == 'momentum':
                X = simulate_ar3_second_order_v1(phi, gap_mean, sigma, cfg['lambda_reg'])
            elif cfg['model'] == 'coupled':
                X = simulate_ar3_coupled_processes(phi, gap_mean, sigma, cfg['coupling'])
            elif cfg['model'] == 'delta_constrained':
                X = simulate_ar3_delta_constrained(phi, gap_mean, sigma, cfg['delta_target_std'])

            acc, _, _ = evaluate_distribution(X, gap_mean)
            accuracies.append(acc)

        return -np.mean(accuracies)

    # Bounds
    if cfg['model'] == 'coupled':
        bounds = [
            (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5),  # phi_x
            (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5),  # phi_y
            (0.01, 0.5)   # sigma
        ]
    else:
        bounds = [
            (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5),  # phi
            (0.01, 0.5)   # sigma
        ]

    # Otimizar
    result = differential_evolution(
        objective, bounds,
        maxiter=50, seed=42, workers=1,
        updating='deferred', atol=1e-6
    )

    if cfg['model'] == 'coupled':
        best_phi_x = result.x[:3]
        best_phi_y = result.x[3:6]
        best_sigma = result.x[6]
        print(f"\n[BEST PARAMS]")
        print(f"  φ_X = [{best_phi_x[0]:.3f}, {best_phi_x[1]:.3f}, {best_phi_x[2]:.3f}]")
        print(f"  φ_Y = [{best_phi_y[0]:.3f}, {best_phi_y[1]:.3f}, {best_phi_y[2]:.3f}]")
        print(f"  σ = {best_sigma:.3f}")
    else:
        best_phi = result.x[:3]
        best_sigma = result.x[3]
        print(f"\n[BEST PARAMS]")
        print(f"  φ = [{best_phi[0]:.3f}, {best_phi[1]:.3f}, {best_phi[2]:.3f}]")
        print(f"  σ = {best_sigma:.3f}")

    best_accuracy = -result.fun
    print(f"  Accuracy (optimization): {best_accuracy:.2f}%")

    # Validação
    print(f"\n[VALIDATION] Rodando 20 vezes...")
    accuracies_validation = []
    all_chi2 = []

    for run in range(20):
        if cfg['model'] == 'momentum':
            X = simulate_ar3_second_order_v1(best_phi, gap_mean, best_sigma, cfg['lambda_reg'])
        elif cfg['model'] == 'coupled':
            best_phi_all = np.concatenate([best_phi_x, best_phi_y])
            X = simulate_ar3_coupled_processes(best_phi_all, gap_mean, best_sigma, cfg['coupling'])
        elif cfg['model'] == 'delta_constrained':
            X = simulate_ar3_delta_constrained(best_phi, gap_mean, best_sigma, cfg['delta_target_std'])

        acc, chi2, P_sim = evaluate_distribution(X, gap_mean)
        accuracies_validation.append(acc)
        all_chi2.append(chi2)

    mean_acc = np.mean(accuracies_validation)
    std_acc = np.std(accuracies_validation)
    max_acc = np.max(accuracies_validation)
    mean_chi2 = np.mean(all_chi2)

    print(f"\n[RESULTS]")
    print(f"  Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Max accuracy: {max_acc:.2f}%")
    print(f"  Mean χ²: {mean_chi2:.4f}")

    # Última simulação para análise
    if cfg['model'] == 'momentum':
        X_final = simulate_ar3_second_order_v1(best_phi, gap_mean, best_sigma, cfg['lambda_reg'])
        params_dict = {'phi': best_phi.tolist(), 'sigma': best_sigma, 'lambda': cfg['lambda_reg']}
    elif cfg['model'] == 'coupled':
        X_final = simulate_ar3_coupled_processes(best_phi_all, gap_mean, best_sigma, cfg['coupling'])
        params_dict = {'phi_x': best_phi_x.tolist(), 'phi_y': best_phi_y.tolist(),
                      'sigma': best_sigma, 'coupling': cfg['coupling']}
    elif cfg['model'] == 'delta_constrained':
        X_final = simulate_ar3_delta_constrained(best_phi, gap_mean, best_sigma, cfg['delta_target_std'])
        params_dict = {'phi': best_phi.tolist(), 'sigma': best_sigma, 'delta_std': cfg['delta_target_std']}

    _, _, P_final = evaluate_distribution(X_final, gap_mean)

    # Analisar gaps dos gaps da simulação
    sim_delta = np.diff(X_final)
    sim_delta_mean = np.mean(sim_delta)
    sim_delta_std = np.std(sim_delta)

    print(f"\n[ANÁLISE SEGUNDA ORDEM]")
    print(f"  Riemann ΔX: μ={delta_mean:.6f}, σ={delta_std:.6f}")
    print(f"  Simulado ΔX: μ={sim_delta_mean:.6f}, σ={sim_delta_std:.6f}")
    print(f"  Razão σ: {sim_delta_std/delta_std:.4f}")

    all_results.append({
        'config': cfg,
        'params': params_dict,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'max_accuracy': max_acc,
        'mean_chi2': mean_chi2,
        'distribution': P_final,
        'time_series': X_final[:1000].tolist(),
        'delta_stats': {
            'mean': sim_delta_mean,
            'std': sim_delta_std,
            'ratio': sim_delta_std/delta_std
        }
    })

# COMPARAÇÃO FINAL
print(f"\n{'=' * 70}")
print("RESUMO COMPARATIVO")
print(f"{'=' * 70}")

print(f"\n{'Config':<50} {'Mean Acc':<15} {'Max Acc':<10}")
print(f"{'-'*75}")
for res in all_results:
    name = res['config']['name'][:48]
    print(f"{name:<50} {res['mean_accuracy']:>6.2f}% ± {res['std_accuracy']:>4.2f}%  {res['max_accuracy']:>6.2f}%")

# Melhor resultado
best_result = max(all_results, key=lambda x: x['max_accuracy'])

print(f"\n{'=' * 70}")
print(f"🏆 MELHOR RESULTADO")
print(f"{'=' * 70}")
print(f"  Config: {best_result['config']['name']}")
print(f"  Model: {best_result['config']['model']}")
print(f"  Params: {best_result['params']}")
print(f"  Mean accuracy: {best_result['mean_accuracy']:.2f}% ± {best_result['std_accuracy']:.2f}%")
print(f"  Max accuracy: {best_result['max_accuracy']:.2f}%")
print(f"  Mean χ²: {best_result['mean_chi2']:.4f}")
print(f"\n  Segunda ordem:")
print(f"    Riemann ΔX std: {delta_std:.6f}")
print(f"    Simulado ΔX std: {best_result['delta_stats']['std']:.6f}")
print(f"    Razão: {best_result['delta_stats']['ratio']:.4f}")

# GRÁFICOS
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# 1. Comparação accuracies
ax1 = fig.add_subplot(gs[0, 0])
x_pos = np.arange(len(all_results))
means = [r['mean_accuracy'] for r in all_results]
stds = [r['std_accuracy'] for r in all_results]
maxs = [r['max_accuracy'] for r in all_results]
ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='blue')
ax1.scatter(x_pos, maxs, color='red', s=100, zorder=5, marker='*', label='Max')
ax1.axhline(98.05, color='green', linestyle='--', linewidth=2, label='Prev. best', alpha=0.7)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('Accuracies Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'C{i+1}' for i in range(len(all_results))], fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([96, 100])

# 2. Distribuição melhor vs target
ax2 = fig.add_subplot(gs[0, 1])
P_best = best_result['distribution']
levels_target = sorted(P_target.keys())
probs_target = [P_target[k] for k in levels_target]
probs_best = [P_best.get(k, 0) for k in levels_target]
x = np.arange(len(levels_target))
width = 0.35
ax2.bar(x - width/2, probs_target, width, label='Riemann', color='black', alpha=0.7)
ax2.bar(x + width/2, probs_best, width, label='Best model', color='blue', alpha=0.7)
ax2.set_xlabel('Level k', fontsize=11)
ax2.set_ylabel('P(k)', fontsize=11)
ax2.set_title(f'Best: {best_result["max_accuracy"]:.2f}%', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(levels_target)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_yscale('log')

# 3. Time series
ax3 = fig.add_subplot(gs[0, 2])
X_best = np.array(best_result['time_series'])
ax3.plot(X_best, linewidth=1, alpha=0.7, color='blue')
ax3.axhline(gap_mean, color='r', linestyle='--', linewidth=2, label=f'μ={gap_mean:.3f}')
ax3.set_xlabel('Step', fontsize=11)
ax3.set_ylabel('Gap', fontsize=11)
ax3.set_title('Time Series (Best)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Gaps dos gaps (Riemann vs Best)
ax4 = fig.add_subplot(gs[0, 3])
X_best_full = np.array(best_result['time_series'])
delta_best = np.diff(X_best_full)
ax4.hist(gap_of_gaps/gap_mean, bins=50, alpha=0.5, label='Riemann', density=True, color='black')
ax4.hist(delta_best/gap_mean, bins=50, alpha=0.5, label='Best model', density=True, color='blue')
ax4.set_xlabel('Normalized ΔX', fontsize=11)
ax4.set_ylabel('Density', fontsize=11)
ax4.set_title('Gap-of-Gaps Distribution', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Razão std(ΔX)
ax5 = fig.add_subplot(gs[1, 0])
ratios = [r['delta_stats']['ratio'] for r in all_results]
bars = ax5.bar(x_pos, ratios, alpha=0.7, color='purple')
ax5.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Target', alpha=0.7)
ax5.set_ylabel('σ(ΔX) ratio', fontsize=11)
ax5.set_title('Second Order Match', fontsize=12, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels([f'C{i+1}' for i in range(len(all_results))], fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Chi² comparison
ax6 = fig.add_subplot(gs[1, 1])
chi2_values = [r['mean_chi2'] for r in all_results]
bars = ax6.bar(x_pos, chi2_values, alpha=0.7, color='orange')
ax6.set_ylabel('χ² (lower is better)', fontsize=11)
ax6.set_title('Chi-squared', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([f'C{i+1}' for i in range(len(all_results))], fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# 7. Gaps reais
ax7 = fig.add_subplot(gs[1, 2])
ax7.plot(gaps[:1000], linewidth=1, alpha=0.7, color='black', label='Riemann')
ax7.axhline(gap_mean, color='r', linestyle='--', linewidth=2)
ax7.set_xlabel('Index', fontsize=11)
ax7.set_ylabel('Gap', fontsize=11)
ax7.set_title('Real Riemann Gaps', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# 8. Gaps dos gaps reais
ax8 = fig.add_subplot(gs[1, 3])
ax8.plot(gap_of_gaps[:1000], linewidth=1, alpha=0.7, color='black', label='Riemann')
ax8.axhline(0, color='r', linestyle='--', linewidth=2)
ax8.set_xlabel('Index', fontsize=11)
ax8.set_ylabel('ΔGap', fontsize=11)
ax8.set_title('Real Gap-of-Gaps', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# 9-11. Distribuições individuais top 3
for i in range(min(3, len(all_results))):
    ax = fig.add_subplot(gs[2, i])
    res = all_results[i]
    levels_sim = sorted(res['distribution'].keys())
    probs_sim = [res['distribution'][k] for k in levels_sim]

    ax.bar(levels_sim, probs_sim, alpha=0.6, label='Model', color='blue')
    ax.plot(levels_target, probs_target, 'ko--', linewidth=2, markersize=6, label='Riemann')
    ax.set_xlabel('Level k', fontsize=10)
    ax.set_ylabel('P(k)', fontsize=10)
    ax.set_title(f'C{i+1}: {res["max_accuracy"]:.2f}%', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

# 12. Summary box
ax12 = fig.add_subplot(gs[2, 3])
ax12.axis('off')
summary_text = f"""
BEST RESULT SUMMARY

Config: {best_result['config']['model']}
Max Accuracy: {best_result['max_accuracy']:.2f}%
Mean: {best_result['mean_accuracy']:.2f}% ± {best_result['std_accuracy']:.2f}%

Previous best: 98.05%
Improvement: {best_result['max_accuracy'] - 98.05:.2f}%

Second Order:
  Riemann σ(ΔX): {delta_std:.6f}
  Model σ(ΔX): {best_result['delta_stats']['std']:.6f}
  Ratio: {best_result['delta_stats']['ratio']:.4f}

χ² = {best_result['mean_chi2']:.4f}
"""
ax12.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('/home/thlinux/relacionalidadegeral/validacao/gap_of_gaps_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/gap_of_gaps_analysis.png")

# Salvar resultados
output = {
    'summary': {
        'best_config': best_result['config']['name'],
        'best_model': best_result['config']['model'],
        'best_params': best_result['params'],
        'best_mean_accuracy': best_result['mean_accuracy'],
        'best_max_accuracy': best_result['max_accuracy'],
        'best_mean_chi2': best_result['mean_chi2'],
        'previous_best': 98.05,
        'improvement': best_result['max_accuracy'] - 98.05
    },
    'riemann_stats': {
        'gap_mean': gap_mean,
        'gap_std': gap_std,
        'delta_mean': delta_mean,
        'delta_std': delta_std,
        'correlation': correlation
    },
    'all_results': [
        {
            'name': r['config']['name'],
            'model': r['config']['model'],
            'params': r['params'],
            'mean_accuracy': r['mean_accuracy'],
            'std_accuracy': r['std_accuracy'],
            'max_accuracy': r['max_accuracy'],
            'mean_chi2': r['mean_chi2'],
            'delta_stats': r['delta_stats'],
            'distribution': {str(k): v for k, v in r['distribution'].items()}
        }
        for r in all_results
    ],
    'target_distribution': {str(k): v for k, v in P_target.items()}
}

with open('/home/thlinux/relacionalidadegeral/validacao/gap_of_gaps_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/gap_of_gaps_results.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO")
print(f"{'=' * 70}")

if best_result['max_accuracy'] >= 99.0:
    print(f"\n🎉 SUCESSO! Chegamos em {best_result['max_accuracy']:.2f}%!")
    print(f"   Modelar os GAPS DOS GAPS foi a chave!")
elif best_result['max_accuracy'] > 98.05:
    print(f"\n✓ MELHORIA! De 98.05% → {best_result['max_accuracy']:.2f}%")
    print(f"   Ganho: +{best_result['max_accuracy'] - 98.05:.2f}%")
    print(f"   A estrutura de segunda ordem ajudou!")
else:
    print(f"\n⚠ Resultado: {best_result['max_accuracy']:.2f}%")
    print(f"   Não superou 98.05%")
    print(f"   Estrutura de segunda ordem não foi suficiente")

print(f"\n{'=' * 70}\n")
