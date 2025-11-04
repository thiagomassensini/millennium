#!/usr/bin/env python3
"""
Processos de Markov de Alta Ordem
Ordem 1 (OU): X_t depende de X_{t-1}
Ordem 2: X_t depende de X_{t-1}, X_{t-2}
Ordem 3: X_t depende de X_{t-1}, X_{t-2}, X_{t-3}

Modelo AR(p): X_t = μ + Σ φ_i (X_{t-i} - μ) + ε_t
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize

print("=" * 70)
print("PROCESSOS DE MARKOV DE ALTA ORDEM")
print("Objetivo: Chegar a 99% de accuracy")
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

print(f"\n[TARGET] Distribuição de Riemann:")
for level in sorted(P_real.keys()):
    print(f"  Level {level}: {100*P_real[level]:.1f}%")

# Constante γ
gamma_euler = 0.5772156649015329

def simulate_markov_order_p(order, phi_params, mu, sigma_noise, n_steps, dt=0.01):
    """
    Simula processo de Markov de ordem p
    X_t = μ + Σ_{i=1}^p φ_i (X_{t-i} - μ) + σ·ε_t
    """
    X = np.zeros(n_steps)

    # Inicialização com média
    for i in range(order):
        X[i] = mu + sigma_noise * np.random.randn()

    # Simulação
    for t in range(order, n_steps):
        # Termo autoregressivo
        ar_term = 0
        for i in range(order):
            ar_term += phi_params[i] * (X[t-1-i] - mu)

        # Ruído
        noise = sigma_noise * np.sqrt(dt) * np.random.randn()

        X[t] = mu + ar_term + noise
        X[t] = max(0.01, X[t])  # Evitar negativos

    return X

def compute_accuracy(X, P_real):
    """Calcular accuracy da distribuição gerada"""
    normalized = X / np.mean(X)
    normalized = np.clip(normalized, 1e-10, None)
    levels = np.floor(np.log2(normalized)).astype(int)
    unique_levels, counts = np.unique(levels, return_counts=True)
    P_emergent = {int(lv): cnt/len(levels) for lv, cnt in zip(unique_levels, counts)}

    # Chi-squared
    chi2 = 0
    for level in P_real.keys():
        obs = P_emergent.get(level, 0)
        exp = P_real[level]
        if exp > 0:
            chi2 += (obs - exp)**2 / exp

    accuracy = max(0, 1 - chi2/10.0) * 100
    return accuracy, chi2, P_emergent

# CONFIGURAÇÕES
configs = [
    {
        'name': 'Ordem 1 (OU baseline)',
        'order': 1,
        'phi': [1.0],  # θ=1 no OU
        'mu': gap_mean,
        'sigma': gap_std * 0.5
    },
    {
        'name': 'Ordem 1 (γ otimizado)',
        'order': 1,
        'phi': [gamma_euler / np.log(gap_mean + 1)],
        'mu': gap_mean,
        'sigma': gap_std * 0.5
    },
    {
        'name': 'Ordem 2 (AR2 simétrico)',
        'order': 2,
        'phi': [0.8, 0.2],  # φ1=0.8, φ2=0.2
        'mu': gap_mean,
        'sigma': gap_std * 0.4
    },
    {
        'name': 'Ordem 2 (AR2 com γ)',
        'order': 2,
        'phi': [gamma_euler, 1.0 - gamma_euler],
        'mu': gap_mean,
        'sigma': gap_std * 0.4
    },
    {
        'name': 'Ordem 3 (AR3 balanceado)',
        'order': 3,
        'phi': [0.6, 0.3, 0.1],
        'mu': gap_mean,
        'sigma': gap_std * 0.35
    },
    {
        'name': 'Ordem 3 (AR3 com γ)',
        'order': 3,
        'phi': [gamma_euler, 0.3, 0.1],
        'mu': gap_mean,
        'sigma': gap_std * 0.35
    },
    {
        'name': 'Ordem 3 (AR3 decaimento exponencial)',
        'order': 3,
        'phi': [0.5, 0.25, 0.125],  # Decai como 2^-n
        'mu': gap_mean,
        'sigma': gap_std * 0.35
    },
]

all_results = []

for cfg in configs:
    print(f"\n{'=' * 70}")
    print(f"TESTANDO: {cfg['name']}")
    print(f"  Ordem: {cfg['order']}")
    print(f"  φ = {cfg['phi']}")
    print(f"  μ = {cfg['mu']:.4f}")
    print(f"  σ = {cfg['sigma']:.4f}")
    print(f"{'=' * 70}")

    # Simular múltiplas vezes para estatística
    n_runs = 10
    accuracies = []

    for run in range(n_runs):
        X = simulate_markov_order_p(
            order=cfg['order'],
            phi_params=cfg['phi'],
            mu=cfg['mu'],
            sigma_noise=cfg['sigma'],
            n_steps=10000
        )

        acc, chi2, P_em = compute_accuracy(X, P_real)
        accuracies.append(acc)

    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)

    # Melhor run
    X_best = simulate_markov_order_p(
        order=cfg['order'],
        phi_params=cfg['phi'],
        mu=cfg['mu'],
        sigma_noise=cfg['sigma'],
        n_steps=10000
    )
    acc_best, chi2_best, P_best = compute_accuracy(X_best, P_real)

    print(f"\n[RESULTADOS]")
    print(f"  Accuracy média: {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f"  Melhor run: {acc_best:.2f}%")
    print(f"  χ² (melhor): {chi2_best:.4f}")

    if acc_mean >= 99.0:
        print(f"  ✅ OBJETIVO ALCANÇADO: ≥99%!")
    elif acc_mean >= 97.0:
        print(f"  🟢 Muito próximo! (≥97%)")
    elif acc_mean >= 95.0:
        print(f"  🟡 Bom resultado (≥95%)")

    all_results.append({
        'config': cfg,
        'accuracy_mean': acc_mean,
        'accuracy_std': acc_std,
        'accuracy_best': acc_best,
        'chi2_best': chi2_best,
        'distribution': P_best,
        'series': X_best
    })

# OTIMIZAÇÃO: Buscar melhor φ para ordem 3
print(f"\n{'=' * 70}")
print("OTIMIZAÇÃO: Buscar melhor φ para AR(3)")
print(f"{'=' * 70}")

def objective_ar3(phi_params):
    """Função objetivo: minimizar -accuracy"""
    if len(phi_params) != 3:
        return 1e10

    # Restrições de estabilidade (soma < 1)
    if np.sum(np.abs(phi_params)) > 1.5:
        return 1e10

    # Simular
    X = simulate_markov_order_p(
        order=3,
        phi_params=phi_params,
        mu=gap_mean,
        sigma_noise=gap_std * 0.35,
        n_steps=10000
    )

    acc, _, _ = compute_accuracy(X, P_real)
    return -acc  # Minimizar = maximizar accuracy

# Otimizar
print(f"\n  Otimizando φ = (φ1, φ2, φ3)...")
initial_guess = [0.6, 0.3, 0.1]
bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

result = minimize(objective_ar3, initial_guess, method='Powell',
                  options={'maxiter': 50, 'disp': False})

phi_optimal = result.x
acc_optimal = -result.fun

print(f"\n[OTIMIZAÇÃO COMPLETA]")
print(f"  φ_optimal = [{phi_optimal[0]:.4f}, {phi_optimal[1]:.4f}, {phi_optimal[2]:.4f}]")
print(f"  Accuracy: {acc_optimal:.2f}%")

# Validar com múltiplas runs
acc_validation = []
for _ in range(10):
    X = simulate_markov_order_p(3, phi_optimal, gap_mean, gap_std * 0.35, 10000)
    acc, _, _ = compute_accuracy(X, P_real)
    acc_validation.append(acc)

acc_val_mean = np.mean(acc_validation)
acc_val_std = np.std(acc_validation)

print(f"  Validação (10 runs): {acc_val_mean:.2f}% ± {acc_val_std:.2f}%")

if acc_val_mean >= 99.0:
    print(f"  ✅✅✅ OBJETIVO ALCANÇADO: ≥99%! ✅✅✅")

# Adicionar aos resultados
X_opt = simulate_markov_order_p(3, phi_optimal, gap_mean, gap_std * 0.35, 10000)
acc_opt, chi2_opt, P_opt = compute_accuracy(X_opt, P_real)

all_results.append({
    'config': {
        'name': 'Ordem 3 (OTIMIZADO)',
        'order': 3,
        'phi': phi_optimal.tolist(),
        'mu': gap_mean,
        'sigma': gap_std * 0.35
    },
    'accuracy_mean': acc_val_mean,
    'accuracy_std': acc_val_std,
    'accuracy_best': acc_opt,
    'chi2_best': chi2_opt,
    'distribution': P_opt,
    'series': X_opt
})

# Comparação final
print(f"\n{'=' * 70}")
print("RANKING FINAL")
print(f"{'=' * 70}")

sorted_results = sorted(all_results, key=lambda x: x['accuracy_mean'], reverse=True)

print(f"\n{'Rank':<6} {'Config':<35} {'Accuracy':<20} {'χ²':<10}")
print(f"{'-'*75}")
for i, res in enumerate(sorted_results):
    name = res['config']['name'][:33]
    acc = res['accuracy_mean']
    std = res['accuracy_std']
    chi2 = res['chi2_best']

    status = "✅" if acc >= 99 else "🟢" if acc >= 97 else "🟡" if acc >= 95 else "  "
    print(f"{i+1:<6} {name:<35} {acc:>6.2f}% ± {std:>4.2f}%    {chi2:>6.4f}  {status}")

best = sorted_results[0]

print(f"\n{'=' * 70}")
print("MELHOR CONFIGURAÇÃO")
print(f"{'=' * 70}")
print(f"\nConfig: {best['config']['name']}")
print(f"  Ordem: {best['config']['order']}")
print(f"  φ = {best['config']['phi']}")
print(f"  Accuracy: {best['accuracy_mean']:.2f}% ± {best['accuracy_std']:.2f}%")
print(f"  χ² = {best['chi2_best']:.4f}")

# Gráficos
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Comparação de accuracies
ax1 = axes[0, 0]
names = [f"O{r['config']['order']}" for r in all_results]
accs = [r['accuracy_mean'] for r in all_results]
stds = [r['accuracy_std'] for r in all_results]
colors = ['darkgreen' if a >= 99 else 'green' if a >= 97 else 'yellow' if a >= 95 else 'orange'
          for a in accs]

bars = ax1.bar(range(len(names)), accs, yerr=stds, color=colors, alpha=0.7, capsize=5)
ax1.axhline(99, color='darkgreen', linestyle='--', linewidth=2, label='99% target')
ax1.axhline(95, color='green', linestyle=':', linewidth=2, label='95%')
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=45, ha='right')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Comparação: Ordens de Markov', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([90, 102])

# 2. Distribuições (top 3)
ax2 = axes[0, 1]
riemann_levels = sorted(P_real.keys())
riemann_probs = [P_real[k] for k in riemann_levels]
ax2.plot(riemann_levels, riemann_probs, 'k^--', linewidth=3, markersize=8,
         label='Riemann', alpha=0.8)

for i, res in enumerate(sorted_results[:3]):
    P_em = res['distribution']
    levels = sorted(P_em.keys())
    probs = [P_em[k] for k in levels]
    ax2.plot(levels, probs, 'o-', linewidth=2, markersize=6,
             label=f"#{i+1}: {res['config']['name'][:15]}", alpha=0.7)

ax2.set_xlabel('Level k', fontsize=12)
ax2.set_ylabel('P(k)', fontsize=12)
ax2.set_title('Top 3 Distribuições', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# 3. Séries temporais (melhor)
ax3 = axes[0, 2]
ax3.plot(best['series'][:500], linewidth=1, alpha=0.7, color='blue')
ax3.axhline(gap_mean, color='r', linestyle='--', linewidth=2, label=f'μ = {gap_mean:.2f}')
ax3.set_xlabel('Step', fontsize=12)
ax3.set_ylabel('Gap value', fontsize=12)
ax3.set_title(f'Série Temporal (Melhor: O{best["config"]["order"]})',
              fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Accuracy vs Ordem
ax4 = axes[1, 0]
orders = [r['config']['order'] for r in all_results]
unique_orders = sorted(set(orders))
acc_by_order = {o: [] for o in unique_orders}
for r in all_results:
    acc_by_order[r['config']['order']].append(r['accuracy_mean'])

order_means = [np.mean(acc_by_order[o]) for o in unique_orders]
order_maxs = [np.max(acc_by_order[o]) for o in unique_orders]

ax4.plot(unique_orders, order_means, 'bo-', linewidth=3, markersize=10, label='Média')
ax4.plot(unique_orders, order_maxs, 'r^--', linewidth=2, markersize=8, label='Máximo')
ax4.axhline(99, color='darkgreen', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_xlabel('Ordem do Processo', fontsize=12)
ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.set_title('Accuracy vs Ordem de Markov', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks(unique_orders)

# 5. χ² comparison
ax5 = axes[1, 1]
chi2_vals = [r['chi2_best'] for r in all_results]
ax5.barh(range(len(names)), chi2_vals, color=colors, alpha=0.7)
ax5.set_yticks(range(len(names)))
ax5.set_yticklabels(names)
ax5.set_xlabel('χ²', fontsize=12)
ax5.set_title('Convergência χ²', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')
ax5.invert_yaxis()

# 6. φ parameters (ordem 3)
ax6 = axes[1, 2]
ar3_results = [r for r in all_results if r['config']['order'] == 3]
phi_matrix = np.array([r['config']['phi'] for r in ar3_results])
labels_ar3 = [r['config']['name'][:20] for r in ar3_results]

x_pos = np.arange(len(ar3_results))
width = 0.25
for i in range(3):
    ax6.bar(x_pos + i*width, phi_matrix[:, i], width, label=f'φ{i+1}', alpha=0.7)

ax6.set_xticks(x_pos + width)
ax6.set_xticklabels(labels_ar3, rotation=45, ha='right', fontsize=8)
ax6.set_ylabel('φ value', fontsize=12)
ax6.set_title('Parâmetros φ para AR(3)', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/thlinux/relacionalidadegeral/validacao/markov_higher_order_test.png',
            dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/markov_higher_order_test.png")

# Salvar resultados
output = {
    'configurations': [
        {
            'name': r['config']['name'],
            'order': r['config']['order'],
            'phi': r['config']['phi'],
            'accuracy_mean': float(r['accuracy_mean']),
            'accuracy_std': float(r['accuracy_std']),
            'chi2_best': float(r['chi2_best'])
        }
        for r in all_results
    ],
    'best': {
        'name': best['config']['name'],
        'order': best['config']['order'],
        'phi': best['config']['phi'],
        'accuracy': float(best['accuracy_mean'])
    }
}

with open('/home/thlinux/relacionalidadegeral/validacao/markov_higher_order_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/markov_higher_order_results.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO")
print(f"{'=' * 70}")

if best['accuracy_mean'] >= 99.0:
    print(f"\n✅✅✅ SUCESSO! Objetivo alcançado!")
    print(f"   Melhor accuracy: {best['accuracy_mean']:.2f}%")
    print(f"   Ordem: {best['config']['order']}")
    print(f"   Parâmetros: φ = {best['config']['phi']}")
elif best['accuracy_mean'] >= 97.0:
    print(f"\n🟢 Muito próximo do objetivo!")
    print(f"   Melhor accuracy: {best['accuracy_mean']:.2f}%")
    print(f"   Falta: {99.0 - best['accuracy_mean']:.2f}%")
else:
    print(f"\n🟡 Melhor resultado: {best['accuracy_mean']:.2f}%")
    print(f"   Ordem superior ajuda, mas platô persiste")

print(f"\n[INSIGHT]")
if best['config']['order'] > 1:
    print(f"  Processos de Markov de ordem {best['config']['order']} capturam")
    print(f"  mais estrutura que ordem 1 (OU)!")
else:
    print(f"  Ordem 1 (OU) já é suficiente - correlações")
    print(f"  de ordem superior não agregam muito.")

print(f"\n{'=' * 70}\n")
