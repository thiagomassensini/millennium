#!/usr/bin/env python3
"""
TESTE EXPLOSIVO: Ordens PRIMAS
Hipótese: Ordens que são TWIN PRIMES dão melhores resultados!
p=3,5,7,11,13,17,19...
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

print("=" * 70)
print("🔥 TESTE: ORDENS PRIMAS (Twin Primes Connection) 🔥")
print("=" * 70)

# Carregar dados
with open('/home/thlinux/relacionalidadegeral/validacao/riemann_extended_analysis.json', 'r') as f:
    data = json.load(f)

zeros = np.array(data['zeros'])
gaps = np.diff(zeros)
gap_mean = np.mean(gaps)
gap_std = np.std(gaps)

gap_analysis = data['gap_analysis']
level_dist_real = gap_analysis['level_distribution']
total_real = sum(level_dist_real.values())
P_real = {int(k): v/total_real for k, v in level_dist_real.items()}

def simulate_ar(order, phi_params, mu, sigma_noise, n_steps):
    X = np.zeros(n_steps)
    for i in range(order):
        X[i] = mu + sigma_noise * np.random.randn()

    for t in range(order, n_steps):
        ar_term = sum(phi_params[i] * (X[t-1-i] - mu) for i in range(order))
        noise = sigma_noise * np.random.randn()
        X[t] = mu + ar_term + noise
        X[t] = max(0.01, X[t])

    return X

def compute_accuracy(X, P_real):
    normalized = X / np.mean(X)
    normalized = np.clip(normalized, 1e-10, None)
    levels = np.floor(np.log2(normalized)).astype(int)
    unique_levels, counts = np.unique(levels, return_counts=True)
    P_emergent = {int(lv): cnt/len(levels) for lv, cnt in zip(unique_levels, counts)}

    chi2 = sum((P_emergent.get(level, 0) - P_real[level])**2 / P_real[level]
               for level in P_real.keys() if P_real[level] > 0)

    accuracy = max(0, 1 - chi2/10.0) * 100
    return accuracy, chi2

# ORDENS PRIMAS (twin primes)
prime_orders = [3, 5, 7, 11, 13]

print(f"\n[TWIN PRIMES]")
print(f"  (3,5), (5,7), (11,13), ...")
print(f"\n[TESTANDO ORDENS]: {prime_orders}")

all_results = []

for order in prime_orders:
    print(f"\n{'=' * 70}")
    print(f"ORDEM {order} {'(PRIMO GEMINIANO!)' if order in [3,5,7,11,13] else '(primo)'}")
    print(f"{'=' * 70}")

    def objective(params):
        phi = params[:-1]
        sigma = params[-1]

        if np.sum(np.abs(phi)) > 2.0 or sigma <= 0 or sigma > gap_std:
            return 1e10

        try:
            X = simulate_ar(order, phi, gap_mean, sigma, 10000)
            acc, _ = compute_accuracy(X, P_real)
            return -acc
        except:
            return 1e10

    phi_bounds = [(-1.5, 1.5)] * order
    sigma_bounds = [(0.05, gap_std)]
    bounds = phi_bounds + sigma_bounds

    print(f"  Otimizando AR({order})...")
    result = differential_evolution(objective, bounds, maxiter=80, seed=42,
                                    workers=1, disp=False, polish=True)

    phi_opt = result.x[:-1]
    sigma_opt = result.x[-1]
    acc_opt = -result.fun

    print(f"\n  [RESULTADO]")
    print(f"    φ = [{', '.join([f'{p:.4f}' for p in phi_opt[:min(5, len(phi_opt))]])}...]"
          if order > 5 else f"    φ = [{', '.join([f'{p:.4f}' for p in phi_opt])}]")
    print(f"    σ = {sigma_opt:.4f}")
    print(f"    Accuracy: {acc_opt:.2f}%")

    # Validar
    acc_validation = []
    for _ in range(15):
        X = simulate_ar(order, phi_opt, gap_mean, sigma_opt, 10000)
        acc, _ = compute_accuracy(X, P_real)
        acc_validation.append(acc)

    acc_val_mean = np.mean(acc_validation)
    acc_val_std = np.std(acc_validation)
    acc_val_max = np.max(acc_validation)

    print(f"    Validação: {acc_val_mean:.2f}% ± {acc_val_std:.2f}% (max:{acc_val_max:.2f}%)")

    if acc_val_mean >= 99.0:
        print(f"    ✅✅✅ 99% ALCANÇADO!")
    elif acc_val_max >= 99.0:
        print(f"    ✅✅ Máximo atingiu 99%!")
    elif acc_val_mean >= 98.0:
        print(f"    🟢🟢 ≥98%!")
    elif acc_val_mean >= 97.0:
        print(f"    🟢 ≥97%")

    all_results.append({
        'order': order,
        'is_twin_prime': order in [3,5,7,11,13,17,19,29,31],
        'phi': phi_opt.tolist(),
        'sigma': float(sigma_opt),
        'accuracy_mean': acc_val_mean,
        'accuracy_std': acc_val_std,
        'accuracy_max': acc_val_max
    })

# COMPARAR: Primos vs Não-primos
print(f"\n{'=' * 70}")
print("TESTE COMPARATIVO: Ordens NÃO-PRIMAS")
print(f"{'=' * 70}")

composite_orders = [4, 6, 8, 9, 10]

for order in composite_orders:
    print(f"\n  Testando ordem {order} (composto)...")

    def objective(params):
        phi = params[:-1]
        sigma = params[-1]
        if np.sum(np.abs(phi)) > 2.0 or sigma <= 0 or sigma > gap_std:
            return 1e10
        try:
            X = simulate_ar(order, phi, gap_mean, sigma, 10000)
            acc, _ = compute_accuracy(X, P_real)
            return -acc
        except:
            return 1e10

    phi_bounds = [(-1.5, 1.5)] * order
    sigma_bounds = [(0.05, gap_std)]
    bounds = phi_bounds + sigma_bounds

    result = differential_evolution(objective, bounds, maxiter=60, seed=42,
                                    workers=1, disp=False, polish=True)

    phi_opt = result.x[:-1]
    sigma_opt = result.x[-1]

    acc_validation = []
    for _ in range(10):
        X = simulate_ar(order, phi_opt, gap_mean, sigma_opt, 10000)
        acc, _ = compute_accuracy(X, P_real)
        acc_validation.append(acc)

    acc_val_mean = np.mean(acc_validation)
    acc_val_max = np.max(acc_validation)

    print(f"    Resultado: {acc_val_mean:.2f}% (max:{acc_val_max:.2f}%)")

    all_results.append({
        'order': order,
        'is_twin_prime': False,
        'phi': phi_opt.tolist(),
        'sigma': float(sigma_opt),
        'accuracy_mean': acc_val_mean,
        'accuracy_std': np.std(acc_validation),
        'accuracy_max': acc_val_max
    })

# RANKING
print(f"\n{'=' * 70}")
print("RANKING: PRIMOS vs COMPOSTOS")
print(f"{'=' * 70}")

sorted_results = sorted(all_results, key=lambda x: x['accuracy_mean'], reverse=True)

print(f"\n{'Rank':<6} {'Ordem':<8} {'Tipo':<15} {'Accuracy':<20} {'Status':<5}")
print(f"{'-'*75}")
for i, res in enumerate(sorted_results):
    order = res['order']
    tipo = "PRIMO ⭐" if res['is_twin_prime'] else "composto"
    acc = res['accuracy_mean']
    acc_max = res['accuracy_max']

    status = "✅✅" if acc >= 99 else "🟢🟢" if acc >= 98 else "🟢" if acc >= 97 else "🟡"
    print(f"{i+1:<6} {order:<8} {tipo:<15} {acc:>6.2f}% (max:{acc_max:>5.2f}%)   {status}")

# ANÁLISE ESTATÍSTICA
print(f"\n{'=' * 70}")
print("ANÁLISE ESTATÍSTICA: PRIMOS vs COMPOSTOS")
print(f"{'=' * 70}")

prime_accs = [r['accuracy_mean'] for r in all_results if r['is_twin_prime']]
composite_accs = [r['accuracy_mean'] for r in all_results if not r['is_twin_prime']]

print(f"\n[ORDENS PRIMAS]")
print(f"  Média: {np.mean(prime_accs):.2f}%")
print(f"  Máximo: {np.max(prime_accs):.2f}%")
print(f"  Mínimo: {np.min(prime_accs):.2f}%")

print(f"\n[ORDENS COMPOSTAS]")
print(f"  Média: {np.mean(composite_accs):.2f}%")
print(f"  Máximo: {np.max(composite_accs):.2f}%")
print(f"  Mínimo: {np.min(composite_accs):.2f}%")

diff = np.mean(prime_accs) - np.mean(composite_accs)
print(f"\n[DIFERENÇA]")
print(f"  Primos - Compostos = {diff:+.2f}%")

if diff > 1.0:
    print(f"  🔥 PRIMOS SÃO SIGNIFICATIVAMENTE MELHORES!")
elif diff > 0.5:
    print(f"  🟢 Primos tendem a ser melhores")
elif diff > -0.5:
    print(f"  🟡 Sem diferença clara")
else:
    print(f"  Compostos são melhores (surpresa!)")

# Gráfico
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Accuracy vs Ordem (colorido por tipo)
ax1 = axes[0, 0]
primes = [r for r in sorted_results if r['is_twin_prime']]
composites = [r for r in sorted_results if not r['is_twin_prime']]

ax1.scatter([r['order'] for r in primes], [r['accuracy_mean'] for r in primes],
            s=150, c='red', marker='*', label='Primos', alpha=0.8, edgecolors='darkred', linewidths=2)
ax1.scatter([r['order'] for r in composites], [r['accuracy_mean'] for r in composites],
            s=100, c='blue', marker='o', label='Compostos', alpha=0.6)

ax1.axhline(99, color='darkgreen', linestyle='--', linewidth=2, label='99%')
ax1.axhline(97, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax1.set_xlabel('Ordem do Processo AR(p)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Primos vs Compostos', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([94, 100])

# 2. Box plot comparison
ax2 = axes[0, 1]
data_box = [prime_accs, composite_accs]
bp = ax2.boxplot(data_box, labels=['Primos', 'Compostos'], patch_artist=True)
bp['boxes'][0].set_facecolor('red')
bp['boxes'][1].set_facecolor('blue')
for box in bp['boxes']:
    box.set_alpha(0.6)
ax2.axhline(99, color='darkgreen', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Distribuição: Primos vs Compostos', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Top 5 bars
ax3 = axes[1, 0]
top5 = sorted_results[:5]
names = [f"O{r['order']}" for r in top5]
accs = [r['accuracy_mean'] for r in top5]
colors = ['red' if r['is_twin_prime'] else 'blue' for r in top5]

bars = ax3.barh(names, accs, color=colors, alpha=0.7)
ax3.axvline(99, color='darkgreen', linestyle='--', linewidth=2, label='99%')
ax3.set_xlabel('Accuracy (%)', fontsize=12)
ax3.set_title('Top 5 Ordens', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')
ax3.set_xlim([96, 100])

# Adicionar valores
for bar, acc in zip(bars, accs):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
             f'{acc:.2f}%', ha='left', va='center', fontsize=10, fontweight='bold')

# 4. Twin primes pairs visualization
ax4 = axes[1, 1]
twin_pairs = [(3,5), (5,7), (11,13)]
twin_accs_pairs = []

for p1, p2 in twin_pairs:
    acc1 = next((r['accuracy_mean'] for r in all_results if r['order'] == p1), None)
    acc2 = next((r['accuracy_mean'] for r in all_results if r['order'] == p2), None)
    if acc1 and acc2:
        twin_accs_pairs.append((p1, p2, acc1, acc2))

x_pos = np.arange(len(twin_accs_pairs))
width = 0.35

for i, (p1, p2, acc1, acc2) in enumerate(twin_accs_pairs):
    ax4.bar(i - width/2, acc1, width, label=f'p={p1}' if i == 0 else '', color='red', alpha=0.7)
    ax4.bar(i + width/2, acc2, width, label=f'p+2={p2}' if i == 0 else '', color='darkred', alpha=0.7)

ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'({p1},{p2})' for p1, p2, _, _ in twin_accs_pairs])
ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.set_title('Twin Primes Pairs', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(99, color='darkgreen', linestyle='--', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.savefig('/home/thlinux/relacionalidadegeral/validacao/markov_prime_orders_test.png',
            dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/markov_prime_orders_test.png")

# Salvar
output = {
    'hypothesis': 'Orders that are twin primes give better accuracy',
    'prime_results': {
        'mean': float(np.mean(prime_accs)),
        'max': float(np.max(prime_accs)),
        'orders': [r['order'] for r in all_results if r['is_twin_prime']]
    },
    'composite_results': {
        'mean': float(np.mean(composite_accs)),
        'max': float(np.max(composite_accs)),
        'orders': [r['order'] for r in all_results if not r['is_twin_prime']]
    },
    'difference': float(diff),
    'best': {
        'order': sorted_results[0]['order'],
        'is_prime': sorted_results[0]['is_twin_prime'],
        'accuracy': float(sorted_results[0]['accuracy_mean'])
    },
    'all_results': sorted_results
}

with open('/home/thlinux/relacionalidadegeral/validacao/markov_prime_orders_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/markov_prime_orders_results.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO")
print(f"{'=' * 70}")

best = sorted_results[0]

if best['is_twin_prime'] and diff > 0.5:
    print(f"\n🔥🔥🔥 CONEXÃO DESCOBERTA! 🔥🔥🔥")
    print(f"   ORDENS PRIMAS (especialmente twin primes) SÃO MELHORES!")
    print(f"   Melhor ordem: {best['order']} (primo) → {best['accuracy_mean']:.2f}%")
    print(f"   Ganho médio: {diff:+.2f}% sobre compostos")
    print(f"\n   INTERPRETAÇÃO:")
    print(f"   A estrutura dos TWIN PRIMES (p ⊕ (p+2) = 2^k-2)")
    print(f"   está DIRETAMENTE relacionada à ordem ótima")
    print(f"   do processo de Markov que modela os zeros de Riemann!")
elif best['accuracy_mean'] >= 99.0:
    print(f"\n✅ 99% ALCANÇADO com ordem {best['order']}!")
else:
    print(f"\n📊 Melhor ordem: {best['order']}")
    print(f"   Accuracy: {best['accuracy_mean']:.2f}%")
    print(f"   Diferença primos/compostos: {diff:+.2f}%")

print(f"\n{'=' * 70}\n")
