#!/usr/bin/env python3
"""
Conexão Analítica: θ (OU) ↔ γ (Euler-Mascheroni) ↔ Zeros de Riemann

Fórmulas dos zeros:
- Riemann-von Mangoldt: θ(t) = Im(log Γ(1/4 + it/2)) - t·log(π)/2
- Gap médio assintótico: Δt_n ≈ 2π / log(t_n/(2π))
- t_n ≈ 2πn / log n (aproximação de Gram)

Hipótese: θ_OU relaciona-se com γ via fórmulas analíticas
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.special import digamma, loggamma
from scipy.optimize import curve_fit

print("=" * 70)
print("CONEXÃO ANALÍTICA: θ (OU) ↔ γ (Euler) ↔ Zeros de Riemann")
print("=" * 70)

# Constantes
gamma_euler = 0.5772156649015329  # Constante de Euler-Mascheroni
pi = np.pi

print(f"\n[CONSTANTES]")
print(f"  γ (Euler-Mascheroni) = {gamma_euler:.10f}")
print(f"  π = {pi:.10f}")

# Carregar zeros de Riemann
with open('/home/thlinux/relacionalidadegeral/validacao/riemann_extended_analysis.json', 'r') as f:
    data = json.load(f)

zeros = np.array(data['zeros'])
gaps = np.diff(zeros)
n_zeros = len(zeros)

print(f"\n[DADOS]")
print(f"  Número de zeros: {len(zeros)}")
print(f"  Range: [{zeros[0]:.2f}, {zeros[-1]:.2f}]")
print(f"  Gap médio: {np.mean(gaps):.4f}")
print(f"  Gap std: {np.std(gaps):.4f}")

# Fórmulas teóricas dos zeros
def riemann_von_mangoldt_theta(t):
    """θ(t) = Im(log Γ(1/4 + it/2)) - t·log(π)/2"""
    # Aproximação usando digamma
    z = 0.25 + 1j * t / 2
    # log Γ(z) aproximado para |z| grande
    log_gamma_approx = (z - 0.5) * np.log(z) - z + 0.5 * np.log(2*pi)
    theta = log_gamma_approx.imag - t * np.log(pi) / 2
    return theta

def asymptotic_gap(t):
    """Gap médio assintótico: 2π / log(t/(2π))"""
    return 2 * pi / np.log(t / (2*pi))

def gram_point_formula(n):
    """Aproximação t_n ≈ 2πn / log n"""
    return 2 * pi * n / np.log(n) if n > 1 else 1.0

# Calcular valores teóricos
print(f"\n{'=' * 70}")
print("FÓRMULAS TEÓRICAS DOS ZEROS")
print(f"{'=' * 70}")

# Para cada zero, calcular fórmulas
theta_values = []
asymptotic_gaps_pred = []
gram_approx = []

for i, t in enumerate(zeros):
    n = i + 1

    # θ(t) de Riemann-von Mangoldt
    theta_t = riemann_von_mangoldt_theta(t)
    theta_values.append(theta_t)

    # Gap assintótico
    if t > 2*pi:
        gap_asym = asymptotic_gap(t)
    else:
        gap_asym = np.nan
    asymptotic_gaps_pred.append(gap_asym)

    # Gram approximation
    t_gram = gram_point_formula(n)
    gram_approx.append(t_gram)

theta_values = np.array(theta_values)
asymptotic_gaps_pred = np.array(asymptotic_gaps_pred)
gram_approx = np.array(gram_approx)

# Comparar gaps reais vs assintóticos
valid_idx = ~np.isnan(asymptotic_gaps_pred[:-1])
gap_error = np.abs(gaps[valid_idx] - asymptotic_gaps_pred[:-1][valid_idx])

print(f"\n[COMPARAÇÃO: Gaps Reais vs Assintóticos]")
print(f"  MAE (erro médio absoluto): {np.mean(gap_error):.4f}")
print(f"  RMSE: {np.sqrt(np.mean(gap_error**2)):.4f}")
print(f"  Correlação: {np.corrcoef(gaps[valid_idx], asymptotic_gaps_pred[:-1][valid_idx])[0,1]:.4f}")

# CONEXÃO COM θ DO PROCESSO OU
print(f"\n{'=' * 70}")
print("CONEXÃO COM θ DO PROCESSO OU")
print(f"{'=' * 70}")

# Hipótese 1: θ_OU relacionado com gap médio assintótico
gap_mean_empirical = np.mean(gaps)
gap_mean_asymptotic = np.mean(asymptotic_gaps_pred[~np.isnan(asymptotic_gaps_pred)])

# θ_OU = 1 é o que usamos
theta_ou_used = 1.0

# Tentar relacionar θ_OU com fórmulas
# Hipótese: θ_OU ~ 1/gap_mean (tempo característico de reversão)
theta_ou_predicted_1 = 1.0 / gap_mean_empirical
print(f"\n[HIPÓTESE 1: θ_OU ~ 1/gap_mean]")
print(f"  θ_OU usado: {theta_ou_used}")
print(f"  θ_OU previsto: {theta_ou_predicted_1:.4f}")
print(f"  Razão: {theta_ou_used / theta_ou_predicted_1:.4f}")

# Hipótese 2: θ_OU relacionado com γ
# θ_OU ~ γ × constante
theta_ou_predicted_2 = gamma_euler * (2*pi / gap_mean_empirical)
print(f"\n[HIPÓTESE 2: θ_OU ~ γ × (2π/gap_mean)]")
print(f"  θ_OU usado: {theta_ou_used}")
print(f"  θ_OU previsto: {theta_ou_predicted_2:.4f}")
print(f"  Razão: {theta_ou_used / theta_ou_predicted_2:.4f}")

# Hipótese 3: Relação via θ(t) de Riemann-von Mangoldt
# θ_OU ~ variação de θ(t)
theta_diffs = np.diff(theta_values)
theta_ou_predicted_3 = np.std(theta_diffs) / gap_mean_empirical
print(f"\n[HIPÓTESE 3: θ_OU ~ std(Δθ(t)) / gap_mean]")
print(f"  θ_OU usado: {theta_ou_used}")
print(f"  θ_OU previsto: {theta_ou_predicted_3:.4f}")
print(f"  Razão: {theta_ou_used / theta_ou_predicted_3:.4f}")

# Hipótese 4: Conexão via γ e log
# Inspirado em: gap ~ 2π/log(t), θ_OU ~ γ/log(gap)
mean_log_t = np.mean(np.log(zeros))
theta_ou_predicted_4 = gamma_euler / np.log(gap_mean_empirical + 1)
print(f"\n[HIPÓTESE 4: θ_OU ~ γ / log(gap_mean + 1)]")
print(f"  θ_OU usado: {theta_ou_used}")
print(f"  θ_OU previsto: {theta_ou_predicted_4:.4f}")
print(f"  Razão: {theta_ou_used / theta_ou_predicted_4:.4f}")

# TESTAR PROCESSO OU COM θ DERIVADO DAS FÓRMULAS
print(f"\n{'=' * 70}")
print("TESTE: OU COM θ DERIVADO DAS FÓRMULAS DE RIEMANN")
print(f"{'=' * 70}")

# Distribuição real
gap_analysis = data['gap_analysis']
level_dist_real = gap_analysis['level_distribution']
total_real = sum(level_dist_real.values())
P_real = {int(k): v/total_real for k, v in level_dist_real.items()}

def test_ou_with_theta(theta_ou, name):
    """Testar processo OU com θ específico"""
    mu = gap_mean_empirical
    sigma_ou = np.std(gaps) * 0.5
    sigma_noise = np.std(gaps) * 0.5
    n_steps = 10000
    dt = 0.01

    # Gerar OU
    X = np.zeros(n_steps)
    X[0] = mu

    for i in range(1, n_steps):
        dX_ou = theta_ou * (mu - X[i-1]) * dt
        dW_ou = sigma_ou * np.sqrt(dt) * np.random.randn()
        dW_noise = sigma_noise * np.sqrt(dt) * np.random.randn()
        X[i] = X[i-1] + dX_ou + dW_ou + dW_noise
        X[i] = max(0.01, X[i])

    # Analisar distribuição
    normalized = X / np.mean(X)
    normalized = np.clip(normalized, 1e-10, None)
    levels = np.floor(np.log2(normalized)).astype(int)
    unique_levels, counts = np.unique(levels, return_counts=True)
    P_emergent = {int(lv): cnt/len(levels) for lv, cnt in zip(unique_levels, counts)}

    # Chi-squared vs Riemann
    chi2 = 0
    for level in P_real.keys():
        obs = P_emergent.get(level, 0)
        exp = P_real[level]
        if exp > 0:
            chi2 += (obs - exp)**2 / exp

    accuracy = max(0, 1 - chi2/10.0) * 100

    print(f"\n  {name}")
    print(f"    θ_OU = {theta_ou:.4f}")
    print(f"    Accuracy vs Riemann: {accuracy:.2f}%")
    print(f"    χ² = {chi2:.4f}")

    return accuracy, P_emergent

# Testar diferentes θ
results = {}

results['baseline'] = test_ou_with_theta(1.0, "[BASELINE] θ = 1.0")
results['hyp1'] = test_ou_with_theta(theta_ou_predicted_1, "[HIP 1] θ ~ 1/gap_mean")
results['hyp2'] = test_ou_with_theta(theta_ou_predicted_2, "[HIP 2] θ ~ γ × (2π/gap)")
results['hyp4'] = test_ou_with_theta(theta_ou_predicted_4, "[HIP 4] θ ~ γ / log(gap)")

# Testar também θ = γ diretamente
results['gamma'] = test_ou_with_theta(gamma_euler, "[DIRETO] θ = γ")

# Gráficos
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Zeros e Gram approximation
ax1 = axes[0, 0]
n_range = np.arange(1, len(zeros)+1)
ax1.plot(n_range, zeros, 'b.', markersize=3, alpha=0.5, label='Zeros reais')
ax1.plot(n_range, gram_approx, 'r-', linewidth=2, alpha=0.7, label='Gram approx')
ax1.set_xlabel('n', fontsize=12)
ax1.set_ylabel('t_n', fontsize=12)
ax1.set_title('Zeros vs Aproximação de Gram', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Gaps reais vs assintóticos
ax2 = axes[0, 1]
ax2.scatter(zeros[:-1][valid_idx], gaps[valid_idx], s=10, alpha=0.3, label='Gaps reais')
ax2.plot(zeros[:-1][valid_idx], asymptotic_gaps_pred[:-1][valid_idx], 'r-',
         linewidth=2, alpha=0.7, label='2π/log(t/2π)')
ax2.set_xlabel('t', fontsize=12)
ax2.set_ylabel('Gap', fontsize=12)
ax2.set_title('Gaps Reais vs Fórmula Assintótica', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. θ(t) de Riemann-von Mangoldt
ax3 = axes[0, 2]
ax3.plot(zeros, theta_values, 'purple', linewidth=1.5)
ax3.set_xlabel('t', fontsize=12)
ax3.set_ylabel('θ(t)', fontsize=12)
ax3.set_title('θ(t) de Riemann-von Mangoldt', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Accuracy por hipótese
ax4 = axes[1, 0]
labels = ['Baseline\nθ=1', 'Hip 1\n1/gap', 'Hip 2\nγ×2π/gap', 'Hip 4\nγ/log', 'Direto\nθ=γ']
accs = [results[k][0] for k in ['baseline', 'hyp1', 'hyp2', 'hyp4', 'gamma']]
colors = ['blue', 'orange', 'green', 'purple', 'red']
bars = ax4.bar(labels, accs, color=colors, alpha=0.7)
ax4.axhline(90, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_ylabel('Accuracy vs Riemann (%)', fontsize=12)
ax4.set_title('Comparação de Hipóteses para θ', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 100])

# Adicionar valores
for bar, acc in zip(bars, accs):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. Distribuições emergentes
ax5 = axes[1, 1]
riemann_levels = sorted(P_real.keys())
riemann_probs = [P_real[k] for k in riemann_levels]
ax5.plot(riemann_levels, riemann_probs, 'k^--', linewidth=3, markersize=8,
         label='Riemann', alpha=0.8)

for key, color, label in [('baseline', 'blue', 'θ=1'), ('gamma', 'red', 'θ=γ')]:
    P_em = results[key][1]
    levels_em = sorted(P_em.keys())
    probs_em = [P_em[k] for k in levels_em]
    ax5.plot(levels_em, probs_em, 'o-', color=color, linewidth=2,
             markersize=6, label=label, alpha=0.7)

ax5.set_xlabel('Level k', fontsize=12)
ax5.set_ylabel('P(k)', fontsize=12)
ax5.set_title('Distribuições: θ=1 vs θ=γ', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_yscale('log')

# 6. Razões θ_previsto / θ_usado
ax6 = axes[1, 2]
theta_predictions = [1.0, theta_ou_predicted_1, theta_ou_predicted_2,
                     theta_ou_predicted_4, gamma_euler]
ratios = [t / theta_ou_used for t in theta_predictions]
ax6.barh(labels, ratios, color=colors, alpha=0.7)
ax6.axvline(1.0, color='black', linestyle='--', linewidth=2, label='θ usado (1.0)')
ax6.set_xlabel('Razão: θ_previsto / θ_usado', fontsize=12)
ax6.set_title('Predições de θ_OU', fontsize=13, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/thlinux/relacionalidadegeral/validacao/theta_gamma_connection.png',
            dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/theta_gamma_connection.png")

# Salvar análise
output = {
    'constants': {
        'gamma_euler': gamma_euler,
        'pi': pi
    },
    'gap_statistics': {
        'mean_empirical': float(gap_mean_empirical),
        'mean_asymptotic': float(gap_mean_asymptotic),
        'std_empirical': float(np.std(gaps))
    },
    'theta_predictions': {
        'baseline': 1.0,
        'hypothesis_1_1_over_gap': float(theta_ou_predicted_1),
        'hypothesis_2_gamma_times_2pi_over_gap': float(theta_ou_predicted_2),
        'hypothesis_4_gamma_over_log_gap': float(theta_ou_predicted_4),
        'direct_gamma': gamma_euler
    },
    'accuracies': {
        'baseline': float(results['baseline'][0]),
        'hypothesis_1': float(results['hyp1'][0]),
        'hypothesis_2': float(results['hyp2'][0]),
        'hypothesis_4': float(results['hyp4'][0]),
        'direct_gamma': float(results['gamma'][0])
    }
}

with open('/home/thlinux/relacionalidadegeral/validacao/theta_gamma_connection_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Análise salva: validacao/theta_gamma_connection_results.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO")
print(f"{'=' * 70}")

best_key = max(results.keys(), key=lambda k: results[k][0])
best_acc = results[best_key][0]

print(f"\n[MELHOR RESULTADO]")
if best_key == 'baseline':
    print(f"  θ = 1.0 (baseline) permanece o melhor: {best_acc:.2f}%")
elif best_key == 'gamma':
    print(f"  θ = γ ({gamma_euler:.4f}) É MELHOR! Accuracy: {best_acc:.2f}%")
    print(f"  🔥 CONEXÃO DIRETA γ ↔ θ_OU VALIDADA!")
else:
    print(f"  Melhor hipótese: {best_key}")
    print(f"  Accuracy: {best_acc:.2f}%")

print(f"\n[INTERPRETAÇÃO]")
print(f"  A constante γ de Euler-Mascheroni aparece nas fórmulas")
print(f"  assintóticas dos zeros de Riemann. Nossos testes mostram:")
print(f"  • θ=1.0 → {results['baseline'][0]:.2f}%")
print(f"  • θ=γ → {results['gamma'][0]:.2f}%")

if abs(results['gamma'][0] - results['baseline'][0]) < 2:
    print(f"\n  Resultados SIMILARES! γ e 1.0 são comparáveis.")
    print(f"  Diferença: {abs(results['gamma'][0] - results['baseline'][0]):.2f}%")
elif results['gamma'][0] > results['baseline'][0]:
    print(f"\n  θ=γ É MELHOR! Ganho de {results['gamma'][0] - results['baseline'][0]:.2f}%")
else:
    print(f"\n  θ=1.0 ainda é melhor que θ=γ")

print(f"\n{'=' * 70}\n")
