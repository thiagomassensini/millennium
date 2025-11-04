#!/usr/bin/env python3
"""
Teste FINAL - XOR Quantização com MUITOS trials
Combinar a melhor abordagem (quantização XOR) com t >> 100
Para tentar quebrar o platô de 92%
"""

import numpy as np
import json
import matplotlib.pyplot as plt

print("=" * 70)
print("TESTE FINAL - XOR QUANTIZAÇÃO + MUITOS TRIALS")
print("Objetivo: Quebrar o platô de 92%")
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

print(f"\n[TARGET] Distribuição real:")
for level in sorted(P_real.keys()):
    print(f"  Level {level}: {100*P_real[level]:.1f}%")

def quantize_to_xor(gap, gap_mean):
    """Quantiza gap para o nível XOR mais próximo"""
    normalized = gap / gap_mean

    # Encontrar k tal que 2^k <= normalized < 2^(k+1)
    if normalized > 0:
        k = int(np.floor(np.log2(normalized)))
    else:
        k = -10

    # Quantizar para 2^(k+1) - 2
    quantized_normalized = 2**(k+1) - 2 if k >= 0 else 2**(k+1)
    quantized_gap = quantized_normalized * gap_mean

    return quantized_gap, k

# Parâmetros
theta = 1.0
mu = gap_mean
sigma_ou = gap_std * 0.5
n_trials = 1000  # MUITOS trials
n_steps = 1000
dt = 0.01
snr_coef = 0.05

print(f"\n[PARÂMETROS]")
print(f"  SNR(t) = {snr_coef} × √t")
print(f"  θ = {theta}")
print(f"  σ_OU = {sigma_ou:.4f}")
print(f"  n_trials = {n_trials}")
print(f"  n_steps = {n_steps}")

# TESTE 2 CONFIGURAÇÕES
configs = [
    {
        'name': 'Config 1: Gaussiano puro (baseline esperado ~92%)',
        'quantize': False
    },
    {
        'name': 'Config 2: Quantização XOR total',
        'quantize': True
    },
]

all_results = []

for cfg in configs:
    print(f"\n{'=' * 70}")
    print(f"TESTANDO: {cfg['name']}")
    print(f"{'=' * 70}")

    results = {
        'trials': [],
        'accuracy': [],
        'chi2': []
    }

    checkpoint_points = [1, 10, 50, 100, 200, 500, 1000]

    # Simulação
    for t in range(1, n_trials + 1):
        if t in checkpoint_points:
            print(f"  Trial {t}/{n_trials}...")

        snr = snr_coef * np.sqrt(t)

        # Simular processo OU + Gaussiano
        X = np.zeros(n_steps)
        X[0] = mu

        for i in range(1, n_steps):
            # OU term
            dX_ou = theta * (mu - X[i-1]) * dt
            dW_ou = sigma_ou * np.sqrt(dt) * np.random.randn()

            # Gaussian noise
            sigma_noise = gap_std * (1.0 / snr) if snr > 0 else gap_std
            dW_noise = sigma_noise * np.sqrt(dt) * np.random.randn()

            X[i] = X[i-1] + dX_ou + dW_ou + dW_noise
            X[i] = max(0.1, X[i])

        # QUANTIZAR se necessário
        if cfg['quantize']:
            X_quantized = np.zeros(n_steps)
            for i in range(n_steps):
                quantized_gap, _ = quantize_to_xor(X[i], gap_mean)
                X_quantized[i] = quantized_gap

            X = X_quantized

        # Analisar níveis
        normalized_gaps = X / np.mean(X)
        normalized_gaps = np.clip(normalized_gaps, 1e-10, None)  # Evitar log(0)
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

    # Análise em diferentes pontos
    analysis_points = [100, 200, 500, 1000]

    print(f"\n[CONVERGÊNCIA]")
    for t_threshold in analysis_points:
        if t_threshold <= len(results['accuracy']):
            idx = t_threshold - 1
            accuracy_post = np.mean(results['accuracy'][idx:])
            accuracy_std = np.std(results['accuracy'][idx:])
            chi2_post = np.mean(results['chi2'][idx:])

            print(f"  t ≥ {t_threshold}: Acc = {accuracy_post:.2f}% ± {accuracy_std:.2f}%, χ² = {chi2_post:.4f}")

    # Melhor ponto
    best_t = 500
    idx = best_t - 1
    accuracy_post_best = np.mean(results['accuracy'][idx:])
    accuracy_std_best = np.std(results['accuracy'][idx:])
    chi2_post_best = np.mean(results['chi2'][idx:])

    all_results.append({
        'config': cfg,
        'accuracy_post500': accuracy_post_best,
        'std_post500': accuracy_std_best,
        'chi2_post500': chi2_post_best,
        'results': results
    })

# Comparação final
print(f"\n{'=' * 70}")
print("COMPARAÇÃO FINAL (t ≥ 500)")
print(f"{'=' * 70}")

for i, res in enumerate(all_results):
    improvement = res['accuracy_post500'] - all_results[0]['accuracy_post500']
    print(f"\n{i+1}. {res['config']['name']}")
    print(f"   Acurácia: {res['accuracy_post500']:.2f}%")
    print(f"   χ²: {res['chi2_post500']:.4f}")
    print(f"   Melhoria: {improvement:+.2f}%")

best_result = max(all_results, key=lambda x: x['accuracy_post500'])
print(f"\n🏆 MELHOR: {best_result['config']['name']}")
print(f"   Acurácia: {best_result['accuracy_post500']:.2f}%")

# Gráfico
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Accuracy evolution
ax1 = axes[0, 0]
colors = ['blue', 'red']
for i, res in enumerate(all_results):
    ax1.semilogx(res['results']['trials'], res['results']['accuracy'],
                 linewidth=1.5, color=colors[i], alpha=0.7, label=res['config']['name'])
ax1.axvline(500, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(92, color='orange', linestyle=':', linewidth=2, label='92% (platô anterior)')
ax1.axhline(95, color='green', linestyle=':', linewidth=2, label='95% target')
ax1.set_xlabel('Trial t (log scale)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Gaussiano vs XOR Quantização', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([80, 100])

# 2. Rolling mean
ax2 = axes[0, 1]
window = 50
for i, res in enumerate(all_results):
    rolling = np.convolve(res['results']['accuracy'], np.ones(window)/window, mode='valid')
    ax2.plot(range(window, len(res['results']['accuracy'])+1), rolling,
             linewidth=3, color=colors[i], alpha=0.8, label=res['config']['name'])
ax2.axhline(92, color='orange', linestyle=':', linewidth=2, label='92%')
ax2.axhline(95, color='green', linestyle=':', linewidth=2, label='95%')
ax2.set_xlabel('Trial t', fontsize=12)
ax2.set_ylabel('Rolling Mean (window=50)', fontsize=12)
ax2.set_title('Média Móvel', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([85, 98])

# 3. Chi2
ax3 = axes[1, 0]
for i, res in enumerate(all_results):
    ax3.loglog(res['results']['trials'], res['results']['chi2'],
               linewidth=1.5, color=colors[i], alpha=0.7, label=res['config']['name'])
ax3.set_xlabel('Trial t (log)', fontsize=12)
ax3.set_ylabel('χ² (log)', fontsize=12)
ax3.set_title('Convergência χ²', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Comparison bars
ax4 = axes[1, 1]
names = ['Gaussiano', 'XOR Quant']
accs = [r['accuracy_post500'] for r in all_results]
bars = ax4.bar(names, accs, color=['blue', 'red'], alpha=0.7, width=0.6)
ax4.axhline(92, color='orange', linestyle='--', linewidth=2, label='Platô 92%')
ax4.axhline(95, color='green', linestyle='--', linewidth=2, label='Target 95%')
ax4.set_ylabel('Acurácia (t≥500) [%]', fontsize=12)
ax4.set_title('Comparação Final', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([88, 98])

# Adicionar valores nas barras
for i, (bar, acc) in enumerate(zip(bars, accs)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/thlinux/relacionalidadegeral/validacao/stochastic_xor_final_test.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/stochastic_xor_final_test.png")

# Salvar resultados
output = {
    'configurations': [
        {
            'name': r['config']['name'],
            'quantize': r['config']['quantize'],
            'accuracy_post500': r['accuracy_post500'],
            'std_post500': r['std_post500'],
            'chi2_post500': r['chi2_post500']
        }
        for r in all_results
    ],
    'best': {
        'name': best_result['config']['name'],
        'accuracy': best_result['accuracy_post500']
    }
}

with open('/home/thlinux/relacionalidadegeral/validacao/stochastic_xor_final_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/stochastic_xor_final_results.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO FINAL")
print(f"{'=' * 70}")

gauss_acc = all_results[0]['accuracy_post500']
xor_acc = all_results[1]['accuracy_post500']
improvement = xor_acc - gauss_acc

if xor_acc >= 95.0:
    print(f"\n✅ BREAKTHROUGH! XOR quantização quebrou o platô!")
    print(f"   Acurácia: {xor_acc:.2f}%")
elif xor_acc > gauss_acc + 1.0:
    print(f"\n🟢 XOR quantização é SIGNIFICATIVAMENTE melhor:")
    print(f"   Gaussiano: {gauss_acc:.2f}%")
    print(f"   XOR Quant: {xor_acc:.2f}%")
    print(f"   Ganho: +{improvement:.2f}%")
elif xor_acc > gauss_acc:
    print(f"\n🟡 XOR quantização é ligeiramente melhor:")
    print(f"   Gaussiano: {gauss_acc:.2f}%")
    print(f"   XOR Quant: {xor_acc:.2f}%")
    print(f"   Ganho: +{improvement:.2f}%")
else:
    print(f"\n🔴 Gaussiano ainda é melhor:")
    print(f"   Gaussiano: {gauss_acc:.2f}%")
    print(f"   XOR Quant: {xor_acc:.2f}%")

print(f"\n[INTERPRETAÇÃO]")
if xor_acc >= 95.0:
    print(f"  A estrutura XOR dos twin primes captura informação")
    print(f"  adicional que não estava presente no modelo Gaussiano!")
else:
    print(f"  O platô de ~92% parece ser fundamental:")
    print(f"  • 92% = estrutura estocástica (OU + Gaussiano)")
    print(f"  • 8% restantes = correlações determinísticas")
    print(f"  (Montgomery, estrutura específica dos zeros)")

print(f"\n{'=' * 70}\n")
