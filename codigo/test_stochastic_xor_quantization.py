#!/usr/bin/env python3
"""
Teste Estocástico com XOR - OPÇÃO 2
Gerar gaps com OU + Gaussiano, DEPOIS quantizar usando níveis XOR
2^(k+1) - 2 para k = 0, 1, 2, ...
"""

import numpy as np
import json
import matplotlib.pyplot as plt

print("=" * 70)
print("TESTE ESTOCÁSTICO - QUANTIZAÇÃO XOR (Opção 2)")
print("OU + Gaussiano → depois quantizar gaps em níveis 2^(k+1) - 2")
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

# Níveis XOR
print(f"\n[NÍVEIS XOR] 2^(k+1) - 2:")
xor_levels = {}
for k in range(-3, 15):  # k negativo também
    xor_val = 2**(k+1) - 2 if k >= 0 else 2**(k+1)  # Ajuste para k < 0
    xor_levels[k] = xor_val
    if -3 <= k <= 5:
        print(f"  k={k}: {xor_val:.4f}")

# Parâmetros
theta = 1.0
mu = gap_mean
sigma_ou = gap_std * 0.5
n_trials = 200
n_steps = 1000
dt = 0.01
snr_coef = 0.05

def quantize_to_xor(gap, gap_mean):
    """Quantiza gap para o nível XOR mais próximo"""
    normalized = gap / gap_mean

    # Encontrar k tal que 2^k <= normalized < 2^(k+1)
    if normalized > 0:
        k = int(np.floor(np.log2(normalized)))
    else:
        k = -10  # muito pequeno

    # Quantizar para 2^(k+1) - 2
    quantized_normalized = 2**(k+1) - 2 if k >= 0 else 2**(k+1)
    quantized_gap = quantized_normalized * gap_mean

    return quantized_gap, k

# TESTE 3 CONFIGURAÇÕES
configs = [
    {
        'name': 'Config 1: Sem quantização (baseline)',
        'quantize': False
    },
    {
        'name': 'Config 2: Quantização XOR total',
        'quantize': True,
        'quantize_strength': 1.0
    },
    {
        'name': 'Config 3: Quantização XOR parcial (70%)',
        'quantize': True,
        'quantize_strength': 0.7
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

    # Simulação
    for t in range(1, n_trials + 1):
        if t % 50 == 0 or t in [1, 10, 50, 100]:
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

                # Interpolar entre gap original e quantizado
                strength = cfg['quantize_strength']
                X_quantized[i] = strength * quantized_gap + (1 - strength) * X[i]

            X = X_quantized

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

    # Análise pós t≥100
    accuracy_post100 = np.mean(results['accuracy'][99:])
    accuracy_std = np.std(results['accuracy'][99:])
    chi2_post100 = np.mean(results['chi2'][99:])

    print(f"\n[RESULTADOS t≥100]")
    print(f"  Acurácia média: {accuracy_post100:.2f}%")
    print(f"  Desvio padrão: {accuracy_std:.2f}%")
    print(f"  χ² médio: {chi2_post100:.4f}")

    if accuracy_post100 >= 95.0:
        print(f"  ✅ CONVERGIU para ~100%!")
    elif accuracy_post100 >= 93.0:
        print(f"  🟢 Quase lá! (>93%)")
    elif accuracy_post100 >= 90.0:
        print(f"  🟡 Muito próximo (>90%)")

    all_results.append({
        'config': cfg,
        'accuracy_post100': accuracy_post100,
        'std_post100': accuracy_std,
        'chi2_post100': chi2_post100,
        'results': results
    })

# Comparação final
print(f"\n{'=' * 70}")
print("COMPARAÇÃO FINAL")
print(f"{'=' * 70}")

for i, res in enumerate(all_results):
    improvement = res['accuracy_post100'] - all_results[0]['accuracy_post100']
    print(f"\n{i+1}. {res['config']['name']}")
    print(f"   Acurácia: {res['accuracy_post100']:.2f}%")
    print(f"   χ²: {res['chi2_post100']:.4f}")
    print(f"   Melhoria: {improvement:+.2f}%")

best_result = max(all_results, key=lambda x: x['accuracy_post100'])
print(f"\n🏆 MELHOR: {best_result['config']['name']}")
print(f"   Acurácia: {best_result['accuracy_post100']:.2f}%")

# Gráfico
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Comparação de acurácia
ax1 = axes[0, 0]
colors = ['blue', 'red', 'orange']
for i, res in enumerate(all_results):
    ax1.plot(res['results']['trials'], res['results']['accuracy'],
             linewidth=2, color=colors[i], alpha=0.7, label=res['config']['name'])
ax1.axvline(100, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(95, color='green', linestyle=':', linewidth=2, label='95%')
ax1.set_xlabel('Trial t', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Comparação: Sem vs Com Quantização XOR', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Chi-squared
ax2 = axes[0, 1]
for i, res in enumerate(all_results):
    ax2.semilogy(res['results']['trials'], res['results']['chi2'],
                 linewidth=2, color=colors[i], alpha=0.7, label=res['config']['name'])
ax2.axvline(100, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Trial t', fontsize=12)
ax2.set_ylabel('χ² (log scale)', fontsize=12)
ax2.set_title('Convergência χ²', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Barras comparativas
ax3 = axes[1, 0]
names = [f"Config {i+1}" for i in range(len(all_results))]
accs = [r['accuracy_post100'] for r in all_results]
colors_bar = ['blue', 'red', 'orange']
bars = ax3.bar(names, accs, color=colors_bar, alpha=0.7)
ax3.axhline(all_results[0]['accuracy_post100'], color='blue', linestyle='--',
            linewidth=2, label=f'Baseline: {all_results[0]["accuracy_post100"]:.1f}%')
ax3.axhline(95, color='green', linestyle='--', linewidth=2, label='95% target')
ax3.set_ylabel('Acurácia (t≥100) [%]', fontsize=12)
ax3.set_title('Comparação Final (Quantização XOR)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim([85, 100])

# 4. Rolling mean
ax4 = axes[1, 1]
window = 20
for i, res in enumerate(all_results):
    rolling = np.convolve(res['results']['accuracy'], np.ones(window)/window, mode='valid')
    ax4.plot(range(window, len(res['results']['accuracy'])+1), rolling,
             linewidth=2, color=colors[i], alpha=0.7, label=res['config']['name'])
ax4.axvline(100, color='gray', linestyle='--', alpha=0.5)
ax4.axhline(95, color='green', linestyle=':', linewidth=2)
ax4.set_xlabel('Trial t', fontsize=12)
ax4.set_ylabel('Rolling Mean (window=20)', fontsize=12)
ax4.set_title('Média Móvel', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/thlinux/relacionalidadegeral/validacao/stochastic_xor_quantization_test.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/stochastic_xor_quantization_test.png")

# Salvar resultados
output = {
    'configurations': [
        {
            'name': r['config']['name'],
            'accuracy_post100': r['accuracy_post100'],
            'std_post100': r['std_post100'],
            'chi2_post100': r['chi2_post100']
        }
        for r in all_results
    ],
    'best': {
        'name': best_result['config']['name'],
        'accuracy': best_result['accuracy_post100']
    }
}

with open('/home/thlinux/relacionalidadegeral/validacao/stochastic_xor_quantization_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/stochastic_xor_quantization_results.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO")
print(f"{'=' * 70}")

if best_result['accuracy_post100'] >= 95.0:
    print(f"\n✅ BREAKTHROUGH! Quantização XOR quebrou o platô de 92%!")
    print(f"   Nova acurácia: {best_result['accuracy_post100']:.2f}%")
elif best_result['accuracy_post100'] > all_results[0]['accuracy_post100']:
    print(f"\n🟢 MELHORIA! Quantização XOR ajuda:")
    improvement = best_result['accuracy_post100'] - all_results[0]['accuracy_post100']
    print(f"   Sem quantização: {all_results[0]['accuracy_post100']:.2f}%")
    print(f"   Com quantização: {best_result['accuracy_post100']:.2f}%")
    print(f"   Ganho: +{improvement:.2f}%")
else:
    print(f"\n🟡 Quantização não melhorou")
    print(f"   Baseline: {all_results[0]['accuracy_post100']:.2f}%")
    print(f"   Melhor quantização: {best_result['accuracy_post100']:.2f}%")

print(f"\n{'=' * 70}\n")
