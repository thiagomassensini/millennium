#!/usr/bin/env python3
"""
Teste Estocástico v2 - AJUSTADO
Tentativa de chegar a 100% de acurácia com ajustes nos parâmetros
"""

import numpy as np
import json
import matplotlib.pyplot as plt

print("=" * 60)
print("TESTE ESTOCÁSTICO V2 - Ajustes para convergir a 100%")
print("=" * 60)

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

# AJUSTES baseados no resultado anterior
configs = [
    {
        'name': 'Config 1: SNR × 2',
        'snr_coef': 0.10,  # dobrar SNR
        'theta': 1.0,
        'sigma_ou': gap_std * 0.5,
        'n_steps': 1000
    },
    {
        'name': 'Config 2: SNR × 2 + θ × 2',
        'snr_coef': 0.10,
        'theta': 2.0,  # aumentar reversão
        'sigma_ou': gap_std * 0.5,
        'n_steps': 1000
    },
    {
        'name': 'Config 3: SNR × 2 + mais steps',
        'snr_coef': 0.10,
        'theta': 1.0,
        'sigma_ou': gap_std * 0.5,
        'n_steps': 2000  # mais passos por rodada
    },
    {
        'name': 'Config 4: SNR × 3 + σ_OU reduzido',
        'snr_coef': 0.15,
        'theta': 1.5,
        'sigma_ou': gap_std * 0.3,  # menos volatilidade intrínseca
        'n_steps': 1500
    },
]

best_config = None
best_accuracy = 0
all_results = []

for cfg in configs:
    print(f"\n{'=' * 60}")
    print(f"TESTANDO: {cfg['name']}")
    print(f"  SNR = {cfg['snr_coef']} × √t")
    print(f"  θ = {cfg['theta']}")
    print(f"  σ_OU = {cfg['sigma_ou']:.4f}")
    print(f"  n_steps = {cfg['n_steps']}")
    print(f"{'=' * 60}")

    # Parâmetros
    theta = cfg['theta']
    mu = gap_mean
    sigma_ou = cfg['sigma_ou']
    n_trials = 100
    n_steps = cfg['n_steps']
    dt = 0.01
    snr_coef = cfg['snr_coef']

    results = {
        'trials': [],
        'accuracy': [],
        'chi2': []
    }

    # Simulação
    for t in range(1, n_trials + 1):
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

    # Análise pós t≥50
    accuracy_post50 = np.mean(results['accuracy'][49:])
    accuracy_std_post50 = np.std(results['accuracy'][49:])
    chi2_final = results['chi2'][-1]

    print(f"\n[RESULTADOS]")
    print(f"  Acurácia média (t≥50): {accuracy_post50:.2f}%")
    print(f"  Desvio padrão (t≥50): {accuracy_std_post50:.2f}%")
    print(f"  χ² final: {chi2_final:.4f}")

    if accuracy_std_post50 < 5.0:
        print(f"  ✓ ESTABILIZADO")
    else:
        print(f"  ✗ Ainda oscilando")

    if accuracy_post50 >= 95.0:
        print(f"  ✅ CONVERGIU para ~100%!")
    elif accuracy_post50 >= 90.0:
        print(f"  🟡 Muito próximo (>90%)")
    elif accuracy_post50 >= 85.0:
        print(f"  🟠 Razoável (>85%)")

    all_results.append({
        'config': cfg,
        'accuracy_post50': accuracy_post50,
        'std_post50': accuracy_std_post50,
        'chi2_final': chi2_final,
        'results': results
    })

    if accuracy_post50 > best_accuracy:
        best_accuracy = accuracy_post50
        best_config = cfg
        best_results = results

# Resultado final
print(f"\n{'=' * 60}")
print("MELHOR CONFIGURAÇÃO")
print(f"{'=' * 60}")
print(f"\nConfig: {best_config['name']}")
print(f"  SNR = {best_config['snr_coef']} × √t")
print(f"  θ = {best_config['theta']}")
print(f"  σ_OU = {best_config['sigma_ou']:.4f}")
print(f"  n_steps = {best_config['n_steps']}")
print(f"\n  Acurácia (t≥50): {best_accuracy:.2f}%")

# Gráfico comparativo
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Comparar todas as configs
ax1 = axes[0, 0]
for i, res in enumerate(all_results):
    ax1.plot(res['results']['trials'], res['results']['accuracy'],
             linewidth=2, label=f"Config {i+1}", alpha=0.7)
ax1.axvline(50, color='r', linestyle='--', alpha=0.5)
ax1.axhline(95, color='orange', linestyle=':', alpha=0.5, label='95% target')
ax1.set_xlabel('Trial t', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Comparação de Configurações', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Melhor config: Accuracy
ax2 = axes[0, 1]
ax2.plot(best_results['trials'], best_results['accuracy'], 'g-', linewidth=2)
ax2.axvline(50, color='r', linestyle='--', label='t=50')
ax2.axhline(95, color='orange', linestyle=':', label='95%')
ax2.fill_between(best_results['trials'][49:],
                  [best_accuracy - all_results[np.argmax([r['accuracy_post50'] for r in all_results])]['std_post50']] * len(best_results['trials'][49:]),
                  [best_accuracy + all_results[np.argmax([r['accuracy_post50'] for r in all_results])]['std_post50']] * len(best_results['trials'][49:]),
                  alpha=0.2, color='green')
ax2.set_xlabel('Trial t', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title(f'Melhor Config: {best_config["name"]}', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([70, 105])

# Chi-squared convergência
ax3 = axes[1, 0]
ax3.semilogy(best_results['trials'], best_results['chi2'], 'purple', linewidth=2)
ax3.axvline(50, color='r', linestyle='--', label='t=50')
ax3.set_xlabel('Trial t', fontsize=12)
ax3.set_ylabel('χ² (log scale)', fontsize=12)
ax3.set_title('Convergência χ² (Melhor Config)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Barras: Acurácia por config
ax4 = axes[1, 1]
config_names = [f"Config {i+1}" for i in range(len(all_results))]
accuracies = [r['accuracy_post50'] for r in all_results]
colors = ['green' if acc >= 95 else 'yellow' if acc >= 90 else 'orange' for acc in accuracies]

ax4.barh(config_names, accuracies, color=colors, alpha=0.7)
ax4.axvline(95, color='red', linestyle='--', linewidth=2, label='95% target')
ax4.set_xlabel('Acurácia média (t≥50) [%]', fontsize=12)
ax4.set_title('Comparação Final de Configs', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='x')
ax4.set_xlim([70, 105])

plt.tight_layout()
plt.savefig('../validacao/stochastic_riemann_test_v2.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/stochastic_riemann_test_v2.png")

# Salvar resultados
output = {
    'best_config': {
        'name': best_config['name'],
        'snr_coef': best_config['snr_coef'],
        'theta': best_config['theta'],
        'sigma_ou': best_config['sigma_ou'],
        'n_steps': best_config['n_steps']
    },
    'best_accuracy': best_accuracy,
    'all_configs': [
        {
            'name': r['config']['name'],
            'accuracy_post50': r['accuracy_post50'],
            'std_post50': r['std_post50'],
            'chi2_final': r['chi2_final']
        }
        for r in all_results
    ]
}

with open('../validacao/stochastic_riemann_test_v2_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/stochastic_riemann_test_v2_results.json")

print(f"\n{'=' * 60}")
print("CONCLUSÃO FINAL")
print(f"{'=' * 60}")

if best_accuracy >= 95.0:
    print(f"\n✅ SUCESSO! Convergência para ~100% alcançada!")
    print(f"   Melhor acurácia: {best_accuracy:.2f}%")
    print(f"   Config vencedora: {best_config['name']}")
elif best_accuracy >= 90.0:
    print(f"\n🟡 MUITO PRÓXIMO! Acurácia {best_accuracy:.2f}%")
    print(f"   Próximos ajustes sugeridos:")
    print(f"   - SNR = {best_config['snr_coef'] * 1.2:.2f} × √t")
    print(f"   - θ = {best_config['theta'] * 1.1:.2f}")
else:
    print(f"\n🟠 Melhor resultado: {best_accuracy:.2f}%")
    print(f"   Pode precisar de abordagem diferente")

print(f"\n{'=' * 60}\n")
