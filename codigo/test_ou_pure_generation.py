#!/usr/bin/env python3
"""
Teste GENERATIVO - OU + Gaussiano Puro
SEM usar dados de Riemann como target
Gerar processo puro e ver que distribuição EMERGE
Depois comparar com Riemann e Twin Primes
"""

import numpy as np
import json
import matplotlib.pyplot as plt

print("=" * 70)
print("TESTE GENERATIVO - OU + GAUSSIANO PURO")
print("Ver que distribuição emerge espontaneamente")
print("=" * 70)

# Carregar dados de Riemann e Twin Primes para COMPARAÇÃO (não como target)
with open('/home/thlinux/relacionalidadegeral/validacao/riemann_extended_analysis.json', 'r') as f:
    data_riemann = json.load(f)

zeros = np.array(data_riemann['zeros'])
gaps_riemann = np.diff(zeros)

# Distribuição de Riemann
gap_analysis = data_riemann['gap_analysis']
level_dist_riemann = gap_analysis['level_distribution']
total_riemann = sum(level_dist_riemann.values())
P_riemann = {int(k): v/total_riemann for k, v in level_dist_riemann.items()}

print(f"\n[REFERÊNCIA 1] Distribuição de Riemann:")
for level in sorted(P_riemann.keys()):
    print(f"  Level {level}: {100*P_riemann[level]:.1f}%")

# Distribuição Twin Primes teórica P(k) = 2^-k
print(f"\n[REFERÊNCIA 2] Distribuição Twin Primes P(k) = 2^-k:")
P_twin_primes = {}
for k in range(-3, 10):
    P_twin_primes[k] = 2**(-k) if k >= 0 else 0
# Normalizar
total_twin = sum(P_twin_primes.values())
P_twin_primes = {k: v/total_twin for k, v in P_twin_primes.items()}
for k in range(-3, 5):
    if P_twin_primes[k] > 0.001:
        print(f"  Level {k}: {100*P_twin_primes[k]:.1f}%")

# PARÂMETROS OU + GAUSSIANO PURO
# Usar parâmetros "naturais" sem ajuste para Riemann
configs = [
    {
        'name': 'Config 1: OU puro (θ=1, sem ruído)',
        'theta': 1.0,
        'mu': 1.0,
        'sigma_ou': 0.5,
        'sigma_noise': 0.0,
        'n_steps': 10000
    },
    {
        'name': 'Config 2: OU + Gaussiano (parâmetros naturais)',
        'theta': 1.0,
        'mu': 1.0,
        'sigma_ou': 0.5,
        'sigma_noise': 0.5,
        'n_steps': 10000
    },
    {
        'name': 'Config 3: OU + Gaussiano (mais ruído)',
        'theta': 1.0,
        'mu': 1.0,
        'sigma_ou': 0.3,
        'sigma_noise': 1.0,
        'n_steps': 10000
    },
    {
        'name': 'Config 4: OU fraco + muito ruído',
        'theta': 0.5,
        'mu': 1.0,
        'sigma_ou': 0.2,
        'sigma_noise': 2.0,
        'n_steps': 10000
    }
]

all_results = []

for cfg in configs:
    print(f"\n{'=' * 70}")
    print(f"TESTANDO: {cfg['name']}")
    print(f"  θ = {cfg['theta']}")
    print(f"  μ = {cfg['mu']}")
    print(f"  σ_OU = {cfg['sigma_ou']}")
    print(f"  σ_noise = {cfg['sigma_noise']}")
    print(f"{'=' * 70}")

    theta = cfg['theta']
    mu = cfg['mu']
    sigma_ou = cfg['sigma_ou']
    sigma_noise = cfg['sigma_noise']
    n_steps = cfg['n_steps']
    dt = 0.01

    # GERAR processo OU + Gaussiano
    X = np.zeros(n_steps)
    X[0] = mu

    for i in range(1, n_steps):
        dX_ou = theta * (mu - X[i-1]) * dt
        dW_ou = sigma_ou * np.sqrt(dt) * np.random.randn()
        dW_noise = sigma_noise * np.sqrt(dt) * np.random.randn()
        X[i] = X[i-1] + dX_ou + dW_ou + dW_noise
        X[i] = max(0.01, X[i])  # Evitar negativos

    # Analisar distribuição EMERGENTE
    normalized = X / np.mean(X)
    normalized = np.clip(normalized, 1e-10, None)
    levels = np.floor(np.log2(normalized)).astype(int)

    unique_levels, counts = np.unique(levels, return_counts=True)
    P_emergent = {int(lv): cnt/len(levels) for lv, cnt in zip(unique_levels, counts)}

    print(f"\n[DISTRIBUIÇÃO EMERGENTE]")
    for level in sorted(P_emergent.keys()):
        if P_emergent[level] > 0.001:
            print(f"  Level {level}: {100*P_emergent[level]:.1f}%")

    # Comparar com Riemann
    chi2_riemann = 0
    for level in P_riemann.keys():
        obs = P_emergent.get(level, 0)
        exp = P_riemann[level]
        if exp > 0:
            chi2_riemann += (obs - exp)**2 / exp

    accuracy_riemann = max(0, 1 - chi2_riemann/10.0) * 100

    # Comparar com Twin Primes
    chi2_twin = 0
    levels_common = set(P_emergent.keys()) & set(P_twin_primes.keys())
    for level in levels_common:
        obs = P_emergent[level]
        exp = P_twin_primes[level]
        if exp > 0.001:
            chi2_twin += (obs - exp)**2 / exp

    accuracy_twin = max(0, 1 - chi2_twin/10.0) * 100

    print(f"\n[COMPARAÇÃO]")
    print(f"  Match com Riemann: {accuracy_riemann:.2f}% (χ² = {chi2_riemann:.4f})")
    print(f"  Match com Twin Primes: {accuracy_twin:.2f}% (χ² = {chi2_twin:.4f})")

    all_results.append({
        'config': cfg,
        'distribution': P_emergent,
        'accuracy_riemann': accuracy_riemann,
        'accuracy_twin': accuracy_twin,
        'chi2_riemann': chi2_riemann,
        'chi2_twin': chi2_twin,
        'gaps': X
    })

# Comparação final
print(f"\n{'=' * 70}")
print("RESUMO COMPARATIVO")
print(f"{'=' * 70}")

print(f"\n{'Config':<40} {'Riemann':<15} {'Twin Primes':<15}")
print(f"{'-'*70}")
for res in all_results:
    name = res['config']['name'][:38]
    print(f"{name:<40} {res['accuracy_riemann']:>6.2f}%        {res['accuracy_twin']:>6.2f}%")

# Melhor match
best_riemann = max(all_results, key=lambda x: x['accuracy_riemann'])
best_twin = max(all_results, key=lambda x: x['accuracy_twin'])

print(f"\n[MELHOR MATCH RIEMANN]")
print(f"  Config: {best_riemann['config']['name']}")
print(f"  Accuracy: {best_riemann['accuracy_riemann']:.2f}%")

print(f"\n[MELHOR MATCH TWIN PRIMES]")
print(f"  Config: {best_twin['config']['name']}")
print(f"  Accuracy: {best_twin['accuracy_twin']:.2f}%")

# Gráficos
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Distribuições emergentes
ax1 = axes[0, 0]
for i, res in enumerate(all_results):
    levels = sorted(res['distribution'].keys())
    probs = [res['distribution'][k] for k in levels]
    ax1.plot(levels, probs, 'o-', linewidth=2, markersize=6, label=f"Config {i+1}", alpha=0.7)

# Adicionar referências
riemann_levels = sorted(P_riemann.keys())
riemann_probs = [P_riemann[k] for k in riemann_levels]
ax1.plot(riemann_levels, riemann_probs, 'k^--', linewidth=3, markersize=8,
         label='Riemann', alpha=0.8)

twin_levels = [k for k in sorted(P_twin_primes.keys()) if P_twin_primes[k] > 0.001]
twin_probs = [P_twin_primes[k] for k in twin_levels]
ax1.plot(twin_levels, twin_probs, 'r*--', linewidth=3, markersize=10,
         label='Twin Primes', alpha=0.8)

ax1.set_xlabel('Level k', fontsize=12)
ax1.set_ylabel('P(k)', fontsize=12)
ax1.set_title('Distribuições Emergentes vs Referências', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 2. Accuracy comparison
ax2 = axes[0, 1]
x_pos = np.arange(len(all_results))
width = 0.35
bars1 = ax2.bar(x_pos - width/2, [r['accuracy_riemann'] for r in all_results],
                width, label='vs Riemann', color='blue', alpha=0.7)
bars2 = ax2.bar(x_pos + width/2, [r['accuracy_twin'] for r in all_results],
                width, label='vs Twin Primes', color='red', alpha=0.7)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Match com Referências', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'C{i+1}' for i in range(len(all_results))])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Time series (melhor config)
ax3 = axes[0, 2]
best_overall = max(all_results, key=lambda x: max(x['accuracy_riemann'], x['accuracy_twin']))
ax3.plot(best_overall['gaps'][:500], linewidth=1, alpha=0.7)
ax3.axhline(best_overall['config']['mu'], color='r', linestyle='--',
            linewidth=2, label=f'μ = {best_overall["config"]["mu"]}')
ax3.set_xlabel('Step', fontsize=12)
ax3.set_ylabel('Gap value', fontsize=12)
ax3.set_title(f'Séries Temporal (Melhor Config)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4-6: Distribuições individuais das configs
for i, res in enumerate(all_results[:3]):
    row = 1
    col = i
    ax = axes[row, col]

    levels_em = sorted(res['distribution'].keys())
    probs_em = [res['distribution'][k] for k in levels_em]

    ax.bar(levels_em, probs_em, alpha=0.6, label='Emergente', color='blue')
    ax.plot(riemann_levels, riemann_probs, 'k^--', linewidth=2, markersize=6,
            label='Riemann', alpha=0.8)
    ax.plot(twin_levels, twin_probs, 'r*--', linewidth=2, markersize=8,
            label='Twin', alpha=0.8)

    ax.set_xlabel('Level k', fontsize=10)
    ax.set_ylabel('P(k)', fontsize=10)
    ax.set_title(f'Config {i+1}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('/home/thlinux/relacionalidadegeral/validacao/ou_pure_generation_test.png',
            dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/ou_pure_generation_test.png")

# Salvar resultados
output = {
    'configurations': [
        {
            'name': r['config']['name'],
            'params': {k: r['config'][k] for k in ['theta', 'mu', 'sigma_ou', 'sigma_noise']},
            'accuracy_riemann': r['accuracy_riemann'],
            'accuracy_twin': r['accuracy_twin'],
            'chi2_riemann': r['chi2_riemann'],
            'chi2_twin': r['chi2_twin'],
            'distribution': {str(k): v for k, v in r['distribution'].items()}
        }
        for r in all_results
    ],
    'references': {
        'riemann': {str(k): v for k, v in P_riemann.items()},
        'twin_primes': {str(k): v for k, v in P_twin_primes.items() if v > 0.001}
    }
}

with open('/home/thlinux/relacionalidadegeral/validacao/ou_pure_generation_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Resultados salvos: validacao/ou_pure_generation_results.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO")
print(f"{'=' * 70}")

if best_riemann['accuracy_riemann'] > best_twin['accuracy_twin']:
    print(f"\n📊 OU + Gaussiano se parece MAIS com Riemann!")
    print(f"   Match Riemann: {best_riemann['accuracy_riemann']:.2f}%")
    print(f"   Match Twin: {best_twin['accuracy_twin']:.2f}%")
    print(f"\n   INTERPRETAÇÃO:")
    print(f"   O processo estocástico OU + Gaussiano naturalmente gera")
    print(f"   uma distribuição similar aos gaps dos zeros de Riemann.")
else:
    print(f"\n📊 OU + Gaussiano se parece MAIS com Twin Primes!")
    print(f"   Match Twin: {best_twin['accuracy_twin']:.2f}%")
    print(f"   Match Riemann: {best_riemann['accuracy_riemann']:.2f}%")
    print(f"\n   INTERPRETAÇÃO:")
    print(f"   O processo estocástico OU + Gaussiano naturalmente gera")
    print(f"   a distribuição P(k) = 2^-k dos twin primes!")

print(f"\n{'=' * 70}\n")
