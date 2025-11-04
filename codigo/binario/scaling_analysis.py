#!/usr/bin/env python3
"""
ANÁLISE REFINADA: Scaling de picos vs tamanho do dataset
Determinar lei de potência: N_picos ∝ N_primos^β
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

print("=" * 80)
print("ANÁLISE DE SCALING: Número de Picos vs Tamanho do Dataset")
print("=" * 80)

# Dados empíricos
datasets = {
    '1M':   {'primos': 1_000_000,   'picos': 8,  'sigma_max': 11.1},
    '10M':  {'primos': 10_000_000,  'picos': 20, 'sigma_max': 24.3},
    '1B':   {'primos': 1_004_800_003, 'picos': None, 'sigma_max': None}  # A determinar
}

# Constantes teóricas
alpha_em_inv = 137.035999084
alpha_grav = 1.751809e-45
alpha_em = 1/alpha_em_inv
log_ratio = np.log10(alpha_em / alpha_grav)

print(f"\nlog₁₀(α_EM/α_grav) = {log_ratio:.2f}")
print(f"Predição teórica: ~{log_ratio:.0f} picos\n")

# Arrays para fitting
N = np.array([datasets['1M']['primos'], datasets['10M']['primos']])
picos = np.array([datasets['1M']['picos'], datasets['10M']['picos']])

print("Dados empíricos:")
print(f"  1M primos  → {datasets['1M']['picos']} picos")
print(f"  10M primos → {datasets['10M']['picos']} picos")

# Fit lei de potência: N_picos = A × N^β
log_N = np.log10(N)
log_picos = np.log10(picos)

# Fit linear no espaço log
coeffs = np.polyfit(log_N, log_picos, 1)
beta = coeffs[0]
log_A = coeffs[1]
A = 10**log_A

print(f"\n📊 LEI DE POTÊNCIA:")
print(f"   N_picos = {A:.6f} × N^{beta:.4f}")
print(f"   β = {beta:.4f}")

# Interpretação do expoente
print(f"\n💡 INTERPRETAÇÃO:")
if abs(beta - 0.5) < 0.05:
    print("   β ≈ 0.5 → Picos ∝ √N (crescimento sub-linear)")
    print("   Consistente com fenômenos de difusão/random walk")
elif abs(beta - 1.0) < 0.05:
    print("   β ≈ 1.0 → Picos ∝ N (crescimento linear)")
    print("   Sugestão: Novo pico a cada X primos (constante)")
elif beta > 1.0:
    print(f"   β > 1.0 → Crescimento super-linear!")
    print("   ⚠️  Implica aceleração - verificar se é real ou artefato")
else:
    print(f"   β = {beta:.3f} → Crescimento intermediário")

# Projeções
N_1B = datasets['1B']['primos']
picos_1B_projetado = A * (N_1B ** beta)

print(f"\n🎯 PROJEÇÃO PARA 1B PRIMOS:")
print(f"   Esperado: {picos_1B_projetado:.1f} picos")
print(f"   Teórico:  {log_ratio:.0f} picos (α_EM/α_grav)")
print(f"   Razão:    {picos_1B_projetado/log_ratio:.2f}")

# Determinar em que tamanho chegamos a ~43 picos
if beta > 0:
    N_para_43_picos = (43 / A) ** (1/beta)
    print(f"\n📈 Para atingir {log_ratio:.0f} picos:")
    print(f"   Precisamos de ~{N_para_43_picos:.2e} primos")
    if N_para_43_picos < N_1B:
        print(f"   ✅ DENTRO do dataset de 1B!")
    else:
        print(f"   ❌ FORA do dataset atual")
        print(f"   Precisaríamos minerar até ~{N_para_43_picos/1e15:.1f}e15")

# Análise alternativa: threshold adaptativo
print("\n" + "=" * 80)
print("HIPÓTESE ALTERNATIVA: Threshold Adaptativo")
print("=" * 80)

print("""
Se o número de picos VERDADEIROS é constante (~43), mas threshold 3σ
captura mais picos em datasets maiores (melhor resolução espectral), então:

  N_picos_detectados(3σ) = N_picos_real + f(N)
  
onde f(N) são picos espúrios que aparecem com melhor resolução.

Para ter exatamente 43 picos, precisaríamos threshold adaptativo:
""")

# Calcular threshold necessário para ter 43 picos
# Aproximação: σ_threshold ≈ σ_base + k × log(N)
for dataset_name, data in datasets.items():
    if data['picos'] is not None:
        # Estimativa: se com 3σ temos X picos, com quanto σ teríamos 43?
        # Relação empírica: N_picos ∝ exp(-σ²/2)
        # Resolver: 43 / X = exp(-(σ_new² - 9)/2)
        X = data['picos']
        if X < 43:
            # Precisamos diminuir threshold (mais picos)
            ratio = 43 / X
            sigma_new = np.sqrt(9 - 2*np.log(ratio))
            print(f"  {dataset_name}: threshold {sigma_new:.2f}σ → 43 picos")
        elif X > 43:
            # Precisamos aumentar threshold (menos picos)
            ratio = X / 43
            sigma_new = np.sqrt(9 + 2*np.log(ratio))
            print(f"  {dataset_name}: threshold {sigma_new:.2f}σ → 43 picos")
        else:
            print(f"  {dataset_name}: threshold 3.00σ → 43 picos ✓")

# Visualização
print("\n" + "=" * 80)
print("Gerando visualização do scaling...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Lei de potência (escala log-log)
ax1 = axes[0, 0]
N_range = np.logspace(6, 9.5, 100)
picos_fit = A * (N_range ** beta)

ax1.loglog(N, picos, 'ro', markersize=15, label='Dados empíricos', zorder=5)
ax1.loglog(N_range, picos_fit, 'b--', linewidth=2, label=f'Fit: N^{beta:.3f}', alpha=0.7)
ax1.axhline(log_ratio, color='orange', linestyle='--', linewidth=2, label=f'α_EM/α_grav = {log_ratio:.0f}')
ax1.axvline(N_1B, color='green', linestyle=':', linewidth=2, alpha=0.5, label='1B primos')
ax1.plot(N_1B, picos_1B_projetado, 'g^', markersize=12, label=f'Projeção 1B: {picos_1B_projetado:.0f}')

ax1.set_xlabel('Número de Primos', fontsize=12)
ax1.set_ylabel('Número de Picos (3σ)', fontsize=12)
ax1.set_title('Lei de Scaling: Picos vs Tamanho Dataset', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# 2. Desvio da predição teórica
ax2 = axes[0, 1]
N_plot = np.array([1e6, 10e6, N_1B])
picos_plot = A * (N_plot ** beta)
desvio = (picos_plot - log_ratio) / log_ratio * 100

ax2.semilogx(N_plot, desvio, 'bo-', linewidth=2, markersize=10)
ax2.axhline(0, color='orange', linestyle='--', linewidth=2, label='Predição teórica')
ax2.fill_between(N_plot, -10, 10, alpha=0.2, color='green', label='±10% erro')
ax2.set_xlabel('Número de Primos', fontsize=12)
ax2.set_ylabel('Desvio da Predição (%)', fontsize=12)
ax2.set_title('Desvio de log₁₀(α_EM/α_grav)', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Significância máxima vs tamanho
ax3 = axes[1, 0]
N_sigma = np.array([1e6, 10e6])
sigma_max = np.array([11.1, 24.3])

# Fit: σ_max ∝ log(N) ou √N ou N^α ?
coeffs_sigma = np.polyfit(np.log10(N_sigma), sigma_max, 1)
sigma_fit = coeffs_sigma[0] * np.log10(N_range) + coeffs_sigma[1]

ax3.semilogx(N_sigma, sigma_max, 'ro', markersize=15, label='Observado', zorder=5)
ax3.semilogx(N_range, sigma_fit, 'b--', linewidth=2, 
             label=f'Fit: {coeffs_sigma[0]:.1f}×log₁₀(N) + {coeffs_sigma[1]:.1f}')
ax3.axvline(N_1B, color='green', linestyle=':', linewidth=2, alpha=0.5)
sigma_1B = coeffs_sigma[0] * np.log10(N_1B) + coeffs_sigma[1]
ax3.plot(N_1B, sigma_1B, 'g^', markersize=12, label=f'Projeção 1B: {sigma_1B:.1f}σ')

ax3.set_xlabel('Número de Primos', fontsize=12)
ax3.set_ylabel('Significância Máxima (σ)', fontsize=12)
ax3.set_title('Pico Mais Forte vs Tamanho', fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Razão picos/predição
ax4 = axes[1, 1]
N_plot_ratio = np.logspace(6, 9.5, 50)
picos_plot_ratio = A * (N_plot_ratio ** beta)
ratio_plot = picos_plot_ratio / log_ratio

ax4.semilogx(N_plot_ratio, ratio_plot, 'b-', linewidth=3)
ax4.axhline(1.0, color='orange', linestyle='--', linewidth=2, label='Concordância perfeita')
ax4.fill_between(N_plot_ratio, 0.9, 1.1, alpha=0.2, color='green', label='±10%')
ax4.axvline(N_1B, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax4.plot(N_1B, picos_1B_projetado/log_ratio, 'g^', markersize=12, 
         label=f'1B: {picos_1B_projetado/log_ratio:.2f}×')

ax4.set_xlabel('Número de Primos', fontsize=12)
ax4.set_ylabel('Razão: Observado / Teórico', fontsize=12)
ax4.set_title('Convergência para α_EM/α_grav', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scaling_analysis_alpha_em.png', dpi=150, bbox_inches='tight')
print("✓ Salvo: scaling_analysis_alpha_em.png\n")

# Conclusão
print("=" * 80)
print("CONCLUSÃO: HIPÓTESE α_EM")
print("=" * 80)

if picos_1B_projetado / log_ratio > 2.5:
    print(f"""
⚠️  PROJEÇÃO MUITO ACIMA DA PREDIÇÃO ({picos_1B_projetado/log_ratio:.1f}×)

Interpretações possíveis:
1. Lei de scaling está errada (apenas 2 pontos)
2. Threshold 3σ captura muitos falsos positivos em N grande
3. A hipótese α_EM está incorreta
4. Número "verdadeiro" de picos ≠ log₁₀(α_EM/α_grav)

TESTE CRÍTICO NECESSÁRIO:
→ Analisar dataset completo de 1B primos
→ Usar múltiplos thresholds (3σ, 5σ, 7σ)
→ Verificar se existe plateau em ~43 picos
""")
elif abs(picos_1B_projetado/log_ratio - 1.0) < 0.2:
    print(f"""
✅ CONCORDÂNCIA EXCELENTE! ({picos_1B_projetado/log_ratio:.2f}×)

A projeção para 1B primos está muito próxima de log₁₀(α_EM/α_grav)!

INTERPRETAÇÃO:
- Periodicidade reflete hierarquia de acoplamentos
- ~{log_ratio:.0f} modos fundamentais
- Cada modo ~ 1 ordem de grandeza em α_EM/α_grav

PRÓXIMO PASSO:
→ Confirmar com análise do dataset completo (1B)
""")
else:
    print(f"""
🔍 DESVIO MODERADO ({picos_1B_projetado/log_ratio:.2f}×)

Diferença: {abs(picos_1B_projetado - log_ratio):.1f} picos

Possíveis ajustes:
1. β pode não ser constante (saturação em N→∞)
2. Threshold ótimo pode ser > 3σ para N grande
3. Número "real" pode ser submúltiplo de 43 (ex: 43/2, 43/3)

RECOMENDAÇÃO:
→ Análise com dataset completo para validar
""")

print("=" * 80)
