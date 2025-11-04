#!/usr/bin/env python3
"""
Análise do padrão de convergência
Dados observados:
t=100: 91.66%
t=500: 92.37%
t=1000: 94.95%

Extrapolação para prever t necessário para 100%
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

print("=" * 70)
print("ANÁLISE DO PADRÃO DE CONVERGÊNCIA")
print("=" * 70)

# Dados observados
t_obs = np.array([100, 500, 1000])
acc_obs = np.array([91.66, 92.37, 94.95])

print(f"\n[DADOS OBSERVADOS]")
for t, acc in zip(t_obs, acc_obs):
    print(f"  t={t}: {acc:.2f}%")

# Testar diferentes fits
print(f"\n{'=' * 70}")
print("TESTANDO DIFERENTES MODELOS")
print(f"{'=' * 70}")

# Modelo 1: Logarítmico
# Acc(t) = a + b * log(t)
def log_model(t, a, b):
    return a + b * np.log(t)

params_log, cov_log = curve_fit(log_model, t_obs, acc_obs)
a_log, b_log = params_log
r2_log = 1 - np.sum((acc_obs - log_model(t_obs, a_log, b_log))**2) / np.sum((acc_obs - np.mean(acc_obs))**2)

print(f"\n[MODELO 1: Logarítmico]")
print(f"  Acc(t) = {a_log:.4f} + {b_log:.4f} × ln(t)")
print(f"  R² = {r2_log:.6f}")

# Prever t para 100%
if b_log > 0:
    t_for_100_log = np.exp((100 - a_log) / b_log)
    print(f"  Para Acc=100%: t ≈ {t_for_100_log:.0f}")
    print(f"  Para Acc=99%: t ≈ {np.exp((99 - a_log) / b_log):.0f}")
    print(f"  Para Acc=98%: t ≈ {np.exp((98 - a_log) / b_log):.0f}")

# Modelo 2: Potência
# Acc(t) = a + b * t^c
def power_model(t, a, b, c):
    return a + b * t**c

try:
    params_power, cov_power = curve_fit(power_model, t_obs, acc_obs, p0=[80, 1, 0.1], maxfev=10000)
    a_pow, b_pow, c_pow = params_power
    r2_pow = 1 - np.sum((acc_obs - power_model(t_obs, a_pow, b_pow, c_pow))**2) / np.sum((acc_obs - np.mean(acc_obs))**2)

    print(f"\n[MODELO 2: Potência]")
    print(f"  Acc(t) = {a_pow:.4f} + {b_pow:.4f} × t^{c_pow:.6f}")
    print(f"  R² = {r2_pow:.6f}")

    # Prever t para 100%
    if b_pow > 0 and c_pow != 0:
        t_for_100_pow = ((100 - a_pow) / b_pow) ** (1/c_pow)
        print(f"  Para Acc=100%: t ≈ {t_for_100_pow:.0f}")
except:
    print(f"\n[MODELO 2: Potência]")
    print(f"  Falha no fit")
    r2_pow = -1

# Modelo 3: Assintótico
# Acc(t) = a - b * exp(-c * t)
def asymptotic_model(t, a, b, c):
    return a - b * np.exp(-c * t)

try:
    params_asym, cov_asym = curve_fit(asymptotic_model, t_obs, acc_obs, p0=[100, 10, 0.001], maxfev=10000)
    a_asym, b_asym, c_asym = params_asym
    r2_asym = 1 - np.sum((acc_obs - asymptotic_model(t_obs, a_asym, b_asym, c_asym))**2) / np.sum((acc_obs - np.mean(acc_obs))**2)

    print(f"\n[MODELO 3: Assintótico]")
    print(f"  Acc(t) = {a_asym:.4f} - {b_asym:.4f} × exp(-{c_asym:.6f} × t)")
    print(f"  R² = {r2_asym:.6f}")
    print(f"  Limite assintótico: {a_asym:.2f}%")

    if b_asym > 0 and c_asym > 0:
        # Para Acc = a - ε, resolver ε = b * exp(-c*t)
        for target_acc in [98, 99, 99.5, 100]:
            if target_acc < a_asym:
                epsilon = a_asym - target_acc
                t_target = -np.log(epsilon / b_asym) / c_asym if epsilon > 0 else np.inf
                print(f"  Para Acc={target_acc:.1f}%: t ≈ {t_target:.0f}")
            else:
                print(f"  Para Acc={target_acc:.1f}%: Acima do limite assintótico!")
except Exception as e:
    print(f"\n[MODELO 3: Assintótico]")
    print(f"  Falha no fit: {e}")
    r2_asym = -1

# Modelo 4: log(100-Acc) vs log(t)  [decaimento tipo lei de potência]
# log(100-Acc) = a + b*log(t)  =>  100-Acc = exp(a) * t^b
error_obs = 100 - acc_obs
log_error = np.log(error_obs)
log_t = np.log(t_obs)

slope_ll, intercept_ll, r_value_ll, p_value_ll, std_err_ll = stats.linregress(log_t, log_error)
r2_ll = r_value_ll**2

print(f"\n[MODELO 4: Log-Log (decaimento erro)]")
print(f"  log(100-Acc) = {intercept_ll:.4f} + {slope_ll:.4f} × log(t)")
print(f"  100 - Acc = {np.exp(intercept_ll):.4f} × t^{slope_ll:.4f}")
print(f"  R² = {r2_ll:.6f}")

if slope_ll < 0:  # convergindo
    for target_acc in [98, 99, 99.5, 100]:
        target_error = 100 - target_acc
        t_target = np.exp((np.log(target_error) - intercept_ll) / slope_ll)
        print(f"  Para Acc={target_acc:.1f}%: t ≈ {t_target:.0f}")

# Escolher melhor modelo
best_r2 = max(r2_log, r2_pow, r2_asym, r2_ll)
if best_r2 == r2_log:
    best_model = "Logarítmico"
elif best_r2 == r2_pow:
    best_model = "Potência"
elif best_r2 == r2_asym:
    best_model = "Assintótico"
else:
    best_model = "Log-Log"

print(f"\n{'=' * 70}")
print(f"MELHOR MODELO: {best_model} (R² = {best_r2:.6f})")
print(f"{'=' * 70}")

# Gráfico
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Dados + fits
ax1 = axes[0, 0]
t_pred = np.logspace(2, 4.5, 100)  # 100 a ~30000

ax1.scatter(t_obs, acc_obs, s=100, c='blue', zorder=5, label='Dados observados')

# Fit logarítmico
acc_pred_log = log_model(t_pred, a_log, b_log)
ax1.semilogx(t_pred, acc_pred_log, 'r-', linewidth=2, label=f'Log: R²={r2_log:.4f}')

# Fit log-log se bom
if r2_ll > 0.9:
    acc_pred_ll = 100 - np.exp(intercept_ll) * t_pred**slope_ll
    ax1.semilogx(t_pred, acc_pred_ll, 'g--', linewidth=2, label=f'Log-Log: R²={r2_ll:.4f}')

ax1.axhline(100, color='darkgreen', linestyle=':', linewidth=2, label='100%')
ax1.axhline(95, color='green', linestyle=':', linewidth=2, label='95%')
ax1.set_xlabel('Trial t (log scale)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Extrapolação da Convergência', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([90, 102])
ax1.set_xlim([80, 30000])

# 2. Log-Log plot (erro vs t)
ax2 = axes[0, 1]
ax2.loglog(t_obs, 100-acc_obs, 'bo', markersize=10, label='Dados')
# Fit line
t_fit = np.logspace(2, 4.5, 100)
error_fit = np.exp(intercept_ll) * t_fit**slope_ll
ax2.loglog(t_fit, error_fit, 'r-', linewidth=2, label=f'Fit: R²={r2_ll:.4f}')
ax2.set_xlabel('Trial t (log)', fontsize=12)
ax2.set_ylabel('100 - Accuracy (log)', fontsize=12)
ax2.set_title('Decaimento do Erro (Log-Log)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Tabela de predições
ax3 = axes[1, 0]
ax3.axis('off')

# Criar tabela
target_accs = [95, 96, 97, 98, 99, 99.5, 100]
table_data = []

for target in target_accs:
    # Log model
    if b_log > 0:
        t_log = np.exp((target - a_log) / b_log)
    else:
        t_log = np.inf

    # Log-log model
    if slope_ll < 0:
        error_target = 100 - target
        if error_target > 0:
            t_ll = np.exp((np.log(error_target) - intercept_ll) / slope_ll)
        else:
            t_ll = np.inf
    else:
        t_ll = np.inf

    table_data.append([f"{target}%", f"{t_log:.0f}" if t_log < 1e6 else "∞",
                       f"{t_ll:.0f}" if t_ll < 1e6 else "∞"])

table = ax3.table(cellText=table_data,
                  colLabels=['Target', 'Log Model', 'Log-Log Model'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0.2, 1, 0.7])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Estilizar header
for i in range(3):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax3.set_title('Predição de Trials Necessários', fontsize=14, fontweight='bold', pad=20)

# 4. Comparação visual
ax4 = axes[1, 1]
models_names = ['Observado\nt=1000', 'Predição\nt=5000', 'Predição\nt=10000']
if slope_ll < 0:
    acc_5000 = 100 - np.exp(intercept_ll) * 5000**slope_ll
    acc_10000 = 100 - np.exp(intercept_ll) * 10000**slope_ll
    accs_pred = [94.95, acc_5000, acc_10000]
else:
    accs_pred = [94.95, 98, 99.5]

colors_bars = ['blue', 'orange', 'green']
bars = ax4.bar(models_names, accs_pred, color=colors_bars, alpha=0.7)
ax4.axhline(100, color='darkgreen', linestyle='--', linewidth=2)
ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.set_title('Predição para t Maiores', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([90, 102])

# Adicionar valores
for bar, acc in zip(bars, accs_pred):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/thlinux/relacionalidadegeral/validacao/convergence_pattern_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: validacao/convergence_pattern_analysis.png")

# Salvar análise
output = {
    'observed_data': {
        't': t_obs.tolist(),
        'accuracy': acc_obs.tolist()
    },
    'models': {
        'logarithmic': {
            'formula': f'Acc(t) = {a_log:.4f} + {b_log:.4f} * ln(t)',
            'r_squared': float(r2_log),
            'params': {'a': float(a_log), 'b': float(b_log)}
        },
        'loglog': {
            'formula': f'100 - Acc = {np.exp(intercept_ll):.4f} * t^{slope_ll:.4f}',
            'r_squared': float(r2_ll),
            'params': {'intercept': float(intercept_ll), 'slope': float(slope_ll)}
        }
    },
    'predictions': {}
}

# Adicionar predições
for target in [95, 98, 99, 99.5, 100]:
    if slope_ll < 0:
        error_target = 100 - target
        if error_target > 0:
            t_pred_ll = float(np.exp((np.log(error_target) - intercept_ll) / slope_ll))
        else:
            t_pred_ll = float('inf')
    else:
        t_pred_ll = float('inf')

    output['predictions'][f'{target}%'] = {
        'loglog_model': t_pred_ll
    }

import json
with open('/home/thlinux/relacionalidadegeral/validacao/convergence_pattern_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Análise salva: validacao/convergence_pattern_analysis.json")

print(f"\n{'=' * 70}")
print("CONCLUSÃO")
print(f"{'=' * 70}")

if r2_ll > 0.99:
    print(f"\n✅ PADRÃO CLARO DE CONVERGÊNCIA!")
    print(f"   Modelo: 100 - Acc = {np.exp(intercept_ll):.4f} × t^{slope_ll:.4f}")
    print(f"   R² = {r2_ll:.6f} (excelente fit!)")
    print(f"\n   PREDIÇÕES:")
    if slope_ll < 0:
        for target in [98, 99, 100]:
            error = 100 - target
            t_needed = np.exp((np.log(error) - intercept_ll) / slope_ll)
            print(f"   • {target}% accuracy: t ≈ {t_needed:.0f} trials")
else:
    print(f"\n🟡 Padrão identificado mas incerto")
    print(f"   R² = {r2_ll:.4f}")
    print(f"   Recomendação: Testar com mais pontos (t=2000, 5000)")

print(f"\n{'=' * 70}\n")
