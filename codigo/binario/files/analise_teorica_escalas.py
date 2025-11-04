#!/usr/bin/env python3
"""
Análise Teórica: Escalas de f_cosmos e Relação com Distribuição de Primos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Constantes fundamentais
G = 6.67430e-11
hbar = 1.054571817e-34
c = 2.99792458e8

M_Pl = np.sqrt(hbar * c / G)
f_Pl = np.sqrt(c**5 / (hbar * G))

# ==============================================================================
# PARTÍCULAS E OBJETOS
# ==============================================================================

objetos = {
    # Partículas elementares
    'elétron': 9.1093837015e-31,
    'múon': 1.883531627e-28,
    'tau': 3.16754e-27,
    'próton': 1.67262192369e-27,
    'nêutron': 1.67492749804e-27,
    
    # Objetos compostos
    'molécula_H2': 3.346e-27,
    'vírus': 1e-18,
    'bactéria': 1e-15,
    'grão_poeira': 1e-9,
    'grão_areia': 1e-6,
    'humano': 70,
    
    # Objetos astronômicos
    'Lua': 7.342e22,
    'Terra': 5.972e24,
    'Júpiter': 1.898e27,
    'Sol': 1.989e30,
    'Sagitário_A*': 8.26e36
}

def calcular_alpha_grav(m):
    return (m / M_Pl)**2

def calcular_f_cosmos(m):
    alpha = calcular_alpha_grav(m)
    return f_Pl * (alpha**(1/3))

# ==============================================================================
# MAPEAR ESCALAS
# ==============================================================================

print("=" * 80)
print("MAPEAMENTO DE ESCALAS: α_grav e f_cosmos")
print("=" * 80)
print()

dados_objetos = []

for nome, massa in objetos.items():
    alpha = calcular_alpha_grav(massa)
    f_c = calcular_f_cosmos(massa)
    
    dados_objetos.append({
        'nome': nome,
        'massa': massa,
        'log_massa': np.log10(massa),
        'alpha_grav': alpha,
        'log_alpha': np.log10(alpha),
        'f_cosmos': f_c,
        'log_f_cosmos': np.log10(f_c)
    })
    
    print(f"{nome:20s} | m={massa:.3e} kg | α_grav={alpha:.3e} | f_cosmos={f_c:.3e} Hz")

print("\n")

# ==============================================================================
# ANÁLISE: REGIÃO DE PRIMOS (~10^15)
# ==============================================================================

print("=" * 80)
print("ANÁLISE TEÓRICA: ESCALAS NO RANGE DE PRIMOS (~10^15)")
print("=" * 80)
print()

# Os primos gêmeos estão na escala ~10^15
# Qual seria a "frequência característica" desse espaço numérico?

N_primo = 1e15  # Escala dos primos sendo analisados

# Se interpretarmos os primos como uma "escala de comprimento" no espaço de números
# Podemos definir uma frequência característica como 1/N

f_caracterico_primo = 1.0 / N_primo
print(f"Frequência característica do espaço N~10^15:")
print(f"  f_char = 1/N = {f_caracterico_primo:.6e} Hz")
print()

# Comparar com f_cosmos de partículas
print("Razões f_cosmos(partícula) / f_char(primo):")
for obj in dados_objetos[:7]:  # Apenas partículas
    razao = obj['f_cosmos'] / f_caracterico_primo
    print(f"  {obj['nome']:15s}: {razao:.6e}")

print("\n")

# ==============================================================================
# HIPÓTESE: MODULAÇÃO HARMÔNICA
# ==============================================================================

print("=" * 80)
print("HIPÓTESE: MODULAÇÃO HARMÔNICA DA DENSIDADE DE PRIMOS")
print("=" * 80)
print()

print("Se a densidade de primos for modulada por f_cosmos, esperaríamos ver")
print("periodicidades em 'espaço log' correspondentes a:")
print()

# A periodicidade não seria em Hz, mas em "número de primos por intervalo"
# A modulação seria sutil, aparecendo como variações de densidade

# Para detectar, precisamos:
# 1. Densidade local em janelas log
# 2. FFT dessa série temporal
# 3. Procurar picos em frequências correspondentes a f_cosmos/f_char

print("Frequências relativas esperadas (f_cosmos / f_char):")
for obj in dados_objetos[:7]:
    f_rel = obj['f_cosmos'] / f_caracterico_primo
    periodo_em_numeros = N_primo / f_rel if f_rel > 0 else np.inf
    
    print(f"  {obj['nome']:15s}: f_rel = {f_rel:.3e} | Período ≈ {periodo_em_numeros:.3e} números")

print("\n")

# ==============================================================================
# VISUALIZAÇÃO: HIERARQUIA DE ESCALAS
# ==============================================================================

fig = plt.figure(figsize=(16, 10))

# --------------------------------------------------
# Plot 1: Massa vs α_grav (log-log)
# --------------------------------------------------
ax1 = plt.subplot(2, 3, 1)

massas_plot = [obj['massa'] for obj in dados_objetos]
alphas_plot = [obj['alpha_grav'] for obj in dados_objetos]
nomes_plot = [obj['nome'] for obj in dados_objetos]

# Linha teórica: α ∝ m²
m_teorico = np.logspace(-31, 37, 100)
alpha_teorico = (m_teorico / M_Pl)**2

ax1.loglog(m_teorico, alpha_teorico, 'k--', alpha=0.5, linewidth=2, label='α = (m/M_Pl)²')
ax1.loglog(massas_plot, alphas_plot, 'o', markersize=8, alpha=0.7)

for i, nome in enumerate(nomes_plot):
    if i % 3 == 0:  # Evitar poluição visual
        ax1.annotate(nome, (massas_plot[i], alphas_plot[i]), 
                    fontsize=7, ha='right', alpha=0.7)

ax1.set_xlabel('Massa (kg)', fontsize=11)
ax1.set_ylabel('α_grav', fontsize=11)
ax1.set_title('Acoplamento Gravitacional vs Massa', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend()

# --------------------------------------------------
# Plot 2: Massa vs f_cosmos (log-log)
# --------------------------------------------------
ax2 = plt.subplot(2, 3, 2)

f_cosmos_plot = [obj['f_cosmos'] for obj in dados_objetos]

# Linha teórica: f ∝ m^(2/3)
f_teorico = f_Pl * ((m_teorico / M_Pl)**2)**(1/3)

ax2.loglog(m_teorico, f_teorico, 'k--', alpha=0.5, linewidth=2, label='f_cosmos ∝ m^(2/3)')
ax2.loglog(massas_plot, f_cosmos_plot, 'o', markersize=8, alpha=0.7, color='C1')

for i, nome in enumerate(nomes_plot):
    if i % 3 == 0:
        ax2.annotate(nome, (massas_plot[i], f_cosmos_plot[i]), 
                    fontsize=7, ha='right', alpha=0.7)

ax2.set_xlabel('Massa (kg)', fontsize=11)
ax2.set_ylabel('f_cosmos (Hz)', fontsize=11)
ax2.set_title('Frequência Gravitacional vs Massa', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend()

# --------------------------------------------------
# Plot 3: α_grav vs f_cosmos
# --------------------------------------------------
ax3 = plt.subplot(2, 3, 3)

ax3.loglog(alphas_plot, f_cosmos_plot, 'o', markersize=8, alpha=0.7, color='C2')

for i, nome in enumerate(nomes_plot):
    if i % 3 == 0:
        ax3.annotate(nome, (alphas_plot[i], f_cosmos_plot[i]), 
                    fontsize=7, ha='right', alpha=0.7)

ax3.set_xlabel('α_grav', fontsize=11)
ax3.set_ylabel('f_cosmos (Hz)', fontsize=11)
ax3.set_title('f_cosmos vs α_grav (relação α^(1/3))', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')

# Adicionar linha de tendência
alpha_fit = np.logspace(np.log10(min(alphas_plot)), np.log10(max(alphas_plot)), 100)
f_fit = f_Pl * (alpha_fit)**(1/3)
ax3.loglog(alpha_fit, f_fit, 'k--', alpha=0.5, linewidth=2, label='f ∝ α^(1/3)')
ax3.legend()

# --------------------------------------------------
# Plot 4: Espectro de Frequências (Linear)
# --------------------------------------------------
ax4 = plt.subplot(2, 3, 4)

# Ordenar por frequência
dados_sorted = sorted(dados_objetos, key=lambda x: x['f_cosmos'])

y_pos = np.arange(len(dados_sorted))
freqs = [obj['f_cosmos'] for obj in dados_sorted]
names = [obj['nome'] for obj in dados_sorted]

# Cores por categoria
cores = []
for obj in dados_sorted:
    m = obj['massa']
    if m < 1e-20:
        cores.append('blue')  # Partículas
    elif m < 1:
        cores.append('green')  # Micro/nano
    elif m < 1e20:
        cores.append('orange')  # Macro
    else:
        cores.append('red')  # Astronômico

bars = ax4.barh(y_pos, freqs, color=cores, alpha=0.7, edgecolor='black')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(names, fontsize=8)
ax4.set_xlabel('f_cosmos (Hz)', fontsize=11)
ax4.set_title('Espectro de f_cosmos por Objeto', fontsize=12, fontweight='bold')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3, axis='x')

# --------------------------------------------------
# Plot 5: Razões Harmônicas
# --------------------------------------------------
ax5 = plt.subplot(2, 3, 5)

# Calcular razões entre f_cosmos consecutivos
razoes = []
pares = []

for i in range(len(dados_sorted)-1):
    f1 = dados_sorted[i]['f_cosmos']
    f2 = dados_sorted[i+1]['f_cosmos']
    razao = f2 / f1
    razoes.append(razao)
    pares.append(f"{dados_sorted[i]['nome'][:8]}\n→\n{dados_sorted[i+1]['nome'][:8]}")

x_pos = np.arange(len(razoes))
ax5.bar(x_pos, razoes, alpha=0.7, color='purple', edgecolor='black')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(pares, fontsize=6, rotation=0)
ax5.set_ylabel('Razão f_j+1 / f_j', fontsize=11)
ax5.set_title('Razões Harmônicas Entre Objetos Consecutivos', fontsize=12, fontweight='bold')
ax5.axhline(1, color='k', linestyle='--', alpha=0.3)
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3, axis='y')

# --------------------------------------------------
# Plot 6: Região de Primos Destacada
# --------------------------------------------------
ax6 = plt.subplot(2, 3, 6)

# Criar visualização mostrando onde ~10^15 se encaixa

# Eixo X: log(N) para números
# Eixo Y: densidade esperada / frequência característica

N_range = np.logspace(10, 20, 100)
f_char_range = 1.0 / N_range

ax6.loglog(N_range, f_char_range, 'b-', linewidth=2, label='f_char = 1/N')

# Destacar região de primos analisados
N_primos = 1e15
f_primos = 1.0 / N_primos
ax6.axvline(N_primos, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Primos analisados (N≈10^15)')
ax6.axhline(f_primos, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Adicionar f_cosmos de partículas como linhas horizontais
for obj in dados_objetos[:5]:
    ax6.axhline(obj['f_cosmos'], color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax6.text(1e19, obj['f_cosmos'], obj['nome'], fontsize=7, va='bottom')

ax6.set_xlabel('N (escala numérica)', fontsize=11)
ax6.set_ylabel('Frequência (Hz)', fontsize=11)
ax6.set_title('Contexto: Região de Primos vs f_cosmos', fontsize=12, fontweight='bold')
ax6.legend(loc='upper right', fontsize=9)
ax6.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/escalas_teoricas_fcosmos.png', dpi=150, bbox_inches='tight')
print("✓ Gráfico salvo: escalas_teoricas_fcosmos.png")

# ==============================================================================
# PREVISÕES TEÓRICAS
# ==============================================================================

print("\n" + "=" * 80)
print("PREVISÕES TEÓRICAS PARA DETECÇÃO")
print("=" * 80)
print()

print("1. PERIODICIDADE ESPERADA:")
print("   Se a densidade de primos é modulada por f_cosmos, a periodicidade")
print("   aparecerá no espaço logarítmico, não no espaço linear.")
print()

print("2. ESCALA DE MODULAÇÃO:")
print("   Para elétron: f_cosmos ≈ 2.236 Hz")
print(f"   Razão: f_cosmos/f_char = {2.236 / f_caracterico_primo:.3e}")
print(f"   Período esperado: ~{N_primo / (2.236 / f_caracterico_primo):.3e} números")
print()

print("3. DETECÇÃO PRÁTICA:")
print("   - Usar janelas deslizantes em log(p)")
print("   - Calcular densidade local (primos/intervalo)")
print("   - FFT da série de densidade")
print("   - Procurar picos em f_rel = f_cosmos/f_char")
print()

print("4. HARMÔNICOS:")
print("   Além da frequência fundamental, esperar harmônicos:")
print("   f, 2f, 3f, 5f, ... (série de Fibonacci?)")
print()

print("=" * 80)
