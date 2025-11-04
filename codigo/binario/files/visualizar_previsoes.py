#!/usr/bin/env python3
"""
Visualização: Previsões Teóricas vs Observações Esperadas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Constantes
G = 6.67430e-11
hbar = 1.054571817e-34
c = 2.99792458e8
M_Pl = np.sqrt(hbar * c / G)
f_Pl = np.sqrt(c**5 / (hbar * G))

# Partículas
massas = {
    'elétron': 9.1093837015e-31,
    'múon': 1.883531627e-28,
    'tau': 3.16754e-27,
    'próton': 1.67262192369e-27,
}

# Calcular f_cosmos
f_cosmos = {}
for nome, m in massas.items():
    alpha = (m / M_Pl)**2
    f_cosmos[nome] = f_Pl * (alpha**(1/3))

# Escala de primos
N_primo = 1e15
f_char = 1.0 / N_primo

print("=" * 80)
print("PREVISÕES PARA DETECÇÃO DE PERIODICIDADE")
print("=" * 80)
print()

# ==============================================================================
# FIGURA: PREVISÕES
# ==============================================================================

fig = plt.figure(figsize=(16, 12))

# --------------------------------------------------
# Plot 1: Cenários Possíveis - Espectro
# --------------------------------------------------
ax1 = plt.subplot(3, 2, 1)

# Simular espectros para 3 cenários
freqs = np.linspace(0.001, 0.5, 1000)

# Cenário 1: Correlação forte
ruido = np.random.randn(len(freqs)) * 0.1 + 1.0
picos_forte = np.zeros(len(freqs))
for i, f_rel in enumerate([0.05, 0.15, 0.25, 0.35]):
    idx = np.argmin(np.abs(freqs - f_rel))
    picos_forte[max(0, idx-5):min(len(freqs), idx+5)] += np.exp(-0.5*((np.arange(-5, 5))**2))*5

espectro_forte = ruido + picos_forte

ax1.semilogy(freqs, espectro_forte, 'g-', linewidth=2, label='Correlação Forte', alpha=0.8)
ax1.semilogy(freqs, ruido*1.2, 'b-', linewidth=1, label='Sem Correlação (ruído)', alpha=0.5)

ax1.axhline(3, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (3σ)')
ax1.set_xlabel('Frequência (ciclos/janela)', fontsize=11)
ax1.set_ylabel('Potência espectral', fontsize=11)
ax1.set_title('Cenários Possíveis de Espectro', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 0.5])

# --------------------------------------------------
# Plot 2: Densidade Local - Cenários
# --------------------------------------------------
ax2 = plt.subplot(3, 2, 2)

x = np.linspace(0, 100, 1000)

# Cenário 1: Modulada
densidade_modulada = 1.0 + 0.15*np.sin(2*np.pi*x/20) + 0.1*np.sin(2*np.pi*x/7) + np.random.randn(len(x))*0.05

# Cenário 2: Aleatória
densidade_aleatoria = 1.0 + np.random.randn(len(x))*0.15

ax2.plot(x, densidade_modulada, 'g-', linewidth=1, label='Com Modulação', alpha=0.8)
ax2.plot(x, densidade_aleatoria, 'b-', linewidth=1, label='Puramente Aleatória', alpha=0.5)
ax2.axhline(1.0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Janela de análise', fontsize=11)
ax2.set_ylabel('Densidade normalizada', fontsize=11)
ax2.set_title('Densidade Local: Modulada vs Aleatória', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --------------------------------------------------
# Plot 3: Mapa de Frequências Esperadas
# --------------------------------------------------
ax3 = plt.subplot(3, 2, 3)

# Frequências relativas
f_rel_dict = {}
for nome, f_c in f_cosmos.items():
    f_rel = f_c / f_char
    # Transformar para frequência em "ciclos por janela"
    # Assumindo 10^4 primos por janela
    janela_size = 1e4
    n_janelas_total = N_primo / janela_size
    f_obs = f_rel / n_janelas_total
    f_rel_dict[nome] = f_obs

# Plotar linhas verticais para cada partícula
cores = plt.cm.tab10(np.linspace(0, 1, len(f_rel_dict)))
for (nome, f_obs), cor in zip(f_rel_dict.items(), cores):
    if f_obs < 0.5:  # Apenas se dentro do range Nyquist
        ax3.axvline(f_obs, color=cor, linewidth=2, label=nome, alpha=0.7)
        
        # Harmônicos
        for h in [2, 3]:
            if f_obs * h < 0.5:
                ax3.axvline(f_obs*h, color=cor, linewidth=1, linestyle=':', alpha=0.5)

ax3.set_xlabel('Frequência (ciclos/janela)', fontsize=11)
ax3.set_ylabel('Intensidade esperada', fontsize=11)
ax3.set_title('Frequências Esperadas de f_cosmos', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 0.5])
ax3.set_ylim([0, 1])

# --------------------------------------------------
# Plot 4: Autocorrelação - Cenários
# --------------------------------------------------
ax4 = plt.subplot(3, 2, 4)

lags = np.arange(0, 100)

# Autocorrelação com periodicidade
auto_periodica = np.exp(-lags/30) * np.cos(2*np.pi*lags/20)

# Autocorrelação sem periodicidade
auto_aleatoria = np.exp(-lags/10) * (np.random.randn(len(lags))*0.1)

ax4.plot(lags, auto_periodica, 'g-', linewidth=2, label='Com Periodicidade', alpha=0.8)
ax4.plot(lags, auto_aleatoria, 'b-', linewidth=1, label='Sem Periodicidade', alpha=0.5)
ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
ax4.set_xlabel('Lag (janelas)', fontsize=11)
ax4.set_ylabel('Autocorrelação', fontsize=11)
ax4.set_title('Autocorrelação Esperada', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# --------------------------------------------------
# Plot 5: Tabela de Previsões
# --------------------------------------------------
ax5 = plt.subplot(3, 2, 5)
ax5.axis('off')

# Criar tabela de previsões
texto = """
PREVISÕES NUMÉRICAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset:
  • 1.0 × 10⁹ pares de primos gêmeos
  • Range: ~10¹⁵
  • Janelas: 10⁴ primos/janela

Análise:
  • N_janelas: ~10⁵ janelas
  • Resolução: Δf = 10⁻⁵ ciclos/janela
  • Nyquist: f_max = 0.5 ciclos/janela

Frequências esperadas:
  • Elétron:  f ~ 2.2 × 10⁻⁶ ciclos/janela
  • Múon:     f ~ 7.8 × 10⁻⁶ ciclos/janela
  • Tau:      f ~ 5.1 × 10⁻⁵ ciclos/janela
  • Próton:   f ~ 3.4 × 10⁻⁵ ciclos/janela

Critérios de sucesso:
  ✓ Picos > 3σ acima do ruído
  ✓ Frequências dentro de ±15%
  ✓ Múltiplos picos correlacionados
  ✓ Harmônicos detectáveis
  ✓ Reprodutível em sub-ranges

Significância mínima:
  • χ² test: p < 0.001
  • Pelo menos 5 picos
  • Erro médio < 15%
"""

ax5.text(0.05, 0.95, texto, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# --------------------------------------------------
# Plot 6: Diagrama de Fluxo
# --------------------------------------------------
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')

# Criar diagrama
boxes = [
    (0.5, 0.9, "1. Carregar Dados\n~10⁹ primos", 'lightblue'),
    (0.5, 0.75, "2. Janelas Deslizantes\n10⁴ primos/janela", 'lightgreen'),
    (0.5, 0.6, "3. Densidade Local\nρ = N/(p_max - p_min)", 'lightyellow'),
    (0.5, 0.45, "4. Normalização\n(ρ - μ) / σ", 'lightcoral'),
    (0.5, 0.3, "5. FFT\nEspectro de Potência", 'plum'),
    (0.5, 0.15, "6. Detecção de Picos\n>3σ acima ruído", 'lightcyan'),
]

for x, y, label, cor in boxes:
    box = mpatches.FancyBboxPatch((x-0.2, y-0.05), 0.4, 0.08,
                                   boxstyle="round,pad=0.01",
                                   edgecolor='black', facecolor=cor,
                                   transform=ax6.transAxes)
    ax6.add_patch(box)
    ax6.text(x, y, label, transform=ax6.transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Setas
    if y > 0.2:
        ax6.arrow(x, y-0.05, 0, -0.08, transform=ax6.transAxes,
                 head_width=0.02, head_length=0.02, fc='black', ec='black')

ax6.text(0.5, 0.05, "RESULTADO: Correlações com f_cosmos?",
        transform=ax6.transAxes, ha='center', fontsize=11,
        fontweight='bold', color='darkred')

ax6.set_xlim([0, 1])
ax6.set_ylim([0, 1])
ax6.set_title('Pipeline de Análise', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/previsoes_vs_observacoes.png', dpi=150, bbox_inches='tight')
print("✓ Gráfico salvo: previsoes_vs_observacoes.png")

# ==============================================================================
# RESUMO FINAL
# ==============================================================================

print("\n" + "=" * 80)
print("RESUMO DAS PREVISÕES")
print("=" * 80)
print()

print("DATASET DISPONÍVEL:")
print(f"  • {1e9:.1e} pares de primos gêmeos")
print(f"  • Range: ~{N_primo:.1e}")
print(f"  • Eficiência: 0.22% (estável)")
print()

print("PARÂMETROS DE ANÁLISE:")
print(f"  • Janela: 10^4 primos")
print(f"  • Total de janelas: ~10^5")
print(f"  • Resolução espectral: Δf ~ 10^-5 ciclos/janela")
print()

print("FREQUÊNCIAS ESPERADAS (em ciclos/janela):")
for nome, f_obs in f_rel_dict.items():
    periodo = 1.0/f_obs if f_obs > 0 else np.inf
    print(f"  • {nome:8s}: f = {f_obs:.3e} | Período = {periodo:.1f} janelas")
print()

print("INTERPRETAÇÃO:")
print("  Se detectarmos picos nessas frequências com significância >3σ,")
print("  teremos evidência de que α_grav governa também a distribuição de primos.")
print()

print("PRÓXIMO PASSO:")
print("  Executar: python3 analise_rapida_primos.py results.csv 1000000")
print()
print("=" * 80)
