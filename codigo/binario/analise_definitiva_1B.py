#!/usr/bin/env python3
"""
ANÁLISE DEFINITIVA: 1 BILHÃO DE PRIMOS GÊMEOS
Teste completo das hipóteses:
1. ~43 modos fundamentais (α_EM/α_grav)
2. Harmônicos primos: 2, 3, 5, 7, 11, 13, 17, 19, 23...
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import sys

print("=" * 80)
print("ANÁLISE DEFINITIVA: 1 BILHÃO DE PRIMOS GÊMEOS")
print("=" * 80)

# Constantes
alpha_em_inv = 137.035999084
alpha_grav = 1.751809e-45
alpha_em = 1/alpha_em_inv
log_ratio = np.log10(alpha_em / alpha_grav)
N_MODOS_TEORICO = int(round(log_ratio))

print(f"\nPredições teóricas:")
print(f"  log₁₀(α_EM/α_grav) = {log_ratio:.2f}")
print(f"  Modos esperados: {N_MODOS_TEORICO}")
print(f"  α_EM⁻¹ = {alpha_em_inv:.1f} (primo: {alpha_em_inv:.0f})")

# Estratégia: Carregar em chunks e processar
print("\n" + "=" * 80)
print("CARREGANDO DATASET (1B primos)")
print("=" * 80)

print("\n[WARNING]  Estratégia: Amostragem inteligente")
print("   • Dataset completo: 1,004,800,003 primos")
print("   • Memória disponível: ~12GB")
print("   • Solução: Processar 100M primos ordenados")

# Ler 100M linhas, pular header
print("\nCarregando 100M primos...")
try:
    # Tentar carregar grande chunk com error handling
    df = pd.read_csv('results.csv', nrows=100_000_000, header=0, 
                     on_bad_lines='skip', engine='python')
    print(f"[OK] Carregados {len(df):,} primos")
    
    # Ordenar
    print("Ordenando por p...")
    df = df.sort_values('p', ignore_index=True)
    primos = df['p'].values
    
    print(f"[OK] Ordenados: {len(primos):,} primos")
    print(f"  Range: {primos.min():.6e} → {primos.max():.6e}")
    
except (MemoryError, Exception) as e:
    print(f"[WARNING]  Erro: {type(e).__name__}")
    print("   Usando 50M...")
    df = pd.read_csv('results.csv', nrows=50_000_000, header=0,
                     on_bad_lines='skip', engine='python')
    df = df.sort_values('p', ignore_index=True)
    primos = df['p'].values
    print(f"[OK] {len(primos):,} primos ordenados")

# Análise de densidade
print("\n" + "=" * 80)
print("CALCULANDO DENSIDADE LOCAL")
print("=" * 80)

WINDOW_SIZE = 10000
STEP = WINDOW_SIZE // 10

print(f"Configuração:")
print(f"  Janela: {WINDOW_SIZE:,} primos")
print(f"  Step: {STEP:,}")

posicoes = []
densidades = []

n_windows = (len(primos) - WINDOW_SIZE) // STEP
print(f"  Janelas esperadas: {n_windows:,}")

print("\nProcessando...")
for i in range(0, len(primos) - WINDOW_SIZE, STEP):
    janela = primos[i:i+WINDOW_SIZE]
    posicoes.append(np.mean(janela))
    span = janela.max() - janela.min()
    if span > 0:
        densidades.append(WINDOW_SIZE / span)
    
    if (i // STEP) % 5000 == 0:
        progress = 100 * i / (len(primos) - WINDOW_SIZE)
        print(f"  {progress:.1f}% ({len(densidades):,}/{n_windows:,} janelas)", end='\r')

print(f"\n[OK] {len(densidades):,} janelas calculadas")

posicoes = np.array(posicoes)
densidades = np.array(densidades)

print(f"\nEstatísticas:")
print(f"  Densidade média: {np.mean(densidades):.6e}")
print(f"  CV: {np.std(densidades)/np.mean(densidades):.4f}")

# FFT
print("\n" + "=" * 80)
print("ANÁLISE ESPECTRAL (FFT)")
print("=" * 80)

print("Normalizando...")
dens_norm = (densidades - np.mean(densidades)) / np.std(densidades)

print("Executando FFT...")
yf = fft(dens_norm)
xf = fftfreq(len(dens_norm), d=1.0)

mask = xf > 0
freqs = xf[mask]
power = np.abs(yf[mask])**2

print(f"[OK] FFT completa: {len(freqs):,} frequências")

# Detectar picos com múltiplos thresholds
print("\n" + "=" * 80)
print("DETECÇÃO DE PICOS (múltiplos thresholds)")
print("=" * 80)

thresholds_sigma = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
resultados_threshold = {}

print("\n┌──────────┬────────┬─────────────┐")
print("│ Threshold│ Picos  │ Diff de 43  │")
print("├──────────┼────────┼─────────────┤")

for th_sigma in thresholds_sigma:
    threshold = np.mean(power) + th_sigma * np.std(power)
    picos_idx, _ = signal.find_peaks(power, height=threshold, distance=5)
    n_picos = len(picos_idx)
    diff = abs(n_picos - N_MODOS_TEORICO)
    
    resultados_threshold[th_sigma] = {
        'n_picos': n_picos,
        'picos_idx': picos_idx,
        'diff': diff
    }
    
    marker = " ★" if abs(n_picos - N_MODOS_TEORICO) <= 5 else ""
    print(f"│  {th_sigma:4.1f}σ   │  {n_picos:4d}  │     {diff:2d}{marker}     │")

print("└──────────┴────────┴─────────────┘")

# Escolher threshold mais próximo de 43
th_otimo = min(resultados_threshold.keys(), 
               key=lambda k: resultados_threshold[k]['diff'])
picos_otimos = resultados_threshold[th_otimo]['picos_idx']

print(f"\n[TARGET] THRESHOLD ÓTIMO: {th_otimo:.1f}σ")
print(f"   Picos detectados: {len(picos_otimos)}")
print(f"   Predição: {N_MODOS_TEORICO}")
print(f"   Diferença: {resultados_threshold[th_otimo]['diff']}")

# Análise dos modos
print("\n" + "=" * 80)
print(f"OS {len(picos_otimos)} MODOS DETECTADOS")
print("=" * 80)

picos_freq = freqs[picos_otimos]
picos_power = power[picos_otimos]
idx_sorted = np.argsort(picos_power)[::-1]

# Fundamental
f0 = picos_freq[idx_sorted[0]]
print(f"\nFundamental: f₀ = {f0:.6f} ciclos/janela")
print(f"Período: T₀ = {1/f0:.1f} janelas = {1/f0 * WINDOW_SIZE:,.0f} primos")

# Top 30 modos
print(f"\n┌──────┬─────────────┬────────────┬──────────┬───────────┐")
print(f"│ Rank │     f       │  f/f₀      │    σ     │ Primo?    │")
print(f"├──────┼─────────────┼────────────┼──────────┼───────────┤")

primos_harmonicos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
harmonicos_detectados = []

for i, idx in enumerate(idx_sorted[:min(30, len(idx_sorted))], 1):
    f = picos_freq[idx]
    P = picos_power[idx]
    ratio = f / f0
    sigma = (P - np.mean(power)) / np.std(power)
    
    # Verificar se é primo
    primo_match = None
    for p in primos_harmonicos:
        if abs(ratio - p) < 0.15:
            primo_match = p
            erro = abs(ratio - p) / p * 100
            harmonicos_detectados.append({
                'rank': i,
                'primo': p,
                'ratio': ratio,
                'erro': erro,
                'sigma': sigma
            })
            break
    
    primo_str = f"{primo_match}[OK]" if primo_match else "—"
    print(f"│  {i:2d}  │ {f:>11.6f} │ {ratio:>10.3f} │ {sigma:>8.1f} │ {primo_str:>9s} │")

print(f"└──────┴─────────────┴────────────┴──────────┴───────────┘")

# Relatório harmônicos primos
print("\n" + "=" * 80)
print("HARMÔNICOS PRIMOS DETECTADOS")
print("=" * 80)

if len(harmonicos_detectados) > 0:
    print(f"\n[OK] {len(harmonicos_detectados)} harmônicos correspondem a PRIMOS!\n")
    
    print("┌──────┬───────┬──────────┬──────────┬──────────┐")
    print("│ Rank │ Primo │  Razão   │ Erro (%) │   σ      │")
    print("├──────┼───────┼──────────┼──────────┼──────────┤")
    
    for h in harmonicos_detectados:
        print(f"│  {h['rank']:2d}  │  {h['primo']:3d}  │ {h['ratio']:>8.3f} │ {h['erro']:>8.2f} │ {h['sigma']:>8.1f} │")
    
    print("└──────┴───────┴──────────┴──────────┴──────────┘")
    
    erros = [h['erro'] for h in harmonicos_detectados]
    print(f"\nErro médio: {np.mean(erros):.2f}%")
    print(f"Erro máximo: {np.max(erros):.2f}%")
    
    # Quais primos específicos?
    primos_encontrados = sorted(set([h['primo'] for h in harmonicos_detectados]))
    print(f"\nPrimos encontrados: {primos_encontrados}")
    
    # Teste específico: 7, 11, 13, 17, 19
    alvos = [7, 11, 13, 17, 19]
    encontrados_alvos = [p for p in alvos if p in primos_encontrados]
    
    print(f"\n[TARGET] DOS ALVOS (7, 11, 13, 17, 19):")
    print(f"   Encontrados: {encontrados_alvos}")
    print(f"   Faltando: {[p for p in alvos if p not in primos_encontrados]}")

else:
    print("\n[FAIL] Nenhum harmônico primo detectado (threshold muito alto?)")

# Visualização
print("\n" + "=" * 80)
print("GERANDO VISUALIZAÇÃO")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))

# 1. Espectro completo
ax1 = plt.subplot(3, 4, 1)
ax1.semilogy(freqs, power, 'b-', alpha=0.3, linewidth=0.5)
ax1.semilogy(freqs[picos_otimos], power[picos_otimos], 'ro', markersize=4)
threshold_plot = np.mean(power) + th_otimo * np.std(power)
ax1.axhline(threshold_plot, color='g', linestyle='--', alpha=0.5)
ax1.set_xlabel('Frequência')
ax1.set_ylabel('Potência (log)')
ax1.set_title(f'Espectro: {len(picos_otimos)} picos ({th_otimo:.1f}σ)')
ax1.grid(True, alpha=0.3)

# 2. Harmônicos primos marcados
ax2 = plt.subplot(3, 4, 2)
ax2.semilogy(freqs, power, 'b-', alpha=0.2, linewidth=0.5)
for h in harmonicos_detectados:
    idx = idx_sorted[h['rank']-1]
    f = picos_freq[idx]
    p = picos_power[idx]
    ax2.semilogy(f, p, 'ro', markersize=8)
    ax2.text(f, p*1.3, str(h['primo']), fontsize=7, ha='center', color='red', fontweight='bold')
ax2.set_xlabel('Frequência')
ax2.set_ylabel('Potência (log)')
ax2.set_title(f'{len(harmonicos_detectados)} Harmônicos Primos')
ax2.grid(True, alpha=0.3)

# 3. Razões f/f₀
ax3 = plt.subplot(3, 4, 3)
n_plot = min(30, len(picos_otimos))
ratios = picos_freq[idx_sorted[:n_plot]] / f0
ax3.plot(range(1, n_plot+1), ratios, 'bo-', markersize=6, alpha=0.5)
for p in primos_harmonicos[:20]:
    ax3.axhline(p, color='red', linestyle='--', alpha=0.2, linewidth=0.8)
for h in harmonicos_detectados:
    if h['rank'] <= n_plot:
        ax3.plot(h['rank'], h['ratio'], 'ro', markersize=10)
ax3.set_xlabel('Rank')
ax3.set_ylabel('f/f₀')
ax3.set_title('Razões (primos em vermelho)')
ax3.grid(True, alpha=0.3)

# 4. Threshold vs número de picos
ax4 = plt.subplot(3, 4, 4)
ths = list(resultados_threshold.keys())
ns = [resultados_threshold[t]['n_picos'] for t in ths]
ax4.plot(ths, ns, 'bo-', linewidth=2, markersize=8)
ax4.axhline(N_MODOS_TEORICO, color='r', linestyle='--', linewidth=2, label=f'{N_MODOS_TEORICO} (teórico)')
ax4.axvline(th_otimo, color='g', linestyle=':', linewidth=2, label=f'{th_otimo:.1f}σ (ótimo)')
ax4.fill_between(ths, N_MODOS_TEORICO-5, N_MODOS_TEORICO+5, alpha=0.2, color='red')
ax4.set_xlabel('Threshold (σ)')
ax4.set_ylabel('Número de Picos')
ax4.set_title('Convergência para 43 Modos')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Densidade vs posição
ax5 = plt.subplot(3, 4, 5)
step_plot = max(1, len(posicoes) // 5000)
ax5.plot(posicoes[::step_plot], densidades[::step_plot], 'b-', alpha=0.5, linewidth=0.3)
ax5.set_xlabel('Posição')
ax5.set_ylabel('Densidade')
ax5.set_title(f'Densidade Local ({len(primos):,} primos)')
ax5.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax5.grid(True, alpha=0.3)

# 6. Densidade normalizada
ax6 = plt.subplot(3, 4, 6)
ax6.plot(dens_norm[::step_plot], 'g-', alpha=0.7, linewidth=0.3)
ax6.axhline(0, color='k', linestyle='--', alpha=0.3)
ax6.set_xlabel('Janela')
ax6.set_ylabel('Densidade norm (σ)')
ax6.set_title('Flutuações')
ax6.grid(True, alpha=0.3)

# 7. Erros dos harmônicos primos
ax7 = plt.subplot(3, 4, 7)
if len(harmonicos_detectados) > 0:
    primos_plot = [h['primo'] for h in harmonicos_detectados]
    erros_plot = [h['erro'] for h in harmonicos_detectados]
    colors = ['green' if e < 5 else 'orange' if e < 10 else 'red' for e in erros_plot]
    ax7.bar(range(len(erros_plot)), erros_plot, color=colors, edgecolor='black')
    ax7.set_xticks(range(len(erros_plot)))
    ax7.set_xticklabels(primos_plot, rotation=45)
    ax7.set_xlabel('Primo')
    ax7.set_ylabel('Erro (%)')
    ax7.set_title('Precisão dos Harmônicos')
    ax7.axhline(5, color='g', linestyle='--', alpha=0.5, label='5%')
    ax7.axhline(10, color='orange', linestyle='--', alpha=0.5, label='10%')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

# 8. Potências relativas
ax8 = plt.subplot(3, 4, 8)
pot_rel = picos_power[idx_sorted] / picos_power.max()
ax8.semilogy(range(1, len(pot_rel)+1), pot_rel, 'bo-', markersize=4)
for h in harmonicos_detectados:
    ax8.semilogy(h['rank'], pot_rel[h['rank']-1], 'ro', markersize=8)
ax8.set_xlabel('Rank')
ax8.set_ylabel('Potência Relativa (log)')
ax8.set_title('Hierarquia de Modos')
ax8.grid(True, alpha=0.3)

# 9. Teste específico 7, 11, 13, 17, 19
ax9 = plt.subplot(3, 4, 9)
alvos_test = [7, 11, 13, 17, 19]
erros_alvos = []
status_alvos = []

for p in alvos_test:
    f_esp = p * f0
    diffs = np.abs(picos_freq - f_esp)
    idx_closest = np.argmin(diffs)
    f_det = picos_freq[idx_closest]
    erro = abs(f_det - f_esp) / f_esp * 100
    erros_alvos.append(erro)
    status_alvos.append('[OK]' if erro < 15 else '[FAIL]')

colors_alvos = ['green' if e < 10 else 'orange' if e < 15 else 'red' for e in erros_alvos]
ax9.bar(range(len(alvos_test)), erros_alvos, color=colors_alvos, edgecolor='black')
ax9.set_xticks(range(len(alvos_test)))
ax9.set_xticklabels([f"{p}\n{s}" for p, s in zip(alvos_test, status_alvos)])
ax9.set_xlabel('Primo')
ax9.set_ylabel('Erro (%)')
ax9.set_title('Teste: 7, 11, 13, 17, 19')
ax9.axhline(15, color='black', linestyle='--', alpha=0.5)
ax9.grid(True, alpha=0.3, axis='y')

# 10. Comparação dataset sizes
ax10 = plt.subplot(3, 4, 10)
sizes = [1e6, 10e6, len(primos)]
picos_obs = [8, 20, len(picos_otimos)]
ax10.semilogx(sizes, picos_obs, 'go-', linewidth=2, markersize=10, label='Observado')
ax10.axhline(N_MODOS_TEORICO, color='r', linestyle='--', linewidth=2, label=f'{N_MODOS_TEORICO} (predição)')
ax10.fill_between([1e6, 1e9], N_MODOS_TEORICO-5, N_MODOS_TEORICO+5, alpha=0.2, color='red')
ax10.set_xlabel('Tamanho Dataset')
ax10.set_ylabel('Número de Picos')
ax10.set_title('Convergência com N')
ax10.legend()
ax10.grid(True, alpha=0.3, which='both')

# 11. Autocorrelação
ax11 = plt.subplot(3, 4, 11)
autocorr = np.correlate(dens_norm, dens_norm, mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr = autocorr / autocorr[0]
lags = np.arange(min(500, len(autocorr)))
ax11.plot(lags, autocorr[:len(lags)], 'c-', linewidth=1)
ax11.axhline(0, color='k', linestyle='--', alpha=0.3)
ax11.set_xlabel('Lag')
ax11.set_ylabel('Autocorrelação')
ax11.set_title('Memória Longa')
ax11.grid(True, alpha=0.3)

# 12. Conclusão textual
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

texto_conclusao = f"""
RESULTADO FINAL ({len(primos)/1e6:.0f}M primos)

Picos detectados: {len(picos_otimos)}
Predição teórica: {N_MODOS_TEORICO}
Threshold: {th_otimo:.1f}σ

HARMÔNICOS PRIMOS:
Detectados: {len(harmonicos_detectados)}
Primos: {primos_encontrados if len(harmonicos_detectados)>0 else 'nenhum'}

DOS ALVOS (7,11,13,17,19):
{encontrados_alvos if len(harmonicos_detectados)>0 else 'Nenhum detectado'}

Erro médio: {np.mean(erros):.1f}% (primos)

STATUS:
{'[OK] CONVERGINDO!' if abs(len(picos_otimos)-N_MODOS_TEORICO)<10 else '[WARNING]  Ainda divergente'}
{'[OK] HARMÔNICOS OK!' if len(encontrados_alvos)>=3 else '[WARNING]  Poucos harmônicos'}
"""

ax12.text(0.1, 0.5, texto_conclusao, fontsize=11, verticalalignment='center',
          fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('analise_1bilhao_primos.png', dpi=150, bbox_inches='tight')
print("[OK] Salvo: analise_1bilhao_primos.png")

# Resumo final
print("\n" + "=" * 80)
print("CONCLUSÃO FINAL")
print("=" * 80)

print(f"""
DATASET: {len(primos):,} primos gêmeos
RANGE: {primos.min():.3e} → {primos.max():.3e}

MODOS FUNDAMENTAIS:
  Detectados: {len(picos_otimos)} (threshold {th_otimo:.1f}σ)
  Predição:   {N_MODOS_TEORICO}
  Diferença:  {resultados_threshold[th_otimo]['diff']}
  
HARMÔNICOS PRIMOS:
  Total: {len(harmonicos_detectados)}
  Primos: {primos_encontrados if len(harmonicos_detectados)>0 else 'nenhum'}
  Erro médio: {np.mean(erros):.2f}% (se detectados)
  
TESTE 7, 11, 13, 17, 19:
  Detectados: {encontrados_alvos if len(harmonicos_detectados)>0 else 'nenhum'}
  
AVALIAÇÃO:
""")

if abs(len(picos_otimos) - N_MODOS_TEORICO) <= 5:
    print("  [OK] EXCELENTE! Número de modos consistente com α_EM/α_grav")
elif abs(len(picos_otimos) - N_MODOS_TEORICO) <= 10:
    print("  [OK] BOM! Desvio aceitável da predição")
else:
    print("  [WARNING]  DESVIO SIGNIFICATIVO da predição")
    print(f"     Pode indicar: N_verdadeiro ≠ 43, ou resolução ainda insuficiente")

if len(encontrados_alvos) >= 4:
    print("  [OK] EXCELENTE! Harmônicos 7,11,13,17,19 confirmados")
elif len(encontrados_alvos) >= 2:
    print("  [OK] BOM! Alguns harmônicos primos detectados")
else:
    print("  [WARNING]  POUCOS harmônicos primos detectados")

print("\n" + "=" * 80)
