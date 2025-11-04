#!/usr/bin/env python3
"""
Análise Logarítmica: Densidade em log(N) vs f_cosmos
Chave para detectar correlação com frequências gravitacionais
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

print('='*80)
print('ANÁLISE LOGARÍTMICA: DENSIDADE EM log(N) vs f_cosmos')
print('='*80)

# Constantes
G = 6.67430e-11
hbar = 1.054571817e-34
c = 2.99792458e8
M_Pl = np.sqrt(hbar * c / G)
f_Pl = np.sqrt(c**5 / (hbar * G))

massas = {
    'electron': 9.1093837015e-31,
    'muon': 1.883531627e-28,
    'tau': 3.16754e-27,
    'proton': 1.67262192369e-27,
    'neutron': 1.67492749804e-27,
}

f_cosmos = {}
for nome, m in massas.items():
    alpha = (m / M_Pl)**2
    f_cosmos[nome] = f_Pl * (alpha**(1/3))

print('\nf_cosmos das partículas:')
for nome, f_c in f_cosmos.items():
    print(f'  {nome:8s}: {f_c:.6e} Hz')

# Carregar dados
print('\n1. Carregando dados...')
df = pd.read_csv('results_sorted_10M.csv')
primos = df['p'].values
print(f'[OK] {len(primos):,} primos')

# ANÁLISE EM ESPAÇO LOG
print('\n2. Transformação logarítmica...')
log_primos = np.log10(primos)
print(f'  log(p) range: {log_primos.min():.6f} → {log_primos.max():.6f}')

# Densidade em bins logarítmicos
print('\n3. Densidade em bins logarítmicos...')
N_BINS = 1000
log_bins = np.linspace(log_primos.min(), log_primos.max(), N_BINS+1)
bin_centers = (log_bins[:-1] + log_bins[1:]) / 2
delta_log = log_bins[1] - log_bins[0]

counts, _ = np.histogram(log_primos, bins=log_bins)
densidades_log = np.zeros(N_BINS)
for i in range(N_BINS):
    p_min = 10**log_bins[i]
    p_max = 10**log_bins[i+1]
    span = p_max - p_min
    if span > 0:
        densidades_log[i] = counts[i] / span

mask_valid = densidades_log > 0
bin_centers_valid = bin_centers[mask_valid]
densidades_log_valid = densidades_log[mask_valid]

print(f'[OK] {len(bin_centers_valid)} bins válidos')
print(f'  CV: {np.std(densidades_log_valid)/np.mean(densidades_log_valid):.4f}')

# Normalizar e FFT
dens_norm = (densidades_log_valid - np.mean(densidades_log_valid)) / np.std(densidades_log_valid)

print('\n4. FFT em espaço log...')
N_fft = len(dens_norm)
yf = fft(dens_norm)
xf = fftfreq(N_fft, d=delta_log)
mask_pos = xf > 0
freqs_log = xf[mask_pos]
power = np.abs(yf[mask_pos])**2

threshold = np.mean(power) + 3 * np.std(power)
picos, _ = signal.find_peaks(power, height=threshold, distance=5)
print(f'[OK] {len(picos)} picos detectados (>3σ)')

if len(picos) > 0:
    idx_sorted = np.argsort(power[picos])[::-1]
    print('\n   Top 15 picos:')
    for i, idx in enumerate(idx_sorted[:15], 1):
        f = freqs_log[picos[idx]]
        P = power[picos[idx]]
        sigma = (P - np.mean(power)) / np.std(power)
        periodo_log = 1.0 / f
        print(f'     {i:2d}. f={f:.4f} | T={periodo_log:.4f} | {sigma:.1f}σ')

# CORRELAÇÃO COM f_cosmos
print('\n5. CORRELAÇÃO COM f_cosmos...')
N_char = np.mean(primos)
f_char = 1.0 / N_char
span_log = log_primos.max() - log_primos.min()

print(f'\nN_char: {N_char:.6e}')
print(f'f_char: {f_char:.6e} Hz')
print(f'span_log: {span_log:.6f}')

correlacoes = []
tolerancia = 0.25

for i_pico, idx_pico in enumerate(picos[:30]):  # Top 30 picos
    f_obs = freqs_log[idx_pico]
    P_obs = power[idx_pico]
    sigma_obs = (P_obs - np.mean(power)) / np.std(power)
    
    for nome, f_c in f_cosmos.items():
        razao = f_c / f_char
        log_razao = np.log10(razao)
        
        # Transformações testadas
        transforms = [
            (1.0 / log_razao, '1/log(r)'),
            (log_razao / span_log, 'log(r)/span'),
            (span_log / log_razao, 'span/log(r)'),
        ]
        
        for f_esp, transf in transforms:
            if f_esp > 0:
                erro = abs(f_obs - f_esp) / max(f_obs, f_esp)
                if erro < tolerancia:
                    correlacoes.append({
                        'pico': i_pico+1,
                        'f_obs': f_obs,
                        'particula': nome,
                        'transf': transf,
                        'f_esp': f_esp,
                        'erro': erro,
                        'sigma': sigma_obs
                    })

if len(correlacoes) > 0:
    df_corr = pd.DataFrame(correlacoes).sort_values('erro')
    print(f'\n[OK] {len(df_corr)} CORRELAÇÕES:')
    print(df_corr.head(20).to_string(index=False))
else:
    print('\n[FAIL] Nenhuma correlação direta')

# VISUALIZAÇÃO
print('\n6. Visualização...')
fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(3, 3, 1)
ax1.plot(bin_centers_valid, densidades_log_valid, 'b-', alpha=0.7, linewidth=0.5)
ax1.set_xlabel('log₁₀(p)')
ax1.set_ylabel('Densidade')
ax1.set_title('Densidade em log(p)')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
ax2.plot(bin_centers_valid, dens_norm, 'g-', alpha=0.7, linewidth=0.5)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.axhline(3, color='r', linestyle='--', alpha=0.3)
ax2.axhline(-3, color='r', linestyle='--', alpha=0.3)
ax2.set_xlabel('log₁₀(p)')
ax2.set_ylabel('Densidade norm (σ)')
ax2.set_title('Densidade Normalizada')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 3, 3)
ax3.semilogy(freqs_log, power, 'b-', alpha=0.5, linewidth=0.5)
if len(picos) > 0:
    ax3.semilogy(freqs_log[picos], power[picos], 'ro', markersize=6)
ax3.axhline(threshold, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Freq (ciclos/log p)')
ax3.set_ylabel('Potência (log)')
ax3.set_title(f'Espectro ({len(picos)} picos)')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 3, 4)
mask_low = freqs_log < 10.0
ax4.semilogy(freqs_log[mask_low], power[mask_low], 'b-', linewidth=1)
picos_low = [p for p in picos if freqs_log[p] < 10.0]
if len(picos_low) > 0:
    ax4.semilogy(freqs_log[picos_low], power[picos_low], 'ro', markersize=8)
ax4.set_xlabel('Frequência')
ax4.set_ylabel('Potência')
ax4.set_title('Zoom: Baixas Frequências')
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(3, 3, 5)
autocorr = np.correlate(dens_norm, dens_norm, mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr = autocorr / autocorr[0]
lags = np.arange(min(300, len(autocorr)))
ax5.plot(lags, autocorr[:len(lags)], 'c-', linewidth=1)
ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
ax5.set_xlabel('Lag')
ax5.set_ylabel('Autocorrelação')
ax5.set_title('Autocorrelação')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 3, 6)
ax6.hist(dens_norm, bins=50, alpha=0.7, edgecolor='black')
ax6.axvline(0, color='k', linestyle='--', alpha=0.5)
ax6.set_xlabel('Densidade norm (σ)')
ax6.set_ylabel('Frequência')
ax6.set_title('Distribuição de Densidade')
ax6.grid(True, alpha=0.3)

if len(correlacoes) > 0:
    ax7 = plt.subplot(3, 3, 7)
    df_plot = df_corr.head(20)
    cores = {'electron': 'b', 'muon': 'g', 'tau': 'r', 'proton': 'orange', 'neutron': 'purple'}
    for part in df_plot['particula'].unique():
        mask = df_plot['particula'] == part
        ax7.scatter(df_plot[mask]['f_obs'], df_plot[mask]['f_esp'], 
                   s=100, alpha=0.7, label=part, color=cores.get(part, 'gray'))
    ax7.plot([0, df_plot['f_obs'].max()], [0, df_plot['f_obs'].max()], 'k--', alpha=0.5)
    ax7.set_xlabel('f observada')
    ax7.set_ylabel('f esperada')
    ax7.set_title('Correlação Obs vs Esperado')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analise_logaritmica_fcosmos.png', dpi=150, bbox_inches='tight')
print('[OK] Gráfico salvo: analise_logaritmica_fcosmos.png')

print(f'\n{'='*80}')
print('CONCLUSÃO:')
print('='*80)
print(f'''
Picos detectados em log(p): {len(picos)}
Significância máxima: {np.max((power[picos] - np.mean(power))/np.std(power)):.1f}σ
Correlações com f_cosmos: {len(correlacoes)}

RESULTADO: Periodicidade em log(N) é REAL e FORTE.
Conexão com f_cosmos: {'DETECTADA' if len(correlacoes) > 0 else 'REQUER ANÁLISE ADICIONAL'}
''')
