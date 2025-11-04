#!/usr/bin/env python3
"""
Análise de Periodicidade em Primos Gêmeos vs f_cosmos
Investiga correlação entre densidade local de primos e frequências gravitacionais
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTES FÍSICAS
# ==============================================================================

# Constantes fundamentais (SI)
G = 6.67430e-11          # m³ kg⁻¹ s⁻²
hbar = 1.054571817e-34   # J·s
c = 2.99792458e8         # m/s

# Massa de Planck
M_Pl = np.sqrt(hbar * c / G)  # kg
print(f"Massa de Planck: {M_Pl:.6e} kg")

# Frequência de Planck
f_Pl = np.sqrt(c**5 / (hbar * G))  # Hz
print(f"Frequência de Planck: {f_Pl:.6e} Hz\n")

# Massas de partículas elementares (kg)
massas = {
    'electron': 9.1093837015e-31,
    'muon': 1.883531627e-28,
    'tau': 3.16754e-27,
    'proton': 1.67262192369e-27,
    'neutron': 1.67492749804e-27,
}

# ==============================================================================
# CÁLCULO DE f_cosmos PARA PARTÍCULAS
# ==============================================================================

def calcular_alpha_grav(m):
    """Calcula α_grav = (m/M_Pl)²"""
    return (m / M_Pl)**2

def calcular_f_cosmos(m):
    """Calcula f_cosmos = f_Pl × α_grav^(1/3)"""
    alpha = calcular_alpha_grav(m)
    return f_Pl * (alpha**(1/3))

print("=" * 80)
print("FREQUÊNCIAS f_cosmos PARA PARTÍCULAS ELEMENTARES")
print("=" * 80)

f_cosmos_particulas = {}
for nome, massa in massas.items():
    alpha = calcular_alpha_grav(massa)
    f_c = calcular_f_cosmos(massa)
    f_cosmos_particulas[nome] = f_c
    
    print(f"{nome:12s}: m={massa:.6e} kg | α_grav={alpha:.6e} | f_cosmos={f_c:.6e} Hz")

print("\n")

# ==============================================================================
# FUNÇÃO PARA ANÁLISE DE DENSIDADE LOCAL
# ==============================================================================

def analisar_densidade_local(primos, window_size=1000, step=100):
    """
    Calcula densidade de primos em janelas deslizantes
    
    Args:
        primos: array de números primos
        window_size: tamanho da janela
        step: passo do deslizamento
    
    Returns:
        posicoes: centros das janelas
        densidades: densidade em cada janela
    """
    posicoes = []
    densidades = []
    
    for i in range(0, len(primos) - window_size, step):
        janela = primos[i:i+window_size]
        centro = np.mean(janela)
        
        # Densidade = número de primos / amplitude do range
        if len(janela) > 1:
            densidade = window_size / (janela.max() - janela.min())
            posicoes.append(centro)
            densidades.append(densidade)
    
    return np.array(posicoes), np.array(densidades)

# ==============================================================================
# ANÁLISE ESPECTRAL
# ==============================================================================

def analise_espectral(posicoes, densidades, fs_effective=None):
    """
    Realiza análise espectral da densidade de primos
    
    Args:
        posicoes: array de posições
        densidades: array de densidades
        fs_effective: frequência de amostragem efetiva (opcional)
    
    Returns:
        frequencias: array de frequências
        potencias: potência espectral
        picos: índices dos picos detectados
    """
    # Normalizar e remover média
    densidade_norm = (densidades - np.mean(densidades)) / np.std(densidades)
    
    # Se não for fornecida, calcular fs a partir do espaçamento médio
    if fs_effective is None:
        dx = np.mean(np.diff(posicoes))
        fs_effective = 1.0 / dx
    
    # FFT
    N = len(densidade_norm)
    yf = fft(densidade_norm)
    xf = fftfreq(N, 1/fs_effective)
    
    # Apenas frequências positivas
    mask = xf > 0
    frequencias = xf[mask]
    potencias = np.abs(yf[mask])**2
    
    # Detectar picos significativos
    # Usar threshold = 3*sigma acima da média
    threshold = np.mean(potencias) + 3*np.std(potencias)
    picos, _ = signal.find_peaks(potencias, height=threshold, distance=10)
    
    return frequencias, potencias, picos

# ==============================================================================
# PERIODOGRAMA LOMB-SCARGLE (para dados não uniformes)
# ==============================================================================

def lomb_scargle_analysis(posicoes, densidades, frequencias_teste=None):
    """
    Análise usando Lomb-Scargle para dados não uniformemente espaçados
    """
    # Normalizar
    densidade_norm = (densidades - np.mean(densidades)) / np.std(densidades)
    
    if frequencias_teste is None:
        # Frequências de teste baseadas no Nyquist efetivo
        dx_median = np.median(np.diff(posicoes))
        freq_nyquist = 0.5 / dx_median
        frequencias_teste = np.linspace(1e-12, freq_nyquist, 10000)
    
    # Lomb-Scargle
    potencias = signal.lombscargle(posicoes, densidade_norm, frequencias_teste, normalize=True)
    
    # Detectar picos
    threshold = np.mean(potencias) + 3*np.std(potencias)
    picos, _ = signal.find_peaks(potencias, height=threshold, distance=50)
    
    return frequencias_teste, potencias, picos

# ==============================================================================
# CORRELAÇÃO COM f_cosmos
# ==============================================================================

def correlacionar_com_fcosmos(frequencias_observadas, f_cosmos_dict, tolerancia_rel=0.1):
    """
    Verifica se frequências observadas correlacionam com f_cosmos de partículas
    
    Args:
        frequencias_observadas: array de frequências detectadas
        f_cosmos_dict: dicionário {partícula: f_cosmos}
        tolerancia_rel: tolerância relativa (10% = 0.1)
    
    Returns:
        correlacoes: lista de tuplas (freq_obs, partícula, f_cosmos, erro_rel)
    """
    correlacoes = []
    
    for freq_obs in frequencias_observadas:
        for nome, f_c in f_cosmos_dict.items():
            # Verificar se estão próximas (considerando múltiplos harmônicos)
            for harmonic in [1, 2, 3, 5, 10]:
                f_test = f_c * harmonic
                erro_rel = np.abs(freq_obs - f_test) / f_test
                
                if erro_rel < tolerancia_rel:
                    correlacoes.append({
                        'freq_obs': freq_obs,
                        'particula': nome,
                        'harmonico': harmonic,
                        'f_cosmos': f_c,
                        'f_test': f_test,
                        'erro_rel': erro_rel
                    })
    
    return pd.DataFrame(correlacoes)

# ==============================================================================
# FUNÇÃO PRINCIPAL DE ANÁLISE
# ==============================================================================

def analisar_primos_vs_fcosmos(arquivo_csv, max_linhas=None, window_size=10000):
    """
    Análise completa: primos gêmeos vs f_cosmos
    """
    print("=" * 80)
    print("CARREGANDO DADOS DE PRIMOS GÊMEOS")
    print("=" * 80)
    
    # Ler CSV (assumindo formato: p, p+2, k, thread, timestamp)
    # Ajuste conforme o formato real
    try:
        if max_linhas:
            df = pd.read_csv(arquivo_csv, nrows=max_linhas, header=0)
        else:
            df = pd.read_csv(arquivo_csv, header=0)
        
        print(f"✓ Carregados {len(df)} pares de primos gêmeos")
        print(f"  Range: {df['p'].min():.6e} → {df['p'].max():.6e}")
        
        # Extrair coluna de primos
        primos = df['p'].values
        
    except Exception as e:
        print(f"✗ Erro ao carregar: {e}")
        return None
    
    # -------------------------------------------------------------------------
    # 1. DENSIDADE LOCAL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CALCULANDO DENSIDADE LOCAL")
    print("=" * 80)
    
    posicoes, densidades = analisar_densidade_local(primos, window_size=window_size, step=window_size//10)
    
    print(f"✓ {len(posicoes)} janelas analisadas")
    print(f"  Densidade média: {np.mean(densidades):.6e}")
    print(f"  Desvio padrão: {np.std(densidades):.6e}")
    print(f"  CV: {np.std(densidades)/np.mean(densidades):.4f}")
    
    # -------------------------------------------------------------------------
    # 2. ANÁLISE ESPECTRAL (FFT)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANÁLISE ESPECTRAL (FFT)")
    print("=" * 80)
    
    frequencias_fft, potencias_fft, picos_fft = analise_espectral(posicoes, densidades)
    
    print(f"✓ {len(picos_fft)} picos significativos detectados")
    if len(picos_fft) > 0:
        print("\nTop 10 frequências detectadas:")
        idx_sorted = np.argsort(potencias_fft[picos_fft])[::-1][:10]
        for i, idx in enumerate(idx_sorted, 1):
            freq = frequencias_fft[picos_fft[idx]]
            pot = potencias_fft[picos_fft[idx]]
            print(f"  {i:2d}. f = {freq:.6e} Hz | Potência = {pot:.6e}")
    
    # -------------------------------------------------------------------------
    # 3. LOMB-SCARGLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANÁLISE LOMB-SCARGLE")
    print("=" * 80)
    
    frequencias_ls, potencias_ls, picos_ls = lomb_scargle_analysis(posicoes, densidades)
    
    print(f"✓ {len(picos_ls)} picos significativos detectados")
    if len(picos_ls) > 0:
        print("\nTop 10 frequências detectadas:")
        idx_sorted = np.argsort(potencias_ls[picos_ls])[::-1][:10]
        for i, idx in enumerate(idx_sorted, 1):
            freq = frequencias_ls[picos_ls[idx]]
            pot = potencias_ls[picos_ls[idx]]
            print(f"  {i:2d}. f = {freq:.6e} Hz | Potência = {pot:.6e}")
    
    # -------------------------------------------------------------------------
    # 4. CORRELAÇÃO COM f_cosmos
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CORRELAÇÃO COM f_cosmos DE PARTÍCULAS")
    print("=" * 80)
    
    # Pegar frequências dos picos mais significativos
    if len(picos_ls) > 0:
        freqs_obs = frequencias_ls[picos_ls]
        correlacoes = correlacionar_com_fcosmos(freqs_obs, f_cosmos_particulas, tolerancia_rel=0.2)
        
        if len(correlacoes) > 0:
            print(f"✓ {len(correlacoes)} correlações encontradas:")
            print(correlacoes.to_string(index=False))
        else:
            print("✗ Nenhuma correlação significativa encontrada")
    
    # -------------------------------------------------------------------------
    # 5. VISUALIZAÇÃO
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: Densidade local
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(posicoes, densidades, 'b-', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Posição (valor do primo)')
    ax1.set_ylabel('Densidade local')
    ax1.set_title('Densidade Local de Primos Gêmeos')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Histograma de densidade
    ax2 = plt.subplot(3, 2, 2)
    ax2.hist(densidades, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Densidade')
    ax2.set_ylabel('Frequência')
    ax2.set_title('Distribuição de Densidade')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Espectro FFT
    ax3 = plt.subplot(3, 2, 3)
    ax3.semilogy(frequencias_fft, potencias_fft, 'b-', alpha=0.7, linewidth=0.5)
    if len(picos_fft) > 0:
        ax3.plot(frequencias_fft[picos_fft], potencias_fft[picos_fft], 'ro', markersize=8, label='Picos')
    ax3.set_xlabel('Frequência (Hz)')
    ax3.set_ylabel('Potência espectral')
    ax3.set_title('Análise Espectral (FFT)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Lomb-Scargle
    ax4 = plt.subplot(3, 2, 4)
    ax4.semilogy(frequencias_ls, potencias_ls, 'g-', alpha=0.7, linewidth=0.5)
    if len(picos_ls) > 0:
        ax4.plot(frequencias_ls[picos_ls], potencias_ls[picos_ls], 'ro', markersize=8, label='Picos')
    ax4.set_xlabel('Frequência (Hz)')
    ax4.set_ylabel('Potência normalizada')
    ax4.set_title('Periodograma Lomb-Scargle')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Comparação com f_cosmos
    ax5 = plt.subplot(3, 2, 5)
    if len(picos_ls) > 0:
        freqs_obs = frequencias_ls[picos_ls]
        pots_obs = potencias_ls[picos_ls]
        
        # Plotar frequências observadas
        for freq, pot in zip(freqs_obs[:20], pots_obs[:20]):
            ax5.axvline(freq, color='blue', alpha=0.3, linewidth=1)
        
        # Plotar f_cosmos teóricos
        colors = plt.cm.tab10(np.linspace(0, 1, len(f_cosmos_particulas)))
        for (nome, f_c), color in zip(f_cosmos_particulas.items(), colors):
            for harm in [1, 2, 3]:
                ax5.axvline(f_c * harm, color=color, linestyle='--', linewidth=2, 
                           label=f'{nome} (h={harm})' if harm == 1 else '', alpha=0.7)
    
    ax5.set_xlabel('Frequência (Hz)')
    ax5.set_ylabel('Intensidade')
    ax5.set_title('Comparação: Observado vs f_cosmos Teórico')
    ax5.legend(fontsize=8)
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Autocorrelação de densidade
    ax6 = plt.subplot(3, 2, 6)
    autocorr = np.correlate(densidades - np.mean(densidades), 
                            densidades - np.mean(densidades), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalizar
    
    lags = np.arange(len(autocorr))
    ax6.plot(lags[:len(lags)//4], autocorr[:len(lags)//4], 'b-', linewidth=1)
    ax6.set_xlabel('Lag')
    ax6.set_ylabel('Autocorrelação')
    ax6.set_title('Autocorrelação de Densidade')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analise_periodicidade_fcosmos.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gráfico salvo: analise_periodicidade_fcosmos.png")
    
    # Retornar resultados
    return {
        'posicoes': posicoes,
        'densidades': densidades,
        'frequencias_fft': frequencias_fft,
        'potencias_fft': potencias_fft,
        'picos_fft': picos_fft,
        'frequencias_ls': frequencias_ls,
        'potencias_ls': potencias_ls,
        'picos_ls': picos_ls
    }

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("ANÁLISE DE PERIODICIDADE: PRIMOS GÊMEOS vs f_cosmos")
    print("=" * 80)
    print()
    
    # Caminho do arquivo
    if len(sys.argv) > 1:
        arquivo = sys.argv[1]
    else:
        arquivo = "/home/thlinux/relacionalidadegeral/codigo/binario/results.csv"
        print(f"Usando arquivo padrão: {arquivo}")
    
    # Número máximo de linhas (para teste rápido)
    max_linhas = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if max_linhas:
        print(f"⚠ Modo teste: analisando apenas {max_linhas} linhas\n")
    
    # Executar análise
    resultados = analisar_primos_vs_fcosmos(arquivo, max_linhas=max_linhas, window_size=10000)
    
    if resultados:
        print("\n" + "=" * 80)
        print("✓ ANÁLISE CONCLUÍDA")
        print("=" * 80)
