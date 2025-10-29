"""
Processo de Ornstein-Uhlenbeck Modificado
========================================

Este módulo implementa o processo de Ornstein-Uhlenbeck com correções
gravitacionais, incluindo simulações numéricas e análise estatística.

Autor: Equipe de Pesquisa
Data: Outubro 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional, Callable
import warnings

# Importar constantes do módulo local
from constantes import *

class OrnsteinUhlenbeckModificado:
    """
    Classe para simulação e análise do processo OU modificado
    """
    
    def __init__(self, gamma: float, D: float, alpha_grav_factor: float = 1.0):
        """
        Inicializa o processo OU modificado
        
        Parameters:
        -----------
        gamma : float
            Coeficiente de atrito/relaxação
        D : float
            Coeficiente de difusão
        alpha_grav_factor : float
            Fator multiplicativo para α_grav (para estudos paramétricos)
        """
        self.gamma = gamma
        self.D = D
        self.alpha_grav = alpha_grav * alpha_grav_factor
        
        # Parâmetros derivados
        self.tau_correlation = 1.0 / gamma  # Tempo de correlação
        self.variance_equilibrium = D / gamma  # Variância no equilíbrio
        
    def forca_gravitacional(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Força gravitacional modificada (exemplo)
        
        Parameters:
        -----------
        x : np.ndarray
            Posição
        t : float
            Tempo
            
        Returns:
        --------
        np.ndarray
            Força gravitacional
        """
        # Exemplo: força oscilatória com frequência cósmica
        f_grav = self.alpha_grav * f_cosmos * np.sin(2 * np.pi * f_cosmos * t)
        return f_grav * np.ones_like(x)
    
    def simular_trajetoria(self, 
                          t_final: float, 
                          dt: float, 
                          x0: float = 0.0,
                          incluir_gravidade: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula uma trajetória do processo OU modificado
        
        Parameters:
        -----------
        t_final : float
            Tempo final da simulação
        dt : float
            Passo de tempo
        x0 : float
            Posição inicial
        incluir_gravidade : bool
            Se True, inclui correções gravitacionais
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Tempos e posições
        """
        N_steps = int(t_final / dt)
        t = np.linspace(0, t_final, N_steps)
        x = np.zeros(N_steps)
        x[0] = x0
        
        # Ruído branco gaussiano
        noise = np.random.normal(0, np.sqrt(2 * self.D * dt), N_steps-1)
        
        for i in range(N_steps - 1):
            # Termo determinístico padrão
            drift = -self.gamma * x[i]
            
            # Correção gravitacional
            if incluir_gravidade:
                f_grav = self.forca_gravitacional(x[i], t[i])
                drift += f_grav
            
            # Evolução temporal
            x[i+1] = x[i] + drift * dt + noise[i]
        
        return t, x
    
    def funcao_correlacao_teorica(self, tau: np.ndarray) -> np.ndarray:
        """
        Função de correlação teórica do processo OU modificado
        
        Parameters:
        -----------
        tau : np.ndarray
            Atraso temporal
            
        Returns:
        --------
        np.ndarray
            Função de correlação
        """
        # Parte padrão do OU
        corr_standard = (self.D / self.gamma) * np.exp(-self.gamma * np.abs(tau))
        
        # Correção gravitacional (exemplo)
        # Modulação devido à frequência cósmica
        modulation = 1 + self.alpha_grav * np.cos(2 * np.pi * f_cosmos * tau)
        
        return corr_standard * modulation
    
    def calcular_correlacao_empirica(self, x: np.ndarray, max_lag: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula função de correlação empírica
        
        Parameters:
        -----------
        x : np.ndarray
            Série temporal
        max_lag : int
            Máximo atraso para cálculo
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Atrasos e autocorrelações
        """
        N = len(x)
        max_lag = min(max_lag, N // 4)  # Limitar lag
        
        lags = np.arange(max_lag)
        correlations = np.zeros(max_lag)
        
        x_mean = np.mean(x)
        x_std = np.std(x)
        
        for lag in lags:
            if lag == 0:
                correlations[lag] = 1.0
            else:
                corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                correlations[lag] = corr if not np.isnan(corr) else 0.0
        
        return lags, correlations
    
    def densidade_espectral_teorica(self, omega: np.ndarray) -> np.ndarray:
        """
        Densidade espectral de potência teórica
        
        Parameters:
        -----------
        omega : np.ndarray
            Frequências angulares
            
        Returns:
        --------
        np.ndarray
            Densidade espectral
        """
        # Densidade espectral padrão do OU
        S_standard = (2 * self.D) / (self.gamma**2 + omega**2)
        
        # Correções gravitacionais (exemplo)
        # Pico próximo à frequência cósmica
        omega_cosmos = 2 * np.pi * f_cosmos
        correction = 1 + self.alpha_grav / (1 + ((omega - omega_cosmos) / (0.1 * omega_cosmos))**2)
        
        return S_standard * correction
    
    def ajustar_parametros(self, t: np.ndarray, x: np.ndarray) -> Dict[str, float]:
        """
        Ajusta parâmetros do modelo aos dados
        
        Parameters:
        -----------
        t : np.ndarray
            Tempos
        x : np.ndarray
            Posições
            
        Returns:
        --------
        Dict[str, float]
            Parâmetros ajustados
        """
        dt = t[1] - t[0]
        
        # Estimar gamma a partir da função de autocorrelação
        lags, correlations = self.calcular_correlacao_empirica(x)
        
        # Ajustar exponencial decrescente
        def exp_decay(lag, gamma_est):
            return np.exp(-gamma_est * lag * dt)
        
        try:
            popt, _ = curve_fit(exp_decay, lags[1:20], correlations[1:20], p0=[self.gamma])
            gamma_est = popt[0]
        except:
            gamma_est = self.gamma
        
        # Estimar D a partir da variância
        variance_est = np.var(x)
        D_est = variance_est * gamma_est
        
        # Estimar correção gravitacional (simplificado)
        alpha_grav_est = self.alpha_grav  # Placeholder
        
        return {
            'gamma': gamma_est,
            'D': D_est,
            'alpha_grav': alpha_grav_est,
            'tau_correlation': 1.0 / gamma_est,
            'variance': variance_est
        }
    
    def teste_estacionariedade(self, x: np.ndarray, n_segments: int = 10) -> Dict[str, float]:
        """
        Testa estacionariedade da série temporal
        
        Parameters:
        -----------
        x : np.ndarray
            Série temporal
        n_segments : int
            Número de segmentos para análise
            
        Returns:
        --------
        Dict[str, float]
            Estatísticas de estacionariedade
        """
        N = len(x)
        segment_size = N // n_segments
        
        means = []
        variances = []
        
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size
            segment = x[start:end]
            
            means.append(np.mean(segment))
            variances.append(np.var(segment))
        
        # Estatísticas
        mean_stability = np.std(means) / np.mean(np.abs(means)) if np.mean(np.abs(means)) > 0 else float('inf')
        variance_stability = np.std(variances) / np.mean(variances)
        
        return {
            'mean_stability': mean_stability,
            'variance_stability': variance_stability,
            'is_stationary': mean_stability < 0.1 and variance_stability < 0.1
        }
    
    def grafico_trajetoria(self, t: np.ndarray, x: np.ndarray, 
                          x_standard: Optional[np.ndarray] = None,
                          salvar: bool = True) -> None:
        """
        Plota trajetória do processo
        
        Parameters:
        -----------
        t : np.ndarray
            Tempos
        x : np.ndarray
            Posições (processo modificado)
        x_standard : np.ndarray, optional
            Posições (processo padrão para comparação)
        salvar : bool
            Se True, salva o gráfico
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Trajetória temporal
        ax1.plot(t, x, 'b-', alpha=0.7, label='OU Modificado')
        if x_standard is not None:
            ax1.plot(t, x_standard, 'r--', alpha=0.5, label='OU Padrão')
        
        ax1.set_xlabel('Tempo')
        ax1.set_ylabel('Posição')
        ax1.set_title('Trajetória do Processo OU Modificado')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Histograma das posições
        ax2.hist(x, bins=50, density=True, alpha=0.7, color='blue', label='Modificado')
        if x_standard is not None:
            ax2.hist(x_standard, bins=50, density=True, alpha=0.5, color='red', label='Padrão')
        
        # Distribuição teórica
        x_theory = np.linspace(np.min(x), np.max(x), 100)
        sigma_theory = np.sqrt(self.variance_equilibrium)
        gaussian = norm.pdf(x_theory, 0, sigma_theory)
        ax2.plot(x_theory, gaussian, 'k-', linewidth=2, label='Teoria')
        
        ax2.set_xlabel('Posição')
        ax2.set_ylabel('Densidade de Probabilidade')
        ax2.set_title('Distribuição Estacionária')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('/home/thlinux/relacionalidadegeral/resultados/graficos/ou_trajetoria.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def grafico_correlacao(self, t: np.ndarray, x: np.ndarray, salvar: bool = True) -> None:
        """
        Plota função de correlação
        
        Parameters:
        -----------
        t : np.ndarray
            Tempos
        x : np.ndarray
            Posições
        salvar : bool
            Se True, salva o gráfico
        """
        dt = t[1] - t[0]
        lags, corr_empirica = self.calcular_correlacao_empirica(x)
        tau_theory = lags * dt
        corr_teorica = self.funcao_correlacao_teorica(tau_theory)
        corr_teorica /= corr_teorica[0]  # Normalizar
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(tau_theory, corr_empirica, 'bo-', markersize=3, 
               label='Empírica', alpha=0.7)
        ax.plot(tau_theory, corr_teorica, 'r-', linewidth=2, 
               label='Teórica (Modificada)')
        
        # Correlação padrão para comparação
        corr_standard = np.exp(-self.gamma * tau_theory)
        ax.plot(tau_theory, corr_standard, 'g--', linewidth=2, 
               label='Teórica (Padrão)')
        
        ax.set_xlabel('Atraso τ')
        ax.set_ylabel('Autocorrelação')
        ax.set_title('Função de Autocorrelação')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 10 * self.tau_correlation)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('/home/thlinux/relacionalidadegeral/resultados/graficos/ou_correlacao.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()

def exemplo_simulacao():
    """
    Exemplo de simulação completa
    """
    print("=== SIMULAÇÃO DO PROCESSO OU MODIFICADO ===\n")
    
    # Parâmetros do processo
    gamma = 1.0  # s^-1
    D = 0.5  # (unidade)^2/s
    
    # Criar processo
    ou_mod = OrnsteinUhlenbeckModificado(gamma, D, alpha_grav_factor=1e40)  # Amplificar para visualização
    
    print(f"Parâmetros do processo:")
    print(f"γ = {gamma} s⁻¹")
    print(f"D = {D} (unidade)²/s")
    print(f"α_grav = {ou_mod.alpha_grav:.2e}")
    print(f"τ_correlação = {ou_mod.tau_correlation:.3f} s")
    print(f"Variância equilíbrio = {ou_mod.variance_equilibrium:.3f}\n")
    
    # Simulação
    t_final = 50.0  # segundos
    dt = 0.01  # s
    
    print("Simulando trajetórias...")
    t, x_mod = ou_mod.simular_trajetoria(t_final, dt, incluir_gravidade=True)
    _, x_std = ou_mod.simular_trajetoria(t_final, dt, incluir_gravidade=False)
    
    # Análise
    print("Analisando resultados...")
    params = ou_mod.ajustar_parametros(t, x_mod)
    stationarity = ou_mod.teste_estacionariedade(x_mod)
    
    print(f"\nParâmetros ajustados:")
    for key, value in params.items():
        print(f"{key} = {value:.6f}")
    
    print(f"\nTeste de estacionariedade:")
    print(f"Estabilidade da média: {stationarity['mean_stability']:.6f}")
    print(f"Estabilidade da variância: {stationarity['variance_stability']:.6f}")
    print(f"É estacionário: {stationarity['is_stationary']}")
    
    # Gráficos
    print("\nGerando gráficos...")
    ou_mod.grafico_trajetoria(t, x_mod, x_std)
    ou_mod.grafico_correlacao(t, x_mod)
    
    print("Simulação completa!")

if __name__ == "__main__":
    exemplo_simulacao()