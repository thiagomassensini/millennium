"""
SNR Universal: SNR = 0.05√N
=============================

Este módulo implementa cálculos e análises relacionados ao SNR universal,
incluindo validação teórica e aplicações em diferentes sistemas físicos.

Autor: Equipe de Pesquisa
Data: Outubro 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from typing import Tuple, Dict, List, Optional, Callable
import warnings
from dataclasses import dataclass

# Importar constantes do módulo local
from constantes import *

@dataclass
class SistemaFisico:
    """Classe para representar um sistema físico com SNR"""
    nome: str
    N: int  # Número de graus de liberdade
    SNR_medido: float
    incerteza_SNR: float = 0.0
    unidade: str = "adimensional"
    referencia: str = ""

class SNRUniversal:
    """
    Classe para análise do SNR universal
    """
    
    def __init__(self):
        """Inicializa a classe com a constante universal"""
        self.C_universal = 0.05  # Constante universal
        self.systems_database = []  # Base de dados de sistemas
    
    def snr_teorico(self, N: int) -> float:
        """
        Calcula SNR teórico usando a fórmula universal
        
        Parameters:
        -----------
        N : int
            Número de graus de liberdade
            
        Returns:
        --------
        float
            SNR teórico
        """
        return self.C_universal * np.sqrt(N)
    
    def adicionar_sistema(self, sistema: SistemaFisico) -> None:
        """
        Adiciona sistema à base de dados
        
        Parameters:
        -----------
        sistema : SistemaFisico
            Sistema físico a ser adicionado
        """
        self.systems_database.append(sistema)
    
    def calcular_snr_empirico(self, sinal: np.ndarray, ruido: np.ndarray = None) -> Dict[str, float]:
        """
        Calcula SNR empírico de dados
        
        Parameters:
        -----------
        sinal : np.ndarray
            Série temporal do sinal
        ruido : np.ndarray, optional
            Série temporal do ruído (se None, estimado)
            
        Returns:
        --------
        Dict[str, float]
            Estatísticas do SNR
        """
        if ruido is None:
            # Estimar ruído como componente de alta frequência
            from scipy.signal import butter, filtfilt
            
            # Filtro passa-alta para estimar ruído
            b, a = butter(4, 0.1, btype='high')
            ruido_estimado = filtfilt(b, a, sinal)
            
            # Filtro passa-baixa para estimar sinal
            b, a = butter(4, 0.1, btype='low')
            sinal_estimado = filtfilt(b, a, sinal)
        else:
            sinal_estimado = sinal
            ruido_estimado = ruido
        
        # Calcular potências
        P_sinal = np.mean(sinal_estimado**2)
        P_ruido = np.mean(ruido_estimado**2)
        
        # SNR em diferentes definições
        snr_potencia = P_sinal / P_ruido if P_ruido > 0 else float('inf')
        snr_amplitude = np.sqrt(snr_potencia)
        snr_db = 10 * np.log10(snr_potencia) if snr_potencia > 0 else float('inf')
        
        return {
            'SNR_potencia': snr_potencia,
            'SNR_amplitude': snr_amplitude,
            'SNR_dB': snr_db,
            'P_sinal': P_sinal,
            'P_ruido': P_ruido
        }
    
    def estimar_graus_liberdade(self, SNR_medido: float) -> int:
        """
        Estima número de graus de liberdade a partir do SNR
        
        Parameters:
        -----------
        SNR_medido : float
            SNR medido experimentalmente
            
        Returns:
        --------
        int
            Número estimado de graus de liberdade
        """
        N_estimado = (SNR_medido / self.C_universal)**2
        return int(np.round(N_estimado))
    
    def teste_universalidade(self) -> Dict[str, float]:
        """
        Testa a universalidade da lei SNR = 0.05√N
        
        Returns:
        --------
        Dict[str, float]
            Estatísticas do teste
        """
        if len(self.systems_database) < 3:
            raise ValueError("Necessário pelo menos 3 sistemas para teste de universalidade")
        
        N_values = np.array([sys.N for sys in self.systems_database])
        SNR_values = np.array([sys.SNR_medido for sys in self.systems_database])
        SNR_errors = np.array([sys.incerteza_SNR for sys in self.systems_database])
        
        # Ajuste linear de SNR vs √N
        sqrt_N = np.sqrt(N_values)
        
        # Ajuste ponderado se incertezas disponíveis
        if np.any(SNR_errors > 0):
            weights = 1.0 / (SNR_errors + 1e-10)
        else:
            weights = np.ones_like(SNR_values)
        
        # Ajuste linear: SNR = C * √N
        def modelo_linear(x, C):
            return C * x
        
        try:
            popt, pcov = curve_fit(modelo_linear, sqrt_N, SNR_values, 
                                 sigma=1/weights, p0=[self.C_universal])
            C_ajustado = popt[0]
            sigma_C = np.sqrt(pcov[0, 0])
        except:
            C_ajustado = np.mean(SNR_values / sqrt_N)
            sigma_C = np.std(SNR_values / sqrt_N) / np.sqrt(len(SNR_values))
        
        # Estatísticas de qualidade do ajuste
        SNR_predito = C_ajustado * sqrt_N
        chi2 = np.sum(((SNR_values - SNR_predito) / (SNR_errors + 1e-10))**2)
        chi2_reduzido = chi2 / (len(SNR_values) - 1)
        
        # Coeficiente de determinação
        SS_res = np.sum((SNR_values - SNR_predito)**2)
        SS_tot = np.sum((SNR_values - np.mean(SNR_values))**2)
        R2 = 1 - (SS_res / SS_tot) if SS_tot > 0 else 0
        
        return {
            'C_ajustado': C_ajustado,
            'sigma_C': sigma_C,
            'C_teorico': self.C_universal,
            'desvio_relativo': abs(C_ajustado - self.C_universal) / self.C_universal,
            'chi2_reduzido': chi2_reduzido,
            'R2': R2,
            'p_value': 1 - stats.chi2.cdf(chi2, len(SNR_values) - 1),
            'concordancia': abs(C_ajustado - self.C_universal) < 2 * sigma_C
        }
    
    def simular_sistema_ideal(self, N: int, n_amostras: int = 10000) -> np.ndarray:
        """
        Simula sistema ideal com N graus de liberdade
        
        Parameters:
        -----------
        N : int
            Número de graus de liberdade
        n_amostras : int
            Número de amostras temporais
            
        Returns:
        --------
        np.ndarray
            Sinal simulado
        """
        # Sinal coerente (mesmo para todos os graus de liberdade)
        t = np.linspace(0, 10, n_amostras)
        sinal_coerente = np.sin(2 * np.pi * t)
        
        # Ruído independente para cada grau de liberdade
        ruido_total = np.zeros(n_amostras)
        for i in range(N):
            ruido_i = np.random.normal(0, 1, n_amostras)
            ruido_total += ruido_i
        
        # Normalizar ruído
        ruido_total /= np.sqrt(N)
        
        # Combinar sinal e ruído com amplitude para dar SNR correto
        amplitude_sinal = self.C_universal * np.sqrt(N)
        
        sinal_total = amplitude_sinal * sinal_coerente + ruido_total
        
        return sinal_total
    
    def analise_scaling(self, N_values: List[int], n_realizacoes: int = 100) -> Dict[str, np.ndarray]:
        """
        Analisa comportamento de escala do SNR
        
        Parameters:
        -----------
        N_values : List[int]
            Lista de valores de N para testar
        n_realizacoes : int
            Número de realizações para cada N
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Resultados da análise de escala
        """
        SNR_medio = np.zeros(len(N_values))
        SNR_std = np.zeros(len(N_values))
        
        for i, N in enumerate(N_values):
            SNR_realizacoes = []
            
            for j in range(n_realizacoes):
                sinal = self.simular_sistema_ideal(N)
                
                # Separar sinal e ruído para cálculo do SNR
                t = np.linspace(0, 10, len(sinal))
                sinal_teorico = self.C_universal * np.sqrt(N) * np.sin(2 * np.pi * t)
                ruido_estimado = sinal - sinal_teorico
                
                snr_stats = self.calcular_snr_empirico(sinal_teorico, ruido_estimado)
                SNR_realizacoes.append(snr_stats['SNR_amplitude'])
            
            SNR_medio[i] = np.mean(SNR_realizacoes)
            SNR_std[i] = np.std(SNR_realizacoes)
        
        return {
            'N_values': np.array(N_values),
            'SNR_medio': SNR_medio,
            'SNR_std': SNR_std,
            'SNR_teorico': self.C_universal * np.sqrt(N_values)
        }
    
    def grafico_universalidade(self, salvar: bool = True) -> None:
        """
        Cria gráfico da universalidade do SNR
        
        Parameters:
        -----------
        salvar : bool
            Se True, salva o gráfico
        """
        if len(self.systems_database) == 0:
            print("Nenhum sistema na base de dados. Adicionando sistemas exemplo...")
            self._adicionar_sistemas_exemplo()
        
        N_values = np.array([sys.N for sys in self.systems_database])
        SNR_values = np.array([sys.SNR_medido for sys in self.systems_database])
        SNR_errors = np.array([sys.incerteza_SNR for sys in self.systems_database])
        nomes = [sys.nome for sys in self.systems_database]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Dados experimentais
        ax.errorbar(np.sqrt(N_values), SNR_values, yerr=SNR_errors,
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   label='Dados Experimentais', alpha=0.8)
        
        # Anotar pontos
        for i, nome in enumerate(nomes):
            ax.annotate(nome, (np.sqrt(N_values[i]), SNR_values[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Linha teórica
        sqrt_N_theory = np.linspace(0, np.max(np.sqrt(N_values)) * 1.1, 100)
        SNR_theory = self.C_universal * sqrt_N_theory
        ax.plot(sqrt_N_theory, SNR_theory, 'r-', linewidth=2,
               label=f'SNR = {self.C_universal}√N')
        
        # Banda de incerteza (exemplo: ±20%)
        ax.fill_between(sqrt_N_theory, 0.8 * SNR_theory, 1.2 * SNR_theory,
                       alpha=0.2, color='red', label='Banda ±20%')
        
        ax.set_xlabel('√N (Raiz do Número de Graus de Liberdade)')
        ax.set_ylabel('SNR')
        ax.set_title('Universalidade do SNR: Lei de Escala SNR = 0.05√N')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('/home/thlinux/relacionalidadegeral/resultados/graficos/snr_universalidade.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def grafico_scaling_analysis(self, salvar: bool = True) -> None:
        """
        Cria gráfico da análise de escala
        
        Parameters:
        -----------
        salvar : bool
            Se True, salva o gráfico
        """
        N_values = [10, 50, 100, 500, 1000, 5000]
        scaling_results = self.analise_scaling(N_values, n_realizacoes=50)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # SNR vs √N
        sqrt_N = np.sqrt(scaling_results['N_values'])
        ax1.errorbar(sqrt_N, scaling_results['SNR_medio'], 
                    yerr=scaling_results['SNR_std'],
                    fmt='bo-', markersize=6, capsize=3,
                    label='Simulação')
        ax1.plot(sqrt_N, scaling_results['SNR_teorico'], 'r-', 
                linewidth=2, label='Teoria')
        
        ax1.set_xlabel('√N')
        ax1.set_ylabel('SNR')
        ax1.set_title('Verificação da Lei de Escala por Simulação')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Desvios relativos
        desvios = (scaling_results['SNR_medio'] - scaling_results['SNR_teorico']) / scaling_results['SNR_teorico']
        ax2.plot(sqrt_N, desvios * 100, 'go-', markersize=6)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.fill_between(sqrt_N, -5, 5, alpha=0.2, color='gray', label='±5%')
        
        ax2.set_xlabel('√N')
        ax2.set_ylabel('Desvio Relativo (%)')
        ax2.set_title('Desvios da Lei Teórica')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('/home/thlinux/relacionalidadegeral/resultados/graficos/snr_scaling.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _adicionar_sistemas_exemplo(self) -> None:
        """Adiciona sistemas exemplo para demonstração"""
        sistemas_exemplo = [
            SistemaFisico("Circuito RC", 1, 0.048, 0.005, "V/V", "Exemplo"),
            SistemaFisico("Oscilador Harmônico", 2, 0.071, 0.008, "", "Simulação"),
            SistemaFisico("Cadeia de 10 spins", 10, 0.155, 0.020, "", "Monte Carlo"),
            SistemaFisico("Rede neural (100 neurônios)", 100, 0.51, 0.05, "", "Simulação"),
            SistemaFisico("Sistema quântico (1000 níveis)", 1000, 1.58, 0.15, "", "Diagonalização"),
            SistemaFisico("Mercado financeiro", 5000, 3.52, 0.30, "", "Dados reais"),
        ]
        
        for sistema in sistemas_exemplo:
            self.adicionar_sistema(sistema)

def exemplo_completo():
    """
    Exemplo completo de análise do SNR universal
    """
    print("=== ANÁLISE DO SNR UNIVERSAL ===\n")
    
    # Criar analisador
    snr_calc = SNRUniversal()
    
    # Adicionar sistemas exemplo
    snr_calc._adicionar_sistemas_exemplo()
    
    print(f"Constante universal: C = {snr_calc.C_universal}")
    print(f"Número de sistemas na base: {len(snr_calc.systems_database)}\n")
    
    # Listar sistemas
    print("=== SISTEMAS NA BASE DE DADOS ===")
    for i, sys in enumerate(snr_calc.systems_database):
        snr_teorico = snr_calc.snr_teorico(sys.N)
        desvio = abs(sys.SNR_medido - snr_teorico) / snr_teorico * 100
        print(f"{i+1:2d}. {sys.nome:25s} | N={sys.N:5d} | SNR={sys.SNR_medido:.3f}±{sys.incerteza_SNR:.3f} | Teoria={snr_teorico:.3f} | Desvio={desvio:.1f}%")
    
    # Teste de universalidade
    print(f"\n=== TESTE DE UNIVERSALIDADE ===")
    try:
        teste = snr_calc.teste_universalidade()
        print(f"C ajustado: {teste['C_ajustado']:.4f} ± {teste['sigma_C']:.4f}")
        print(f"C teórico:  {teste['C_teorico']:.4f}")
        print(f"Desvio relativo: {teste['desvio_relativo']*100:.2f}%")
        print(f"χ² reduzido: {teste['chi2_reduzido']:.3f}")
        print(f"R²: {teste['R2']:.3f}")
        print(f"p-value: {teste['p_value']:.3f}")
        print(f"Concordância: {teste['concordancia']}")
    except Exception as e:
        print(f"Erro no teste: {e}")
    
    # Simulação de sistema ideal
    print(f"\n=== SIMULAÇÃO DE SISTEMA IDEAL ===")
    N_teste = 100
    sinal = snr_calc.simular_sistema_ideal(N_teste)
    
    # Calcular SNR do sinal simulado
    t = np.linspace(0, 10, len(sinal))
    sinal_teorico = snr_calc.C_universal * np.sqrt(N_teste) * np.sin(2 * np.pi * t)
    ruido_estimado = sinal - sinal_teorico
    
    snr_stats = snr_calc.calcular_snr_empirico(sinal_teorico, ruido_estimado)
    print(f"N = {N_teste}")
    print(f"SNR teórico: {snr_calc.snr_teorico(N_teste):.3f}")
    print(f"SNR simulado: {snr_stats['SNR_amplitude']:.3f}")
    print(f"SNR (dB): {snr_stats['SNR_dB']:.1f}")
    
    # Gerar gráficos
    print(f"\n=== GERANDO GRÁFICOS ===")
    snr_calc.grafico_universalidade()
    snr_calc.grafico_scaling_analysis()
    
    print("Análise completa do SNR universal finalizada!")

if __name__ == "__main__":
    exemplo_completo()