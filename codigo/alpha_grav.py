"""
Cálculo da Constante de Acoplamento Gravitacional α_grav
======================================================

Este módulo implementa o cálculo detalhado de α_grav e suas propriedades,
incluindo análise de incertezas e comparações com outras constantes.

Autor: Equipe de Pesquisa
Data: Outubro 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from typing import Tuple, Dict, List
import warnings

# Importar constantes do módulo local
from constantes import *

class AlphaGravCalculator:
    """
    Classe para cálculos relacionados à constante α_grav
    """
    
    def __init__(self):
        """Inicializa o calculador com valores das constantes"""
        self.G = constants.G
        self.m_e = constants.m_e
        self.c = constants.c
        self.hbar = constants.hbar
        
        # Incertezas relativas (CODATA 2018)
        self.sigma_G_rel = 2.2e-5  # Incerteza em G
        self.sigma_me_rel = 3.0e-10  # Incerteza em m_e
        # c e ℏ são exatos por definição
        
    def calcular_alpha_grav(self) -> float:
        """
        Calcula α_grav = (G * m_e * c) / ℏ
        
        Returns:
        --------
        float
            Valor de α_grav
        """
        return (self.G * self.m_e * self.c) / self.hbar
    
    def incerteza_alpha_grav(self) -> float:
        """
        Calcula a incerteza em α_grav usando propagação de incertezas
        
        Returns:
        --------
        float
            Incerteza absoluta em α_grav
        """
        alpha_grav = self.calcular_alpha_grav()
        
        # Propagação de incertezas (soma quadrática das incertezas relativas)
        sigma_rel_total = np.sqrt(self.sigma_G_rel**2 + self.sigma_me_rel**2)
        
        return alpha_grav * sigma_rel_total
    
    def comparar_constantes(self) -> Dict[str, float]:
        """
        Compara α_grav com outras constantes de acoplamento
        
        Returns:
        --------
        Dict[str, float]
            Dicionário com as razões entre constantes
        """
        alpha_grav = self.calcular_alpha_grav()
        alpha_em = constants.alpha
        
        # Estimativas para outras constantes (dependem da energia)
        alpha_s = 0.118  # Constante forte a ~91 GeV (Z boson)
        alpha_w = alpha_em / (np.sin(28.7 * np.pi/180)**2)  # Ângulo de Weinberg
        
        return {
            'alpha_grav': alpha_grav,
            'alpha_em': alpha_em,
            'alpha_s': alpha_s,
            'alpha_w': alpha_w,
            'alpha_grav/alpha_em': alpha_grav / alpha_em,
            'alpha_grav/alpha_s': alpha_grav / alpha_s,
            'alpha_grav/alpha_w': alpha_grav / alpha_w,
            'log10(alpha_grav/alpha_em)': np.log10(alpha_grav / alpha_em)
        }
    
    def escala_energia_caracteristica(self) -> Dict[str, float]:
        """
        Calcula escalas de energia características relacionadas a α_grav
        
        Returns:
        --------
        Dict[str, float]
            Dicionário com energias características
        """
        alpha_grav = self.calcular_alpha_grav()
        
        # Energia onde efeitos gravitacionais quânticos se tornam importantes
        E_grav = self.m_e * self.c**2 / alpha_grav
        
        # Energia de Planck
        E_planck = np.sqrt(self.hbar * self.c**5 / self.G)
        
        # Energia da massa do elétron
        E_electron = self.m_e * self.c**2
        
        return {
            'E_grav_GeV': E_grav / (1.602e-19 * 1e9),  # em GeV
            'E_planck_GeV': E_planck / (1.602e-19 * 1e9),  # em GeV
            'E_electron_MeV': E_electron / (1.602e-19 * 1e6),  # em MeV
            'E_grav/E_planck': E_grav / E_planck,
            'E_grav/E_electron': E_grav / E_electron
        }
    
    def correcoes_quanticas(self, energia_GeV: float) -> Dict[str, float]:
        """
        Calcula correções quânticas esperadas em diferentes energias
        
        Parameters:
        -----------
        energia_GeV : float
            Energia em GeV
            
        Returns:
        --------
        Dict[str, float]
            Correções em diferentes observáveis
        """
        alpha_grav = self.calcular_alpha_grav()
        E = energia_GeV * 1e9 * 1.602e-19  # Conversão para Joules
        E_electron = self.m_e * self.c**2
        
        # Correções proporcionais a α_grav
        correcao_energia = alpha_grav * (E / E_electron)
        correcao_momento_magnetico = alpha_grav * (E / E_electron)**2
        correcao_tempo_vida = alpha_grav * (E / E_electron)
        
        return {
            'energia_relativa': correcao_energia,
            'momento_magnetico_relativa': correcao_momento_magnetico,
            'tempo_vida_relativa': correcao_tempo_vida,
            'detectabilidade': correcao_energia > 1e-15
        }
    
    def evolucao_RG(self, escala_inicial_GeV: float, escala_final_GeV: float) -> float:
        """
        Evolução de α_grav sob o grupo de renormalização (estimativa)
        
        Parameters:
        -----------
        escala_inicial_GeV : float
            Escala inicial em GeV
        escala_final_GeV : float
            Escala final em GeV
            
        Returns:
        --------
        float
            α_grav na escala final
        """
        alpha_grav_inicial = self.calcular_alpha_grav()
        
        # Função beta estimada (modelo simplificado)
        beta_0 = -2  # Coeficiente leading-order
        t = np.log(escala_final_GeV / escala_inicial_GeV)
        
        # Solução aproximada da equação RG
        alpha_grav_final = alpha_grav_inicial / (1 + beta_0 * alpha_grav_inicial * t / (2 * np.pi))
        
        return alpha_grav_final
    
    def grafico_comparacao_constantes(self, salvar: bool = True) -> None:
        """
        Cria gráfico comparando as constantes de acoplamento
        
        Parameters:
        -----------
        salvar : bool
            Se True, salva o gráfico
        """
        comparacao = self.comparar_constantes()
        
        constantes = ['α_em', 'α_w', 'α_s', 'α_grav']
        valores = [
            comparacao['alpha_em'],
            comparacao['alpha_w'],
            comparacao['alpha_s'],
            comparacao['alpha_grav']
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(constantes, valores, 
                     color=['blue', 'green', 'red', 'purple'],
                     alpha=0.7)
        
        ax.set_yscale('log')
        ax.set_ylabel('Valor da Constante de Acoplamento')
        ax.set_title('Comparação das Constantes de Acoplamento Fundamentais')
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nos bars
        for bar, valor in zip(bars, valores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{valor:.2e}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('/home/thlinux/relacionalidadegeral/resultados/graficos/comparacao_constantes.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def grafico_correcoes_energia(self, energias_GeV: List[float], salvar: bool = True) -> None:
        """
        Cria gráfico das correções vs energia
        
        Parameters:
        -----------
        energias_GeV : List[float]
            Lista de energias em GeV
        salvar : bool
            Se True, salva o gráfico
        """
        correcoes = []
        
        for E in energias_GeV:
            corr = self.correcoes_quanticas(E)
            correcoes.append(corr['energia_relativa'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(energias_GeV, correcoes, 'b-', linewidth=2, label='Correção Gravitacional')
        ax.axhline(y=1e-15, color='r', linestyle='--', label='Limite de Detectabilidade')
        
        ax.set_xlabel('Energia (GeV)')
        ax.set_ylabel('Correção Relativa')
        ax.set_title('Correções Gravitacionais Quânticas vs Energia')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('/home/thlinux/relacionalidadegeral/resultados/graficos/correcoes_energia.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()

def analise_completa():
    """
    Executa análise completa de α_grav
    """
    calc = AlphaGravCalculator()
    
    print("=== ANÁLISE COMPLETA DE α_grav ===\n")
    
    # Valor principal
    alpha_grav = calc.calcular_alpha_grav()
    incerteza = calc.incerteza_alpha_grav()
    
    print(f"α_grav = ({alpha_grav:.6e} ± {incerteza:.6e})")
    print(f"Incerteza relativa: {incerteza/alpha_grav:.2e}\n")
    
    # Comparação com outras constantes
    print("=== COMPARAÇÃO COM OUTRAS CONSTANTES ===")
    comp = calc.comparar_constantes()
    for chave, valor in comp.items():
        if 'alpha_' in chave and '/' not in chave:
            print(f"{chave} = {valor:.6e}")
        elif '/' in chave:
            print(f"{chave} = {valor:.6e}")
    print()
    
    # Escalas de energia
    print("=== ESCALAS DE ENERGIA CARACTERÍSTICAS ===")
    escalas = calc.escala_energia_caracteristica()
    for chave, valor in escalas.items():
        print(f"{chave} = {valor:.6e}")
    print()
    
    # Correções em diferentes energias
    print("=== CORREÇÕES QUÂNTICAS (exemplos) ===")
    energias_teste = [0.001, 0.1, 1, 100, 1000]  # GeV
    
    for E in energias_teste:
        corr = calc.correcoes_quanticas(E)
        print(f"E = {E:6.3f} GeV: correção = {corr['energia_relativa']:.2e}, "
              f"detectável = {corr['detectabilidade']}")
    
    # Gerar gráficos
    print("\n=== GERANDO GRÁFICOS ===")
    calc.grafico_comparacao_constantes()
    
    energias = np.logspace(-3, 4, 100)  # 1 MeV a 10 TeV
    calc.grafico_correcoes_energia(energias.tolist())
    
    print("Análise completa finalizada!")

if __name__ == "__main__":
    analise_completa()