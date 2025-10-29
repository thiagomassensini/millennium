"""
Frequência Cósmica f_cosmos
===========================

Este módulo implementa cálculos relacionados à frequência cósmica fundamental,
incluindo sua derivação, propriedades e implicações físicas.

Autor: Equipe de Pesquisa
Data: Outubro 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, special
from scipy.optimize import fsolve
from typing import Tuple, Dict, List, Optional
import warnings

# Importar constantes do módulo local
from constantes import *

class FrequenciaCosmica:
    """
    Classe para cálculos relacionados à frequência cósmica
    """
    
    def __init__(self):
        """Inicializa com constantes fundamentais"""
        self.c = constants.c
        self.G = constants.G
        self.hbar = constants.hbar
        self.M_planck = M_Planck
        
        # Calcular frequência cósmica
        self.f_cosmos = self.calcular_f_cosmos()
        
    def calcular_f_cosmos(self) -> float:
        """
        Calcula a frequência cósmica usando diferentes métodos
        
        Returns:
        --------
        float
            Frequência cósmica em Hz
        """
        # Método 1: Usando massa de Planck
        f1 = self.c**3 / (self.G * self.M_planck)
        
        # Método 2: Diretamente das constantes fundamentais
        f2 = np.sqrt(self.c**5 / (self.G * self.hbar)) / (2 * np.pi)
        
        # Método 3: A partir do tempo de Planck
        t_planck = np.sqrt(self.hbar * self.G / self.c**5)
        f3 = 1 / (2 * np.pi * t_planck)
        
        # Verificar consistência
        if not (np.isclose(f1, f2, rtol=1e-10) and np.isclose(f2, f3, rtol=1e-10)):
            warnings.warn("Inconsistência nos cálculos de f_cosmos")
        
        return f1
    
    def escalas_temporais_relacionadas(self) -> Dict[str, float]:
        """
        Calcula escalas temporais relacionadas à frequência cósmica
        
        Returns:
        --------
        Dict[str, float]
            Dicionário com diferentes escalas temporais
        """
        T_cosmos = 1 / self.f_cosmos  # Período cósmico
        t_planck = np.sqrt(self.hbar * self.G / self.c**5)
        
        # Outras escalas temporais relevantes
        t_compton_electron = self.hbar / (constants.m_e * self.c**2)
        t_luz_raio_classico = constants.e**2 / (6 * np.pi * constants.epsilon_0 * constants.m_e * self.c**3)
        
        # Escalas cosmológicas
        idade_universo = 13.8e9 * 365.25 * 24 * 3600  # segundos
        tempo_hubble = 1 / (70 * 1000 / (3.086e22))  # 1/H_0
        
        return {
            'T_cosmos': T_cosmos,
            't_planck': t_planck,
            't_compton_electron': t_compton_electron,
            't_radiacao_electron': t_luz_raio_classico,
            'idade_universo': idade_universo,
            'tempo_hubble': tempo_hubble,
            'razao_T_cosmos_t_planck': T_cosmos / t_planck,
            'razao_idade_T_cosmos': idade_universo / T_cosmos
        }
    
    def frequencias_harmonicas(self, n_harmonicos: int = 10) -> np.ndarray:
        """
        Calcula frequências harmônicas de f_cosmos
        
        Parameters:
        -----------
        n_harmonicos : int
            Número de harmônicos a calcular
            
        Returns:
        --------
        np.ndarray
            Array com frequências harmônicas
        """
        harmonicos = np.arange(1, n_harmonicos + 1)
        return harmonicos * self.f_cosmos
    
    def frequencias_subharmonicas(self, n_subharmonicos: int = 10) -> np.ndarray:
        """
        Calcula frequências sub-harmônicas de f_cosmos
        
        Parameters:
        -----------
        n_subharmonicos : int
            Número de sub-harmônicos a calcular
            
        Returns:
        --------
        np.ndarray
            Array com frequências sub-harmônicas
        """
        divisores = np.arange(1, n_subharmonicos + 1)
        return self.f_cosmos / divisores
    
    def modulacao_temporal(self, t: np.ndarray, amplitude: float = 1.0, fase: float = 0.0) -> np.ndarray:
        """
        Calcula modulação temporal na frequência cósmica
        
        Parameters:
        -----------
        t : np.ndarray
            Array de tempos
        amplitude : float
            Amplitude da modulação
        fase : float
            Fase da modulação
            
        Returns:
        --------
        np.ndarray
            Sinal modulado
        """
        return amplitude * np.cos(2 * np.pi * self.f_cosmos * t + fase)
    
    def espectro_potencia_cosmico(self, frequencias: np.ndarray) -> np.ndarray:
        """
        Calcula espectro de potência com características cósmicas
        
        Parameters:
        -----------
        frequencias : np.ndarray
            Array de frequências
            
        Returns:
        --------
        np.ndarray
            Espectro de potência
        """
        # Espectro com pico em f_cosmos
        omega = 2 * np.pi * frequencias
        omega_cosmos = 2 * np.pi * self.f_cosmos
        
        # Forma lorentziana centrada em f_cosmos
        gamma = omega_cosmos / 1000  # Largura espectral
        espectro = gamma**2 / ((omega - omega_cosmos)**2 + gamma**2)
        
        # Adicionar componente de lei de potência
        espectro += (frequencias / self.f_cosmos)**(-2) * np.exp(-frequencias / self.f_cosmos)
        
        return espectro
    
    def ressonancia_gravitacional(self, massa_kg: float, raio_m: float) -> Dict[str, float]:
        """
        Calcula frequências de ressonância gravitacional de um objeto
        
        Parameters:
        -----------
        massa_kg : float
            Massa do objeto em kg
        raio_m : float
            Raio do objeto em m
            
        Returns:
        --------
        Dict[str, float]
            Frequências de ressonância
        """
        # Frequência gravitacional natural
        f_grav_natural = np.sqrt(self.G * massa_kg) / (2 * np.pi * raio_m**(3/2))
        
        # Frequência modificada pela f_cosmos
        f_modificada = f_grav_natural * np.sqrt(1 + (f_grav_natural / self.f_cosmos)**2)
        
        # Frequências de Bragg gravitacionais (análogo óptico)
        lambda_grav = self.c / self.f_cosmos
        f_bragg = self.c / (2 * raio_m)  # Primeira ordem
        
        return {
            'f_gravitacional_natural': f_grav_natural,
            'f_modificada_cosmos': f_modificada,
            'f_bragg_gravitacional': f_bragg,
            'lambda_cosmos': lambda_grav,
            'Q_ressonancia': f_grav_natural / (f_grav_natural - self.f_cosmos) if abs(f_grav_natural - self.f_cosmos) > 0 else float('inf')
        }
    
    def interacao_com_particulas(self, massa_particula_kg: float) -> Dict[str, float]:
        """
        Calcula interação da frequência cósmica com partículas
        
        Parameters:
        -----------
        massa_particula_kg : float
            Massa da partícula em kg
            
        Returns:
        --------
        Dict[str, float]
            Parâmetros de interação
        """
        # Frequência Compton da partícula
        f_compton = massa_particula_kg * self.c**2 / constants.h
        
        # Comprimento de onda Compton
        lambda_compton = constants.h / (massa_particula_kg * self.c)
        
        # Acoplamento com f_cosmos
        acoplamento = alpha_grav * (f_compton / self.f_cosmos)
        
        # Probabilidade de transição (estimativa)
        prob_transicao = acoplamento**2 / (1 + (f_compton / self.f_cosmos)**2)
        
        return {
            'f_compton': f_compton,
            'lambda_compton': lambda_compton,
            'acoplamento_cosmos': acoplamento,
            'probabilidade_transicao': prob_transicao,
            'razao_f_compton_f_cosmos': f_compton / self.f_cosmos
        }
    
    def correcoes_cosmologicas(self, z_redshift: float = 0.0) -> Dict[str, float]:
        """
        Calcula correções cosmológicas à frequência cósmica
        
        Parameters:
        -----------
        z_redshift : float
            Redshift cosmológico
            
        Returns:
        --------
        Dict[str, float]
            Frequência corrigida por efeitos cosmológicos
        """
        # Correção por redshift
        f_cosmos_observed = self.f_cosmos / (1 + z_redshift)
        
        # Correção por expansão do universo (aproximação)
        H0 = 2.2e-18  # Constante de Hubble em s^-1
        idade_universo = 1 / H0
        
        # Deriva temporal da frequência cósmica
        df_dt = -self.f_cosmos * H0 / 2  # Estimativa
        
        return {
            'f_cosmos_z0': self.f_cosmos,
            'f_cosmos_observed': f_cosmos_observed,
            'redshift': z_redshift,
            'deriva_temporal': df_dt,
            'tempo_variacao_relativa': 1 / (df_dt / self.f_cosmos) if df_dt != 0 else float('inf')
        }
    
    def grafico_espectro_cosmico(self, salvar: bool = True) -> None:
        """
        Cria gráfico do espectro cósmico
        
        Parameters:
        -----------
        salvar : bool
            Se True, salva o gráfico
        """
        # Frequências de 0.1*f_cosmos a 10*f_cosmos
        f_min = 0.1 * self.f_cosmos
        f_max = 10 * self.f_cosmos
        frequencias = np.logspace(np.log10(f_min), np.log10(f_max), 1000)
        
        espectro = self.espectro_potencia_cosmico(frequencias)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(frequencias / self.f_cosmos, espectro, 'b-', linewidth=2)
        ax.axvline(x=1, color='r', linestyle='--', linewidth=2, 
                  label=f'f_cosmos = {self.f_cosmos:.2e} Hz')
        
        # Marcar harmônicos
        harmonicos = self.frequencias_harmonicas(5)
        for i, f_h in enumerate(harmonicos):
            if f_h <= f_max:
                ax.axvline(x=f_h/self.f_cosmos, color='g', linestyle=':', alpha=0.7)
                if i < 3:  # Anotar só os primeiros
                    ax.text(f_h/self.f_cosmos, np.max(espectro)/2, f'{i+1}H', 
                           rotation=90, ha='right', va='bottom')
        
        ax.set_xlabel('Frequência / f_cosmos')
        ax.set_ylabel('Densidade Espectral de Potência (u.a.)')
        ax.set_title('Espectro de Potência Cósmico')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('/home/thlinux/relacionalidadegeral/resultados/graficos/espectro_cosmico.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def grafico_escalas_temporais(self, salvar: bool = True) -> None:
        """
        Cria gráfico comparando escalas temporais
        
        Parameters:
        -----------
        salvar : bool
            Se True, salva o gráfico
        """
        escalas = self.escalas_temporais_relacionadas()
        
        # Selecionar escalas para o gráfico
        nomes = ['t_planck', 'T_cosmos', 't_compton_electron', 't_radiacao_electron', 
                'tempo_hubble', 'idade_universo']
        valores = [escalas[nome] for nome in nomes]
        labels = ['Tempo de Planck', 'Período Cósmico', 'Tempo Compton (e⁻)', 
                 'Tempo Radiação (e⁻)', 'Tempo de Hubble', 'Idade do Universo']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Gráfico de barras em escala log
        bars = ax.bar(range(len(valores)), np.log10(valores), 
                     color=['purple', 'red', 'blue', 'green', 'orange', 'brown'],
                     alpha=0.7)
        
        ax.set_ylabel('log₁₀(Tempo em segundos)')
        ax.set_title('Escalas Temporais Fundamentais')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (bar, valor) in enumerate(zip(bars, valores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{valor:.2e}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('/home/thlinux/relacionalidadegeral/resultados/graficos/escalas_temporais.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def grafico_ressonancia_objetos(self, salvar: bool = True) -> None:
        """
        Cria gráfico de ressonâncias para diferentes objetos
        
        Parameters:
        -----------
        salvar : bool
            Se True, salva o gráfico
        """
        # Objetos astronômicos típicos
        objetos = {
            'Próton': (constants.m_p, 1e-15),
            'Núcleo Atômico': (100 * constants.m_p, 5e-15),
            'Asteroide': (1e15, 1e3),
            'Lua': (7.34e22, 1.74e6),
            'Terra': (5.97e24, 6.37e6),
            'Sol': (1.99e30, 6.96e8),
            'Estrela de Nêutrons': (2.8e30, 1e4),
            'Buraco Negro Estelar': (20 * 1.99e30, 6e4),
        }
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        nomes = []
        f_naturais = []
        f_modificadas = []
        Q_factors = []
        
        for nome, (massa, raio) in objetos.items():
            res = self.ressonancia_gravitacional(massa, raio)
            nomes.append(nome)
            f_naturais.append(res['f_gravitacional_natural'])
            f_modificadas.append(res['f_modificada_cosmos'])
            Q_factors.append(min(res['Q_ressonancia'], 1e10))  # Limitar Q
        
        x = range(len(nomes))
        
        # Frequências de ressonância
        ax1.semilogy(x, f_naturais, 'bo-', label='Natural', markersize=6)
        ax1.semilogy(x, f_modificadas, 'ro-', label='Modificada', markersize=6)
        ax1.axhline(y=self.f_cosmos, color='g', linestyle='--', linewidth=2,
                   label=f'f_cosmos = {self.f_cosmos:.2e} Hz')
        
        ax1.set_ylabel('Frequência (Hz)')
        ax1.set_title('Frequências de Ressonância Gravitacional')
        ax1.set_xticks(x)
        ax1.set_xticklabels(nomes, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Fatores Q
        ax2.semilogy(x, Q_factors, 'go-', markersize=6)
        ax2.set_ylabel('Fator Q')
        ax2.set_xlabel('Objeto')
        ax2.set_title('Fatores de Qualidade de Ressonância')
        ax2.set_xticks(x)
        ax2.set_xticklabels(nomes, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('/home/thlinux/relacionalidadegeral/resultados/graficos/ressonancia_objetos.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()

def analise_completa_f_cosmos():
    """
    Análise completa da frequência cósmica
    """
    print("=== ANÁLISE COMPLETA DA FREQUÊNCIA CÓSMICA ===\n")
    
    # Criar calculadora
    f_calc = FrequenciaCosmica()
    
    print(f"f_cosmos = {f_calc.f_cosmos:.6e} Hz")
    print(f"T_cosmos = {1/f_calc.f_cosmos:.6e} s")
    print(f"λ_cosmos = {constants.c/f_calc.f_cosmos:.6e} m\n")
    
    # Escalas temporais
    print("=== ESCALAS TEMPORAIS RELACIONADAS ===")
    escalas = f_calc.escalas_temporais_relacionadas()
    for nome, valor in escalas.items():
        if 'razao' not in nome:
            print(f"{nome:20s} = {valor:.6e} s")
        else:
            print(f"{nome:20s} = {valor:.6e}")
    print()
    
    # Harmônicos e sub-harmônicos
    print("=== HARMÔNICOS E SUB-HARMÔNICOS ===")
    harmonicos = f_calc.frequencias_harmonicas(5)
    subharmonicos = f_calc.frequencias_subharmonicas(5)
    
    print("Harmônicos:")
    for i, f_h in enumerate(harmonicos):
        print(f"  {i+1}H: {f_h:.6e} Hz")
    
    print("Sub-harmônicos:")
    for i, f_s in enumerate(subharmonicos):
        print(f"  f/{i+1}: {f_s:.6e} Hz")
    print()
    
    # Interação com partículas fundamentais
    print("=== INTERAÇÃO COM PARTÍCULAS ===")
    particulas = {
        'Elétron': constants.m_e,
        'Múon': 1.883e-28,  # kg
        'Próton': constants.m_p,
        'Nêutron': constants.m_n
    }
    
    for nome, massa in particulas.items():
        interacao = f_calc.interacao_com_particulas(massa)
        print(f"{nome:8s}: f_Compton = {interacao['f_compton']:.2e} Hz, "
              f"razão = {interacao['razao_f_compton_f_cosmos']:.2e}, "
              f"acoplamento = {interacao['acoplamento_cosmos']:.2e}")
    print()
    
    # Ressonâncias gravitacionais
    print("=== RESSONÂNCIAS GRAVITACIONAIS (exemplos) ===")
    objetos_exemplo = [
        ('Terra', 5.97e24, 6.37e6),
        ('Sol', 1.99e30, 6.96e8),
        ('Estrela de Nêutrons', 2.8e30, 1e4),
    ]
    
    for nome, massa, raio in objetos_exemplo:
        res = f_calc.ressonancia_gravitacional(massa, raio)
        print(f"{nome:18s}: f_nat = {res['f_gravitacional_natural']:.2e} Hz, "
              f"Q = {res['Q_ressonancia']:.2e}")
    print()
    
    # Correções cosmológicas
    print("=== CORREÇÕES COSMOLÓGICAS ===")
    redshifts = [0, 1, 5, 10]
    for z in redshifts:
        corr = f_calc.correcoes_cosmologicas(z)
        print(f"z = {z:2d}: f_obs = {corr['f_cosmos_observed']:.2e} Hz, "
              f"redução = {(1-corr['f_cosmos_observed']/f_calc.f_cosmos)*100:.1f}%")
    
    # Gerar gráficos
    print(f"\n=== GERANDO GRÁFICOS ===")
    f_calc.grafico_espectro_cosmico()
    f_calc.grafico_escalas_temporais()
    f_calc.grafico_ressonancia_objetos()
    
    print("Análise completa da frequência cósmica finalizada!")

if __name__ == "__main__":
    analise_completa_f_cosmos()