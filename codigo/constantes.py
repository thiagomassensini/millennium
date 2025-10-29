"""
Constantes Físicas Fundamentais
==============================

Este módulo contém todas as constantes físicas fundamentais utilizadas
na Teoria da Relacionalidade Geral, com alta precisão numérica.

Autor: Equipe de Pesquisa
Data: Outubro 2025
"""

import numpy as np
from scipy import constants

# =============================================================================
# CONSTANTES FUNDAMENTAIS (CODATA 2018/2022)
# =============================================================================

# Velocidade da luz no vácuo
c = constants.c  # 299792458.0 m/s (exato por definição)

# Constante de Planck
h = constants.h  # 6.62607015e-34 J⋅s (exato por definição)
hbar = constants.hbar  # 1.0545718176461565e-34 J⋅s

# Constante gravitacional de Newton
G = constants.G  # 6.67430e-11 m³⋅kg⁻¹⋅s⁻²

# Massa do elétron
m_e = constants.m_e  # 9.1093837015e-31 kg

# Carga elementar
e = constants.e  # 1.602176634e-19 C (exato por definição)

# Constante de Boltzmann
k_B = constants.k  # 1.380649e-23 J/K (exato por definição)

# Permeabilidade magnética do vácuo
mu_0 = constants.mu_0  # 1.25663706212e-6 H/m

# Permissividade elétrica do vácuo
epsilon_0 = constants.epsilon_0  # 8.8541878128e-12 F/m

# =============================================================================
# CONSTANTES DERIVADAS IMPORTANTES
# =============================================================================

# Constante de estrutura fina
alpha_em = constants.alpha  # 7.2973525693e-3 ≈ 1/137.036

# Raio clássico do elétron
r_e = constants.e**2 / (4 * np.pi * constants.epsilon_0 * constants.m_e * constants.c**2)  # 2.8179403262e-15 m

# Comprimento de onda Compton do elétron
lambda_C = constants.h / (constants.m_e * constants.c)  # 2.42631023867e-12 m

# Raio de Bohr
a_0 = constants.hbar / (constants.m_e * constants.c * constants.alpha)  # 5.29177210903e-11 m

# =============================================================================
# UNIDADES DE PLANCK
# =============================================================================

# Comprimento de Planck
l_Planck = np.sqrt(constants.hbar * constants.G / constants.c**3)  # 1.616255e-35 m

# Tempo de Planck
t_Planck = np.sqrt(constants.hbar * constants.G / constants.c**5)  # 5.391247e-44 s

# Massa de Planck
M_Planck = np.sqrt(constants.hbar * constants.c / constants.G)  # 2.176434e-8 kg

# Energia de Planck
E_Planck = M_Planck * constants.c**2  # 1.956082e9 J

# Temperatura de Planck
T_Planck = E_Planck / constants.k  # 1.416784e32 K

# =============================================================================
# CONSTANTES ASTROFÍSICAS
# =============================================================================

# Massa do Sol
M_sun = 1.98847e30  # kg

# Raio do Sol
R_sun = 6.96e8  # m

# Ano luz
ly = 9.4607304725808e15  # m

# Parsec
pc = 3.0856775814913673e16  # m

# Unidade astronômica
AU = constants.au  # 1.495978707e11 m

# =============================================================================
# CONSTANTES DA TEORIA DA RELACIONALIDADE GERAL
# =============================================================================

# Constante de acoplamento gravitacional
alpha_grav = (constants.G * constants.m_e * constants.c) / constants.hbar
print(f"α_grav = {alpha_grav:.6e}")

# Frequência cósmica (usando massa de Planck)
f_cosmos = constants.c**3 / (constants.G * M_Planck)
print(f"f_cosmos = {f_cosmos:.6e} Hz")

# Frequência de Planck alternativa
f_Planck = constants.c / l_Planck
print(f"f_Planck = {f_Planck:.6e} Hz")

# Coeficiente do SNR universal
SNR_coefficient = 0.05
print(f"Coeficiente SNR = {SNR_coefficient}")

# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def energia_massa(massa_kg):
    """
    Calcula a energia de repouso usando E = mc²
    
    Parameters:
    -----------
    massa_kg : float
        Massa em quilogramas
        
    Returns:
    --------
    float
        Energia em Joules
    """
    return massa_kg * constants.c**2

def frequencia_planck(energia_J):
    """
    Calcula a frequência correspondente a uma energia usando E = hf
    
    Parameters:
    -----------
    energia_J : float
        Energia em Joules
        
    Returns:
    --------
    float
        Frequência em Hz
    """
    return energia_J / constants.h

def comprimento_onda_de_broglie(massa_kg, velocidade_ms):
    """
    Calcula o comprimento de onda de de Broglie
    
    Parameters:
    -----------
    massa_kg : float
        Massa em kg
    velocidade_ms : float
        Velocidade em m/s
        
    Returns:
    --------
    float
        Comprimento de onda em metros
    """
    momentum = massa_kg * velocidade_ms
    return constants.h / momentum

def raio_schwarzschild(massa_kg):
    """
    Calcula o raio de Schwarzschild
    
    Parameters:
    -----------
    massa_kg : float
        Massa em kg
        
    Returns:
    --------
    float
        Raio de Schwarzschild em metros
    """
    return 2 * constants.G * massa_kg / constants.c**2

def temperatura_hawking(massa_kg):
    """
    Calcula a temperatura de Hawking de um buraco negro
    
    Parameters:
    -----------
    massa_kg : float
        Massa do buraco negro em kg
        
    Returns:
    --------
    float
        Temperatura em Kelvin
    """
    return constants.hbar * constants.c**3 / (8 * np.pi * constants.G * massa_kg * constants.k)

# =============================================================================
# VERIFICAÇÕES DE CONSISTÊNCIA
# =============================================================================

if __name__ == "__main__":
    print("=== Verificações de Consistência ===")
    print(f"c = {c:.0f} m/s")
    print(f"G = {G:.6e} m³⋅kg⁻¹⋅s⁻²")
    print(f"ℏ = {hbar:.6e} J⋅s")
    print(f"m_e = {m_e:.6e} kg")
    print(f"α_em = {alpha_em:.6f}")
    print(f"α_grav = {alpha_grav:.6e}")
    print(f"α_grav/α_em = {alpha_grav/alpha_em:.6e}")
    print(f"f_cosmos = {f_cosmos:.6e} Hz")
    print(f"1/f_cosmos = {1/f_cosmos:.6e} s")
    print(f"t_Planck = {t_Planck:.6e} s")
    print(f"Razão: (1/f_cosmos)/t_Planck = {(1/f_cosmos)/t_Planck:.3f}")