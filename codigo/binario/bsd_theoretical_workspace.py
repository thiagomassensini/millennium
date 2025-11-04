#!/usr/bin/env python3
"""
BSD THEORETICAL WORKSPACE
Ferramentas para investigar teoricamente rank(E_p) = (log₂(k)+1)//2

Objetivo: Provar (ou encontrar contraexemplo) que para k=2^n:
  E_p: y² = x³ + (p mod k²)·x + k  com p primo gêmeo
  rank(E_p) = (n+1)//2

Estratégias:
1. Calcular Selmer groups (2-descent)
2. Estudar discriminante e fatoração
3. Analisar função L
4. Investigar torção
5. Conectar estrutura XOR com álgebra
"""

import cypari2
import json
from collections import defaultdict

pari = cypari2.Pari()

# ==================== BASIC CURVE TOOLS ====================

def get_curve_info(p, k):
    """
    Retorna informações completas sobre E_p: y²=x³+(p mod k²)x+k
    """
    a = int(p % (k * k)) if k > 0 else 0
    b = k
    
    try:
        E = pari.ellinit([0, 0, 0, a, b])
        
        # Discriminante
        disc = pari.ellglobalred(E)[0]
        
        # Rank (analítico)
        rank_data = pari.ellanalyticrank(E)
        rank = int(rank_data[0])
        L_value = float(rank_data[1])
        
        # Torção
        torsion = pari.elltors(E)
        tors_order = int(torsion[0])
        
        # Minimal model
        minimal = pari.ellminimalmodel(E)
        
        return {
            'p': p,
            'k': k,
            'a': a,
            'b': b,
            'discriminant': int(disc),
            'rank': rank,
            'L_value': L_value,
            'torsion_order': tors_order,
            'torsion_structure': str(torsion),
            'minimal_model': str(minimal),
            'curve': E
        }
    except Exception as e:
        return {'error': str(e), 'p': p, 'k': k}

def factorize_discriminant(disc):
    """Fatoração do discriminante"""
    try:
        factors = pari.factor(abs(disc))
        return str(factors)
    except:
        return None

# ==================== 2-DESCENT TOOLS ====================

def compute_2_descent(p, k):
    """
    Calcula 2-descent usando PARI/GP
    Retorna dimensão do grupo de Selmer
    """
    a = int(p % (k * k))
    b = k
    
    try:
        E = pari.ellinit([0, 0, 0, a, b])
        
        # ellrank faz 2-descent automaticamente
        # Retorna [rank_lower_bound, rank_upper_bound, Sha_bound]
        descent = pari.ellrank(E)
        
        return {
            'p': p,
            'k': k,
            'rank_lower': int(descent[0]),
            'rank_upper': int(descent[1]),
            'sha_bound': int(descent[2]) if len(descent) > 2 else None,
            'descent_info': str(descent)
        }
    except Exception as e:
        return {'error': str(e), 'p': p, 'k': k}

# ==================== XOR STRUCTURE ====================

def analyze_xor_structure(p):
    """
    Analisa estrutura do XOR e sua relação com k
    """
    xor = p ^ (p + 2)
    xor_plus_2 = xor + 2
    
    # k_real
    if xor_plus_2 & (xor_plus_2 - 1) == 0:  # é potência de 2
        k = xor_plus_2.bit_length() - 2
    else:
        k = None
    
    return {
        'p': p,
        'xor': xor,
        'xor_binary': bin(xor)[2:],
        'xor_plus_2': xor_plus_2,
        'xor_plus_2_binary': bin(xor_plus_2)[2:],
        'k': k,
        'is_power_of_2': (xor_plus_2 & (xor_plus_2 - 1)) == 0,
        'bit_length': xor_plus_2.bit_length()
    }

# ==================== L-FUNCTION ANALYSIS ====================

def analyze_L_function(p, k, precision=100):
    """
    Analisa função L da curva
    """
    a = int(p % (k * k))
    b = k
    
    try:
        E = pari.ellinit([0, 0, 0, a, b])
        
        # Alta precisão
        pari.default('realprecision', precision)
        
        # Rank analítico (ordem do zero em s=1)
        rank_data = pari.ellanalyticrank(E)
        rank = int(rank_data[0])
        L_derivative = float(rank_data[1])
        
        # Período (para verificar fórmula BSD)
        omega = pari.ellperiods(E)[0]
        
        return {
            'p': p,
            'k': k,
            'rank': rank,
            'L_derivative': L_derivative,
            'omega': float(omega.real()),
            'L_over_omega': float(L_derivative / omega.real()) if rank == 0 else None
        }
    except Exception as e:
        return {'error': str(e), 'p': p, 'k': k}

# ==================== BATCH ANALYSIS ====================

def comprehensive_analysis(prime_k_pairs, output_file='bsd_comprehensive.json'):
    """
    Análise completa de múltiplas curvas
    """
    results = []
    
    for p, k in prime_k_pairs:
        print(f"\n{'='*60}")
        print(f"Analyzing p={p}, k={k} (k=2^{k.bit_length()-1 if k > 0 and (k & (k-1)) == 0 else '?'})")
        print('='*60)
        
        # 1. Informações básicas
        info = get_curve_info(p, k)
        print(f"  Rank: {info.get('rank', '?')}")
        print(f"  Discriminant: {info.get('discriminant', '?')}")
        print(f"  Torsion: {info.get('torsion_order', '?')}")
        
        # 2. XOR structure
        xor_info = analyze_xor_structure(p)
        print(f"  XOR: {xor_info['xor']} = {xor_info['xor_binary']}")
        
        # 3. 2-descent
        descent = compute_2_descent(p, k)
        print(f"  2-descent: rank ∈ [{descent.get('rank_lower', '?')}, {descent.get('rank_upper', '?')}]")
        
        # 4. L-function
        L_info = analyze_L_function(p, k)
        print(f"  L-function rank: {L_info.get('rank', '?')}")
        
        # Combinar tudo
        result = {
            'p': p,
            'k': k,
            'basic_info': info,
            'xor_structure': xor_info,
            'descent': descent,
            'L_function': L_info
        }
        
        results.append(result)
    
    # Salvar
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")
    return results

# ==================== PATTERN DETECTION ====================

def detect_patterns(results):
    """
    Procura padrões nos resultados
    """
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    # Agrupar por k
    by_k = defaultdict(list)
    for r in results:
        k = r['k']
        rank = r['basic_info'].get('rank')
        tors = r['basic_info'].get('torsion_order')
        disc = r['basic_info'].get('discriminant')
        
        if rank is not None:
            by_k[k].append({
                'p': r['p'],
                'rank': rank,
                'torsion': tors,
                'disc': disc,
                'xor': r['xor_structure']['xor']
            })
    
    # Analisar cada k
    for k in sorted(by_k.keys()):
        data = by_k[k]
        ranks = [d['rank'] for d in data]
        
        print(f"\nk={k}:")
        print(f"  Curves: {len(data)}")
        print(f"  Ranks: {set(ranks)}")
        
        if len(set(ranks)) == 1:
            print(f"  ✓ DETERMINISTIC: rank = {ranks[0]}")
            
            # Verificar fórmula
            if k > 0 and (k & (k-1)) == 0:  # potência de 2
                n = k.bit_length() - 1
                expected = (n + 1) // 2
                if ranks[0] == expected:
                    print(f"  ✓✓✓ FORMULA CONFIRMED: rank = ({n}+1)//2 = {expected}")
                else:
                    print(f"  ⚠️  FORMULA FAILED: expected {expected}, got {ranks[0]}")
        
        # Analisar torção
        torsions = [d['torsion'] for d in data]
        print(f"  Torsions: {set(torsions)}")
        
        # Analisar discriminantes
        discs = [d['disc'] for d in data]
        print(f"  Sample discriminants: {discs[:3]}")

# ==================== MAIN ====================

if __name__ == '__main__':
    print("="*80)
    print("BSD THEORETICAL WORKSPACE")
    print("="*80)
    print()
    print("Este ambiente está preparado para:")
    print("  1. Calcular Selmer groups (2-descent)")
    print("  2. Analisar função L")
    print("  3. Estudar discriminantes e torção")
    print("  4. Conectar estrutura XOR com álgebra")
    print()
    print("Use as funções acima para investigar teoricamente.")
    print()
    print("Exemplo:")
    print("  >>> from bsd_theoretical_workspace import *")
    print("  >>> info = get_curve_info(p=11, k=2)")
    print("  >>> descent = compute_2_descent(p=11, k=2)")
    print()
