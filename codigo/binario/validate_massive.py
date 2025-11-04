#!/usr/bin/env python3
"""Validação massiva de 100k amostras aleatórias de primos gêmeos"""

import random

def is_prime_miller_rabin(n, k=7):
    """Teste de primalidade Miller-Rabin determinístico para n < 2^64"""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False
    
    # Bases que garantem teste determinístico para n < 2^64
    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    
    # Escrever n-1 como 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def calc_k_real(p):
    """Calcula o k_real de um par de primos gêmeos"""
    if p % 2 == 0:
        return -1
    x = p ^ (p + 2)
    v = x + 2
    # Verificar se v é potência de 2
    if v & (v - 1) != 0:
        return -1
    k = v.bit_length() - 2
    return k if 0 <= k < 25 else -1

print("🔬 VALIDAÇÃO MASSIVA: 100.000 AMOSTRAS ALEATÓRIAS")
print("=" * 70)

# Ler arquivo e pegar linhas aleatórias
print("📁 Lendo arquivo com amostragem reservatório...")
# Algoritmo de amostragem reservatório - eficiente para arquivos grandes
sample_size = 100000
samples = []
total_lines = 0

with open('results.csv', 'r') as f:
    next(f)  # Pular header
    
    for i, line in enumerate(f):
        total_lines = i + 1
        
        if i < sample_size:
            # Primeiras sample_size linhas vão direto
            samples.append(line.strip())
        else:
            # Depois, substituir com probabilidade decrescente
            j = random.randint(0, i)
            if j < sample_size:
                samples[j] = line.strip()
        
        if total_lines % 100000000 == 0:
            print(f"   Processadas: {total_lines:,} linhas...")

print(f"✅ Total de primos no arquivo: {total_lines:,}")
print(f"🎲 Selecionadas {len(samples):,} amostras aleatórias\n")

print("🔍 VALIDANDO AMOSTRAS...")
print("-" * 70)

valid = 0
invalid = 0
errors = []

for i, line in enumerate(samples):
    if (i + 1) % 10000 == 0:
        print(f"   Progresso: {i+1:,}/{sample_size:,} ({100*(i+1)/sample_size:.1f}%)")
    
    parts = line.strip().split(',')
    if len(parts) < 3:
        continue
        
    p = int(parts[0])
    p_plus_2 = int(parts[1])
    k_reported = int(parts[2])
    
    # Verificações
    is_twin = (p_plus_2 == p + 2)
    p_is_prime = is_prime_miller_rabin(p)
    p2_is_prime = is_prime_miller_rabin(p_plus_2)
    k_calc = calc_k_real(p)
    k_matches = (k_calc == k_reported)
    
    is_valid = is_twin and p_is_prime and p2_is_prime and k_matches
    
    if is_valid:
        valid += 1
    else:
        invalid += 1
        errors.append({
            'p': p,
            'p_plus_2': p_plus_2,
            'k_reported': k_reported,
            'k_calc': k_calc,
            'is_twin': is_twin,
            'p_prime': p_is_prime,
            'p2_prime': p2_is_prime
        })

print()
print("=" * 70)
print("📊 RESULTADO FINAL:")
print("=" * 70)
print(f"✅ Válidos:   {valid:,} ({100*valid/sample_size:.4f}%)")
print(f"❌ Inválidos: {invalid:,} ({100*invalid/sample_size:.4f}%)")
print()

if invalid == 0:
    print("🎉 PERFEITO! TODOS OS 100.000 PRIMOS SÃO VÁLIDOS!")
    print("   ✅ Teste de primalidade Miller-Rabin: 100% correto")
    print("   ✅ Pares gêmeos (p, p+2): 100% correto")
    print("   ✅ k_real calculado: 100% correto")
    print()
    print("🏆 O MINERADOR ESTÁ FUNCIONANDO PERFEITAMENTE!")
else:
    print(f"⚠️  Encontrados {invalid} primos com problemas:")
    for err in errors[:10]:  # Mostrar primeiros 10 erros
        print(f"   p={err['p']}, twin={err['is_twin']}, "
              f"p_prime={err['p_prime']}, p2_prime={err['p2_prime']}, "
              f"k={err['k_reported']} (calc={err['k_calc']})")

print()
print("=" * 70)
print(f"📈 ESTATÍSTICAS:")
print(f"   Arquivo analisado: results.csv")
print(f"   Total de primos: {total_lines:,}")
print(f"   Amostras testadas: {sample_size:,}")
print(f"   Taxa de sucesso: {100*valid/sample_size:.6f}%")
print("=" * 70)
