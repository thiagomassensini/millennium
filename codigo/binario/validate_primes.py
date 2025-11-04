#!/usr/bin/env python3
"""Valida se os números encontrados são realmente primos gêmeos"""

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
    k = (v.bit_length() - 1) - 1
    return k if 0 <= k < 25 else -1

# Casos de teste - AMOSTRAS DO ARQUIVO ATUAL (989M primos)
test_cases = [
    "1000008609026507,1000008609026509,2",
    "1000008609027059,1000008609027061,2",
    "1000008609027161,1000008609027163,1",
    "1000008609028937,1000008609028939,1",
    "1000008609028997,1000008609028999,1",
    "1000008609030059,1000008609030061,2",
    "1000008609030071,1000008609030073,3",
    "1000008609030929,1000008609030931,1",
    "1000008609032579,1000008609032581,2",
    "1000008609032687,1000008609032689,4",
]

print("[SEARCH] VALIDAÇÃO DE PRIMOS GÊMEOS")
print("=" * 60)
print()

valid = 0
invalid = 0

for line in test_cases:
    parts = line.split(',')
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
        status = "[OK]"
    else:
        invalid += 1
        status = "[FAIL]"
    
    print(f"{status} p = {p}")
    print(f"   p+2 = {p_plus_2} | Twin: {is_twin}")
    print(f"   p é primo: {p_is_prime}")
    print(f"   p+2 é primo: {p2_is_prime}")
    print(f"   k_real: {k_reported} (calculado: {k_calc}) | Match: {k_matches}")
    
    if not is_valid:
        print(f"   [WARNING]  INVÁLIDO!")
    print()

print("=" * 60)
print(f"[DATA] RESULTADO: {valid} válidos, {invalid} inválidos")
print(f"   Taxa de sucesso: {100*valid/(valid+invalid):.1f}%")

# Verificar distribuição de k
print()
print("[UP] VERIFICANDO DISTRIBUIÇÃO TEÓRICA:")
print("   Esperado: ~50% k=1, ~25% k=2, ~12.5% k=3, etc.")
print("   (proporção 2^(-k))")
