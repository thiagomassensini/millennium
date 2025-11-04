#!/bin/bash
# Script definitivo para validação massiva após importação

set -euo pipefail

DB_PASS='TwinPrimes@2025!XOR'
VALIDATOR="/home/thlinux/relacionalidadegeral/validacao/massive_validator"

echo "=========================================="
echo "  AGUARDANDO IMPORTAÇÃO TERMINAR"
echo "=========================================="

# Aguardar importação completar
while true; do
    COUNT=$(mysql -u prime_miner -p"$DB_PASS" twin_primes_db -sN -e "SELECT COUNT(*) FROM twin_primes;" 2>/dev/null || echo "0")
    
    if [ "$COUNT" -gt 1000000000 ]; then
        echo "Importação concluída: $(echo $COUNT | numfmt --grouping) registros"
        break
    elif [ "$COUNT" -gt 0 ]; then
        echo "Importando... $(echo $COUNT | numfmt --grouping) registros até agora"
    else
        echo "Aguardando importação iniciar..."
    fi
    
    sleep 10
done

echo ""
echo "=========================================="
echo "  ESTATÍSTICAS DO BANCO"
echo "=========================================="

mysql -u prime_miner -p"$DB_PASS" twin_primes_db <<'EOF'
SELECT 
    FORMAT(COUNT(*), 0) as 'Total Primos',
    MIN(p) as 'Primeiro',
    MAX(p) as 'Último',
    COUNT(DISTINCT k_real) as 'K Únicos'
FROM twin_primes;

SELECT 
    k_real as K,
    FORMAT(COUNT(*), 0) as Quantidade,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM twin_primes), 2), '%') as Porcentagem
FROM twin_primes
GROUP BY k_real
ORDER BY k_real
LIMIT 15;
EOF

echo ""
echo "=========================================="
echo "  INICIANDO VALIDAÇÃO MASSIVA"
echo "=========================================="
echo "Amostras: 1.000.000"
echo "Cores: 56"
echo "Tempo estimado: 5 minutos"
echo ""

cd /home/thlinux/relacionalidadegeral/validacao

export PRIME_DB_PASS="$DB_PASS"

"$VALIDATOR" 1000000 56

echo ""
echo "=========================================="
echo "  VALIDAÇÃO CONCLUÍDA!"
echo "=========================================="
echo ""
echo "Resultados salvos em:"
echo "  - validation_results.json"
echo "  - validation_results.csv"
echo "  - validation_report.tex"
echo ""
echo "Próximo passo:"
echo "  cd /home/thlinux/relacionalidadegeral/papers"
echo "  make validate_and_update"
