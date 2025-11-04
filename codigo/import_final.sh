#!/bin/bash
# Importação definitiva de 1 bilhão de primos gêmeos
# Tempo estimado: 15-25 minutos

set -euo pipefail

CSV="/home/thlinux/relacionalidadegeral/codigo/binario/results.csv"
DB="twin_primes_db"
USER="prime_miner"
PASS="TwinPrimes@2025!XOR"

echo "============================================="
echo "  IMPORTAÇÃO: 1 BILHÃO DE PRIMOS GÊMEOS"
echo "============================================="
echo "Arquivo: $CSV ($(ls -lh $CSV | awk '{print $5}'))"
echo "Linhas: $(wc -l < $CSV | numfmt --grouping)"
echo "Início: $(date '+%H:%M:%S')"
echo ""

START=$(date +%s)

# Método 1: Usar mysqlimport (mais rápido)
echo "Preparando dados..."

# Criar arquivo temporário sem header
TMPFILE="/tmp/twin_primes_data_$$.txt"
tail -n +2 "$CSV" > "$TMPFILE"

echo "Configurando MySQL para importação massiva..."
sudo mysql -u root twin_primes_db -e "
SET GLOBAL local_infile = 1;
ALTER TABLE twin_primes DISABLE KEYS;
"

echo "Importando dados (isso vai demorar 15-25 minutos)..."
echo "Progresso: aguarde..."

# Importação via LOAD DATA
mysql -u "$USER" -p"$PASS" --local-infile=1 "$DB" <<EOF
SET autocommit = 0;
SET unique_checks = 0;
SET foreign_key_checks = 0;

LOAD DATA LOCAL INFILE '$TMPFILE'
INTO TABLE twin_primes
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(p, p_plus_2, k_real, thread_id, range_start);

COMMIT;
SET unique_checks = 1;
SET foreign_key_checks = 1;
SET autocommit = 1;
EOF

echo "Reconstruindo índices..."
sudo mysql -u root twin_primes_db -e "ALTER TABLE twin_primes ENABLE KEYS;"

# Limpar
rm -f "$TMPFILE"

END=$(date +%s)
DURATION=$((END - START))
MIN=$((DURATION / 60))
SEC=$((DURATION % 60))

echo ""
echo "✓ IMPORTAÇÃO CONCLUÍDA!"
echo "  Tempo: ${MIN}m ${SEC}s"
echo "  Fim: $(date '+%H:%M:%S')"
echo ""

# Estatísticas
echo "═══════════════════════════════════════"
echo "  ESTATÍSTICAS DA IMPORTAÇÃO"
echo "═══════════════════════════════════════"

mysql -u "$USER" -p"$PASS" "$DB" <<'EOF'
SELECT 
    FORMAT(COUNT(*), 0) as 'Total de Primos',
    MIN(p) as 'Primeiro',
    MAX(p) as 'Último',
    COUNT(DISTINCT k_real) as 'Diferentes K'
FROM twin_primes;
EOF

echo ""
echo "Distribuição por K:"
mysql -u "$USER" -p"$PASS" "$DB" <<'EOF'
SELECT 
    k_real as K,
    FORMAT(COUNT(*), 0) as Quantidade,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM twin_primes), 2), '%') as Porcentagem
FROM twin_primes
GROUP BY k_real
ORDER BY k_real;
EOF

echo ""
echo "═══════════════════════════════════════"
echo "✓ Banco populado e pronto para validação!"
echo ""
echo "Próximos passos:"
echo "  cd /home/thlinux/relacionalidadegeral/validacao"
echo "  export PRIME_DB_PASS='TwinPrimes@2025!XOR'"
echo "  ./massive_validator 1000000 56"
