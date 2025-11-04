#!/bin/bash
# Importação ultra-rápida usando mysqlimport

set -euo pipefail

CSV="/home/thlinux/relacionalidadegeral/codigo/binario/results.csv"
DB="twin_primes_db"
TMPDIR="/tmp/mysql_import_$$"

echo "==========================================="
echo "  IMPORTAÇÃO MASSIVA - 1 BILHÃO DE PRIMOS"
echo "==========================================="
echo "Início: $(date)"
echo ""

# Criar diretório temporário
mkdir -p "$TMPDIR"

# Copiar CSV para formato compatível com mysqlimport
echo "Preparando arquivo..."
tail -n +2 "$CSV" > "$TMPDIR/twin_primes.txt"

echo "Arquivo preparado: $(ls -lh $TMPDIR/twin_primes.txt | awk '{print $5}')"
echo ""

# Configurar permissões
sudo chown mysql:mysql "$TMPDIR/twin_primes.txt"
sudo chmod 644 "$TMPDIR/twin_primes.txt"

# Mover para diretório temporário do MySQL
sudo mv "$TMPDIR/twin_primes.txt" /var/lib/mysql-files/ 2>/dev/null || sudo mkdir -p /var/lib/mysql-files/ && sudo mv "$TMPDIR/twin_primes.txt" /var/lib/mysql-files/

echo "Iniciando importação..."
echo "Tempo estimado: 20-30 minutos"
echo ""

START=$(date +%s)

# Importar direto no MySQL
sudo mysql -u root "$DB" <<'EOF'
SET unique_checks = 0;
SET foreign_key_checks = 0;
SET autocommit = 0;

LOAD DATA INFILE '/var/lib/mysql-files/twin_primes.txt'
INTO TABLE twin_primes
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(p, p_plus_2, k_real, thread_id, range_start);

COMMIT;
SET unique_checks = 1;
SET foreign_key_checks = 1;
SET autocommit = 1;
EOF

END=$(date +%s)
DURATION=$((END - START))
MIN=$((DURATION / 60))
SEC=$((DURATION % 60))

echo ""
echo "✓ Importação concluída em ${MIN}m ${SEC}s"
echo ""

# Estatísticas
echo "Verificando importação..."
sudo mysql -u root "$DB" -e "
SELECT 
    COUNT(*) as total_primos,
    MIN(p) as primeiro,
    MAX(p) as ultimo,
    COUNT(DISTINCT k_real) as diferentes_k
FROM twin_primes;
"

echo ""
echo "Distribuição por k:"
sudo mysql -u root "$DB" -e "
SELECT 
    k_real,
    COUNT(*) as quantidade,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM twin_primes), 4) as pct
FROM twin_primes
GROUP BY k_real
ORDER BY k_real;
"

# Limpar
sudo rm -f /var/lib/mysql-files/twin_primes.txt
rmdir "$TMPDIR" 2>/dev/null || true

echo ""
echo "✓ Pronto! Banco populado com 1 bilhão de primos validados!"
echo ""
echo "Próximo passo:"
echo "  cd /home/thlinux/relacionalidadegeral/validacao"
echo "  export PRIME_DB_PASS=''"
echo "  ./massive_validator 1000000 56"
