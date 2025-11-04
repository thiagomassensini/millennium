#!/bin/bash
# Importação otimizada de 1 bilhão de primos gêmeos para MySQL
# Tempo estimado: 30-40 minutos com 56 cores

set -euo pipefail

CSV_FILE="/home/thlinux/relacionalidadegeral/codigo/binario/results.csv"
DB_NAME="twin_primes_db"
TABLE_NAME="twin_primes"
CHUNK_SIZE=1000000  # 1M linhas por chunk

echo "=========================================="
echo "  IMPORTAÇÃO MASSIVA DE PRIMOS GÊMEOS"
echo "=========================================="
echo "CSV: $CSV_FILE"
echo "Tamanho: $(ls -lh $CSV_FILE | awk '{print $5}')"
echo "Linhas: $(wc -l < $CSV_FILE | numfmt --grouping)"
echo ""

# Verificar MySQL
if ! systemctl is-active --quiet mysql; then
    echo "ERROR: MySQL não está rodando!"
    exit 1
fi

# Verificar espaço em disco
AVAILABLE_GB=$(df -BG /var/lib/mysql | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Espaço disponível: ${AVAILABLE_GB}GB"
if [ "$AVAILABLE_GB" -lt 30 ]; then
    echo "WARNING: Espaço em disco baixo! Pode não comportar a importação completa."
    read -p "Continuar mesmo assim? (s/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Configurando MySQL para importação em massa..."

# Otimizar MySQL para importação massiva
sudo mysql -u root "$DB_NAME" <<'EOF'
SET GLOBAL innodb_buffer_pool_size = 8589934592;  -- 8GB
SET GLOBAL innodb_log_file_size = 2147483648;     -- 2GB
SET GLOBAL innodb_flush_log_at_trx_commit = 0;
SET GLOBAL sync_binlog = 0;
SET GLOBAL max_allowed_packet = 1073741824;       -- 1GB
SET GLOBAL bulk_insert_buffer_size = 268435456;   -- 256MB
EOF

echo "MySQL otimizado"
echo ""

# Limpar tabela se necessário
read -p "Limpar dados existentes na tabela? (s/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "Limpando tabela..."
    sudo mysql -u root "$DB_NAME" -e "TRUNCATE TABLE $TABLE_NAME;"
    echo "Tabela limpa"
fi

echo ""
echo "Iniciando importação em massa..."
echo "Tempo estimado: 30-40 minutos"
echo ""

START_TIME=$(date +%s)

# Importação direta com LOAD DATA INFILE (muito mais rápido)
sudo mysql -u root "$DB_NAME" <<EOF
LOAD DATA LOCAL INFILE '$CSV_FILE'
INTO TABLE $TABLE_NAME
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(p, p_plus_2, k_real, thread_id, range_start);
EOF

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "  IMPORTAÇÃO CONCLUÍDA!"
echo "=========================================="
echo "Tempo total: ${MINUTES}m ${SECONDS}s"
echo ""

# Estatísticas finais
echo "Estatísticas da tabela:"
sudo mysql -u root "$DB_NAME" -e "
SELECT 
    COUNT(*) as total_records,
    MIN(p) as primeiro_primo,
    MAX(p) as ultimo_primo,
    COUNT(DISTINCT k_real) as diferentes_k
FROM $TABLE_NAME;
"

echo ""
echo "Distribuição por k:"
sudo mysql -u root "$DB_NAME" -e "
SELECT 
    k_real as k,
    COUNT(*) as quantidade,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM $TABLE_NAME), 4) as porcentagem
FROM $TABLE_NAME
GROUP BY k_real
ORDER BY k_real;
"

echo ""
echo "Restaurando configurações normais do MySQL..."
sudo mysql -u root "$DB_NAME" <<'EOF'
SET GLOBAL innodb_flush_log_at_trx_commit = 1;
SET GLOBAL sync_binlog = 1;
EOF

echo "Pronto para validação massiva!"
echo ""
echo "Execute agora:"
echo "  cd /home/thlinux/relacionalidadegeral/validacao"
echo "  export PRIME_DB_PASS=''"
echo "  ./massive_validator 1000000 56"
