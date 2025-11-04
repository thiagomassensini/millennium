#!/usr/bin/env bash
set -euo pipefail

echo "🚀 TWIN PRIME MINER v5 ULTRA - DEPLOY"
echo "======================================"

# --- 0) Pré-checagens ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "❌ falta $1"; exit 1; }; }
need g++ ; need mysql ; need awk ; need sed ; need nproc ; need pkg-config

# Dependências dev (Debian/Ubuntu): libmysqlclient-dev
# Verificar se mysql_config existe como alternativa ao pkg-config
if ! command -v mysql_config >/dev/null 2>&1 && ! pkg-config --exists mysqlclient 2>/dev/null; then
  echo "⚠️ libmysqlclient-dev não encontrado"
  echo "   Instale: sudo apt-get install -y libmysqlclient-dev"
  exit 1
fi

CORES=$(nproc)
echo "🧠 Núcleos detectados: $CORES"

# --- 1) Senha e env ---
ENV_FILE=".env.miner"
if [ ! -f "$ENV_FILE" ]; then
  # Gerar senha forte: letras maiúsculas, minúsculas, números e símbolos
  DB_PASS="Aa1!$(openssl rand -base64 24 | tr -d '\n' | sed 's/[\/\=+]/X/g')"
  echo "PRIME_DB_PASS=$DB_PASS" > "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "🔐 Senha gerada e salva em $ENV_FILE"
else
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  DB_PASS="${PRIME_DB_PASS:-}"
  if [ -z "${DB_PASS}" ]; then echo "❌ $ENV_FILE inválido"; exit 1; fi
  echo "🔐 Usando senha do $ENV_FILE"
fi

export PRIME_DB_PASS="$DB_PASS"

# --- 2) Banco de dados ---
SQL_SRC="setup_database_v5_ultra.sql"
SQL_TMP="$(mktemp)"
sed "s/REPLACE_WITH_SECRET/$DB_PASS/g" "$SQL_SRC" > "$SQL_TMP"

echo "📊 Configurando MySQL..."
mysql < "$SQL_TMP"
rm -f "$SQL_TMP"
echo "✅ Banco pronto"

# --- 3) Compilação agressiva ---
CPP="twin_prime_miner_v5_ultra_mpmc.cpp"
OUT="miner_v5_ultra"

CXXFLAGS="-O3 -march=native -mtune=native -flto -fopenmp -funroll-loops -ffast-math -DNDEBUG -pthread"
# Usar mysql_config se disponível, senão pkg-config
if command -v mysql_config >/dev/null 2>&1; then
  LIBS="$(mysql_config --libs)"
else
  LIBS="$(pkg-config --libs mysqlclient)"
fi

echo "🔨 Compilando..."
g++ $CXXFLAGS "$CPP" $LIBS -o "$OUT"
strip "$OUT" || true
echo "✅ Compilado: ./$OUT"

# --- 4) Script de início ---
cat > start_mining.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$ENV_FILE"
export PRIME_DB_PASS="\$PRIME_DB_PASS"
./$OUT --threads $CORES --start 1000000000000000 --end 1010000000000000 --chunk 100000000 --batch 50000
EOF
chmod +x start_mining.sh
echo "✅ start_mining.sh criado"

# --- 5) Dicas de tuning (opcional, manual) ---
cat <<'TIPS'

⚙️  Tuning opcional (avaliar antes de aplicar):
  # MySQL (requer SUPER ou edição do my.cnf):
  # innodb_buffer_pool_size = 40G
  # innodb_log_file_size    = 2G
  # innodb_flush_log_at_trx_commit = 2
  # max_allowed_packet = 256M
  # innodb_flush_method = O_DIRECT

  # Kernel (grandes filas IO, opcional):
  # sudo sysctl -w vm.swappiness=10
  # sudo sysctl -w vm.dirty_ratio=5 vm.dirty_background_ratio=2

Para iniciar:
  ./start_mining.sh

Para acompanhar:
  tail -f nohup.out  (caso rode com nohup)

TIPS
