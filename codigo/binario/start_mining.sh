#!/usr/bin/env bash
set -euo pipefail
source ".env.miner"
export PRIME_DB_PASS="$PRIME_DB_PASS"
# Batch médio (10k) com 4 threads escritoras = 40k batch efetivo, chunk grande
./miner_v5_ultra --threads 56 --start 1000000000000000 --end 1010000000000000 --chunk 500000000 --batch 10000
