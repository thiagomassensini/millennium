#!/bin/bash
# Monitor em tempo real do Twin Prime Miner

DB_PASS=$(grep PRIME_DB_PASS .env.miner | cut -d= -f2)

echo "🔍 TWIN PRIME MINER - MONITOR EM TEMPO REAL"
echo "=========================================="
echo ""

while true; do
    clear
    echo "🔍 TWIN PRIME MINER - MONITOR"
    echo "$(date)"
    echo "=========================================="
    echo ""
    
    # Status do processo
    if ps aux | grep miner_v5_ultra | grep -v grep > /dev/null; then
        CPU=$(ps aux | grep miner_v5_ultra | grep -v grep | awk '{print $3}')
        MEM=$(ps aux | grep miner_v5_ultra | grep -v grep | awk '{print $4}')
        TIME=$(ps aux | grep miner_v5_ultra | grep -v grep | awk '{print $10}')
        echo "✅ PROCESSO ATIVO"
        echo "   CPU: ${CPU}% | MEM: ${MEM}% | Tempo: ${TIME}"
    else
        echo "❌ PROCESSO NÃO ENCONTRADO"
    fi
    echo ""
    
    # Estatísticas do banco
    mysql -u prime_miner -p${DB_PASS} twin_primes_db -N -e "
        SELECT CONCAT('📊 TOTAL: ', FORMAT(COUNT(*), 0), ' primos gêmeos') FROM twin_primes;
    " 2>/dev/null
    
    mysql -u prime_miner -p${DB_PASS} twin_primes_db -N -e "
        SELECT CONCAT('⚡ Últimos 60s: ', FORMAT(COUNT(*), 0), ' primos (', 
               FORMAT(COUNT(*)/60, 1), '/s)') 
        FROM twin_primes 
        WHERE discovered_at > DATE_SUB(NOW(), INTERVAL 60 SECOND);
    " 2>/dev/null
    
    mysql -u prime_miner -p${DB_PASS} twin_primes_db -N -e "
        SELECT CONCAT('🕐 Última inserção: ', 
               TIMESTAMPDIFF(SECOND, MAX(discovered_at), NOW()), 's atrás')
        FROM twin_primes;
    " 2>/dev/null
    
    echo ""
    echo "📈 RANGE ATUAL:"
    mysql -u prime_miner -p${DB_PASS} twin_primes_db -N -e "
        SELECT CONCAT('   Min: ', FORMAT(MIN(p), 0)) FROM twin_primes;
        SELECT CONCAT('   Max: ', FORMAT(MAX(p), 0)) FROM twin_primes;
    " 2>/dev/null
    
    echo ""
    echo "🔢 TOP 5 K_REAL:"
    mysql -u prime_miner -p${DB_PASS} twin_primes_db -t -e "
        SELECT k_real as K, FORMAT(COUNT(*), 0) as Count, 
               CONCAT(FORMAT(100.0*COUNT(*)/(SELECT COUNT(*) FROM twin_primes), 2), '%') as Percent
        FROM twin_primes 
        GROUP BY k_real 
        ORDER BY k_real 
        LIMIT 5;
    " 2>/dev/null
    
    echo ""
    echo "Pressione Ctrl+C para sair | Atualiza a cada 5s"
    sleep 5
done
