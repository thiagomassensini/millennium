#!/bin/bash
# Monitor em tempo real do Twin Prime Miner (versão CSV)

CSV_FILE="results.csv"
LOG_FILE="miner_csv.log"

echo "🔍 TWIN PRIME MINER CSV - MONITOR EM TEMPO REAL"
echo "================================================"
echo ""

# Pegar timestamp da primeira medição do CSV
FIRST_MEASURE_TIME=$(date +%s)
FIRST_COUNT=0
if [ -f "$CSV_FILE" ]; then
    FIRST_COUNT=$(wc -l < "$CSV_FILE")
fi

while true; do
    clear
    echo "🔍 TWIN PRIME MINER CSV - MONITOR"
    echo "$(date)"
    echo "================================================"
    echo ""
    
    # Status do processo
    if ps aux | grep miner_csv | grep -v grep | grep -v monitor > /dev/null; then
        CPU=$(ps aux | grep miner_csv | grep -v grep | grep -v monitor | awk '{print $3}')
        MEM=$(ps aux | grep miner_csv | grep -v grep | grep -v monitor | awk '{print $4}')
        MEM_KB=$(ps aux | grep miner_csv | grep -v grep | grep -v monitor | awk '{print $6}')
        ELAPSED=$(ps -p $(pgrep miner_csv | head -1) -o etime= 2>/dev/null | xargs)
        echo "✅ PROCESSO ATIVO"
        echo "   CPU: ${CPU}% ($(echo "scale=1; $CPU/56" | bc) núcleos)"
        echo "   MEM: ${MEM}% (~$((MEM_KB/1024)) MB)"
        echo "   Tempo rodando: ${ELAPSED}"
    else
        echo "❌ PROCESSO NÃO ENCONTRADO"
    fi
    echo ""
    
    # Estatísticas do arquivo CSV
    if [ -f "$CSV_FILE" ]; then
        CURRENT_COUNT=$(wc -l < "$CSV_FILE")
        CURRENT_TIME=$(date +%s)
        
        # Taxa instantânea (últimos 3 segundos)
        if [ ! -z "$LAST_COUNT" ] && [ ! -z "$LAST_TIME" ]; then
            ELAPSED=$((CURRENT_TIME - LAST_TIME))
            if [ $ELAPSED -gt 0 ]; then
                DELTA=$((CURRENT_COUNT - LAST_COUNT))
                RATE=$(echo "scale=1; $DELTA / $ELAPSED" | bc)
            else
                RATE="0.0"
            fi
        else
            RATE="calculando..."
        fi
        
        # Taxa média desde o início do monitor
        TOTAL_ELAPSED=$((CURRENT_TIME - FIRST_MEASURE_TIME))
        if [ $TOTAL_ELAPSED -gt 0 ]; then
            TOTAL_DELTA=$((CURRENT_COUNT - FIRST_COUNT))
            AVG_RATE=$(echo "scale=1; $TOTAL_DELTA / $TOTAL_ELAPSED" | bc)
        else
            AVG_RATE="0.0"
        fi
        
        echo "📊 ESTATÍSTICAS"
        echo "   Total de linhas: $(printf "%'d" $CURRENT_COUNT) (incluindo header)"
        echo "   Primos encontrados: $(printf "%'d" $((CURRENT_COUNT - 1)))"
        echo "   Taxa instantânea: ${RATE} primos/s"
        echo "   Taxa média (monitor): ${AVG_RATE} primos/s"
        echo "   Tamanho do arquivo: $(du -h "$CSV_FILE" | cut -f1)"
        
        LAST_COUNT=$CURRENT_COUNT
        LAST_TIME=$CURRENT_TIME
    else
        echo "⏳ Arquivo CSV ainda não criado..."
    fi
    echo ""
    
    # Últimas 5 linhas do arquivo
    if [ -f "$CSV_FILE" ]; then
        echo "📝 ÚLTIMOS 5 PRIMOS ENCONTRADOS:"
        tail -5 "$CSV_FILE" | awk -F',' '{printf "   p=%s, k=%s\n", $1, $3}'
    fi
    echo ""
    
    # Log recente
    if [ -f "$LOG_FILE" ]; then
        echo "📋 LOG RECENTE:"
        tail -3 "$LOG_FILE" | sed 's/^/   /'
    fi
    
    echo ""
    echo "================================================"
    echo "Pressione Ctrl+C para sair | Atualiza a cada 3s"
    sleep 3
done
