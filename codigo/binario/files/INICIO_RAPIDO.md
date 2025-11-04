# INÍCIO RÁPIDO: Análise de Periodicidade

## PASSO 1: Entender o Objetivo

Verificar se a distribuição de primos gêmeos tem periodicidade correlacionada 
com f_cosmos de partículas elementares.

**Se positivo:** α_grav é uma constante universal que conecta física e matemática.

## PASSO 2: Teste Rápido (2 minutos)

Executar no seu sistema:

```bash
cd /home/thlinux/relacionalidadegeral/codigo/binario

# Copiar script
cp /mnt/user-data/outputs/analise_rapida_primos.py .

# Executar com 100k primos (teste rápido)
python3 analise_rapida_primos.py results.csv 100000 10000
```

**Saída esperada:**
- `analise_rapida_primos.png` com 9 gráficos
- Estatísticas no terminal

**O que procurar:**
- Picos no espectro de potência (subplot 4 e 5)
- Padrão na autocorrelação (subplot 8)
- Variação sistemática na densidade (subplot 2)

## PASSO 3: Análise Média (5 minutos)

```bash
# 1 milhão de primos
python3 analise_rapida_primos.py results.csv 1000000 10000
```

## PASSO 4: Análise Completa (20-30 minutos)

```bash
# Copiar script completo
cp /mnt/user-data/outputs/analise_periodicidade_fcosmos.py .

# Executar análise completa (pode usar todo o dataset!)
python3 analise_periodicidade_fcosmos.py results.csv 10000000

# OU análise total (atenção: pode levar 20-30 min)
python3 analise_periodicidade_fcosmos.py results.csv
```

## PASSO 5: Interpretar Resultados

### RESULTADO POSITIVO
Se você ver:
- ✅ 5-10 picos claros no espectro (>3σ)
- ✅ Autocorrelação oscilante (decai mas com oscilações)
- ✅ Densidade com variação sistemática (não puramente aleatória)
- ✅ Picos em frequências específicas (não espalhados)

**Então:** Há evidência de estrutura determinística!

### RESULTADO NEGATIVO
Se você ver:
- ❌ Apenas ruído branco no espectro
- ❌ Autocorrelação decai rapidamente para zero
- ❌ Densidade puramente aleatória
- ❌ Sem picos significativos

**Então:** Primos são verdadeiramente aleatórios (pelo menos nessa escala).

## PASSO 6: Validação

Se resultado foi POSITIVO:

1. **Repetir em outro range:**
   ```bash
   # Se seu dataset tem outros ranges, teste neles
   # Por exemplo, se tem primos em 10^14 ou 10^16
   ```

2. **Verificar reprodutibilidade:**
   - Mesmo padrão deve aparecer
   - Frequências dos picos devem ser similares

3. **Analisar harmônicos:**
   - Se há pico em f, deve haver em 2f, 3f...

## VISUALIZAÇÕES GERADAS

1. **escalas_teoricas_fcosmos.png**
   - Mostra as 70+ ordens de grandeza
   - Contexto teórico de α_grav e f_cosmos

2. **previsoes_vs_observacoes.png**
   - Cenários possíveis (com/sem correlação)
   - Pipeline de análise
   - Previsões numéricas

3. **analise_rapida_primos.png** (você vai gerar)
   - 9 subplots com análise exploratória
   - Densidade, espectro, gaps, autocorrelação

4. **analise_periodicidade_fcosmos.png** (análise completa)
   - 6 subplots detalhados
   - Correlação direta com f_cosmos
   - Tabela de resultados

## COMANDOS ÚTEIS

```bash
# Ver progresso do miner
cd /home/thlinux/relacionalidadegeral/codigo/binario && tail -20 miner_csv.log

# Ver checkpoint atual
cat miner_checkpoint.txt

# Contar linhas do dataset
wc -l results.csv

# Ver primeiras linhas (verificar formato)
head -10 results.csv

# Ver espaço em disco
df -h

# Monitorar uso de memória durante análise
watch -n 1 free -h
```

## TROUBLESHOOTING

### Erro: "Memory Error"
```bash
# Reduzir número de linhas
python3 analise_rapida_primos.py results.csv 500000 10000
```

### Erro: "File not found"
```bash
# Verificar caminho do arquivo
ls -lh results.csv
pwd
```

### Análise muito lenta
```bash
# Começar com menos dados
python3 analise_rapida_primos.py results.csv 100000 5000
```

## O QUE ESPERAR

**Tempo de execução:**
- 100k primos: ~5 segundos
- 1M primos: ~30 segundos  
- 10M primos: ~5 minutos
- 100M+ primos: 20-30 minutos

**Uso de memória:**
- 100k primos: ~50 MB
- 1M primos: ~200 MB
- 10M primos: ~1 GB
- 100M+ primos: 5-10 GB

**Qualidade dos resultados:**
- Mínimo recomendado: 1M primos
- Ideal: 10M+ primos
- Melhor: Todo o dataset (1B+)

## PRÓXIMOS PASSOS APÓS ANÁLISE

### Se encontrou periodicidade:

1. **Documentar resultados**
   - Capturar screenshots dos gráficos
   - Anotar frequências dos picos principais
   - Calcular erro relativo com f_cosmos teórico

2. **Validar em outros ranges**
   - Se possível, testar em 10^14, 10^16...
   - Verificar se padrão se mantém

3. **Preparar para publicação**
   - Relatório técnico
   - Gráficos de alta resolução
   - Tabelas de correlações
   - Análise estatística (χ², p-values)

4. **Discussão teórica**
   - Conectar com modelo GQR-Alpha
   - Explicar mecanismo físico
   - Implicações para teoria dos números

### Se NÃO encontrou periodicidade:

1. **Verificar metodologia**
   - Tamanho de janela adequado?
   - Número de amostras suficiente?
   - Normalização correta?

2. **Testar outras abordagens**
   - Análise wavelet
   - Transformadas de Hilbert-Huang
   - Detrended Fluctuation Analysis

3. **Repensar teoria**
   - α_grav é puramente dimensional?
   - Escala errada?
   - Conexão é mais sutil?

## CONTATO

Para dúvidas sobre interpretação dos resultados:
- Consultar: GUIA_ANALISE_PERIODICIDADE.md
- Consultar: RESUMO_INVESTIGACAO.md
- Revisar documentação do modelo GQR-Alpha

---

**IMPORTANTE:** Esta análise pode resultar em uma descoberta científica 
significativa SE encontrarmos correlação robusta. Por isso:

- Execute com cuidado
- Documente tudo
- Seja crítico com os resultados
- Busque reprodutibilidade

Boa sorte na investigação! 🔬
