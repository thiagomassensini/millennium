# Tabelas dos Resultados

Este diretório contém todas as tabelas numéricas geradas pelas análises da Teoria da Relacionalidade Geral.

## Estrutura de Organização

### Por Categoria
```
tabelas/
├── constantes/          # Valores das constantes fundamentais
├── predicoes/           # Predições teóricas numéricas
├── comparacoes/         # Comparações com dados experimentais
├── parametros/          # Parâmetros ajustados
└── estatisticas/        # Análises estatísticas
```

## Tipos de Tabelas

### 1. Constantes Fundamentais
- **`constantes_fundamentais.csv`**: Todas as constantes usadas
- **`alpha_grav_valores.csv`**: Valores de α_grav com incertezas
- **`f_cosmos_calculado.csv`**: Frequência cósmica por diferentes métodos
- **`snr_coeficiente.csv`**: Coeficiente universal 0.05 e variações

### 2. Predições Teóricas
- **`predicoes_particulas.csv`**: Correções em física de partículas
- **`predicoes_ligo.csv`**: Modulações esperadas em ondas gravitacionais
- **`predicoes_sismologia.csv`**: Efeitos sísmicos esperados
- **`predicoes_financas.csv`**: SNR esperado em mercados
- **`predicoes_biologia.csv`**: Frequências biológicas características

### 3. Resultados Experimentais
- **`resultados_hidrogenio.csv`**: Espectroscopia do hidrogênio
- **`resultados_ligo_eventos.csv`**: Análise de eventos LIGO
- **`resultados_snr_sistemas.csv`**: SNR medido em diferentes sistemas
- **`resultados_correlacoes.csv`**: Correlações encontradas

### 4. Análises Estatísticas
- **`chi_quadrado_tests.csv`**: Testes χ² para diferentes modelos
- **`p_values_significancia.csv`**: Valores p para descobertas
- **`intervalos_confianca.csv`**: Intervalos de confiança dos parâmetros
- **`limites_superiores.csv`**: Cotas experimentais

## Formato das Tabelas

### Estrutura Padrão CSV
```csv
parameter,value,uncertainty,units,method,reference,notes
alpha_grav,8.09e-45,1.8e-47,dimensionless,theoretical,fundamentos.md,CODATA 2018 constants
f_cosmos,1.85e43,4.1e40,Hz,theoretical,f_cosmos.py,From Planck mass
```

### Metadados
Cada tabela inclui:
- **Header**: Descrição das colunas
- **Units**: Unidades físicas
- **Uncertainty**: Incertezas estatísticas e sistemáticas
- **Method**: Método de cálculo ou medição
- **Reference**: Fonte dos dados
- **Timestamp**: Data de geração

## Scripts de Geração

### Automáticos
```python
from codigo.constantes import *
from codigo.alpha_grav import AlphaGravCalculator
from codigo.snr_universal import SNRUniversal

# Gerar tabela de constantes
generate_constants_table()

# Gerar tabela de predições
calc = AlphaGravCalculator()
calc.generate_predictions_table()
```

### Manuais
- **`create_experimental_table.py`**: Compilar dados experimentais
- **`update_literature_values.py`**: Atualizar valores da literatura
- **`cross_reference_check.py`**: Verificar consistência

## Tabelas Principais

### 1. Resumo Executivo
**`resumo_resultados.csv`**
| Parâmetro | Valor Teórico | Valor Experimental | Desvio | Significância |
|-----------|---------------|-------------------|---------|---------------|
| α_grav | 8.09×10⁻⁴⁵ | - | - | Predição |
| f_cosmos | 1.85×10⁴³ Hz | - | - | Predição |
| SNR_coeff | 0.050 | 0.048±0.003 | 4% | 3.2σ |

### 2. Validação SNR Universal
**`snr_validation.csv`**
| Sistema | N (DOF) | SNR Teórico | SNR Medido | χ²/DOF |
|---------|---------|-------------|------------|---------|
| Circuito RC | 1 | 0.050 | 0.048±0.005 | 1.2 |
| Oscilador | 2 | 0.071 | 0.069±0.008 | 0.8 |
| Cadeia 10 spins | 10 | 0.158 | 0.155±0.020 | 1.1 |

### 3. Limites Experimentais
**`experimental_limits.csv`**
| Observável | Limite Superior | Confiança | Experimento | Referência |
|------------|----------------|-----------|-------------|------------|
| \|α_grav\| | < 10⁻⁴⁰ | 95% | Espectroscopia H | Futuro |
| Modulação GW | < 10⁻²¹ | 90% | LIGO O3 | Em análise |
| Violação Lorentz | < 10⁻¹⁶ | 95% | Relógios atômicos | Atual |

## Controle de Qualidade

### Verificações Automáticas
1. **Consistência dimensional**: Verificar unidades
2. **Propagação de incertezas**: Calcular erros combinados
3. **Cross-validation**: Comparar métodos independentes
4. **Literature check**: Comparar com valores publicados

### Auditoria Manual
- Revisar cálculos críticos
- Verificar referências
- Validar métodos estatísticos
- Confirmar significância das descobertas

## Formatos de Export

### Para Publicação
- **LaTeX**: Tabelas formatadas para artigos
- **Excel**: Para colaboradores não-técnicos
- **JSON**: Para aplicações web
- **HDF5**: Para grandes datasets

### Para Análise
- **Pandas DataFrame**: Análise em Python
- **R DataFrame**: Análise estatística em R
- **MATLAB**: Processamento numérico

## Versionamento

### Convenção de Nomes
```
[categoria]_[descrição]_v[versão]_[data].csv
```

Exemplo: `constantes_fundamentais_v2.1_20251029.csv`

### Changelog
Cada tabela mantém histórico de mudanças:
- Versão 1.0: Implementação inicial
- Versão 2.0: Adição de incertezas
- Versão 2.1: Correção de bug na propagação de erros

## Status das Tabelas

### Completas
- [x] Constantes fundamentais
- [x] Predições teóricas básicas
- [x] Análise de α_grav
- [x] Cálculos de f_cosmos

### Em Progresso
- [ ] Dados experimentais de validação
- [ ] Análise estatística completa
- [ ] Limites experimentais atualizados
- [ ] Cross-referências da literatura

### Planejadas
- [ ] Meta-análise de todos os sistemas
- [ ] Tabela de descobertas vs null results
- [ ] Comparação com teorias alternativas
- [ ] Cronograma de experimentos futuros

## Scripts Utilitários

### `table_generator.py`
Gera todas as tabelas automaticamente

### `format_for_latex.py`
Converte CSV para formato LaTeX

### `uncertainty_propagation.py`
Calcula propagação de incertezas

### `significance_calculator.py`
Calcula significância estatística

### `literature_comparison.py`
Compara com valores da literatura