# Validação com Dados LIGO/Virgo - Ondas Gravitacionais

## Objetivo
Este diretório contém dados e análises para validar as predições da Teoria da Relacionalidade Geral usando observações de ondas gravitacionais dos detectores LIGO e Virgo.

## Dados Esperados

### 1. Dados de Strain dos Detectores
- **Arquivos**: `H1_strain_*.hdf5`, `L1_strain_*.hdf5`, `V1_strain_*.hdf5`
- **Conteúdo**: Séries temporais de strain de alta resolução
- **Fonte**: LIGO Open Science Center (LOSC)
- **Formato**: HDF5 com metadata completo
- **Taxa de amostragem**: 16 kHz (análise) e 4 kHz (detecção)

### 2. Catálogos de Eventos
- **Arquivo**: `GWTC-3_catalog.json`
- **Conteúdo**: Parâmetros estimados de eventos confirmados
- **Fonte**: LIGO-Virgo Collaboration
- **Predição**: Modulação de frequência f_GW(t) = f₀ * [1 + A * cos(2π * f_cosmos * t)]

### 3. Fundo Estocástico
- **Arquivo**: `stochastic_background_data.csv`
- **Conteúdo**: Limites e medições do fundo estocástico
- **Predição**: Ω_GW(f) = Ω₀ * (f/f_cosmos)^α * exp(-f/f_cosmos)

### 4. Dados de Calibração
- **Arquivo**: `calibration_factors.txt`
- **Conteúdo**: Fatores de calibração temporal dos detectores
- **Uso**: Correção de deriva instrumental vs predições teóricas

## Predições Específicas para Teste

### 1. Modulação Temporal da Frequência
```python
# Amplitude esperada: A ~ 10^-20 (limite de detecção do LIGO)
f_modulada = f_central * (1 + A * cos(2*pi * f_cosmos * t))
```

### 2. Fundo Estocástico Modificado
```python
# Espectro com pico em f_cosmos/e
Omega_GW = Omega_0 * (f/f_cosmos)**(2/3) * exp(-f/f_cosmos)
```

### 3. Polarização Extra
```python
# Modo escalar com amplitude
h_scalar ~ alpha_grav * h_tensorial
```

## Scripts de Análise

### `modulacao_frequencia.py`
Busca modulações periódicas na frequência de ondas gravitacionais com período T_cosmos.

### `analise_fundo_estocastico.py`
Analisa o espectro do fundo estocástico buscando assinatura de f_cosmos.

### `polarizacao_adicional.py`
Busca evidências de modos de polarização adicionais.

### `correlacao_temporal.py`
Analisa correlações temporais entre detectores com delay de f_cosmos.

### `processamento_eventos.py`
Processa catálogo de eventos buscando padrões sistemáticos.

## Estrutura de Dados

```
ligo/
├── dados_strain/         # Dados de strain H1, L1, V1
├── eventos/             # Dados de eventos individuais (GW150914, etc.)
├── catalogos/           # Catálogos GWTC-1, GWTC-2, GWTC-3
├── calibracao/          # Dados de calibração
├── ruido/               # Caracterização de ruído
├── resultados/          # Resultados das análises
├── templates/           # Templates modificados pela teoria
└── software/            # Scripts e ferramentas customizadas
```

## Metodologia de Análise

### 1. Análise de Modulação
- **Técnica**: Transformada de Fourier de curto prazo (STFT)
- **Busca**: Picos espectrais em f_cosmos
- **Estatística**: Teste Z modificado para detecção

### 2. Análise Espectral
- **Método**: Densidade espectral de potência
- **Comparação**: Modelos ΛCDM vs teoria modificada
- **Bayesian**: Evidência para diferentes modelos

### 3. Correlação Cruzada
- **Entre detectores**: H1-L1, H1-V1, L1-V1
- **Delay esperado**: Baseado em f_cosmos
- **Significância**: > 5σ para descoberta

## Parâmetros de Busca

### Frequência Cósmica
- **Valor central**: f_cosmos = 1.85 × 10⁴³ Hz
- **Incerteza**: ±10% (conservativa)
- **Harmônicos**: 1H, 2H, 3H, ..., 10H

### Amplitude de Modulação
- **Range**: 10⁻²² a 10⁻¹⁸
- **Sensibilidade LIGO**: ~10⁻²¹ para f > 10 Hz
- **Advanced LIGO+**: ~10⁻²² (próxima geração)

### Janelas Temporais
- **O1**: Sep 2015 - Jan 2016
- **O2**: Nov 2016 - Aug 2017  
- **O3a**: Apr 2019 - Oct 2019
- **O3b**: Nov 2019 - Mar 2020

## Ferramentas de Software

### Principais
- **LALSuite**: Biblioteca oficial LIGO
- **PyCBC**: Análise de dados de ondas gravitacionais
- **Bilby**: Inferência bayesiana
- **GWpy**: Manipulação de dados LIGO

### Customizadas
- **cosmos_search.py**: Busca específica por f_cosmos
- **modified_templates.py**: Templates com correções da teoria
- **stochastic_modified.py**: Análise de fundo modificado

## Status dos Dados

- [x] GWTC-1 (11 eventos)
- [x] GWTC-2.1 (47 eventos)  
- [ ] GWTC-3 (90+ eventos)
- [ ] Dados de strain O1/O2/O3
- [ ] Dados de calibração completos
- [ ] Templates modificados pela teoria

## Colaborações e Acesso

### LIGO Scientific Collaboration
- **Acesso**: Membro da colaboração necessário para dados proprietários
- **Público**: LOSC fornece subset dos dados
- **Proposta**: Submeter analysis proposal ao LSC

### Recursos Computacionais
- **Cluster**: Necessário para análise de múltiplos detectores
- **GPU**: Aceleração para matched filtering
- **Storage**: ~TB de dados por run observacional

## Cronograma de Análise

### Fase 1 (Meses 1-3)
- Download e organização dos dados públicos
- Implementação de pipelines básicos
- Teste com eventos conhecidos (GW150914)

### Fase 2 (Meses 4-6)
- Análise sistemática de modulação de frequência
- Busca no fundo estocástico
- Desenvolvimento de templates modificados

### Fase 3 (Meses 7-9)
- Análise bayesiana completa
- Estimativa de limites superiores
- Preparação de publicação

## Referências Chave

1. Abbott, B.P. et al. "Observation of Gravitational Waves from a Binary Black Hole Merger" (2016)
2. Abbott, B.P. et al. "GWTC-1: A Gravitational-Wave Transient Catalog" (2019)
3. Abbott, R. et al. "GWTC-3: Compact Binary Coalescences Observed by LIGO and Virgo" (2021)
4. Cornish, N.J. "Detection strategies for extreme mass ratio inspirals" (2011)
5. Thrane, E. & Romano, J.D. "Sensitivity curves for searches for gravitational-wave backgrounds" (2013)