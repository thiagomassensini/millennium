# Validação com Dados Sismológicos

## Objetivo
Este diretório contém dados e análises para validar as predições da Teoria da Relacionalidade Geral usando dados sísmicos e geofísicos globais.

## Dados Esperados

### 1. Dados Sísmicos Globais
- **Fonte**: IRIS (Incorporated Research Institutions for Seismology)
- **Formato**: SEED/miniSEED
- **Cobertura**: Rede global GSN (Global Seismographic Network)
- **Taxa de amostragem**: 1-100 Hz
- **Período**: 1990-presente

### 2. Catálogos de Terremotos
- **Arquivo**: `global_earthquake_catalog.csv`
- **Fonte**: USGS NEIC, ISC
- **Conteúdo**: Magnitude, localização, tempo, profundidade
- **Predição**: Correlação com fase de f_cosmos

### 3. Dados de Marés Terrestres
- **Arquivo**: `earth_tides_data.dat`
- **Fonte**: Superconducting Gravimeters (SG)
- **Predição**: Modulação ultra-rápida com f_cosmos

### 4. Campo Magnético Terrestre
- **Arquivo**: `geomagnetic_observatories.csv`
- **Fonte**: INTERMAGNET
- **Predição**: B(t) = B₀ + B₁ * cos(f_cosmos * t + φ)

## Predições Específicas

### 1. Terremotos e f_cosmos
```python
# Probabilidade de terremoto vs fase cósmica
P(earthquake|phase) = P0 * [1 + correlation * cos(phase)]
```

### 2. Rotação da Terra
```python
# Modulação ultrarrápida na duração do dia
delta_T_day = 10^-21 * cos(f_cosmos * t) segundos
```

### 3. Oscillações Sísmicas Livres
```python
# Modificação nas frequências dos modos normais
f_modified = f_PREM * [1 + alpha_grav * correction_factor]
```

## Scripts de Análise

### `correlacao_terremotos.py`
Analisa correlação temporal entre grandes terremotos e f_cosmos.

### `analise_modos_normais.py`
Busca modificações nas frequências dos modos normais da Terra.

### `tides_analysis.py`
Processa dados de marés terrestres buscando modulações de alta frequência.

### `geomagnetic_oscillations.py`
Analisa oscilações de alta frequência no campo magnético.

### `earth_rotation_variations.py`
Analisa variações ultrarrápidas na rotação terrestre.

## Estrutura de Dados

```
sismologia/
├── waveforms/           # Formas de onda sísmicas brutas
├── catalogs/           # Catálogos de eventos sísmicos
├── tides/              # Dados de marés terrestres
├── magnetic/           # Dados do campo magnético
├── rotation/           # Dados de rotação da Terra
├── stations/           # Informações das estações
├── noise/              # Caracterização de ruído sísmico
└── processed/          # Dados processados e filtrados
```

## Redes e Estações Prioritárias

### Global Seismographic Network (GSN)
- **Estações**: ~150 estações globalmente distribuídas
- **Instrumentos**: Broadband seismometers (STS-1, STS-2)  
- **Banda**: 0.00278 Hz - 50 Hz (360s - 0.02s)

### Superconducting Gravimeters
- **Locais**: Membach, Strasbourg, Cantley, etc.
- **Sensibilidade**: 10⁻¹² m/s²
- **Banda**: DC - 1 Hz

### INTERMAGNET Observatory Network
- **Estações**: ~130 observatórios magnéticos
- **Resolução**: 1 nT
- **Taxa**: 1 Hz - 1 min

## Metodologia de Análise

### 1. Análise Espectral
- **Transformada de Fourier**: Busca picos em f_cosmos
- **Wavelets**: Análise tempo-frequência
- **Multitaper**: Estimação espectral robusta

### 2. Análise de Correlação
- **Cross-correlation**: Entre diferentes estações
- **Phase coherence**: Coerência de fase global
- **Stacking**: Empilhamento de eventos

### 3. Análise Estatística
- **Bootstrap**: Estimação de incertezas
- **Teste de permutação**: Significância estatística
- **False Discovery Rate**: Correção para múltiplos testes

## Desafios Técnicos

### 1. Ruído Sísmico
- **Microseismic noise**: 0.1-1 Hz (oceanos)
- **Cultural noise**: >1 Hz (atividade humana)
- **Instrumental noise**: Limita alta frequência

### 2. Resolução Temporal
- **f_cosmos ~ 10⁴³ Hz**: Muito além da banda sísmica
- **Subharmônicos**: Busca em f_cosmos/N
- **Modulação de amplitude**: Efeitos indiretos

### 3. Efeitos Locais vs Globais
- **Geologia local**: Modifica propagação
- **Efeitos atmosféricos**: Contamina dados
- **Atividade humana**: Varia com localização

## Parâmetros de Busca

### Janelas Temporais
- **Análise contínua**: 1990-presente
- **Eventos específicos**: Grandes terremotos (M>7.0)
- **Eclipses**: Períodos de baixo ruído cultural

### Bandas de Frequência
- **Ultra-baixa**: 10⁻⁵ - 10⁻³ Hz (marés, rotação)
- **Baixa**: 10⁻³ - 0.1 Hz (modos normais)
- **Banda larga**: 0.1 - 10 Hz (terremotos locais)

### Critérios de Seleção
- **Magnitude**: M ≥ 6.0 para análise global
- **Profundidade**: < 100 km (eventos crustais)
- **Qualidade**: Gaps < 10% nos dados

## Ferramentas de Software

### Standard
- **ObsPy**: Processamento de dados sísmicos em Python
- **SAC**: Seismic Analysis Code
- **GMT**: Generic Mapping Tools
- **MATLAB Seismic Toolbox**

### Customizadas
- **cosmos_seismic.py**: Busca específica por f_cosmos
- **global_correlation.py**: Correlação entre estações
- **tidal_modulation.py**: Análise de modulação de marés

## Cronograma

### Fase 1: Coleta de Dados (Meses 1-2)
- Download dados IRIS/USGS
- Organização por evento e estação
- Controle de qualidade básico

### Fase 2: Processamento (Meses 3-4)
- Filtragem e deconvolução instrumental
- Alinhamento temporal preciso
- Remoção de ruído e outliers

### Fase 3: Análise (Meses 5-7)
- Análise espectral sistemática
- Correlações temporais e espaciais
- Testes de significância estatística

### Fase 4: Interpretação (Meses 8-9)
- Comparação com predições teóricas
- Estimativa de limites superiores
- Preparação de manuscrito

## Colaborações

### Instituições
- **USGS**: Earthquake Hazards Program
- **IRIS**: Data Management Center
- **GFZ Potsdam**: GEOFON network
- **IPGP**: Global seismology expertise

### Projetos Relacionados
- **Global CMT**: Centroid moment tensor catalog
- **ISC**: International Seismological Centre
- **CTBTO**: Comprehensive Test Ban Treaty Organization

## Limitações e Caveats

1. **Resolução temporal**: f_cosmos está muito além da banda sísmica
2. **Ruído ambiental**: Contamina sinais de baixa amplitude
3. **Heterogeneidade terrestre**: Complica interpretação global
4. **Efeitos não-lineares**: Sistema Terra é complexo e acoplado

## Status dos Dados

- [ ] Catálogo global de terremotos (1990-2023)
- [ ] Formas de onda GSN para eventos M>7
- [ ] Dados de marés de gravimetria supercondutora
- [ ] Séries temporais do campo magnético INTERMAGNET
- [ ] Dados de rotação da Terra (IERS)

## Referências Chave

1. Dziewonski, A.M. & Anderson, D.L. "Preliminary reference Earth model" (1981)
2. Park, J. et al. "Free oscillations excited by the Sumatra earthquake" (2005)
3. Agnew, D.C. "Earth tides" (2007)
4. Lay, T. & Wallace, T.C. "Modern Global Seismology" (1995)
5. Stein, S. & Wysession, M. "An Introduction to Seismology" (2003)