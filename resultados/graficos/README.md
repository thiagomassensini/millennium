# Gráficos dos Resultados

Este diretório contém todas as visualizações geradas pelos análises da Teoria da Relacionalidade Geral.

## Estrutura de Organização

### Por Módulo Teórico
```
graficos/
├── alpha_grav/          # Gráficos relacionados a α_grav
├── f_cosmos/            # Gráficos da frequência cósmica
├── snr_universal/       # Visualizações do SNR universal
├── processo_ou/         # Gráficos do processo OU modificado
└── comparacoes/         # Comparações entre diferentes teorias
```

### Por Domínio de Validação
```
graficos/
├── particulas/          # Física de partículas
├── ligo/               # Ondas gravitacionais
├── sismologia/         # Dados sísmicos
├── financas/           # Mercados financeiros
├── biologia/           # Sistemas biológicos
└── cosmologia/         # Dados cosmológicos
```

## Tipos de Gráficos

### 1. Gráficos Teóricos
- **Espectros de potência**: Densidade espectral com picos em f_cosmos
- **Funções de correlação**: Correlações modificadas por α_grav
- **Superfícies paramétricas**: Variação de predições com parâmetros
- **Diagramas de fase**: Estados do sistema em função de α_grav e f_cosmos

### 2. Comparações Experimentais
- **Dados vs Teoria**: Sobreposição de dados experimentais e predições
- **Residuais**: Diferenças entre dados e modelo padrão
- **Significância estatística**: P-values e intervalos de confiança
- **Limites superiores**: Cotas experimentais nos parâmetros

### 3. Análises de Validação
- **SNR vs √N**: Validação da lei universal
- **Modulações temporais**: Sinais modulados por f_cosmos
- **Cross-correlações**: Correlações entre diferentes sistemas
- **Análise espectral**: FFT e análise de frequências

## Convenções de Nomenclatura

### Formato dos Arquivos
```
[modulo]_[tipo]_[data].[extensao]
```

Exemplos:
- `alpha_grav_comparacao_20251029.png`
- `snr_universal_validacao_20251029.pdf`
- `f_cosmos_espectro_20251029.svg`

### Resoluções
- **PNG**: 300 DPI para publicações
- **PDF**: Gráficos vetoriais para artigos
- **SVG**: Gráficos interativos para web
- **EPS**: Compatibilidade com LaTeX

## Scripts de Geração

Cada módulo Python possui métodos para gerar gráficos:

```python
# Exemplo de uso
from codigo.alpha_grav import AlphaGravCalculator
calc = AlphaGravCalculator()
calc.grafico_comparacao_constantes(salvar=True)
```

## Controle de Versão

- Todos os gráficos são versionados com timestamp
- Scripts de geração são mantidos em `codigo/`
- Configurações de plot em `config/plot_settings.json`

## Status dos Gráficos

### Gerados Automaticamente
- [x] Comparação de constantes de acoplamento
- [x] Correções gravitacionais vs energia
- [x] Universalidade do SNR
- [x] Análise de scaling do SNR
- [x] Espectro cósmico
- [x] Escalas temporais fundamentais
- [x] Ressonâncias gravitacionais

### Pendentes
- [ ] Validação experimental de α_grav
- [ ] Dados do LIGO vs predições
- [ ] Análise sísmica global
- [ ] SNR em mercados financeiros
- [ ] Sinais biológicos vs f_cosmos
- [ ] Comparação com modelos alternativos