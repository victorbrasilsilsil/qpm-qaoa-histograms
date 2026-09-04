# QPM para dados histograma-valorados (SDA 2026)

Código do resumo "Beyond interval bounds: quantile-parametrized regression
for histogram data from quantum circuits" (Brasil, Duarte e Souza, SDA 2026).
Estende o Parametrized Method de Souza et al. (2017, *Knowledge-Based
Systems*) de dados intervalares para dados histograma-valorados, e aplica a
saída de circuitos QAOA como estudo de caso.

## Ideia

O PM escreve qualquer ponto de um intervalo como `p(λ) = x_lo*(1-λ) + x_hi*λ`
e deixa o próprio ajuste de mínimos quadrados encontrar o melhor λ. Isso é o
quantil λ da uniforme em `[x_lo, x_hi]`. Trocando a uniforme por uma função
quantil de verdade e avaliando em L níveis em vez de 2, o mesmo truque
funciona para histogramas: PM volta a aparecer como o caso `L=2`.

## Módulos

| Arquivo             | Papel                                                     |
| -------------------- | ---------------------------------------------------------- |
| `graphs.py`          | grafos cúbicos aleatórios, não-isomorfos; valores de corte |
| `qaoa.py`             | simulador QAOA statevector (NumPy puro); COBYLA multi-start |
| `symbolic.py`         | distribuição do circuito → histograma → função quantil     |
| `qpm.py`              | o método do artigo: regressão por níveis de referência, rearranjo monótono, W2 |
| `baselines.py`        | CM, PM (Souza et al. 2017) e Wasserstein LS (Irpino & Verde 2015) |
| `run_experiment.py`   | roda a simulação QAOA e monta o dataset (`data.npz`)        |
| `analyze.py`          | ajusta os modelos, valida por leave-one-out, checa coerência |
| `stats.py`            | testes de Wilcoxon pareados entre os modelos                |
| `plots.py`            | gera a Figura 1 do artigo                                   |

## Setup

```
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Rodando do zero

A simulação inteira (60 grafos, n até 14) passa de uma hora numa máquina
comum, então `run_experiment.py` roda um tamanho de grafo por vez:

```
python run_experiment.py 8  5
python run_experiment.py 10 15
python run_experiment.py 12 20
python run_experiment.py 14 10
python run_experiment.py 14 20 --offset 10 --tag b   # segunda metade
```

Cada chamada grava um `chunk_<n>[tag].npz`. Depois:

```python
from run_experiment import merge_chunks
merge_chunks(["chunk_8.npz", "chunk_10.npz", "chunk_12.npz",
              "chunk_14.npz", "chunk_14b.npz"])
```

monta o `data.npz` completo. Com ele:

```
python analyze.py   # ajusta os modelos -> results.json, results_in.json, fit*.npz, curves.npz
python stats.py      # Wilcoxon pareado -> confere os p-valores e win rates do artigo
python plots.py      # figs/fig1_qpm.png
```

## Reprodutibilidade

Seeds fixas por tamanho de grafo (`generate_graphs`) e por otimização
(`run_size`), então o benchmark e os números batem entre re-execuções. Os
resultados crus (`data.npz`, `results*.json`, `fit*.npz`, `curves.npz`) já
estão commitados — não é preciso rodar a simulação de novo só para conferir
os números do artigo; `analyze.py`/`stats.py`/`plots.py` já leem o `data.npz`
salvo.

## Dado degenerado, de propósito

`Q_in` (espectro de cortes, entrada do circuito) e `Q_p2` (saída em p=2)
compartilham exatamente os mesmos limites — a string toda-zeros e uma
ótima estão sempre no suporte, então `x_hi = 1 - x_lo` nos dois lados.
`analyze.py` confere isso (`input_hi_check`, `output_hi_check`, ambos 0) e
é o ponto central do artigo: um modelo intervalar não distingue entrada de
saída; toda a informação está na forma do histograma, não nos limites.
