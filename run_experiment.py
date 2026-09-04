"""Gera o benchmark: para cada grafo, o espectro de cortes (histograma de
entrada), a saída do QAOA em p=1 e p=2 (histogramas), tudo como funções
quantil na grade TG. Roda um tamanho de grafo por vez -- em CPU comum, o
conjunto inteiro (60 grafos, n até 14) passa de uma hora, então convém
paralelizar por tamanho em vez de rodar tudo numa chamada só.

Uso:
    python run_experiment.py 8  5            # n=8, 5 grafos
    python run_experiment.py 14 20 --offset 10 --tag b   # segunda metade de n=14
"""
import argparse
import time
import numpy as np

from graphs import generate_graphs, cut_values
from qaoa import qaoa_state, optimize
from symbolic import TG, circuit_to_quantile

SIZES_COUNTS = [(8, 5), (10, 15), (12, 20), (14, 20)]  # o benchmark completo do artigo


def run_size(n, count, offset=0, tag=""):
    rng = np.random.default_rng(4242 + n + 97 * offset)
    graphs = generate_graphs([(n, count)])[offset:]
    print(f"n={n}: processando {len(graphs)} grafos")

    Q_in, Q_p1, Q_p2, r1s, r2s = [], [], [], [], []
    t0 = time.time()
    for k, (_, g) in enumerate(graphs):
        cv = cut_values(g, n)
        cmax = float(cv.max())
        probs_unif = np.full(1 << n, 1.0 / (1 << n))

        th1, v1 = optimize(cv, n, 1, n_starts=8, maxiter=110, rng=rng)
        th2, v2 = optimize(cv, n, 2, n_starts=16, maxiter=220, rng=rng)
        p1 = np.abs(qaoa_state(cv, n, th1[:1], th1[1:])) ** 2
        p2 = np.abs(qaoa_state(cv, n, th2[:2], th2[2:])) ** 2

        Q_in.append(circuit_to_quantile(cv, probs_unif, cmax))
        Q_p1.append(circuit_to_quantile(cv, p1, cmax))
        Q_p2.append(circuit_to_quantile(cv, p2, cmax))
        r1s.append(v1 / cmax)
        r2s.append(v2 / cmax)
        print(f"  {k+1}/{len(graphs)}  r1={v1/cmax:.4f}  r2={v2/cmax:.4f}  "
              f"[{time.time()-t0:.0f}s]")

    out = f"chunk_{n}{tag}.npz"
    np.savez(out, TG=TG, Q_in=np.array(Q_in), Q_p1=np.array(Q_p1),
             Q_p2=np.array(Q_p2), r1=np.array(r1s), r2=np.array(r2s),
             n=np.full(len(graphs), n))
    print(f"salvo em {out} ({time.time()-t0:.0f}s)")


def merge_chunks(paths, out="data.npz"):
    """Concatena os chunk_*.npz gerados por run_size em um único data.npz."""
    parts = [np.load(p) for p in paths]
    merged = {k: np.concatenate([p[k] for p in parts])
              for k in ["Q_in", "Q_p1", "Q_p2", "r1", "r2", "n"]}
    merged["TG"] = parts[0]["TG"]
    np.savez(out, **merged)
    print(f"{out}: {len(merged['r1'])} grafos, tamanhos "
          f"{dict(zip(*np.unique(merged['n'], return_counts=True)))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("count", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run_size(args.n, args.count, args.offset, args.tag)
