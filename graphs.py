"""Instâncias do benchmark: grafos cúbicos aleatórios, não-isomorfos, conexos."""
import numpy as np
import networkx as nx


def generate_graphs(sizes_counts, seed_offset=1000, max_tries=20000):
    """sizes_counts: lista de (n, quantidade). Retorna [(n, G), ...]."""
    out = []
    for n, count in sizes_counts:
        rng = np.random.default_rng(seed_offset + n)
        accepted, tries = [], 0
        while len(accepted) < count and tries < max_tries:
            tries += 1
            seed = int(rng.integers(0, 2**31 - 1))
            g = nx.random_regular_graph(3, n, seed=seed)
            if not nx.is_connected(g):
                continue
            if any(nx.is_isomorphic(g, h) for h in accepted):
                continue
            accepted.append(g)
        print(f"n={n}: {len(accepted)} grafos ({tries} tentativas)")
        out.extend((n, g) for g in accepted)
    return out


def cut_values(g, n):
    """Valor do corte de cada uma das 2^n strings, por enumeração completa."""
    z = np.arange(1 << n, dtype=np.int64)
    bits = ((z[:, None] >> np.arange(n)) & 1).astype(np.int8)
    c = np.zeros(1 << n, dtype=np.float64)
    for i, j in g.edges():
        c += bits[:, i] ^ bits[:, j]
    return c
