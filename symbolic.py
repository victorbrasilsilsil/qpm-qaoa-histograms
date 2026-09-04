"""Conversão da saída do circuito (uma distribuição sobre bitstrings) em dado
histograma-valorado, na forma de função quantil amostrada numa grade fixa."""
import numpy as np

TG = np.linspace(0.0, 1.0, 201)  # grade de níveis t usada no artigo todo


def bin_distribution(cvals, probs, cmax):
    """Agrega a massa de probabilidade nos valores inteiros de corte e
    devolve as bordas dos bins (normalizadas por cmax) e o peso de cada um.
    Dentro de um bin a massa é tratada como uniforme -- convenção usual para
    dado histograma-valorado."""
    kmax = int(round(cmax))
    w = np.bincount(cvals.astype(int), weights=probs, minlength=kmax + 1)
    edges = (np.arange(kmax + 2) - 0.5) / cmax
    return edges, w


def quantile_fn(edges, w, tgrid=TG):
    w = w + 1e-14
    w = w / w.sum()
    cum = np.concatenate([[0.0], np.cumsum(w)])
    cum[-1] = 1.0
    return np.interp(tgrid, cum, edges)


def circuit_to_quantile(cvals, probs, cmax, tgrid=TG):
    return quantile_fn(*bin_distribution(cvals, probs, cmax), tgrid)
