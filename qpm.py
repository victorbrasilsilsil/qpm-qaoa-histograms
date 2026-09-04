"""QPM: regressão da Eq. (1) do artigo. Para cada nível de resposta t, ajusta
um modelo linear nos quantis do regressor avaliados em L níveis de referência.
Generaliza o PM (Souza et al. 2017): PM é o caso L=2 com s=(0,1)."""
import numpy as np


def knots(Q, L, tgrid):
    """L níveis de referência equiespaçados em [0,1], incluindo os extremos."""
    s = np.linspace(0.0, 1.0, L)
    idx = np.round(s * (len(tgrid) - 1)).astype(int)
    return Q[:, idx], s


def loo_fit(Z, Y, ridge=0.0):
    """Predição leave-one-out: para cada unidade, ajusta em todas as outras
    e prediz a que ficou de fora. Z e Y têm uma linha por grafo."""
    n = Z.shape[0]
    P = np.zeros_like(Y, dtype=float)
    idx = np.arange(n)
    for i in range(n):
        tr = idx != i
        Zt, Yt = Z[tr], Y[tr]
        mu, sd = Zt.mean(0), Zt.std(0)
        sd[sd < 1e-12] = 1.0
        A = np.column_stack([np.ones(tr.sum()), (Zt - mu) / sd])
        G = A.T @ A + ridge * np.eye(A.shape[1])
        G[0, 0] -= ridge  # não penaliza o intercepto
        try:
            B = np.linalg.solve(G, A.T @ Yt)
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(A, Yt, rcond=None)[0]
        P[i] = np.concatenate([[1.0], (Z[i] - mu) / sd]) @ B
    return P


def rearrange(P):
    """Rearranjo monótono (Chernozhukov, Fernández-Val e Galichon, 2010):
    ordenar cada curva prevista nunca aumenta o erro L2 contra uma função
    quantil verdadeira, e resolve qualquer violação de coerência."""
    return np.sort(P, axis=1)


def wasserstein2(P, Y, tgrid):
    """Distância W2 entre cada par de funções quantil previstas/observadas."""
    return np.sqrt(np.trapezoid((P - Y) ** 2, tgrid, axis=1))


def hat_matrix(Z, ridge=0.0):
    """Matriz H usada no diagnóstico de coerência (increments = H @ Delta)."""
    n = Z.shape[0]
    mu, sd = Z.mean(0), Z.std(0)
    sd[sd < 1e-12] = 1.0
    A = np.column_stack([np.ones(n), (Z - mu) / sd])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    G[0, 0] -= ridge
    return A @ np.linalg.solve(G, A.T)
