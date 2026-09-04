"""Os três modelos contra os quais o QPM é comparado no artigo."""
import numpy as np
from scipy.optimize import lsq_linear
from qpm import loo_fit


def center_method(mX, QY, ridge=0.0):
    """CM: regride só a média do regressor. Caso L=1 do QPM."""
    return loo_fit(mX[:, None], QY, ridge)


def parametrized_method(Zbounds, QY, tgrid):
    """PM (Souza et al. 2017): entra com os dois limites do regressor, sai
    com os dois limites da resposta, e assume densidade uniforme dentro do
    intervalo previsto. Caso L=2 do QPM, mas com resposta reduzida a bounds."""
    Ybounds = np.column_stack([QY[:, 0], QY[:, -1]])
    Pb = loo_fit(Zbounds, Ybounds, 0.0)
    lo, hi = Pb[:, 0:1], Pb[:, 1:2]
    return lo * (1 - tgrid)[None, :] + hi * tgrid[None, :]


def wasserstein_ls(mX, QX, QY, tgrid):
    """Irpino & Verde (2015): média + coeficiente de forma restrito a ser
    não-negativo (garante coerência a priori, ao custo de um programa
    quadrático em vez de mínimos quadrados fechados)."""
    n, nt = QY.shape
    P = np.zeros_like(QY)
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        A = np.array([[1.0, mX[k], QX[k, t] - mX[k]] for k in tr for t in range(nt)])
        b = np.array([QY[k, t] for k in tr for t in range(nt)])
        c = lsq_linear(A, b, bounds=(np.array([-np.inf, -np.inf, 0.0]),
                                     np.full(3, np.inf))).x
        P[i] = c[0] + c[1] * mX[i] + c[2] * (QX[i] - mX[i])
    return P
