"""Simulador QAOA (statevector, NumPy puro) e ajuste dos ângulos por COBYLA."""
import numpy as np
from scipy.optimize import minimize


def qaoa_state(cvals, n, gammas, betas):
    dim = 1 << n
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    for gam, bet in zip(gammas, betas):
        psi *= np.exp(-1j * gam * cvals)              # separador de fase
        c, s = np.cos(bet), np.sin(bet)                # mixer = prod RX(2*beta)
        for q in range(n):
            a = psi.reshape(1 << q, 2, dim >> (q + 1))
            a0, a1 = a[:, 0, :].copy(), a[:, 1, :].copy()
            a[:, 0, :] = c * a0 - 1j * s * a1
            a[:, 1, :] = -1j * s * a0 + c * a1
    return psi


def expected_cut(cvals, n, theta, p):
    psi = qaoa_state(cvals, n, theta[:p], theta[p:])
    return float(np.dot(np.abs(psi) ** 2, cvals))


def optimize(cvals, n, p, n_starts, maxiter, rng):
    """Multi-start COBYLA; devolve o melhor (theta, valor)."""
    best_val, best_theta = -np.inf, None
    for _ in range(n_starts):
        x0 = np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
        res = minimize(lambda t: -expected_cut(cvals, n, t, p), x0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.4, "tol": 1e-7})
        if -res.fun > best_val:
            best_val, best_theta = -res.fun, res.x
    return best_theta, best_val
