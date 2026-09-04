"""Ajusta CM, PM, Wasserstein-LS e o sweep do QPM para os dois regressores
usados no artigo (histograma de entrada e saída rasa em p=1), valida por
leave-one-out e checa a coerência das previsões. Grava:
  results.json / results_in.json  -- métricas por modelo (saída / entrada)
  fit.npz / fit_in.npz            -- coeficientes, para plots.py
  curves.npz                      -- erro W2 por grafo, para stats.py
"""
import json
import numpy as np

from qpm import knots, loo_fit, rearrange, wasserstein2, hat_matrix
from baselines import center_method, parametrized_method, wasserstein_ls

RIDGES = [0.0, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0]
SWEEP_LS = [2, 3, 5, 9, 17, 33]


def summarize(P, QY, tgrid):
    P = rearrange(P)
    d = wasserstein2(P, QY, tgrid)
    i95 = int(round(0.95 * (len(tgrid) - 1)))
    n = len(d)
    return dict(W2=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(n)),
                MAEmean=float(np.abs(np.trapezoid(P, tgrid, 1)
                                     - np.trapezoid(QY, tgrid, 1)).mean()),
                MAEtail=float(np.abs(P[:, i95] - QY[:, i95]).mean())), d


def qpm_sweep(QX, QY, tgrid):
    sweep, curves = {}, {}
    for L in SWEEP_LS:
        K, s = knots(QX, L, tgrid)
        best = None
        for rg in RIDGES:
            sc, d = summarize(loo_fit(K, QY, rg), QY, tgrid)
            if best is None or sc["W2"] < best[1]["W2"]:
                best = (rg, sc, d)
        rg, sc, d = best
        sweep[L] = dict(ridge=rg, k=L + 1, **sc)
        curves[L] = d
        print(f"  L={L:3d} ridge={rg:<5g} W2={sc['W2']:.5f} "
              f"mean={sc['MAEmean']:.5f} tail={sc['MAEtail']:.5f}")
    return sweep, curves


def fit_regressor(QX, QY, tgrid, label):
    print(f"[{label}]")
    out, curves = {}, {}

    mX = np.trapezoid(QX, tgrid, 1)
    sc, curves["CM"] = summarize(center_method(mX, QY), QY, tgrid)
    out["CM"] = dict(k=2, **sc)

    Zb = np.column_stack([QX[:, 0], QX[:, -1]])
    out["rank_interval"] = int(np.linalg.matrix_rank(
        np.column_stack([np.ones(len(QX)), Zb]), tol=1e-8))
    sc, curves["PM"] = summarize(parametrized_method(Zb, QY, tgrid), QY, tgrid)
    out["PM"] = dict(k=3, **sc)

    sc, curves["WLS"] = summarize(wasserstein_ls(mX, QX, QY, tgrid), QY, tgrid)
    out["WLS"] = dict(k=3, **sc)

    sweep, sweep_curves = qpm_sweep(QX, QY, tgrid)
    Lb = min(sweep, key=lambda L: sweep[L]["W2"])
    out["QPM"] = {k: v for k, v in sweep[Lb].items() if k != "ridge"}
    out["QPM_L"], out["QPM_ridge"], out["sweep"] = Lb, sweep[Lb]["ridge"], sweep
    curves["QPM"] = sweep_curves[Lb]

    return out, curves, Lb, sweep[Lb]["ridge"]


def main():
    D = np.load("data.npz")
    tgrid, Q_in, Q_p1, QY = D["TG"], D["Q_in"], D["Q_p1"], D["Q_p2"]
    n = len(QY)

    R_sh, C_sh, L_sh, ridge_sh = fit_regressor(Q_p1, QY, tgrid, "saída rasa (p=1)")
    R_in, C_in, L_in, ridge_in = fit_regressor(Q_in, QY, tgrid, "entrada (espectro de cortes)")

    # baseline "sem ajuste": prevê Y = saída rasa, copiada direto
    sc, C_sh["ID"] = summarize(Q_p1.copy(), QY, tgrid)
    R_sh["ID"] = dict(k=0, **sc)

    # entrada + saída rasa juntas, no melhor L encontrado para cada uma
    best = None
    for L in [9, 17, 33]:
        K1, _ = knots(Q_in, L, tgrid)
        K2, _ = knots(Q_p1, L, tgrid)
        Z = np.column_stack([K1, K2])
        for rg in RIDGES:
            sc, d = summarize(loo_fit(Z, QY, rg), QY, tgrid)
            if best is None or sc["W2"] < best[2]["W2"]:
                best = (L, rg, sc, d)
    L_both, ridge_both, sc, C_in["both"] = best
    R_in["QPM_both"] = dict(k=2 * L_both + 1, **sc)
    R_in["QPM_both_L"], R_in["QPM_both_ridge"] = L_both, ridge_both

    # degenerescência dos limites: entrada e saída compartilham xbar = 1 - x
    R_in["input_hi_check"] = float(np.abs(Q_in[:, -1] - (1 - Q_in[:, 0])).max())
    R_sh["output_hi_check"] = float(np.abs(QY[:, -1] - (1 - QY[:, 0])).max())
    R_in["distinct_bound_pairs"] = int(
        len(set(map(tuple, np.round(np.c_[Q_in[:, 0], Q_in[:, -1]], 9)))))

    # coerência: matriz H do regressor de saída rasa, no L escolhido
    K, _ = knots(Q_p1, L_sh, tgrid)
    H = hat_matrix(K, ridge_sh)
    pinc = H @ np.diff(QY, axis=1)
    R_sh["coh_cells_ok"] = float((pinc >= -1e-12).mean())
    R_sh["coh_obs_ok"] = float((pinc.min(1) >= -1e-12).mean())
    Praw = loo_fit(K, QY, ridge_sh)
    R_sh["W2_raw"] = float(wasserstein2(Praw, QY, tgrid).mean())
    R_sh["W2_rearranged"] = float(wasserstein2(rearrange(Praw), QY, tgrid).mean())

    R_sh["r1_mean"], R_sh["r2_mean"], R_sh["N"] = float(D["r1"].mean()), float(D["r2"].mean()), n

    json.dump(R_sh, open("results.json", "w"), indent=2)
    json.dump(R_in, open("results_in.json", "w"), indent=2)
    np.savez("curves.npz", **{f"sh_{k}": v for k, v in C_sh.items()},
             **{f"in_{k}": v for k, v in C_in.items()})

    for tag, QX, L, ridge, path in [("fit", Q_p1, L_sh, ridge_sh, "fit.npz"),
                                    ("fit_in", Q_in, L_in, ridge_in, "fit_in.npz")]:
        K, s = knots(QX, L, tgrid)
        mu, sd = K.mean(0), K.std(0)
        sd[sd < 1e-12] = 1.0
        A = np.column_stack([np.ones(n), (K - mu) / sd])
        G = A.T @ A + ridge * np.eye(A.shape[1])
        G[0, 0] -= ridge
        B = np.linalg.solve(G, A.T @ QY)
        np.savez(path, coef=B[1:], s=s, TG=tgrid, L=L)

    print("\nresultados salvos: results.json, results_in.json, "
          "fit.npz, fit_in.npz, curves.npz")


if __name__ == "__main__":
    main()
