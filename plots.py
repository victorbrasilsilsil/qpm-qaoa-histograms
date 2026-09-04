"""Gera a Figura 1: erro vs. resolução do regressor, e onde o modelo lê o
histograma de entrada. Lê results.json/results_in.json e fit.npz/fit_in.npz,
gravados por analyze.py."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fs = np.load("fit.npz")
fi = np.load("fit_in.npz")
R = json.load(open("results.json"))
Ri = json.load(open("results_in.json"))
TG = fs["TG"]

Ls = np.array(sorted(int(k) for k in R["sweep"]))
w_sh = np.array([R["sweep"][str(L)]["W2"] for L in Ls])
w_in = np.array([Ri["sweep"][str(L)]["W2"] for L in Ls])
e_sh = np.array([R["sweep"][str(L)]["se"] for L in Ls])
e_in = np.array([Ri["sweep"][str(L)]["se"] for L in Ls])

BLUE, ORANGE, RED, GREY = "#1f4e79", "#c1651a", "#c00000", "#7f7f7f"

plt.rcParams.update({"font.size": 10.2, "axes.linewidth": 0.7,
                     "font.family": "serif", "mathtext.fontset": "dejavuserif"})
fig, ax = plt.subplots(1, 2, figsize=(6.0, 1.72))

a = ax[0]
a.errorbar(Ls, w_in, yerr=e_in, marker="s", ms=3.2, lw=1.3, color=ORANGE,
           capsize=1.8, label="input histogram")
a.errorbar(Ls, w_sh, yerr=e_sh, marker="o", ms=3.2, lw=1.3, color=BLUE,
           capsize=1.8, label="depth-1 output")
a.plot([2], [w_in[0]], marker="o", ms=8, mfc="none", mec="k", mew=1.0)
a.set_xscale("log"); a.set_xticks(Ls)
a.set_xticklabels([str(int(v)) for v in Ls]); a.minorticks_off()
a.set_yticks([0, 0.01, 0.02, 0.03])
a.set_ylim(0, 0.040)
a.set_xlabel("reference levels $L$")
a.set_ylabel(r"LOO $\mathcal{W}_2$")
a.legend(frameon=False, fontsize=8.6, loc="upper right", handlelength=1.4,
         borderaxespad=0.1, labelspacing=0.2)
a.set_title("(a) resolution of the regressor", fontsize=10.2)

b = ax[1]
for A, s, c, lab in [(fi["coef"], fi["s"], ORANGE, "input"),
                     (fs["coef"], fs["s"], BLUE, "depth-1 output")]:
    W = np.abs(A); W = W / W.sum(0, keepdims=True)
    lam = (W * s[:, None]).sum(0)
    m = (TG > 0.01) & (TG < 0.99)
    b.plot(TG[m], lam[m], lw=1.6, color=c, label=lab)
b.axhline(0.0, ls="--", lw=0.9, color=GREY)
b.axhline(1.0, ls="--", lw=0.9, color=GREY)
b.axhline(0.5, ls="--", lw=0.9, color=RED)
b.text(0.985, 0.05, "MinMax / PM bounds", fontsize=8.2, color=GREY, ha="right")
b.text(0.985, 1.05, "MinMax / PM bounds", fontsize=8.2, color=GREY, ha="right")
b.text(0.5, 0.27, "CM midpoint", fontsize=8.2, color=RED, ha="center")
b.set_xlabel(r"response level $t$")
b.set_ylabel(r"selected $\lambda(t)$")
b.set_ylim(-0.26, 1.22); b.set_xlim(0, 1)
b.set_title("(b) where the model reads", fontsize=10.2)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
    a_.tick_params(width=0.7, length=3)

fig.tight_layout(pad=0.4, w_pad=1.6)
fig.savefig("figs/fig1_qpm.png", dpi=400)
print("figs/fig1_qpm.png salvo")
