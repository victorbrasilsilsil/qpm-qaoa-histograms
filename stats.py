"""Testa QPM contra cada baseline, grafo a grafo (Wilcoxon pareado), a
partir do curves.npz gravado por analyze.py."""
import numpy as np
from scipy.stats import wilcoxon

C = np.load("curves.npz")


def report(prefix, against):
    qpm = C[f"{prefix}_QPM"]
    print(f"\n[{prefix}] QPM vs.:")
    for name in against:
        other = C[f"{prefix}_{name}"]
        st = wilcoxon(qpm, other)
        win = float((qpm < other).mean())
        print(f"  {name:6s}  p={st.pvalue:.2e}  QPM melhor em {win*100:.0f}% dos grafos")


if __name__ == "__main__":
    report("sh", ["CM", "WLS", "PM", "ID"])
    report("in", ["CM", "WLS", "PM"])
    st = wilcoxon(C["in_both"], C["sh_QPM"])
    win = float((C["in_both"] < C["sh_QPM"]).mean())
    print(f"\n[entrada+saída] vs. [só saída rasa]  p={st.pvalue:.2e}  "
          f"melhor em {win*100:.0f}% dos grafos")
