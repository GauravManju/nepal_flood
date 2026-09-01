"""
Sensitivity Analysis - Nepal Flood Model
=========================================
Tornado-diagram analysis: ranks every stochastic input in the Monte Carlo
model by how strongly it correlates (Spearman rank correlation) with the
final total-death estimate. Answers "which assumption actually drives the
number?" rather than just reporting the output distribution.

Re-runs the same simulation as nepal_flood_model.py (same seed, same draw
order) but also records each intermediate random draw so it can be
correlated against the output.

Run:
    python sensitivity_analysis.py
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import config
from nepal_flood_model import (
    N_AFFECTED, D_CONFIRMED, M_REPORTED,
    missing_categories, EXPOSED_FRACTION, N_SIMS,
)

rng = np.random.default_rng(seed=config.SEED)


def run_traced_simulation():
    """Re-run the full model, capturing every stochastic input alongside
    the final total, so each can be correlated against the output."""
    traces = {}

    # --- missing_dead ---
    d_missing_dead = np.zeros(N_SIMS)
    for name, (share, a, b) in missing_categories.items():
        n_cat = int(round(M_REPORTED * share))
        p_m = rng.beta(a, b, size=N_SIMS)
        traces[f"p_missing_dead[{name}]"] = p_m
        d_missing_dead += rng.binomial(n_cat, p_m)

    # --- unrecovered ---
    exposed_frac = rng.uniform(*EXPOSED_FRACTION, size=N_SIMS)
    traces["exposed_fraction"] = exposed_frac
    n_exposed = (N_AFFECTED * exposed_frac).astype(int)
    p_f = rng.beta(*config.MORTALITY_EXPOSED_BETA, size=N_SIMS)
    traces["p_mortality_exposed"] = p_f
    n_fatal = rng.binomial(n_exposed, p_f)
    p_u = rng.beta(*config.BODY_UNRECOVERED_BETA, size=N_SIMS)
    traces["p_body_unrecovered"] = p_u
    d_unrecovered = n_fatal * p_u

    # --- capture-recapture / unregistered ---
    n_A = rng.normal(*config.CR_NEPAL_LIST, size=N_SIMS).clip(min=1)
    n_B = rng.normal(*config.CR_INDIA_LIST, size=N_SIMS).clip(min=1)
    overlap_frac = rng.beta(*config.CR_OVERLAP_FRAC_BETA, size=N_SIMS)
    m = (overlap_frac * np.minimum(n_A, n_B)).clip(min=1)
    traces["capture_recapture_n_A"] = n_A
    traces["capture_recapture_n_B"] = n_B
    traces["capture_recapture_overlap_frac"] = overlap_frac
    traces["capture_recapture_overlap_m"] = m
    N_hat = ((n_A + 1) * (n_B + 1) / (m + 1)) - 1
    n_unreported = (N_hat - n_A).clip(min=0)
    p_death = rng.beta(*config.DEATH_UNREGISTERED_BETA, size=N_SIMS)
    traces["p_death_unregistered"] = p_death
    d_unregistered = n_unreported * p_death

    d_total = D_CONFIRMED + d_missing_dead + d_unrecovered + d_unregistered
    return d_total, traces


def compute_sensitivity(d_total, traces):
    results = []
    for name, values in traces.items():
        rho, _ = stats.spearmanr(values, d_total)
        results.append((name, rho))
    results.sort(key=lambda r: abs(r[1]), reverse=True)
    return results


def plot_tornado(results):
    names = [r[0] for r in results]
    rhos = [r[1] for r in results]
    colors = ["#c0392b" if r < 0 else "#2f6db5" for r in rhos]

    plt.figure(figsize=(9, 6))
    y_pos = np.arange(len(names))
    plt.barh(y_pos, rhos, color=colors, edgecolor="white")
    plt.yticks(y_pos, names)
    plt.gca().invert_yaxis()
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Spearman rank correlation with total deaths")
    plt.title("Sensitivity Analysis: Which Inputs Drive the Estimate?")
    plt.tight_layout()
    plt.savefig("nepal_flood_sensitivity.png", dpi=150)
    print("\nSaved plot -> nepal_flood_sensitivity.png")


if __name__ == "__main__":
    d_total, traces = run_traced_simulation()
    results = compute_sensitivity(d_total, traces)

    print("=" * 62)
    print("  SENSITIVITY ANALYSIS - RANKED BY |CORRELATION| WITH TOTAL")
    print("=" * 62)
    for name, rho in results:
        print(f"  {name:<34}: rho = {rho:+.3f}")
    print("=" * 62)

    plot_tornado(results)
