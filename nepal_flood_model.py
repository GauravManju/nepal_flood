"""
Nepal Flood Disaster - Death Estimation Model
=============================================
A probabilistic, data-driven Monte Carlo model that estimates the TRUE total
number of deaths by accounting for what official data misses.

Reproduces the poster model:
    D_total = D_confirmed + D_missing_dead + D_unrecovered + D_unregistered

Free-software replacement for the MATLAB version. Requires only numpy,
scipy and matplotlib.

Run:
    python nepal_flood_model.py
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import config

# ----------------------------------------------------------------------
# 0.  Reproducibility + simulation size
# ----------------------------------------------------------------------
rng = np.random.default_rng(seed=config.SEED)
N_SIMS = config.N_SIMS

# ----------------------------------------------------------------------
# 1.  Known numbers (Panel 2)
# ----------------------------------------------------------------------
N_AFFECTED  = config.N_AFFECTED
D_CONFIRMED = config.D_CONFIRMED
M_REPORTED  = config.M_REPORTED
RESCUED     = config.RESCUED

# ----------------------------------------------------------------------
# 2.  D_missing_dead  (Panels 5)
#     Missing persons who ultimately died.
#     D_missing_dead ~ Binomial(M, p_m),  p_m ~ Beta (per category)
# ----------------------------------------------------------------------
missing_categories = config.MISSING_CATEGORIES

def sample_missing_dead():
    total = np.zeros(N_SIMS)
    for share, a, b in missing_categories.values():
        n_cat = int(round(M_REPORTED * share))          # people in this category
        p_m   = rng.beta(a, b, size=N_SIMS)             # Beta-distributed death prob
        total += rng.binomial(n_cat, p_m)               # Binomial deaths
    return total

# ----------------------------------------------------------------------
# 3.  D_unrecovered  (Panel 6)
#     Bodies never found.
#     N_fatal ~ Binomial(N_exposed, p_f)
#     D_unrecovered = N_fatal * p_u        (p_u = prob body never recovered)
# ----------------------------------------------------------------------
# N_exposed: people directly exposed to the flood/debris flow (a fraction of
# the affected population who were in the path, not the whole 109,700).
EXPOSED_FRACTION = config.EXPOSED_FRACTION

def sample_unrecovered():
    exposed_frac = rng.uniform(*EXPOSED_FRACTION, size=N_SIMS)
    n_exposed    = (N_AFFECTED * exposed_frac).astype(int)
    p_f          = rng.beta(*config.MORTALITY_EXPOSED_BETA, size=N_SIMS)
    n_fatal      = rng.binomial(n_exposed, p_f)
    p_u          = rng.beta(*config.BODY_UNRECOVERED_BETA, size=N_SIMS)
    return n_fatal * p_u

# ----------------------------------------------------------------------
# 4.  D_unregistered  (Panel 8)
#     People never on any rescue or missing list.
#     D_unregistered = N_unreported * p_death
# ----------------------------------------------------------------------
def sample_unregistered():
    # N_unreported estimated via capture-recapture (Panel 10) below, but we
    # also add a Poisson-style informal-population term for people on NEITHER
    # list. Here we draw N_unreported from the capture-recapture posterior.
    n_unreported = sample_capture_recapture()
    p_death      = rng.beta(*config.DEATH_UNREGISTERED_BETA, size=N_SIMS)
    return n_unreported * p_death

# ----------------------------------------------------------------------
# 5.  Capture-Recapture (Panel 10 / 11) - Chapman estimator
#     N_hat = ((n_A + 1)(n_B + 1) / (m + 1)) - 1
#     n_A = Nepal missing list, n_B = India missing list, m = overlap
# ----------------------------------------------------------------------
def sample_capture_recapture():
    # Treat the counts as uncertain: draw around plausible values.
    n_A = rng.normal(*config.CR_NEPAL_LIST, size=N_SIMS).clip(min=1)   # Nepal list
    n_B = rng.normal(*config.CR_INDIA_LIST, size=N_SIMS).clip(min=1)   # India list
    # Overlap can never exceed the smaller of the two lists, so draw it as a
    # fraction of min(n_A, n_B) rather than as an independent Normal (which
    # could otherwise put m above n_A or n_B in some simulations).
    overlap_frac = rng.beta(*config.CR_OVERLAP_FRAC_BETA, size=N_SIMS)
    m = (overlap_frac * np.minimum(n_A, n_B)).clip(min=1)
    N_hat = ((n_A + 1) * (n_B + 1) / (m + 1)) - 1
    # People on NEITHER list = total estimate minus those already counted
    n_unreported = (N_hat - n_A).clip(min=0)
    return n_unreported

# ----------------------------------------------------------------------
# 6.  Weibull survival check for tunnel workers (Panel 11 / 12)
#     S(t) = exp(-(t/eta)^k)   -> probability still alive after t hours trapped
#     This modulates the tunnel-worker death probability rather than assuming
#     all trapped workers die.
# ----------------------------------------------------------------------
def tunnel_survival_probability(t_hours, eta=config.WEIBULL_ETA, k=config.WEIBULL_K):
    """Probability a trapped worker is still alive after t hours."""
    return np.exp(-((t_hours / eta) ** k))

# ----------------------------------------------------------------------
# 7.  Run the full Monte Carlo simulation
# ----------------------------------------------------------------------
def run_simulation():
    d_missing_dead = sample_missing_dead()
    d_unrecovered  = sample_unrecovered()
    d_unregistered = sample_unregistered()

    d_total = D_CONFIRMED + d_missing_dead + d_unrecovered + d_unregistered
    return d_total, d_missing_dead, d_unrecovered, d_unregistered


def summarise(d_total):
    median = np.median(d_total)
    ci80   = np.percentile(d_total, [10, 90])   # 80% credible interval
    ci90   = np.percentile(d_total, [5, 95])    # 90% credible interval
    p_7000 = np.mean(d_total >= 7_000) * 100
    p_8000 = np.mean(d_total >= 8_000) * 100

    print("=" * 55)
    print("  NEPAL FLOOD - ESTIMATED TOTAL DEATHS (Monte Carlo)")
    print("=" * 55)
    print(f"  Simulations run          : {N_SIMS:,}")
    print(f"  Median estimate          : ~ {median:,.0f}")
    print(f"  80% credible interval    : ~ {ci80[0]:,.0f} - {ci80[1]:,.0f}")
    print(f"  90% credible interval    : ~ {ci90[0]:,.0f} - {ci90[1]:,.0f}")
    print(f"  P(>= 7,000 deaths)       : ~ {p_7000:.0f}%")
    print(f"  P(>= 8,000 deaths)       : ~ {p_8000:.0f}%")
    print("=" * 55)
    print("  Note: highly uncertain. Update as ground truth emerges.")
    print("=" * 55)
    return median, ci80, ci90


def plot_distribution(d_total, median, ci90):
    # Clip to a sensible viewing window (rare capture-recapture outliers can be
    # huge and would otherwise squash the curve into a single spike).
    lo, hi = np.percentile(d_total, [0.5, 99.5])
    view = d_total[(d_total >= lo) & (d_total <= hi)]

    plt.figure(figsize=(9, 5))
    plt.hist(view, bins=80, color="#2f6db5", edgecolor="white", linewidth=0.3)
    plt.axvline(median, color="red", linewidth=2, label=f"Median ~ {median:,.0f}")
    plt.axvline(ci90[0], color="red", linestyle="--", linewidth=1.5,
                label=f"5% ~ {ci90[0]:,.0f}")
    plt.axvline(ci90[1], color="red", linestyle="--", linewidth=1.5,
                label=f"95% ~ {ci90[1]:,.0f}")
    plt.title("Simulated Distribution of Total Deaths")
    plt.xlabel("Total Deaths")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nepal_flood_distribution.png", dpi=150)
    print("\nSaved plot -> nepal_flood_distribution.png")


if __name__ == "__main__":
    d_total, d_md, d_ur, d_un = run_simulation()
    median, ci80, ci90 = summarise(d_total)

    print("\nBreakdown of the hidden categories (median contributions):")
    print(f"  Confirmed (fixed)        : {D_CONFIRMED:,}")
    print(f"  Missing who died         : ~ {np.median(d_md):,.0f}")
    print(f"  Unrecovered bodies       : ~ {np.median(d_ur):,.0f}")
    print(f"  Unregistered / unlisted  : ~ {np.median(d_un):,.0f}")

    # Example of the tunnel-worker survival curve
    print("\nTunnel-worker survival probability (Weibull, eta=48h, k=1.5):")
    for t in (6, 12, 24, 48, 72):
        print(f"  after {t:>2}h trapped       : {tunnel_survival_probability(t)*100:5.1f}% alive")

    plot_distribution(d_total, median, ci90)
