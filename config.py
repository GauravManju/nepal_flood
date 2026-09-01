"""
Configuration - Nepal Flood Model
==================================
Single source of truth for every parameter used by nepal_flood_model.py and
sensitivity_analysis.py. Both scripts import from here instead of hardcoding
their own copies, so the two can never silently drift out of sync (this bit
us once already: sensitivity_analysis.py kept an old capture-recapture
formula for a bit after nepal_flood_model.py was fixed).

Edit values here to explore different scenarios; no need to touch the model
or analysis code.
"""

# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
SEED = 42
N_SIMS = 100_000            # Panel 12: 100,000 Monte Carlo simulations

# ----------------------------------------------------------------------
# Known numbers (Panel 2)
# ----------------------------------------------------------------------
N_AFFECTED = 109_700        # Directly affected population
D_CONFIRMED = 1_003         # Confirmed deaths (recovered / identified)
M_REPORTED = 3_916          # Reported missing persons
RESCUED = 11_814            # Rescued / safe (reported)

# ----------------------------------------------------------------------
# D_missing_dead (Panel 5)
# Missing persons who ultimately died: D_missing_dead ~ Binomial(M, p_m),
# p_m ~ Beta (per category). (alpha, beta) chosen so the mean sits where
# the poster's bell curve peaks (~0.5-0.8) but with category-specific
# optimism/pessimism.
# ----------------------------------------------------------------------
MISSING_CATEGORIES = {
    # category          : (share_of_missing, beta_alpha, beta_beta)
    "residents":         (0.45, 6, 4),   # mean p ~0.60
    "tunnel_workers":    (0.15, 8, 2),   # mean p ~0.80 (trapped underground)
    "tourists_foreign":  (0.10, 5, 5),   # mean p ~0.50
    "indian_workers":    (0.20, 6, 4),   # mean p ~0.60
    "others":            (0.10, 5, 5),   # mean p ~0.50
}

# ----------------------------------------------------------------------
# D_unrecovered (Panel 6)
# Bodies never found: N_fatal ~ Binomial(N_exposed, p_f),
# D_unrecovered = N_fatal * p_u (p_u = prob body never recovered)
# ----------------------------------------------------------------------
EXPOSED_FRACTION = (0.08, 0.20)          # 8%-20% of affected were directly exposed
MORTALITY_EXPOSED_BETA = (4, 43.96)      # mortality among exposed, mean 8.34%
BODY_UNRECOVERED_BETA = (4, 8)           # ~33% of bodies never recovered

# ----------------------------------------------------------------------
# D_unregistered (Panel 8)
# People never on any rescue or missing list: D_unregistered = N_unreported * p_death
# ----------------------------------------------------------------------
DEATH_UNREGISTERED_BETA = (4, 6)         # ~40% death prob among the unregistered

# ----------------------------------------------------------------------
# Capture-Recapture (Panel 10 / 11) - Chapman estimator
# N_hat = ((n_A + 1)(n_B + 1) / (m + 1)) - 1
# n_A = Nepal missing list, n_B = India missing list, m = overlap
# ----------------------------------------------------------------------
CR_NEPAL_LIST = (3_916, 300)             # (mean, sd) for n_A
CR_INDIA_LIST = (1_800, 400)             # (mean, sd) for n_B
# Overlap can never exceed the smaller of the two lists, so it's drawn as a
# Beta-distributed fraction of min(n_A, n_B) rather than an independent
# Normal (mean ~44%, matching the original 800/1800 ratio).
CR_OVERLAP_FRAC_BETA = (4, 5)

# ----------------------------------------------------------------------
# Weibull survival check for tunnel workers (Panel 11 / 12)
# S(t) = exp(-(t/eta)^k) -> probability still alive after t hours trapped
# ----------------------------------------------------------------------
WEIBULL_ETA = 48.0
WEIBULL_K = 1.5
