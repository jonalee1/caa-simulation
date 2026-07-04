"""
CAA Simulation Study 4: Robustness (revised, full dual-mode battery)
====================================================================

Robustness of the profile-consistent (PC) CAA estimator under stress
conditions, rebuilt on caa_foundation_vf (all model code imported; nothing
model-related is redefined here).

DUAL-MODE DESIGN: every cell in the battery is run in BOTH estimation modes,
so the manuscript table pairs, per condition:
    profile_consistent  — the applied estimator (headline result)
    supervised          — the known-target benchmark (NOT a ceiling or
                          oracle: a trained benchmark using generating
                          allocations as labels)
The pairing decomposes degradation per condition: the benchmark shows what is
condition-inherent; PC minus benchmark shows what is estimator-inherent; and
anchor recovery explains the PC ceiling (the PC target is built from the
noisy profile, so measurement noise enters the target itself).

Metrics per cell:
  recovery_r   mean within-person r between estimated and generating
               allocation (diagnostic; never a PC training target)
  anchor_r     recovery of the closed-form anchor a* = softmax(z/tau) itself;
               delta_r = recovery_r - anchor_r separates denoising by the
               learned mapping from reproduction of the target
  jsd_*        mean Jensen-Shannon divergence to the generating allocation
               (magnitude-sensitive; correlation alone conceals attenuation,
               e.g. the benchmark's ADI collapse under noise)
  ADI triple   generating / anchor / learned
  rho_a        allocation reliability, where the simulation replicate source
               is valid (sample size, noise, initialization, heterogeneity
               complexity, item count); skipped for perturbed conditions,
               where regenerated replicates would use the wrong error process
  recovery_r_clean  (corruption test only) recovery among uncontaminated
               persons — the spillover diagnostic; corrupted persons are
               unrecoverable by construction

Battery (eight tests, every cell x two modes):
    1. Sample size              N in {300, 600, 800, 1000, 1500}
    2. Noise tolerance          sigma^2 in {0.00, 0.05, 0.10, 0.20, 0.35, 0.50}
    3. Initialization           one dataset, R fit seeds
    4. Heterogeneity complexity K = 3 vs. K = 4 generating archetypes
    5. Distributional form      normal / skewed / heavy-tailed errors at
                                matched variance (MATCHED_ERROR_VARIANCE)
    6. Profile corruption       {0%, 5%, 10%, 15%} of persons replaced
    7. Ordinal response scales  continuous oracle; aligned-threshold
                                generation x {EV, direct integer} coding at
                                K in {7, 5, 4} (J = 5); threshold-mismatch
                                generation (floor-skewed, PHQ-9-like
                                marginals) x {EV, direct} at K = 4 (J = 5);
                                and the PHQ-9 configuration J = 9, K = 4
                                under mismatch x {EV, direct}. The
                                expected-value (EV) transform is inherited
                                from CSR (fixed symmetric logistic
                                thresholds, K-dependent, sample-independent)
                                and is the declared CAA default for ordinal
                                data; direct integer coding is the
                                sensitivity benchmark. Aligned cells test
                                information loss from coarsening; mismatch
                                cells test robustness of the fixed EV
                                convention when generating thresholds are
                                floor-skewed. No positive-part step is
                                applied (unlike CSR): signed EV values enter
                                within-person standardization directly.
    8. Item count               J in {4, 5, 9}

Other design decisions (retained; all confirmed):
  * DGP: identifiable mode, so stress effects are not confounded with the
    composition-allocation gap dissected in Study 3.
  * Replication: R = 10 per cell by default; each replicate redraws BOTH the
    dataset and the fit seed (ddof = 1 SDs reflect full Monte Carlo
    variability), except the initialization test (one dataset, fresh fit
    seeds). Replicate-level rows are saved for failure auditing.
  * Global batch size 128 (the N = 300 instability under batch 256 left one
    batch per epoch after the validation split).
  * The target-dispersion parameter tau_0 was fixed at its prespecified
    calibration value; sensitivity of ADI to tau_0 is treated as calibration
    rather than estimator robustness. No tau_0 test is run.

Approximate cost: 29 cells x 2 modes x R fits (~580 fits at R = 10; roughly
double the previous single-mode run).

Usage:
    python study4_robustness.py                     # full battery, R = 10
    python study4_robustness.py --replicates=3      # reduced replication
    python study4_robustness.py --quick             # R = 3, smoke test
    python study4_robustness.py --test=corruption   # single test
        (test names: sample_size, noise, init, heterogeneity,
         distributional, corruption, ordinal, item_count)

Outputs:
    study4_robustness_results.csv     one row per cell x mode (means, SDs)
    study4_robustness_replicates.csv  one row per replicate (failure audit)

Author: Jonathan Lee
Requires: caa_foundation_vf.py (v3.0.0) on the import path
"""
from __future__ import annotations

import csv
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from caa_foundation_vf import (
    RANDOM_SEED,
    CAATrainer,
    UniversalPersonaGenerator,
    compute_adi,
    evaluate_attention_recovery,
    profile_anchor_allocation,
    reliability_from_simulation,
    set_reproducible_state,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

N_ITEMS = 5
N_ARCHETYPES_BASELINE = 3          # generating archetypes
N_SAMPLES_DEFAULT = 800
N_EPOCHS = 220
N_REPLICATES_DEFAULT = 10
BATCH_SIZE = 128                   # global estimator default (see docstring)
RHO_A_REPLICATES = 8               # replicate predictions for rho_a

# Both estimation modes run for every cell. Order fixes table column order.
MODES = ("profile_consistent", "supervised")
MODE_TAGS = {"profile_consistent": "[PC]", "supervised": "[benchmark]"}

# Matched total error variance for the distributional-form test. All three
# error shapes are standardized to unit variance and scaled to this value, so
# the comparison isolates distributional form from error magnitude.
MATCHED_ERROR_VARIANCE = 0.10

DATA_SEED_STRIDE = 1000
INIT_SEED_STRIDE = 137

RESULTS_CSV = "study4_robustness_results.csv"
REPLICATES_CSV = "study4_robustness_replicates.csv"

# Module-level store for replicate-level rows (failure audit).
REPLICATE_ROWS: List[Dict] = []


# =============================================================================
# METRIC HELPERS (numpy only)
# =============================================================================

def mean_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Mean row-wise Jensen-Shannon divergence between allocation matrices."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m), axis=1) + \
          0.5 * np.sum(q * np.log(q / m), axis=1)
    return float(np.mean(jsd))


def subset_recovery(pred: np.ndarray, true: np.ndarray,
                    mask: np.ndarray) -> float:
    """Mean individual recovery r restricted to mask == True."""
    rec = evaluate_attention_recovery(pred[mask], true[mask])
    return rec["mean_individual_correlation"]


# =============================================================================
# DATA GENERATION AND PERTURBATION
# =============================================================================

def generate_identifiable(n_samples: int = N_SAMPLES_DEFAULT,
                          n_items: int = N_ITEMS,
                          n_archetypes: int = N_ARCHETYPES_BASELINE,
                          noise_variance: Optional[float] = None,
                          seed: int = RANDOM_SEED) -> Dict:
    """Identifiable-DGP dataset from the foundation generator."""
    gen = UniversalPersonaGenerator(n_items=n_items, n_personas=n_archetypes)
    kwargs = dict(n_samples=n_samples, dgp_mode="identifiable", seed=seed)
    if noise_variance is not None:
        kwargs["noise_variance"] = noise_variance
    return gen.generate_dataset(**kwargs)


def make_matched_error_dataset(shape: str, seed: int,
                               sigma2: float = MATCHED_ERROR_VARIANCE) -> Dict:
    """
    Distributional-form dataset: clean responses plus errors of a given shape,
    standardized to unit variance and scaled to a common sigma2. Shapes:
      normal  -> Z
      skewed  -> (Z^2 - 1) / sqrt(2)   (standardized chi-square(1), skew 2.83)
      heavy   -> t_3 / sqrt(3)         (standardized Student t, excess kurtosis)
    All three conditions share the same clean signal and the same error
    variance, so any recovery difference is attributable to distributional
    form alone.
    """
    ds = dict(generate_identifiable(noise_variance=0.0, seed=seed))
    clean = np.asarray(ds["clean_responses"], dtype=np.float64)
    rng = np.random.default_rng(seed + 40_000)
    z = rng.normal(0.0, 1.0, clean.shape)
    if shape == "normal":
        e = z
    elif shape == "skewed":
        e = (z ** 2 - 1.0) / np.sqrt(2.0)
    elif shape == "heavy":
        e = rng.standard_t(3, size=clean.shape) / np.sqrt(3.0)
    else:
        raise ValueError(f"Unknown error shape '{shape}'")
    responses = np.clip(clean + np.sqrt(sigma2) * e, 0.0, None)
    ds["responses"] = responses.astype(np.float32)
    ds["noise_variance"] = float(sigma2)   # bookkeeping only; rho_a skipped
    return ds


def apply_profile_corruption(dataset: Dict, seed: int,
                             corruption_rate: float) -> Dict:
    """
    Adversarial person-level contamination: replace a fraction of persons'
    full response vectors with random extreme profiles while leaving their
    generating allocations unchanged. Corrupted persons are unrecoverable by
    construction; the diagnostic question is whether corruption propagates
    through the shared mapping to degrade UNCONTAMINATED persons. A boolean
    'clean_mask' (True = uncontaminated) is attached for subset recovery.
    """
    ds = dict(dataset)
    rng = np.random.default_rng(seed + 70_000)
    x = ds["responses"].copy()
    n, j = x.shape
    n_out = int(round(n * corruption_rate))
    mask = np.ones(n, dtype=bool)
    if n_out > 0:
        idx = rng.choice(n, size=n_out, replace=False)
        lo = float(x.min() - 2.0 * x.std())
        hi = float(x.max() + 2.0 * x.std())
        x[idx] = rng.uniform(lo, hi, size=(n_out, j)).astype(np.float32)
        mask[idx] = False
    ds["responses"] = np.clip(x, 0.0, None)
    ds["clean_mask"] = mask
    return ds


# -----------------------------------------------------------------------------
# Ordinal machinery (EV transform inherited from CSR)
# -----------------------------------------------------------------------------
# The CAA ordinal chain is  y_i -> x_i^EV -> z_i -> a_i -> c_i:
# observed categories are mapped to the continuous working response metric by
# the fixed expected-value (EV) transform, then enter within-person
# standardization directly (no positive-part step; softmax needs no
# truncation). Direct integer coding is retained as the sensitivity benchmark.
# Because the EV mapping is NONLINEAR in the category labels, it is not
# absorbed by within-person standardization: z(T_EV(y)) != z(y) in general.
# -----------------------------------------------------------------------------

# PHQ-9-like floor-skewed cumulative proportions for K = 4 mismatch cells
# (category frequencies 0.55 / 0.25 / 0.12 / 0.08).
FLOOR_SKEW_CUM_PROBS = (0.55, 0.80, 0.92)


def logistic_thresholds(n_categories: int) -> np.ndarray:
    """
    Fixed symmetric logistic thresholds, tau_k = k - K/2 (k = 1..K-1),
    depending on K but not the item or the sample (CSR convention; for
    K = 5 this gives -1.5, -0.5, 0.5, 1.5).
    """
    return np.arange(1, n_categories) - n_categories / 2.0


def ev_values(n_categories: int) -> np.ndarray:
    """
    Expected latent value per category under the standard logistic
    distribution with the fixed symmetric thresholds:

        T_EV(k) = E[z | tau_{k-1} < z <= tau_k]

    (CSR Eq. 9; for K = 5 the values are -2.60, -0.96, 0.00, +0.96, +2.60).
    Computed by numerical integration of the logistic density
    f(z) = (1/4) sech^2(z/2).
    """
    tau = np.concatenate([[-30.0], logistic_thresholds(n_categories), [30.0]])
    grid = np.linspace(-30.0, 30.0, 200_001)
    dens = 0.25 / np.cosh(grid / 2.0) ** 2
    vals = np.zeros(n_categories)
    for k in range(n_categories):
        m = (grid > tau[k]) & (grid <= tau[k + 1])
        # Uniform grid: the step size cancels in the ratio, so plain sums
        # suffice (avoids np.trapz, removed in numpy 2.x).
        vals[k] = float(np.sum(grid[m] * dens[m]) / np.sum(dens[m]))
    return vals


def make_ordinal_dataset(dataset: Dict, n_categories: int,
                         threshold_mode: str, coding: str) -> Dict:
    """
    Ordinal dataset from a continuous one: per-item quantile categorization
    followed by the chosen coding of the resulting categories.

    threshold_mode:
      'aligned'  — per-item cut points at the logistic-implied cumulative
                   probabilities F(tau_k) of the fixed thresholds, so the
                   category-generating geometry matches the geometry the EV
                   transform assumes (tests pure coarsening loss).
      'mismatch' — per-item cut points at floor-skewed, PHQ-9-like cumulative
                   proportions (FLOOR_SKEW_CUM_PROBS; K = 4 only), while the
                   EV transform still uses the fixed default thresholds
                   (tests robustness of the inherited EV convention).

    coding:
      'ev'     — categories mapped to fixed EV values (the CAA default);
                 signed values enter within-person standardization directly.
      'direct' — categories as integer codes 0..K-1 (sensitivity benchmark).

    Per-item monotone categorization is not a per-person affine transform, so
    it legitimately perturbs the within-person profile in both modes.
    """
    if threshold_mode == "aligned":
        tau = logistic_thresholds(n_categories)
        cum = 1.0 / (1.0 + np.exp(-tau))             # logistic CDF at tau_k
    elif threshold_mode == "mismatch":
        if n_categories != len(FLOOR_SKEW_CUM_PROBS) + 1:
            raise ValueError("mismatch cells are defined for K = 4 only")
        cum = np.asarray(FLOOR_SKEW_CUM_PROBS, dtype=np.float64)
    else:
        raise ValueError(f"Unknown threshold_mode '{threshold_mode}'")

    ds = dict(dataset)
    x = np.asarray(ds["responses"], dtype=np.float64)
    y = np.zeros_like(x, dtype=int)
    for j in range(x.shape[1]):
        cuts = np.quantile(x[:, j], cum)
        y[:, j] = np.searchsorted(cuts, x[:, j], side="right")  # 0..K-1

    if coding == "ev":
        ds["responses"] = ev_values(n_categories)[y].astype(np.float32)
    elif coding == "direct":
        ds["responses"] = y.astype(np.float32)
    else:
        raise ValueError(f"Unknown coding '{coding}'")
    return ds


# =============================================================================
# CORE FIT-AND-EVALUATE
# =============================================================================

def fit_and_evaluate(dataset: Dict, fit_seed: int,
                     mode: str = "profile_consistent",
                     n_epochs: int = N_EPOCHS,
                     compute_rho_a: bool = False) -> Dict[str, float]:
    """
    Train one CAA model and return recovery, anchor, JSD, and ADI metrics.

    In profile_consistent mode the generating allocation is passed only as a
    recovery diagnostic (never a target in this mode); in supervised mode it
    is the training label (the known-target benchmark). Anchor recovery is
    mode-independent (a property of the responses), but is reported with both
    modes for table completeness.
    """
    set_reproducible_state(fit_seed)
    trainer = CAATrainer(n_items=dataset["n_items"], n_attention_heads=3,
                         mode=mode)
    trainer.train(responses=dataset["responses"],
                  true_attention_patterns=dataset["true_attention_patterns"],
                  n_epochs=n_epochs, batch_size=BATCH_SIZE, verbose=False)

    pred = trainer.predict_attention_patterns(dataset["responses"])
    a_hat = pred["attention_weights"]
    true_a = dataset["true_attention_patterns"]
    anchor = profile_anchor_allocation(dataset["responses"],
                                       trainer.pc_target_tau)

    rec = evaluate_attention_recovery(a_hat, true_a)
    anchor_rec = evaluate_attention_recovery(anchor, true_a)

    out: Dict[str, float] = {
        "recovery_r": rec["mean_individual_correlation"],
        "anchor_r": anchor_rec["mean_individual_correlation"],
        "delta_r": (rec["mean_individual_correlation"] -
                    anchor_rec["mean_individual_correlation"]),
        "strong_rate": rec["strong_recovery_rate"],
        "jsd_learned": mean_jsd(a_hat, true_a),
        "jsd_anchor": mean_jsd(anchor, true_a),
        "adi_generating": compute_adi(true_a),
        "adi_anchor": compute_adi(anchor),
        "adi_learned": compute_adi(a_hat),
    }

    if "clean_mask" in dataset:
        mask = np.asarray(dataset["clean_mask"], dtype=bool)
        out["recovery_r_clean"] = subset_recovery(a_hat, true_a, mask)

    if compute_rho_a:
        rel = reliability_from_simulation(trainer, dataset,
                                          n_replicates=RHO_A_REPLICATES,
                                          verbose=False)
        out["rho_a"] = float(rel["rho_a"])

    return out


def replicate_cell(make_dataset: Callable[[int], Dict],
                   n_replicates: int,
                   test: str, cell: str, mode: str,
                   base_seed: int = RANDOM_SEED,
                   redraw_data: bool = True,
                   compute_rho_a: bool = False,
                   label: str = "",
                   verbose: bool = True) -> Dict[str, float]:
    """
    Run one design cell x mode with R replicates and aggregate (ddof = 1 SDs).
    Replicate-level rows are appended to REPLICATE_ROWS for the audit CSV.

    redraw_data=True  -> each replicate gets a fresh dataset AND fresh fit
                         seed (full Monte Carlo variability; default).
    redraw_data=False -> one dataset, fresh fit seeds (initialization test).
    Replicate seeds are identical across modes, so the two modes see the same
    sequence of datasets and the mode contrast is paired, not confounded.
    """
    rows: List[Dict[str, float]] = []
    fixed_dataset = None if redraw_data else make_dataset(base_seed)

    for r in range(n_replicates):
        data_seed = base_seed + DATA_SEED_STRIDE * r
        fit_seed = (data_seed if redraw_data
                    else base_seed + INIT_SEED_STRIDE * r)
        ds = make_dataset(data_seed) if redraw_data else fixed_dataset
        row = fit_and_evaluate(ds, fit_seed, mode=mode,
                               compute_rho_a=compute_rho_a)
        rows.append(row)
        REPLICATE_ROWS.append({"test": test, "cell": cell, "mode": mode,
                               "replicate": r, "data_seed": data_seed,
                               "fit_seed": fit_seed, **row})

    agg: Dict[str, float] = {"test": test, "cell": cell, "mode": mode,
                             "n_replicates": float(n_replicates)}
    for key in rows[0]:
        vals = np.array([row[key] for row in rows], dtype=float)
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_sd"] = float(np.std(vals, ddof=1))

    if verbose:
        tag = MODE_TAGS.get(mode, mode)
        msg = (f"  {(label + ' ' + tag):<40} r = {agg['recovery_r_mean']:.3f} "
               f"(SD = {agg['recovery_r_sd']:.3f}) | "
               f"anchor r = {agg['anchor_r_mean']:.3f} | "
               f"JSD l/a = {agg['jsd_learned_mean']:.4f}/"
               f"{agg['jsd_anchor_mean']:.4f}")
        if "recovery_r_clean_mean" in agg:
            msg += f" | clean r = {agg['recovery_r_clean_mean']:.3f}"
        if "rho_a_mean" in agg:
            msg += f" | rho_a = {agg['rho_a_mean']:.3f}"
        print(msg)
        print(f"  {'':<40} ADI g/a/l = {agg['adi_generating_mean']:.3f}/"
              f"{agg['adi_anchor_mean']:.3f}/{agg['adi_learned_mean']:.3f}")
    return agg


def run_cell_both_modes(make_dataset: Callable[[int], Dict],
                        n_replicates: int,
                        test: str, cell: str,
                        base_seed: int = RANDOM_SEED,
                        redraw_data: bool = True,
                        compute_rho_a: bool = False,
                        label: str = "",
                        verbose: bool = True) -> List[Dict]:
    """One design cell in both estimation modes (paired replicate seeds)."""
    return [replicate_cell(make_dataset, n_replicates, test=test, cell=cell,
                           mode=mode, base_seed=base_seed,
                           redraw_data=redraw_data,
                           compute_rho_a=compute_rho_a,
                           label=label, verbose=verbose)
            for mode in MODES]


# =============================================================================
# TEST 1: SAMPLE SIZE
# =============================================================================

def test_sample_size(n_replicates: int, verbose: bool = True) -> List[Dict]:
    if verbose:
        print("\nTest 1: Sample size (batch size 128)")
        print("-" * 78)
    results = []
    for n in (300, 600, 800, 1000, 1500):
        results.extend(run_cell_both_modes(
            lambda s, n=n: generate_identifiable(n_samples=n, seed=s),
            n_replicates, test="sample_size", cell=f"N={n}",
            compute_rho_a=True, label=f"N = {n}", verbose=verbose))
    return results


# =============================================================================
# TEST 2: NOISE TOLERANCE
# =============================================================================

def test_noise(n_replicates: int, verbose: bool = True) -> List[Dict]:
    if verbose:
        print("\nTest 2: Noise tolerance")
        print("-" * 78)
    results = []
    for sigma2 in (0.00, 0.05, 0.10, 0.20, 0.35, 0.50):
        results.extend(run_cell_both_modes(
            lambda s, v=sigma2: generate_identifiable(noise_variance=v, seed=s),
            n_replicates, test="noise", cell=f"sigma2={sigma2:.2f}",
            compute_rho_a=True, label=f"sigma^2 = {sigma2:.2f}",
            verbose=verbose))
    return results


# =============================================================================
# TEST 3: INITIALIZATION SENSITIVITY
# =============================================================================

def test_initialization(n_replicates: int, verbose: bool = True) -> List[Dict]:
    if verbose:
        print("\nTest 3: Initialization sensitivity (fixed dataset, "
              f"{n_replicates} fit seeds)")
        print("-" * 78)
    return run_cell_both_modes(
        lambda s: generate_identifiable(seed=RANDOM_SEED),
        n_replicates, test="init", cell=f"{n_replicates}_inits",
        redraw_data=False, compute_rho_a=True,
        label=f"{n_replicates} initializations", verbose=verbose)


# =============================================================================
# TEST 4: HETEROGENEITY COMPLEXITY
# =============================================================================
# The estimator contains no K parameter, so varying the number of generating
# archetypes cannot misspecify the model; it varies the complexity of the
# heterogeneity structure the shared mapping must represent.
# =============================================================================

def test_heterogeneity(n_replicates: int, verbose: bool = True) -> List[Dict]:
    if verbose:
        print("\nTest 4: Heterogeneity complexity (number of generating "
              "archetypes)")
        print("-" * 78)
    results = []
    for k in (3, 4):
        results.extend(run_cell_both_modes(
            lambda s, k=k: generate_identifiable(n_archetypes=k, seed=s),
            n_replicates, test="heterogeneity", cell=f"K={k}",
            compute_rho_a=True, label=f"K = {k} generating archetypes",
            verbose=verbose))
    return results


# =============================================================================
# TEST 5: DISTRIBUTIONAL FORM (matched error variance)
# =============================================================================

def test_distributional(n_replicates: int, verbose: bool = True) -> List[Dict]:
    if verbose:
        print(f"\nTest 5: Distributional form (matched error variance "
              f"sigma^2 = {MATCHED_ERROR_VARIANCE:.2f})")
        print("-" * 78)
    results = []
    for shape, label in (("normal", "normal errors"),
                         ("skewed", "skewed errors (standardized chi-sq)"),
                         ("heavy", "heavy-tailed errors (standardized t_3)")):
        results.extend(run_cell_both_modes(
            lambda s, sh=shape: make_matched_error_dataset(sh, seed=s),
            n_replicates, test="distributional", cell=shape,
            label=label, verbose=verbose))
    return results


# =============================================================================
# TEST 6: PROFILE CORRUPTION (adversarial person-level contamination)
# =============================================================================

def test_corruption(n_replicates: int, verbose: bool = True) -> List[Dict]:
    if verbose:
        print("\nTest 6: Profile corruption (corrupted persons unrecoverable "
              "by construction; clean r is the spillover test)")
        print("-" * 78)
    results = []
    for rate in (0.00, 0.05, 0.10, 0.15):
        results.extend(run_cell_both_modes(
            lambda s, rt=rate: apply_profile_corruption(
                generate_identifiable(seed=s), seed=s, corruption_rate=rt),
            n_replicates, test="corruption", cell=f"{int(rate * 100)}pct",
            label=f"{int(rate * 100)}% corruption", verbose=verbose))
    return results


# =============================================================================
# TEST 7: ORDINAL RESPONSE SCALES (threshold discretization)
# =============================================================================

def test_ordinal(n_replicates: int, verbose: bool = True) -> List[Dict]:
    if verbose:
        print("\nTest 7: Ordinal response scales (EV transform = CAA default; "
              "direct integer coding = sensitivity benchmark)")
        print("-" * 78)
    results = []

    results.extend(run_cell_both_modes(
        lambda s: generate_identifiable(seed=s),
        n_replicates, test="ordinal", cell="continuous",
        label="continuous (oracle)", verbose=verbose))

    # Aligned-threshold generation: coarsening loss at K = 7, 5, 4 (J = 5)
    for cats in (7, 5, 4):
        for coding in ("ev", "direct"):
            results.extend(run_cell_both_modes(
                lambda s, c=cats, cd=coding: make_ordinal_dataset(
                    generate_identifiable(seed=s), n_categories=c,
                    threshold_mode="aligned", coding=cd),
                n_replicates, test="ordinal",
                cell=f"J5_K{cats}_aligned_{coding}",
                label=f"J = 5, K = {cats}, aligned, {coding.upper()}",
                verbose=verbose))

    # Threshold mismatch (floor-skewed, PHQ-9-like marginals), K = 4, J = 5
    for coding in ("ev", "direct"):
        results.extend(run_cell_both_modes(
            lambda s, cd=coding: make_ordinal_dataset(
                generate_identifiable(seed=s), n_categories=4,
                threshold_mode="mismatch", coding=cd),
            n_replicates, test="ordinal", cell=f"J5_K4_mismatch_{coding}",
            label=f"J = 5, K = 4, mismatch, {coding.upper()}",
            verbose=verbose))

    # PHQ-9 configuration: J = 9, K = 4 under threshold mismatch
    for coding in ("ev", "direct"):
        results.extend(run_cell_both_modes(
            lambda s, cd=coding: make_ordinal_dataset(
                generate_identifiable(n_items=9, seed=s), n_categories=4,
                threshold_mode="mismatch", coding=cd),
            n_replicates, test="ordinal", cell=f"J9_K4_mismatch_{coding}",
            label=f"J = 9, K = 4, mismatch, {coding.upper()} (PHQ-9 config)",
            verbose=verbose))

    return results


# =============================================================================
# TEST 8: ITEM COUNT
# =============================================================================

def test_item_count(n_replicates: int, verbose: bool = True) -> List[Dict]:
    if verbose:
        print("\nTest 8: Item count (J = 9 matches the PHQ-9 application)")
        print("-" * 78)
    results = []
    for j in (4, 5, 9):
        results.extend(run_cell_both_modes(
            lambda s, j=j: generate_identifiable(n_items=j, seed=s),
            n_replicates, test="item_count", cell=f"J={j}",
            compute_rho_a=True, label=f"J = {j}", verbose=verbose))
    return results


# =============================================================================
# AGGREGATION, CSV EXPORT, SUMMARY
# =============================================================================

def _union_fieldnames(rows: List[Dict], leading: List[str]) -> List[str]:
    seen = list(leading)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def write_csvs(all_results: List[Dict]):
    lead = ["test", "cell", "mode", "n_replicates"]
    with open(RESULTS_CSV, "w", newline="") as f:
        fields = _union_fieldnames(all_results, lead)
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\nCell-level results written to {RESULTS_CSV}")

    lead_rep = ["test", "cell", "mode", "replicate", "data_seed", "fit_seed"]
    with open(REPLICATES_CSV, "w", newline="") as f:
        fields = _union_fieldnames(REPLICATE_ROWS, lead_rep)
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in REPLICATE_ROWS:
            writer.writerow(row)
    print(f"Replicate-level results written to {REPLICATES_CSV}")


def print_summary(all_results: List[Dict]):
    print("\n" + "=" * 78)
    print("STUDY 4 SUMMARY (per test x mode; recovery vs. generating "
          "allocation, diagnostic)")
    print("=" * 78)
    by_group: Dict[tuple, List[Dict]] = {}
    for row in all_results:
        by_group.setdefault((row["test"], row["mode"]), []).append(row)

    for (test, mode), rows in by_group.items():
        rs = [r["recovery_r_mean"] for r in rows]
        deltas = [r["delta_r_mean"] for r in rows]
        sds = [r["recovery_r_sd"] for r in rows]
        tag = MODE_TAGS.get(mode, mode)
        print(f"  {test:<16}{tag:<12} cells = {len(rows):>2} | "
              f"r range = {min(rs):.3f} - {max(rs):.3f} | "
              f"delta_r range = {min(deltas):+.3f} - {max(deltas):+.3f} | "
              f"max cell SD = {max(sds):.3f}")
    print("=" * 78)


# =============================================================================
# CLI ENTRY
# =============================================================================

TESTS = {
    "sample_size": test_sample_size,
    "noise": test_noise,
    "init": test_initialization,
    "heterogeneity": test_heterogeneity,
    "distributional": test_distributional,
    "corruption": test_corruption,
    "ordinal": test_ordinal,
    "item_count": test_item_count,
}


def run_study4(n_replicates: int = N_REPLICATES_DEFAULT,
               only: Optional[str] = None, verbose: bool = True) -> List[Dict]:
    print("=" * 78)
    print("CAA SIMULATION STUDY 4: ROBUSTNESS (dual-mode full battery)")
    print(f"Modes: profile_consistent + supervised benchmark | "
          f"DGP: identifiable | J = {N_ITEMS} | "
          f"R = {n_replicates} per cell x mode | batch = {BATCH_SIZE}")
    print("=" * 78)
    start = time.time()

    REPLICATE_ROWS.clear()
    all_results: List[Dict] = []
    for name, fn in TESTS.items():
        if only is not None and name != only:
            continue
        all_results.extend(fn(n_replicates, verbose=verbose))

    print_summary(all_results)
    write_csvs(all_results)
    print(f"Total time: {time.time() - start:.1f}s")
    return all_results


if __name__ == "__main__":
    args = sys.argv[1:]
    n_replicates = N_REPLICATES_DEFAULT
    only = None
    for a in args:
        if a.startswith("--replicates="):
            n_replicates = max(2, int(a.split("=")[1]))
        elif a == "--quick":
            n_replicates = 3
        elif a.startswith("--test="):
            only = a.split("=")[1]
            if only not in TESTS:
                raise SystemExit(f"Unknown test '{only}'. "
                                 f"Options: {', '.join(TESTS)}")
    run_study4(n_replicates=n_replicates, only=only)
