"""
CAA Foundation — Supervised + Profile-Consistent
=============================================

Clean single-file implementation of the Confirmatory Attention Analysis (CAA)
measurement framework with two estimation modes:

    supervised      — target = generating allocation (known in simulation only).
                      Probes the representational capacity of the allocation
                      mapping A_theta. Simulation diagnostic; not an applied
                      estimator.

    profile_consistent — the applied estimator. The allocation is trained against an
                      invariant, profile-implied target under stop-gradient:

                          z_i   = (x_i - mean_j x_ij) / (sd_j x_ij + eps)   (profile)
                          a_i   = A_theta(z_i)                              (allocation)
                          a_i*  = softmax(z_i / tau)        (entropy-smoothed target)
                          loss  = JSD(a_i, a_i*) + regularizers
                          c_i   = a_i^T x_i                 (construct score; reported)

                      The target is the entropy-smoothed allocation implied by the
                      within-person response SHAPE. Because the whole objective is
                      a function of the invariant profile z_i (and a_i) only, the
                      ESTIMATOR is invariant to per-person affine transforms
                      x_i -> alpha_i x_i + beta_i, not merely the prediction of a
                      pre-fit model. The construct score c_i = a_i^T x_i reads the
                      ORIGINAL responses and stays level-aware; it is reported but
                      plays no role in the training objective.

Architecture (invariant allocation input)
-----------------------------------------
The allocation reads the WITHIN-PERSON standardized profile

    z_i = (x_i - mean_j x_ij) / (sd_j x_ij + epsilon),

so A_theta(alpha * x + beta * 1) = A_theta(x) for alpha > 0 — exact invariance to
uniform additive shifts and positive rescaling. The construct score reads the
ORIGINAL responses, c_i = a_i^T x_i, so it remains level-aware.

Reliability
-----------
Section 6b provides construct-score and allocation reliability (rho_c, rho_a)
from a person-by-replicate variance decomposition, and the reliability-correction
of ADI. See run_reliability_demo / reliability_from_simulation / reliability_from_fit.

Usage:
    python caa_foundation_vf.py                 # supervised vs PC comparison (default)
    python caa_foundation_vf.py --supervised    # supervised capacity diagnostics
    python caa_foundation_vf.py --pc            # profile-consistent diagnostics
    python caa_foundation_vf.py --reliability   # reliability demo (both replicate sources)
    python caa_foundation_vf.py --items=4       # 4 items instead of 5

Author: Jonathan Lee
Version: 3.0.0
"""
from __future__ import annotations
import os
import sys
import copy
import time
import warnings
from typing import Optional, Dict, List, Tuple

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split


# =============================================================================
# SECTION 1. CONSTANTS & REPRODUCIBILITY
# =============================================================================

__version__ = "3.0.0"

RANDOM_SEED = 123
DEFAULT_NOISE_VARIANCE = 0.15 ** 2
DEFAULT_DIRICHLET_CONCENTRATION = 60.0
PROFILE_EPSILON = 1e-6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_reproducible_state(seed: int = RANDOM_SEED):
    """Set numpy and torch seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_reproducible_state()


def within_person_standardize(responses: np.ndarray,
                              epsilon: float = PROFILE_EPSILON) -> np.ndarray:
    """
    Map each response vector to its within-person standardized profile z_i:

        z_i = (x_i - mean_j x_ij) / (sd_j x_ij + epsilon)

    The mean and SD are taken WITHIN each individual (across items, axis=1), so
    z_i is invariant to a uniform additive shift and to any positive rescaling
    of x_i. A flat profile (all items equal) yields z_i = 0.
    """
    responses = np.asarray(responses, dtype=np.float64)
    mean = responses.mean(axis=1, keepdims=True)
    sd = responses.std(axis=1, keepdims=True)  # ddof=0 -> (1/J) normalization
    z = (responses - mean) / (sd + epsilon)
    return z.astype(np.float32)


# =============================================================================
# SECTION 2. DATA GENERATION
# =============================================================================
# Persona-based generation with realistic, overlapping allocation templates.
# Two DGP modes:
#   "identifiable": nonnegative construct scores, no baseline -> response
#                   composition approximates the generating allocation.
#   "realistic":    signed construct scores + baseline heterogeneity +
#                   standardization (typical applied complications).
# =============================================================================

def _make_item_templates(n_items: int, n_personas: int,
                         primary_strength: float = 0.45,
                         secondary_strength: float = 0.30,
                         baseline: float = 0.08) -> np.ndarray:
    """Realistic overlapping allocation templates (primary + secondary focus)."""
    templates = np.zeros((n_personas, n_items), dtype=float)
    primary_indices = np.linspace(0, n_items - 1, num=n_personas, dtype=int)

    for k in range(n_personas):
        primary_idx = primary_indices[k]
        if n_items > 1:
            offset = 1 if k % 2 == 0 else -1
            secondary_idx = (primary_idx + offset) % n_items
        else:
            secondary_idx = primary_idx

        pattern = np.full(n_items, baseline / max(1, n_items - 2))
        pattern[primary_idx] = primary_strength
        if secondary_idx != primary_idx:
            pattern[secondary_idx] = secondary_strength

        rng = np.random.default_rng(RANDOM_SEED + k)
        pattern = pattern + rng.normal(0, 0.015, n_items)
        pattern = np.maximum(pattern, 0.01)
        pattern = pattern / pattern.sum()
        templates[k] = pattern

    return templates


def _sample_attention_from_template(template: np.ndarray, concentration: float = 60.0,
                                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample an individual allocation from a template via the Dirichlet."""
    if rng is None:
        rng = np.random.default_rng()
    alpha = np.clip(template, 1e-8, None) * float(concentration)
    return rng.dirichlet(alpha)


def _softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Numerically stable softplus."""
    return np.where(x * beta > 20, x, np.log1p(np.exp(beta * x)) / beta)


class UniversalPersonaGenerator:
    """Persona generator across domains and item counts."""

    def __init__(self, n_items: int = 5, n_personas: int = 3):
        self.n_items = n_items
        self.n_personas = n_personas
        self.templates = _make_item_templates(n_items, n_personas)
        self.persona_names = [f"persona_{k}" for k in range(n_personas)]
        self.item_labels = [f"Item_{j+1}" for j in range(n_items)]

    def set_domain_labels(self, item_labels: List[str],
                          persona_names: Optional[List[str]] = None):
        if len(item_labels) != self.n_items:
            raise ValueError(f"Must provide exactly {self.n_items} item labels")
        self.item_labels = item_labels
        if persona_names is not None:
            if len(persona_names) != self.n_personas:
                raise ValueError(f"Must provide exactly {self.n_personas} persona names")
            self.persona_names = persona_names

    def generate_dataset(self, n_samples: int = 800, signal_strength: float = 5.0,
                         noise_variance: Optional[float] = None,
                         baseline_sd: Optional[float] = None,
                         person_scale_sd: Optional[float] = None,
                         construct_shift: Optional[float] = None,
                         construct_sd: Optional[float] = None,
                         concentration: float = DEFAULT_DIRICHLET_CONCENTRATION,
                         standardize: Optional[bool] = None,
                         dgp_mode: str = "realistic",
                         seed: int = RANDOM_SEED) -> Dict:
        """Generate a dataset under the chosen DGP mode."""
        if dgp_mode not in ("identifiable", "realistic"):
            raise ValueError(f"dgp_mode must be 'identifiable' or 'realistic', got '{dgp_mode}'")

        rng = np.random.default_rng(seed)

        if noise_variance is None:
            noise_variance = 0.05 ** 2 if dgp_mode == "identifiable" else DEFAULT_NOISE_VARIANCE
        if standardize is None:
            # The within-person invariance architecture supersedes column
            # standardization. Leaving it off keeps the construct score c = a^T x
            # on its original nonnegative scale (responses are nonnegative), so c
            # remains an interpretable positive level. The profile-consistent
            # target is softmax(z); it is built from the standardized profile and
            # is independent of c, so it is sign-stable regardless.
            standardize = False
        if baseline_sd is None:
            baseline_sd = 0.0 if dgp_mode == "identifiable" else 0.03
        if person_scale_sd is None:
            person_scale_sd = 0.05 if dgp_mode == "identifiable" else 0.10
        if construct_sd is None:
            construct_sd = 1.0 if dgp_mode == "identifiable" else 0.5
        if construct_shift is None:
            construct_shift = 0.0 if dgp_mode == "identifiable" else 1.5

        persona_assignments = rng.integers(0, self.n_personas, size=n_samples)
        individual_assignments = [self.persona_names[k] for k in persona_assignments]

        true_attention_patterns = np.zeros((n_samples, self.n_items))
        for i in range(n_samples):
            template = self.templates[persona_assignments[i]]
            true_attention_patterns[i] = _sample_attention_from_template(
                template, concentration, rng
            )

        if dgp_mode == "identifiable":
            z = rng.normal(0, 1, n_samples)
            true_construct_scores = _softplus(z)
        else:
            true_construct_scores = rng.normal(
                loc=construct_shift, scale=construct_sd, size=n_samples
            )

        responses = np.zeros((n_samples, self.n_items))
        clean_responses = np.zeros((n_samples, self.n_items))  # noise-free signal + baseline

        for i in range(n_samples):
            person_scale = rng.normal(1.0, person_scale_sd)
            base_response = (true_attention_patterns[i] *
                             true_construct_scores[i] *
                             signal_strength * person_scale)

            if dgp_mode == "identifiable":
                baseline = np.zeros(self.n_items)
            else:
                baseline = rng.normal(2, baseline_sd, self.n_items)

            # Construct score, true allocation, person scale, and baseline are
            # person-level and held fixed across measurement replicates; only the
            # noise term is redrawn (see generate_response_replicates).
            clean_responses[i] = base_response + baseline
            noise = rng.normal(0, np.sqrt(noise_variance), self.n_items)
            responses[i] = np.clip(base_response + baseline + noise, 0.0, None)

        if standardize:
            mu = responses.mean(axis=0, keepdims=True)
            sd = responses.std(axis=0, keepdims=True) + 1e-8
            responses = (responses - mu) / sd

        return {
            'responses': responses.astype(np.float32),
            'clean_responses': clean_responses.astype(np.float32),
            'true_attention_patterns': true_attention_patterns.astype(np.float32),
            'true_construct_scores': true_construct_scores.astype(np.float32),
            'individual_assignments': individual_assignments,
            'persona_templates': self.templates.astype(np.float32),
            'item_labels': self.item_labels,
            'persona_names': self.persona_names,
            'n_samples': n_samples,
            'generation_seed': seed,
            'n_items': self.n_items,
            'dgp_mode': dgp_mode,
            'standardize': standardize,
            'noise_variance': noise_variance,
            'baseline_sd': baseline_sd,
            'person_scale_sd': person_scale_sd,
            'construct_shift': construct_shift,
            'construct_sd': construct_sd,
            'concentration': concentration,
        }


# =============================================================================
# SECTION 3. NETWORK ARCHITECTURE
# =============================================================================
# Dual-encoder allocation mapping A_theta. The allocation reads the within-person
# profile z_i; the construct score reads the original responses x_i.
# =============================================================================

class CAANetwork(nn.Module):
    """Dual-encoder allocation network with multi-head attention."""

    def __init__(self, n_items: int = 5, hidden_dim: int = 64,
                 fusion_dim: int = 32, n_attention_heads: int = 3,
                 temperature_init: float = 1.5):
        super().__init__()
        self.n_items = n_items
        self.n_attention_heads = n_attention_heads

        self.primary_encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32)
        )
        self.secondary_encoder = nn.Sequential(
            nn.Linear(n_items, 24), nn.ReLU(), nn.Linear(24, 16)
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(32 + 16, 48), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(48, fusion_dim)
        )
        self.attention_heads = nn.ModuleList([
            nn.Linear(fusion_dim, n_items) for _ in range(n_attention_heads)
        ])
        self.attention_combiner = nn.Linear(n_attention_heads * n_items, n_items)

        temperature_logit_init = np.log(np.exp(max(0.1, temperature_init - 0.5)) - 1.0)
        self.temperature_logit = nn.Parameter(torch.tensor(float(temperature_logit_init)))

        self._initialize_attention_weights()

    def _initialize_attention_weights(self):
        for head in self.attention_heads:
            nn.init.trunc_normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        nn.init.trunc_normal_(self.attention_combiner.weight, std=0.01)
        nn.init.zeros_(self.attention_combiner.bias)

    def forward(self, alloc_input: torch.Tensor, score_input: torch.Tensor,
                return_attention_heads: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            alloc_input: within-person profile z_i [batch, n_items] (drives allocation)
            score_input: original responses x_i [batch, n_items] (forms construct score)
        Returns:
            attention_weights, construct_score[, head_weights]
            where construct_score = a_i^T x_i (the single, manuscript-aligned score).
        """
        primary_context = self.primary_encoder(alloc_input)
        secondary_context = self.secondary_encoder(alloc_input)
        fused_context = self.context_fusion(
            torch.cat([primary_context, secondary_context], dim=1)
        )
        attention_head_logits = [head(fused_context) for head in self.attention_heads]
        raw_attention_logits = self.attention_combiner(
            torch.cat(attention_head_logits, dim=1)
        )
        temperature = torch.clamp(
            F.softplus(self.temperature_logit) + 0.5, min=0.3, max=3.0
        )
        attention_weights = F.softmax(raw_attention_logits / temperature, dim=1)

        # Construct score reads the ORIGINAL responses (level-aware): c_i = a_i^T x_i.
        construct_score = torch.sum(attention_weights * score_input, dim=1, keepdim=True)

        if return_attention_heads:
            head_weights = [
                F.softmax(h / temperature, dim=1) for h in attention_head_logits
            ]
            return attention_weights, construct_score, head_weights
        return attention_weights, construct_score


# =============================================================================
# SECTION 4. LOSS HELPERS & SCHEDULER
# =============================================================================

def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor,
                              eps: float = 1e-8) -> torch.Tensor:
    """Row-wise Jensen-Shannon divergence between probability distributions."""
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum(dim=-1) + (q * (q / m).log()).sum(dim=-1))


def correlation_loss(predicted: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """Mean (1 - r) across rows."""
    p = predicted - predicted.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    num = torch.sum(p * t, dim=1)
    den = torch.sqrt(torch.sum(p ** 2, dim=1) * torch.sum(t ** 2, dim=1) + eps) + eps
    return torch.mean(1 - num / den)


def _compute_entropy(attention: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise entropy of allocation distributions."""
    return -torch.sum(attention * torch.log(torch.clamp(attention, eps, 1.0)), dim=1)


def pc_target(alloc_input: torch.Tensor, tau: float = 1.0,
              eps: float = 1e-8) -> torch.Tensor:
    """
    Invariant, entropy-smoothed profile-implied allocation target:

        a*_i = softmax(z_i / tau)

    z_i is the within-person standardized profile, so the target is a function of
    the response SHAPE only and is invariant to per-person affine transforms
    x_i -> alpha_i x_i + beta_i. The construct score is deliberately NOT used here:
    including the response level would make the target sharpness scale-dependent
    and break ESTIMATOR invariance. tau controls target concentration (calibrate
    so the target ADI matches the expected heterogeneity; tau = 1.0 is the
    default and matches the persona-template ADI used in the simulations).

    This is the entropy-smoothed surrogate for the profile-implied allocation,
    not an exact least-squares simplex projection.
    """
    return F.softmax(alloc_input / max(tau, eps), dim=1)


def profile_anchor_allocation(responses: np.ndarray, tau: float = 1.0,
                              epsilon: float = PROFILE_EPSILON) -> np.ndarray:
    """
    Profile-implied anchor allocation as a numpy array, a*_i = softmax(z_i / tau).

    This is the reference dispersion the profile-consistent estimator is anchored
    to. Use compute_adi(profile_anchor_allocation(x, tau)) to report ADI_anchor
    alongside ADI_true and ADI_learned, so a reader can see whether the learned
    allocation over- or under-states heterogeneity relative to both the truth and
    the invariant profile the estimator is calibrated against. tau is the single
    dispersion knob: larger tau flattens the anchor (lower ADI), smaller sharpens.
    """
    z = within_person_standardize(responses, epsilon)
    z_t = torch.as_tensor(z, dtype=torch.float32)
    a = F.softmax(z_t / max(tau, 1e-8), dim=1)
    return a.detach().cpu().numpy()


class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer, warmup_steps: int = 200, total_steps: int = 2000):
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.current_step = 0
        self.base_learning_rates = [g['lr'] for g in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_learning_rates[i]
            if self.current_step < self.warmup_steps:
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                progress = ((self.current_step - self.warmup_steps) /
                            (self.total_steps - self.warmup_steps))
                lr = base_lr * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))
            group['lr'] = float(lr)


# =============================================================================
# SECTION 5. TRAINER
# =============================================================================
# Two modes:
#   supervised      — target = generating allocation (a-matching; simulation only)
#   profile_consistent — target a* = softmax(z / tau), the profile-implied
#                     allocation; a function of the invariant profile z and a
#                     only (no reconstruction term, so the level c never enters)
# =============================================================================

class CAATrainer:
    """Training procedure for CAA in supervised or profile-consistent mode."""

    SUPERVISED_LOSS_WEIGHTS = {
        'attention_supervision': 12.5,
        # Reconstruction is OFF for the supervised capacity study. c = a^T x has no
        # parameters of its own, so a nonzero reconstruction term only reshapes the
        # allocation toward response peaks, sharpening a beyond the profile anchor
        # and inflating ADI (the recurring Study 2 inflation). Capacity needs only
        # the supervision target; raise this only to deliberately probe c-calibration.
        'reconstruction': 0.00,
        'entropy_floor': 0.05,
        'head_diversity': 0.02,
    }

    # profile-consistent weights. The objective is a function of the invariant
    # profile z and the allocation a ONLY (estimator invariance): JSD to the
    # profile-implied target, with entropy and head-diversity safeguards. No
    # raw-response reconstruction term, since it would reintroduce the level c.
    PC_LOSS_WEIGHTS = {
        'pc_jsd': 1.0,
        'entropy_floor': 0.05,
        'head_diversity': 0.05,
    }

    def __init__(self, n_items: int = 5, n_attention_heads: int = 3,
                 hidden_dim: int = 64, mode: str = 'profile_consistent',
                 loss_weights: Optional[Dict[str, float]] = None,
                 pc_target_tau: float = 1.0,
                 profile_epsilon: float = PROFILE_EPSILON):
        if mode not in ('supervised', 'profile_consistent'):
            raise ValueError(f"mode must be 'supervised' or 'profile_consistent', got '{mode}'")

        self.n_items = n_items
        self.n_attention_heads = n_attention_heads
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.pc_target_tau = pc_target_tau
        self.profile_epsilon = profile_epsilon
        self.entropy_floor = 0.30 * np.log(n_items)

        if loss_weights is not None:
            self.loss_weights = loss_weights
        elif mode == 'profile_consistent':
            self.loss_weights = self.PC_LOSS_WEIGHTS.copy()
        else:
            self.loss_weights = self.SUPERVISED_LOSS_WEIGHTS.copy()

        self.network: Optional[CAANetwork] = None
        self.training_history: Dict[str, list] = {}
        self.best_model_state = None

    # ---- input preparation --------------------------------------------------
    def _prepare_inputs(self, responses: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (alloc_input = z_i, score_input = x_i)."""
        responses = np.asarray(responses, dtype=np.float32)
        alloc_np = within_person_standardize(responses, self.profile_epsilon)
        return (torch.as_tensor(alloc_np, dtype=torch.float32, device=DEVICE),
                torch.as_tensor(responses, dtype=torch.float32, device=DEVICE))

    # ---- losses -------------------------------------------------------------
    def _compute_losses(self, preds: Dict[str, torch.Tensor],
                        targets: Dict[str, torch.Tensor],
                        alloc_input: torch.Tensor,
                        score_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        device = score_input.device
        a = preds['attention_weights']
        c = preds['construct_score'].reshape(-1, 1)

        if self.mode == 'supervised':
            # Capacity probe: match the generating allocation. A reconstruction
            # term (centered, using c = a^T x) helps calibrate the construct
            # score; invariance is not claimed for this diagnostic mode.
            t = torch.clamp(targets['attention_weights'], 1e-8, 1.0)
            p = torch.clamp(a, 1e-8, 1.0)
            js = jensen_shannon_divergence(p, t).mean()
            losses['attention_supervision'] = js + 0.5 * correlation_loss(p, t)

            x_hat = c * a
            x_hat_c = x_hat - x_hat.mean(dim=1, keepdim=True)
            score_c = score_input - score_input.mean(dim=1, keepdim=True)
            losses['reconstruction'] = F.huber_loss(x_hat_c, score_c, delta=0.35)
        else:  # profile_consistent — invariant objective, function of z and a only
            target_a = pc_target(alloc_input, self.pc_target_tau)
            losses['pc_jsd'] = jensen_shannon_divergence(a, target_a).mean()

        # Entropy floor (prevents degenerate collapse)
        entropy_violation = F.relu(self.entropy_floor - _compute_entropy(a))
        losses['entropy_floor'] = torch.mean(entropy_violation)

        # Head diversity (complementary specialization)
        if 'attention_heads' in preds:
            heads = preds['attention_heads']
            hd = torch.tensor(0.0, device=device)
            for i in range(len(heads)):
                for j in range(i + 1, len(heads)):
                    hd = hd + F.cosine_similarity(heads[i], heads[j], dim=1).mean() ** 2
            losses['head_diversity'] = hd
        else:
            losses['head_diversity'] = torch.tensor(0.0, device=device)

        return losses

    # ---- validation metric --------------------------------------------------
    def _validation_metric(self, alloc_val: torch.Tensor, score_val: torch.Tensor,
                           y_val: Optional[torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            a_val, _ = self.network(alloc_val, score_val)

            if self.mode == 'supervised' and y_val is not None:
                corrs = [torch.corrcoef(torch.stack([a_val[i], y_val[i]]))[0, 1]
                         for i in range(len(a_val))]
                corr = float(np.nanmean([c.item() for c in corrs]))
                return {'val_metric': corr, 'val_corr': corr, 'val_pc_jsd': float('nan')}

            # profile_consistent: early-stop on the validation PC loss so the
            # criterion cannot be gamed by a degenerate near-uniform allocation.
            target = pc_target(alloc_val, self.pc_target_tau)
            jsd = jensen_shannon_divergence(a_val, target).mean().item()
            ent = F.relu(self.entropy_floor - _compute_entropy(a_val)).mean().item()

            w = self.loss_weights
            val_loss = (w.get('pc_jsd', 1.0) * jsd +
                        w.get('entropy_floor', 0.05) * ent)
            corrs = [torch.corrcoef(torch.stack([a_val[i], target[i]]))[0, 1]
                     for i in range(len(a_val))]
            corr = float(np.nanmean([cc.item() for cc in corrs]))
            return {'val_metric': -val_loss, 'val_corr': corr, 'val_pc_jsd': jsd}

    # ---- training -----------------------------------------------------------
    def train(self, responses: np.ndarray,
              true_attention_patterns: Optional[np.ndarray] = None,
              n_epochs: int = 220, learning_rate: float = 0.002,
              batch_size: int = 256, validation_split: float = 0.15,
              early_stopping_patience: int = 30, min_improvement: float = 1e-4,
              verbose: bool = True) -> Dict[str, float]:
        """
        Train the CAA network.

        supervised:      true_attention_patterns are the training target.
        profile_consistent: true_attention_patterns (if given) are used only for a
                         diagnostic recovery correlation, never for training.
        """
        start_time = time.time()
        if verbose:
            print(f"Training in '{self.mode}' mode | {self.n_items} items")

        alloc_t, score_t = self._prepare_inputs(responses)

        y_train = y_val = y_val_gt = None
        if true_attention_patterns is not None:
            y_tensor = torch.as_tensor(true_attention_patterns, dtype=torch.float32, device=DEVICE)
            if self.mode == 'supervised':
                y_train = 0.98 * y_tensor + 0.02 * (1.0 / self.n_items)
            y_val_gt = y_tensor

        idx = np.arange(len(score_t))
        if validation_split > 0:
            tr, va = train_test_split(idx, test_size=validation_split, random_state=RANDOM_SEED)
            alloc_tr, alloc_va = alloc_t[tr], alloc_t[va]
            score_tr, score_va = score_t[tr], score_t[va]
            y_train = y_train[tr] if y_train is not None else None
            y_val_gt = y_val_gt[va] if y_val_gt is not None else None
        else:
            alloc_tr, score_tr = alloc_t, score_t
            alloc_va = score_va = None

        self.network = CAANetwork(
            n_items=self.n_items, n_attention_heads=self.n_attention_heads,
            hidden_dim=self.hidden_dim
        ).to(DEVICE)

        optimizer = optim.Adam([
            {'params': self.network.primary_encoder.parameters(), 'lr': learning_rate, 'weight_decay': 1e-5},
            {'params': self.network.secondary_encoder.parameters(), 'lr': learning_rate * 0.8, 'weight_decay': 1e-5},
            {'params': self.network.context_fusion.parameters(), 'lr': learning_rate, 'weight_decay': 1e-6},
            {'params': self.network.attention_heads.parameters(), 'lr': learning_rate * 1.2, 'weight_decay': 5e-6},
            {'params': self.network.attention_combiner.parameters(), 'lr': learning_rate * 1.1, 'weight_decay': 5e-6},
            {'params': [self.network.temperature_logit], 'lr': learning_rate * 0.5, 'weight_decay': 0.0},
        ])
        n_batches = int(np.ceil(len(alloc_tr) / batch_size))
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=min(100, n_batches * 2), total_steps=n_epochs * n_batches
        )

        best_metric = -np.inf
        patience = 0
        self.training_history = {
            'train_loss': [], 'val_metric': [], 'val_corr': [],
            'val_pc_jsd': [], 'val_gt_corr': [], 'temperature': []
        }

        epoch = 0
        for epoch in range(n_epochs):
            self.network.train()
            epoch_loss = 0.0
            perm = torch.randperm(len(alloc_tr), device=DEVICE)

            for b in range(n_batches):
                bidx = perm[b * batch_size:(b + 1) * batch_size]
                ba, bs = alloc_tr[bidx], score_tr[bidx]
                targets = {'attention_weights': y_train[bidx]} if y_train is not None else {}

                optimizer.zero_grad()
                att, c, heads = self.network(ba, bs, return_attention_heads=True)
                preds = {'attention_weights': att, 'construct_score': c,
                         'attention_heads': heads}
                losses = self._compute_losses(preds, targets, ba, bs)
                loss = sum(self.loss_weights.get(k, 0.0) * v for k, v in losses.items())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            val = {'val_metric': -1.0, 'val_corr': float('nan'), 'val_pc_jsd': float('nan')}
            val_gt = float('nan')
            if alloc_va is not None:
                self.network.eval()
                with torch.no_grad():
                    a_va, _ = self.network(alloc_va, score_va)
                    val = self._validation_metric(alloc_va, score_va, y_val_gt)
                    if y_val_gt is not None:
                        gt = [torch.corrcoef(torch.stack([a_va[k], y_val_gt[k]]))[0, 1].item()
                              for k in range(len(a_va))]
                        val_gt = float(np.nanmean(gt))

            with torch.no_grad():
                cur_temp = float(torch.clamp(
                    F.softplus(self.network.temperature_logit) + 0.5, min=0.3, max=3.0
                ).item())

            self.training_history['train_loss'].append(epoch_loss / n_batches)
            self.training_history['val_metric'].append(val['val_metric'])
            self.training_history['val_corr'].append(val.get('val_corr', float('nan')))
            self.training_history['val_pc_jsd'].append(val.get('val_pc_jsd', float('nan')))
            self.training_history['val_gt_corr'].append(val_gt)
            self.training_history['temperature'].append(cur_temp)

            if val['val_metric'] > best_metric + min_improvement:
                best_metric = val['val_metric']
                patience = 0
                self.best_model_state = copy.deepcopy(self.network.state_dict())
            else:
                patience += 1

            if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                msg = f"Epoch {epoch+1}: loss={epoch_loss / n_batches:.4f}"
                if self.mode == 'profile_consistent':
                    msg += f", PC-JSD={val.get('val_pc_jsd', float('nan')):.4f}, corr={val.get('val_corr', float('nan')):.3f}"
                    if not np.isnan(val_gt):
                        msg += f", GT={val_gt:.3f}(diag)"
                else:
                    msg += f", corr={val.get('val_corr', float('nan')):.3f}"
                print(msg)

            if patience >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if self.best_model_state is not None:
            self.network.load_state_dict(self.best_model_state)

        return {
            'training_time': time.time() - start_time,
            'epochs_trained': epoch + 1,
            'final_val_corr': self.training_history['val_corr'][-1],
            'final_val_pc_jsd': self.training_history['val_pc_jsd'][-1],
            'final_gt_corr': self.training_history['val_gt_corr'][-1],
        }

    def predict_attention_patterns(self, responses: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict allocation and construct score for new responses."""
        if self.network is None:
            raise ValueError("Model not trained — call train() first.")
        alloc_t, score_t = self._prepare_inputs(responses)
        self.network.eval()
        with torch.no_grad():
            att, c = self.network(alloc_t, score_t)
        return {
            'attention_weights': att.detach().cpu().numpy(),
            'construct_score': c.detach().cpu().numpy().flatten(),
        }


def within_person_standardize_tensor(x: torch.Tensor, eps: float = PROFILE_EPSILON) -> torch.Tensor:
    """Within-person standardization for a torch tensor (kept as a utility)."""
    mean = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, unbiased=False, keepdim=True)
    return (x - mean) / (sd + eps)


# =============================================================================
# SECTION 6. EVALUATION UTILITIES
# =============================================================================

def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def compute_adi(attention_patterns: np.ndarray) -> float:
    """
    Allocation Diversity Index (ADI), Eq. 14:

        ADI = sum_j Var_i(a_ij) / (1 - 1/J)

    Near 0 indicates homogeneous measurement; larger values indicate meaningful
    individual differences in allocation.
    """
    n_items = attention_patterns.shape[1]
    max_var = (1 / n_items) * (1 - 1 / n_items)
    observed_var = np.sum(np.var(attention_patterns, axis=0))
    return float(observed_var / (n_items * max_var))


def evaluate_attention_recovery(
    predicted: Optional[np.ndarray] = None,
    true: Optional[np.ndarray] = None,
    *,
    predicted_attention: Optional[np.ndarray] = None,
    true_attention: Optional[np.ndarray] = None,
    individual_assignments: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Mean individual-level allocation recovery (correlation with true).

    Supports both legacy names (`predicted`, `true`) and explicit names
    (`predicted_attention`, `true_attention`) for compatibility across scripts.
    """
    if predicted is None:
        predicted = predicted_attention
    if true is None:
        true = true_attention
    if predicted is None or true is None:
        raise TypeError(
            "evaluate_attention_recovery requires predicted/true arrays "
            "(or predicted_attention/true_attention)."
        )

    corrs = []
    for i in range(len(predicted)):
        c = np.corrcoef(predicted[i], true[i])[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    corrs = np.array(corrs)
    return {
        'mean_individual_correlation': float(np.mean(corrs)),
        'std_individual_correlation': float(np.std(corrs)),
        'strong_recovery_rate': float(np.mean(corrs > 0.7)),
        'excellent_recovery_rate': float(np.mean(corrs > 0.9)),
    }


# =============================================================================
# SECTION 6b. RELIABILITY (PERSON-BY-REPLICATE PRECISION)
# =============================================================================
#   rho_c = sigma2_{b,c} / (sigma2_{b,c} + sigma2_{w,c})                 (Eq. 15)
#   rho_a = sum_j sigma2_{b,j} / sum_j (sigma2_{b,j} + sigma2_{w,j})     (Eq. 16)
#   ADI_true = rho_a * ADI_obs                                           (Eq. 17)
#
# b = between-person (reproducible signal); w = within-person across replicates
# (estimation error). Two replicate sources:
#   simulation  — fresh measurement error on a fixed latent (generate_response_replicates)
#   application — model-based replicates from x_hat = c * a (reliability_from_fit)
# =============================================================================

def _icc_components(y_rn: np.ndarray) -> Dict[str, float]:
    """One-way random-effects variance components for a scalar quantity [R, N]."""
    y = np.asarray(y_rn, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y_rn must be [n_replicates, n_persons].")
    R, N = y.shape
    if R < 2:
        raise ValueError("Reliability requires at least 2 replicates.")
    person_mean = y.mean(axis=0)
    grand_mean = person_mean.mean()
    ms_between = R * np.sum((person_mean - grand_mean) ** 2) / max(N - 1, 1)
    ms_within = np.sum((y - person_mean[None, :]) ** 2) / max(N * (R - 1), 1)
    sigma2_w = float(ms_within)
    sigma2_b = float(max((ms_between - ms_within) / R, 0.0))
    denom = sigma2_b + sigma2_w
    return {'sigma2_b': sigma2_b, 'sigma2_w': sigma2_w,
            'rho': float(sigma2_b / denom) if denom > 0 else float('nan')}


def compute_reliability(allocation_replicates: np.ndarray,
                        construct_replicates: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Construct-score and allocation reliability from replicate estimates.

    Args:
        allocation_replicates: [n_replicates, n_persons, n_items]
        construct_replicates:  optional [n_replicates, n_persons]
    Returns dict: rho_a, rho_c, sigma2_b, sigma2_w, adi_obs, adi_corrected,
                  n_replicates, n_persons.
    """
    a = np.asarray(allocation_replicates, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError("allocation_replicates must be [n_replicates, n_persons, n_items].")
    R, N, J = a.shape

    sigma2_b = np.zeros(J)
    sigma2_w = np.zeros(J)
    for j in range(J):
        comp = _icc_components(a[:, :, j])
        sigma2_b[j], sigma2_w[j] = comp['sigma2_b'], comp['sigma2_w']

    between_total = float(np.sum(sigma2_b))
    within_total = float(np.sum(sigma2_w))
    observed_total = between_total + within_total
    rho_a = float(between_total / observed_total) if observed_total > 0 else float('nan')

    norm = 1.0 - 1.0 / J
    adi_obs = observed_total / norm
    # Replicate-error correction ONLY: removes within-person (replicate) variance.
    # It does NOT correct systematic shrinkage/bias in the learned mapping, so it
    # is NOT an estimate of the true generating ADI. Compare against ADI_gen and
    # ADI_learned separately to see estimator bias.
    adi_corrected = rho_a * adi_obs if not np.isnan(rho_a) else float('nan')

    rho_c = float('nan')
    if construct_replicates is not None:
        c = np.asarray(construct_replicates, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError("construct_replicates must be [n_replicates, n_persons].")
        rho_c = _icc_components(c)['rho']

    return {'rho_a': rho_a, 'rho_c': rho_c, 'sigma2_b': sigma2_b, 'sigma2_w': sigma2_w,
            'adi_obs': float(adi_obs), 'adi_corrected': float(adi_corrected),
            'n_replicates': int(R), 'n_persons': int(N)}


def generate_response_replicates(base_dataset: Dict, n_replicates: int = 10,
                                 seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Simulation replicate source. Regenerates responses that SHARE each person's
    construct score, true allocation, person scale, and baseline (carried in
    'clean_responses') but carry INDEPENDENT measurement error.
    Returns [n_replicates, n_persons, n_items].
    """
    if 'clean_responses' not in base_dataset:
        raise KeyError("base_dataset has no 'clean_responses'; regenerate with generate_dataset.")
    clean = np.asarray(base_dataset['clean_responses'], dtype=np.float64)
    noise_var = float(base_dataset['noise_variance'])
    standardize = bool(base_dataset.get('standardize', False))
    N, J = clean.shape
    rng = np.random.default_rng(seed)

    reps = np.zeros((n_replicates, N, J), dtype=np.float32)
    for r in range(n_replicates):
        obs = np.clip(clean + rng.normal(0.0, np.sqrt(noise_var), size=(N, J)), 0.0, None)
        if standardize:
            mu = obs.mean(axis=0, keepdims=True)
            sd = obs.std(axis=0, keepdims=True) + 1e-8
            obs = (obs - mu) / sd
        reps[r] = obs.astype(np.float32)
    return reps


def reliability_from_simulation(trainer: 'CAATrainer', base_dataset: Dict,
                                n_replicates: int = 10, seed: int = RANDOM_SEED,
                                verbose: bool = True) -> Dict[str, object]:
    """
    Reliability via the simulation replicate source (true allocation known),
    conditional on the fitted allocation mapping (replicates pass through the
    same trained model; this does not include uncertainty from refitting).
    """
    reps = generate_response_replicates(base_dataset, n_replicates, seed)
    alloc_reps, con_reps = [], []
    for r in range(reps.shape[0]):
        pred = trainer.predict_attention_patterns(reps[r])
        alloc_reps.append(pred['attention_weights'])
        con_reps.append(pred['construct_score'])
    result = compute_reliability(np.stack(alloc_reps, 0), np.stack(con_reps, 0))
    result['conditional_on_fitted_mapping'] = True

    # ADI_learned: dispersion of the single-fit allocation on the base data.
    # Comparing it to ADI_generating exposes estimator bias (shrinkage / inflation)
    # that the replicate-error correction cannot repair.
    base_pred = trainer.predict_attention_patterns(base_dataset['responses'])
    result['adi_learned'] = compute_adi(base_pred['attention_weights'])
    if 'true_attention_patterns' in base_dataset:
        result['adi_generating'] = compute_adi(
            np.asarray(base_dataset['true_attention_patterns'])
        )
    if verbose:
        _print_reliability("simulation replicates", result)
    return result


def reliability_from_fit(trainer: 'CAATrainer', responses: np.ndarray,
                         n_replicates: int = 10, residual_sd: Optional[np.ndarray] = None,
                         seed: int = RANDOM_SEED, verbose: bool = True) -> Dict[str, object]:
    """
    Reliability via model-based replicates (application route; no true parameter),
    conditional on the fitted allocation mapping.
    """
    responses = np.asarray(responses, dtype=np.float64)
    base = trainer.predict_attention_patterns(responses)
    a_hat, c_hat = base['attention_weights'], base['construct_score']
    x_hat = c_hat[:, None] * a_hat
    if residual_sd is None:
        residual_sd = (responses - x_hat).std(axis=0)
    residual_sd = np.asarray(residual_sd, dtype=np.float64).reshape(-1)

    N, J = x_hat.shape
    rng = np.random.default_rng(seed)
    alloc_reps, con_reps = [], []
    for _ in range(n_replicates):
        rep = (x_hat + rng.normal(0.0, 1.0, size=(N, J)) * residual_sd[None, :]).astype(np.float32)
        pred = trainer.predict_attention_patterns(rep)
        alloc_reps.append(pred['attention_weights'])
        con_reps.append(pred['construct_score'])
    result = compute_reliability(np.stack(alloc_reps, 0), np.stack(con_reps, 0))
    result['residual_sd'] = residual_sd
    result['conditional_on_fitted_mapping'] = True
    result['adi_learned'] = compute_adi(a_hat)
    if verbose:
        _print_reliability("model-based replicates", result)
    return result


def _print_reliability(label: str, result: Dict[str, object]):
    print(f"\nReliability ({label}; conditional on the fitted allocation mapping)")
    print("-" * 64)
    print(f"  Replicates: {result['n_replicates']} | persons: {result['n_persons']}")
    print(f"  rho_a (allocation):      {result['rho_a']:.3f}")
    print(f"  rho_c (construct score): {result['rho_c']:.3f}")
    if 'adi_generating' in result:
        print(f"  ADI generating (true):   {result['adi_generating']:.3f}")
    if 'adi_learned' in result:
        print(f"  ADI learned (single fit):{result['adi_learned']:.3f}   <- estimator bias vs generating")
    print(f"  ADI observed (w/ replicate error): {result['adi_obs']:.3f}")
    print(f"  ADI corrected (replicate error only): {result['adi_corrected']:.3f}")
    print("  Note: correction removes replicate error only, not mapping bias.")


# =============================================================================
# SECTION 7. DIAGNOSTIC DRIVERS
# =============================================================================

def _default_generator(n_items: int) -> UniversalPersonaGenerator:
    gen = UniversalPersonaGenerator(n_items=n_items, n_personas=3)
    if n_items == 5:
        gen.set_domain_labels(
            item_labels=['Mood', 'Anhedonia', 'Cognitive', 'Somatic', 'Social'],
            persona_names=['mood_focused', 'somatic_focused', 'cognitive_social']
        )
    elif n_items == 4:
        gen.set_domain_labels(
            item_labels=['Social Withdrawal', 'Somatic Symptoms', 'Sleep/Appetite', 'Cognitive/Anhedonia'],
            persona_names=['cognitive_focused', 'somatic_focused', 'social_focused']
        )
    return gen


def run_diagnostics(n_samples: int = 800, n_items: int = 5,
                    mode: str = 'profile_consistent', dgp_mode: str = 'realistic',
                    verbose: bool = True) -> Dict:
    """Train one model and report allocation recovery, ADI, and profile-consistency."""
    set_reproducible_state(RANDOM_SEED)
    gen = _default_generator(n_items)
    dataset = gen.generate_dataset(n_samples=n_samples, dgp_mode=dgp_mode)
    true_a = dataset['true_attention_patterns']

    if verbose:
        print("=" * 72)
        print(f"CAA Diagnostics | mode={mode} | DGP={dgp_mode} | items={n_items} | N={n_samples}")
        print("=" * 72)

    trainer = CAATrainer(n_items=n_items, n_attention_heads=3, mode=mode)
    train_res = trainer.train(responses=dataset['responses'],
                              true_attention_patterns=true_a,
                              n_epochs=220, verbose=verbose)

    pred = trainer.predict_attention_patterns(dataset['responses'])
    a_hat = pred['attention_weights']
    rec = evaluate_attention_recovery(a_hat, true_a)

    if verbose:
        print("\nRecovery vs generating allocation"
              + (" (diagnostic only — not a training target)" if mode == 'profile_consistent' else ""))
        print(f"  Mean individual r:   {rec['mean_individual_correlation']:.3f}")
        print(f"  Strong (r>0.7):      {rec['strong_recovery_rate']:.1%}")
        print(f"  Excellent (r>0.9):   {rec['excellent_recovery_rate']:.1%}")
        print(f"  ADI true / anchor / learned:  {compute_adi(true_a):.3f} / "
              f"{compute_adi(profile_anchor_allocation(dataset['responses'], trainer.pc_target_tau)):.3f} / "
              f"{compute_adi(a_hat):.3f}")
        if mode == 'profile_consistent':
            print(f"  Final PC-JSD:        {train_res['final_val_pc_jsd']:.4f}")
            print(f"  Final PC corr:       {train_res['final_val_corr']:.3f}")

    return {'dataset': dataset, 'trainer': trainer, 'predictions': pred,
            'recovery': rec, 'training_results': train_res}


def compare_supervised_pc(n_samples: int = 800, n_items: int = 5,
                          dgp_mode: str = 'realistic', verbose: bool = True) -> Dict:
    """Side-by-side supervised (capacity) vs profile-consistent (applied) on one DGP."""
    if verbose:
        print("\n>>> SUPERVISED (architectural capacity) <<<")
    sup = run_diagnostics(n_samples, n_items, 'supervised', dgp_mode, verbose)
    if verbose:
        print("\n>>> PROFILE-CONSISTENT (applied estimator) <<<")
    pc = run_diagnostics(n_samples, n_items, 'profile_consistent', dgp_mode, verbose)

    if verbose:
        s, c = sup['recovery'], pc['recovery']
        print("\n" + "=" * 72)
        print(f"{'Metric':<34}{'Supervised':>18}{'Profile-Consistent':>18}")
        print("-" * 72)
        print(f"{'Mean allocation recovery r':<34}"
              f"{s['mean_individual_correlation']:>18.3f}{c['mean_individual_correlation']:>18.3f}")
        print(f"{'Strong recovery (r>0.7)':<34}"
              f"{s['strong_recovery_rate']:>18.1%}{c['strong_recovery_rate']:>18.1%}")
        print("-" * 72)
        print("Supervised probes capacity (target = generating allocation).")
        print("Profile-consistent is the applied estimator (model-induced target).")
    return {'supervised': sup, 'profile_consistent': pc}


def run_reliability_demo(n_samples: int = 800, n_items: int = 5,
                         dgp_mode: str = 'realistic', mode: str = 'profile_consistent',
                         n_replicates: int = 12, verbose: bool = True) -> Dict:
    """Train once, then report reliability via both replicate sources."""
    set_reproducible_state(RANDOM_SEED)
    gen = _default_generator(n_items)
    dataset = gen.generate_dataset(n_samples=n_samples, dgp_mode=dgp_mode)

    trainer = CAATrainer(n_items=n_items, n_attention_heads=3, mode=mode)
    trainer.train(responses=dataset['responses'],
                  true_attention_patterns=dataset['true_attention_patterns'],
                  n_epochs=220, verbose=False)

    if verbose:
        print("=" * 72)
        print(f"RELIABILITY DEMO | mode={mode} | DGP={dgp_mode} | items={n_items} "
              f"| N={n_samples} | R={n_replicates}")
        print("=" * 72)
    sim = reliability_from_simulation(trainer, dataset, n_replicates, verbose=verbose)
    fit = reliability_from_fit(trainer, dataset['responses'], n_replicates, verbose=verbose)
    return {'trainer': trainer, 'dataset': dataset,
            'reliability_simulation': sim, 'reliability_fit': fit}


# =============================================================================
# SECTION 7c. CLOSED FORM vs LEARNED ALLOCATION
# =============================================================================
# Does the network add anything over the one-line closed form a* = softmax(z/tau)?
# Compared across two regimes (linear: response proportional to allocation;
# nonlinear: allocation a function of the construct level, non-monotone in the
# profile) and three estimators (raw closed form; profile-consistent A_theta;
# supervised A_theta). The honest result this is designed to surface:
#   linear    -> closed_form ~= pc_learned ~= supervised  (graceful reduction)
#   nonlinear -> closed_form low; pc_learned ~= closed_form (the PC target IS the
#                monotone closed form, so the applied estimator inherits its
#                ceiling); supervised recovers (architectural capacity).
# =============================================================================

class NonlinearLevelGenerator:
    """
    J=4 generator where allocation is a function of the construct level eta.
    Two patterns (Saturation, Curvilinear) are non-monotone in the response
    profile, so the monotone closed form softmax(z/tau) cannot recover them.
    """

    def __init__(self):
        self.n_items = 4
        self.item_labels = ['Threshold', 'Saturation', 'Curvilinear', 'Reference']

    def _alloc(self, eta: np.ndarray) -> np.ndarray:
        a = np.zeros((len(eta), 4))
        a[:, 0] = np.where(eta > 2.5, 0.40, 0.15)                       # threshold step
        a[:, 1] = 0.12 + 0.28 * np.exp(-0.8 * np.maximum(eta, 0.0))      # saturation (declines)
        a[:, 2] = 0.16 + (0.42 - 0.16) * np.exp(-0.5 * ((eta - 2.5) / 1.5) ** 2)  # curvilinear peak
        a[:, 3] = 0.25                                                   # reference (constant raw)
        return a / a.sum(axis=1, keepdims=True)

    def generate_dataset(self, n_samples: int = 800, signal_strength: float = 2.5,
                         noise_sd: float = 0.10, baseline_mean: float = 2.0,
                         seed: int = RANDOM_SEED) -> Dict:
        rng = np.random.default_rng(seed)
        eta = rng.uniform(0.5, 4.5, n_samples)
        true_a = self._alloc(eta)
        signal = true_a * eta[:, None] * signal_strength
        baseline = rng.normal(baseline_mean, 0.08, size=(n_samples, 4))  # fixed per person-item
        clean = signal + baseline
        responses = np.clip(clean + rng.normal(0.0, noise_sd, size=(n_samples, 4)), 1.0, 7.0)
        return {
            'responses': responses.astype(np.float32),
            'clean_responses': clean.astype(np.float32),
            'true_attention_patterns': true_a.astype(np.float32),
            'true_construct_scores': eta.astype(np.float32),
            'item_labels': self.item_labels,
            'n_items': 4, 'n_samples': n_samples,
            'dgp_mode': 'nonlinear', 'standardize': False,
            'noise_variance': float(noise_sd ** 2), 'baseline_sd': 0.08,
        }


def _mean_recovery(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean within-person correlation between predicted and true allocation."""
    cs = [np.corrcoef(pred[i], true[i])[0, 1] for i in range(len(pred))]
    cs = [c for c in cs if not np.isnan(c)]
    return float(np.mean(cs)) if cs else float('nan')


def _rho_a_for_allocator(alloc_fn, base_dataset: Dict,
                         n_replicates: int = 8, seed: int = RANDOM_SEED) -> float:
    """Allocation reliability rho_a for any allocator alloc_fn(responses)->[N,J]."""
    reps = generate_response_replicates(base_dataset, n_replicates, seed)
    stacks = np.stack([alloc_fn(reps[r]) for r in range(reps.shape[0])], axis=0)
    return compute_reliability(stacks)['rho_a']


def run_closed_form_vs_learned(n_samples: int = 800, n_epochs: int = 220,
                               n_replicates: int = 8, tau: float = 1.0,
                               verbose: bool = True) -> Dict:
    """
    Head-to-head: raw closed form softmax(z/tau) vs learned A_theta
    (profile-consistent and supervised), on linear vs nonlinear allocation,
    scored on recovery (vs generating allocation) and rho_a (reliability).
    """
    set_reproducible_state(RANDOM_SEED)
    lin = _default_generator(5).generate_dataset(n_samples=n_samples, dgp_mode='identifiable')
    nl = NonlinearLevelGenerator().generate_dataset(n_samples=n_samples)
    regimes = {
        'linear (J=5, response proportional to allocation)': (lin, 5),
        'nonlinear (J=4, level-dependent allocation)': (nl, 4),
    }

    if verbose:
        print("=" * 92)
        print(f"CLOSED FORM vs LEARNED | softmax(z/tau={tau}) vs A_theta | N={n_samples}")
        print("=" * 92)

    results: Dict[str, Dict] = {}
    for label, (ds, J) in regimes.items():
        true_a = ds['true_attention_patterns']
        adi_true = compute_adi(true_a)
        row: Dict[str, Dict[str, float]] = {}

        # 1. Raw closed form (no network)
        cf = profile_anchor_allocation(ds['responses'], tau)
        row['closed_form'] = {
            'recovery': _mean_recovery(cf, true_a),
            'rho_a': _rho_a_for_allocator(
                lambda r: profile_anchor_allocation(r, tau), ds, n_replicates),
            'adi': compute_adi(cf),
        }

        # 2. Profile-consistent A_theta (the applied estimator)
        set_reproducible_state(RANDOM_SEED)
        tr_pc = CAATrainer(n_items=J, n_attention_heads=3,
                           mode='profile_consistent', pc_target_tau=tau)
        tr_pc.train(responses=ds['responses'], true_attention_patterns=true_a,
                    n_epochs=n_epochs, verbose=False)
        pc = tr_pc.predict_attention_patterns(ds['responses'])['attention_weights']
        row['pc_learned'] = {
            'recovery': _mean_recovery(pc, true_a),
            'rho_a': reliability_from_simulation(tr_pc, ds, n_replicates, verbose=False)['rho_a'],
            'adi': compute_adi(pc),
        }

        # 3. Supervised A_theta (capacity ceiling)
        set_reproducible_state(RANDOM_SEED)
        tr_sup = CAATrainer(n_items=J, n_attention_heads=3, mode='supervised')
        tr_sup.train(responses=ds['responses'], true_attention_patterns=true_a,
                     n_epochs=n_epochs, verbose=False)
        sup = tr_sup.predict_attention_patterns(ds['responses'])['attention_weights']
        row['supervised'] = {
            'recovery': _mean_recovery(sup, true_a),
            'rho_a': reliability_from_simulation(tr_sup, ds, n_replicates, verbose=False)['rho_a'],
            'adi': compute_adi(sup),
        }

        results[label] = {'adi_true': adi_true, 'estimators': row}

        if verbose:
            print(f"\n{label}   ADI_true={adi_true:.3f}")
            print(f"  {'estimator':<14}{'recovery_r':>12}{'rho_a':>10}{'ADI_learn':>11}")
            print("  " + "-" * 45)
            for est in ('closed_form', 'pc_learned', 'supervised'):
                m = row[est]
                print(f"  {est:<14}{m['recovery']:>12.3f}{m['rho_a']:>10.3f}{m['adi']:>11.3f}")

    if verbose:
        print("\n" + "=" * 92)
        print("Reading:")
        print("  LINEAR: closed_form ~= pc_learned ~= supervised -> graceful reduction.")
        print("          The network reproduces the one-line transform where it suffices;")
        print("          on simple heterogeneity CAA coincides with the closed-form (iCSR) estimator.")
        print("  NONLINEAR: closed_form is low (it is monotone in the profile), and")
        print("          pc_learned ~= closed_form because the profile-consistent TARGET *is*")
        print("          the monotone closed form -- the applied estimator inherits its ceiling.")
        print("          supervised recovers the non-monotone patterns: the architecture has")
        print("          capacity the closed form lacks, not accessible from an anchor target.")
        print("  => Neural value over the closed form = capacity (supervised) + extensibility,")
        print("     not a better applied allocation under the profile-consistent objective.")
        print("=" * 92)

    return results


# =============================================================================
# SECTION 8. CLI ENTRY
# =============================================================================

if __name__ == "__main__":
    args = sys.argv[1:]
    n_items = 4 if '--items=4' in args else 5

    if '--supervised' in args:
        run_diagnostics(n_samples=800, n_items=n_items, mode='supervised',
                        dgp_mode='realistic', verbose=True)
    elif '--pc' in args or '--profile-consistent' in args:
        run_diagnostics(n_samples=800, n_items=n_items, mode='profile_consistent',
                        dgp_mode='realistic', verbose=True)
    elif '--reliability' in args or '-r' in args:
        run_reliability_demo(n_samples=800, n_items=n_items, verbose=True)
    elif '--closed-vs-learned' in args:
        run_closed_form_vs_learned(n_samples=800, verbose=True)
    else:
        print("=" * 72)
        print(">>> CAA FOUNDATION: SUPERVISED vs PROFILE-CONSISTENT (DEFAULT) <<<")
        print("=" * 72)
        print("Options: --supervised | --pc | --reliability | --closed-vs-learned | --items=4\n")
        compare_supervised_pc(n_samples=800, n_items=n_items,
                              dgp_mode='realistic', verbose=True)
