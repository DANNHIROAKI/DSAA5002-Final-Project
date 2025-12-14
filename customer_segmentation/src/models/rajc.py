# src/models/rajc.py
"""Response-aware joint clustering for customer segmentation.

This module implements the **RAJC** family and the upgraded main method
**RAMoE / HyRAMoE** (Response-Aware Mixture-of-Experts).

The code is designed to be:
- leakage-safe: segmentation/gating uses **behavior** features only;
- practical: outputs both cluster assignments and response probabilities;
- robust: falls back to constant-prob experts when a component becomes degenerate.

Public API
----------
The class :class:`~customer_segmentation.src.models.rajc.RAJCModel` exposes:

- ``fit(X_beh, y, full_features=X_full)``
- ``predict_clusters(X_beh)``
- ``predict_response(X_beh, full_features=X_full)``

Modes
-----
Controlled via :class:`~customer_segmentation.src.models.rajc.RAJCConfig.model_type`:

- ``"ramoe"`` (default):
    EM-style *soft* mixture-of-experts.
    Gating is a lightweight diagonal-covariance GMM (or soft-kmeans), and experts
    can be Logistic Regression or HistGradientBoostingClassifier.

- ``"logreg"``:
    Hard-assignment joint optimization with per-cluster Logistic Regression experts.

- ``"constant_prob"``:
    RAJC-CP++ style clustering with a cluster-wise constant response probability.

Notes
-----
- This implementation intentionally keeps compatibility with the repository's
  experiment scripts (which import RAJCConfig/RAJCModel and expect the same
  method signatures).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

try:  # HistGradientBoosting is available in sklearn>=0.21
    from sklearn.ensemble import HistGradientBoostingClassifier

    _HGBDT_AVAILABLE = True
except Exception:  # pragma: no cover
    HistGradientBoostingClassifier = object  # type: ignore
    _HGBDT_AVAILABLE = False


ModelType = Literal["ramoe", "logreg", "constant_prob"]
GatingType = Literal["gmm", "soft_kmeans"]
CovarianceType = Literal["diag", "spherical"]
ExpertType = Literal["logreg", "hgbdt"]


class _ConstantProbClassifier:
    """A robust fallback expert that outputs a fixed probability."""

    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-7, 1.0 - 1e-7))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = int(X.shape[0])
        p = np.full(n, self.p, dtype=float)
        return np.column_stack([1.0 - p, p])


def _row_softmax(logits: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax."""
    m = np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits - m)
    denom = np.sum(ex, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return ex / denom


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0))


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """Stable logsumexp over a given axis."""
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out.squeeze(axis)


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _as_int_labels(y) -> np.ndarray:
    yy = np.asarray(y).reshape(-1)
    if yy.dtype.kind not in {"i", "u"}:
        yy = yy.astype(int)
    return yy


def _balanced_sample_weight(y: np.ndarray, base_weight: np.ndarray) -> np.ndarray:
    """Apply a simple 'balanced' scheme via sample_weight rescaling.

    This mimics class_weight='balanced' behaviour but works for any estimator
    that accepts sample_weight.

    The scaling is computed using the *weighted* class mass:
        w_pos = neg_mass / pos_mass
        w_neg = 1
    """
    y = y.reshape(-1).astype(int)
    w = base_weight.reshape(-1).astype(float)

    pos_mass = float(np.sum(w * (y == 1)))
    neg_mass = float(np.sum(w * (y == 0)))

    if pos_mass <= 1e-12 or neg_mass <= 1e-12:
        return w

    scale_pos = neg_mass / pos_mass
    w_adj = w.copy()
    w_adj[y == 1] *= scale_pos
    return w_adj


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RAJCConfig:
    """Configuration for RAJC / RAMoE.

    Parameters
    ----------
    n_clusters:
        Number of segments K.
    model_type:
        One of {"ramoe", "logreg", "constant_prob"}.
    lambda_:
        Controls how strongly response likelihood influences assignments.
        Larger values push the model to discover clusters with different
        response propensities.

    RAMoE / HyRAMoE specific
    ------------------------
    gating_type:
        "gmm" (default) uses a diagonal/spherical covariance Gaussian gating.
        "soft_kmeans" reproduces the earlier soft-kmeans gating behaviour.
    covariance_type:
        "diag" (default) or "spherical" (single variance per component).
    covariance_reg:
        Added to variances for numerical stability.
    temperature:
        Posterior temperature for the E-step (smaller -> crisper assignments).
    gating_temperature:
        Temperature used for *inference* gating g(z|x_beh).

    expert_type:
        "hgbdt" (default) uses HistGradientBoosting experts (nonlinear).
        "logreg" uses weighted Logistic Regression experts.

    use_global_expert / hybrid_alpha:
        Optional hybridization with a global expert for extra stability.
        Final p = (1-alpha)*p_moe + alpha*p_global.

    budget_reweight_alpha:
        If >0, upweights samples currently predicted in the top budget fraction
        (budget_top_frac). This directly nudges training toward lift@top-q.

    constant_prob specific
    ----------------------
    gamma:
        Cluster polarization penalty.
    smoothing:
        Beta smoothing for cluster-wise constant probabilities.
    """

    # Core
    n_clusters: int = 4
    model_type: ModelType = "ramoe"
    lambda_: float = 1.0

    # Optimization
    max_iter: int = 50
    tol: float = 1e-3
    random_state: Optional[int] = 42
    kmeans_n_init: int = 10

    # -------- constant_prob (RAJC-CP++) specific --------
    gamma: float = 0.0
    smoothing: float = 1.0

    # -------- RAMoE gating settings --------
    gating_type: GatingType = "gmm"
    covariance_type: CovarianceType = "diag"
    covariance_reg: float = 1e-6
    temperature: float = 1.0
    gating_temperature: float = 1.0
    min_component_weight: float = 1e-6

    # -------- Expert settings --------
    expert_type: ExpertType = "hgbdt"
    expert_class_weight: Optional[object] = "balanced"  # "balanced"|None

    # Logistic Regression experts
    logreg_C: float = 1.0
    logreg_max_iter: int = 400
    logreg_solver: str = "lbfgs"
    logreg_class_weight: Optional[object] = None  # We handle balancing via sample_weight.

    # HistGradientBoosting experts (if available)
    hgbdt_max_depth: int = 3
    hgbdt_learning_rate: float = 0.05
    hgbdt_max_iter: int = 200
    hgbdt_min_samples_leaf: int = 20
    hgbdt_l2_regularization: float = 0.0
    hgbdt_early_stopping: bool = True
    hgbdt_validation_fraction: float = 0.1
    hgbdt_n_iter_no_change: int = 20

    # -------- Hybrid global expert (optional) --------
    use_global_expert: bool = True
    global_expert_type: ExpertType = "hgbdt"
    hybrid_alpha: float = 0.2

    # -------- Budget-aware reweighting (optional) --------
    budget_top_frac: float = 0.2
    budget_reweight_alpha: float = 0.0


# ---------------------------------------------------------------------------
# RAJC-CP++ helpers (constant_prob)
# ---------------------------------------------------------------------------


def _update_cluster_probs_constant(
    labels: np.ndarray,
    assignments: np.ndarray,
    n_clusters: int,
    smoothing: float,
    global_rate: float,
) -> np.ndarray:
    """Compute smoothed cluster response probabilities p_k."""
    alpha = float(smoothing)
    probs = np.zeros(int(n_clusters), dtype=float)

    for k in range(int(n_clusters)):
        mask = assignments == k
        n_k = int(mask.sum())
        if n_k == 0:
            probs[k] = float(global_rate)
            continue

        pos = float(labels[mask].sum())
        p_k = (pos + alpha) / (n_k + 2.0 * alpha)
        probs[k] = p_k

    return np.clip(probs, 1e-7, 1.0 - 1e-7)


def _compute_cost_matrix_constant(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_probs: np.ndarray,
    config: RAJCConfig,
) -> np.ndarray:
    """Compute RAJC-CP++ cost matrix.

    cost_{ik} = ||x_i - mu_k||^2 + lambda * CE(y_i; p_k) + gamma * p_k(1-p_k)

    The polarization term is constant per cluster.
    """
    diff = features[:, None, :] - centers[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)

    y = labels.reshape(-1, 1).astype(float)
    p = cluster_probs.reshape(1, -1).astype(float)
    ce = -(y * _safe_log(p) + (1.0 - y) * _safe_log(1.0 - p))

    polarization = cluster_probs * (1.0 - cluster_probs)
    pol_pen = float(config.gamma) * polarization.reshape(1, -1)

    return sq_dist + float(config.lambda_) * ce + pol_pen


# ---------------------------------------------------------------------------
# RAMoE helpers (gating + experts)
# ---------------------------------------------------------------------------


def _experts_predict_matrix(experts: List[object], X_full: np.ndarray) -> np.ndarray:
    """Return matrix P where P[i,k] = expert_k.predict_proba(X_full)[i,1]."""
    probs: List[np.ndarray] = []
    for ex in experts:
        p = ex.predict_proba(X_full)[:, 1]
        probs.append(p)
    P = np.column_stack(probs)
    return np.clip(P, 1e-7, 1.0 - 1e-7)


def _fit_weighted_expert(
    X_full: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    cfg: RAJCConfig,
    fallback_p: float,
    *,
    expert_type: Optional[ExpertType] = None,
) -> object:
    """Fit one expert under (possibly soft) sample weights.

    Returns a scikit-learn estimator with `predict_proba`, or a constant-prob fallback.
    """
    y = y.reshape(-1).astype(int)
    sw = np.asarray(sample_weight, dtype=float).reshape(-1)

    sw_sum = float(np.sum(sw))
    if not np.isfinite(sw_sum) or sw_sum <= 1e-12:
        return _ConstantProbClassifier(fallback_p)

    # Ensure both classes exist under the weight mass
    pos_mass = float(np.sum(sw * (y == 1)))
    neg_mass = float(np.sum(sw * (y == 0)))
    if pos_mass <= 1e-12 or neg_mass <= 1e-12:
        p = pos_mass / max(sw_sum, 1e-12)
        return _ConstantProbClassifier(p)

    # Optional balancing via weight rescaling
    if cfg.expert_class_weight == "balanced":
        sw_eff = _balanced_sample_weight(y, sw)
    else:
        sw_eff = sw

    et = expert_type or cfg.expert_type

    if et == "logreg":
        clf = LogisticRegression(
            C=float(cfg.logreg_C),
            max_iter=int(cfg.logreg_max_iter),
            solver=str(cfg.logreg_solver),
            class_weight=None if cfg.expert_class_weight == "balanced" else cfg.logreg_class_weight,
        )
        clf.fit(X_full, y, sample_weight=sw_eff)
        return clf

    if et == "hgbdt":
        if not _HGBDT_AVAILABLE:
            # fallback to logistic regression if HGBDT is missing
            clf = LogisticRegression(
                C=float(cfg.logreg_C),
                max_iter=int(cfg.logreg_max_iter),
                solver=str(cfg.logreg_solver),
                class_weight=None,
            )
            clf.fit(X_full, y, sample_weight=sw_eff)
            return clf

        clf = HistGradientBoostingClassifier(
            learning_rate=float(cfg.hgbdt_learning_rate),
            max_depth=int(cfg.hgbdt_max_depth),
            max_iter=int(cfg.hgbdt_max_iter),
            min_samples_leaf=int(cfg.hgbdt_min_samples_leaf),
            l2_regularization=float(cfg.hgbdt_l2_regularization),
            early_stopping=bool(cfg.hgbdt_early_stopping),
            validation_fraction=float(cfg.hgbdt_validation_fraction),
            n_iter_no_change=int(cfg.hgbdt_n_iter_no_change),
            random_state=None if cfg.random_state is None else int(cfg.random_state),
        )
        clf.fit(X_full, y, sample_weight=sw_eff)
        return clf

    raise ValueError(f"Unsupported expert_type: {et}")


def _log_gaussian_diag(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log density of a diagonal Gaussian for every (sample, component).

    Parameters
    ----------
    X : (n, d)
    mu : (K, d)
    var : (K, d)  (diagonal variances)
    """
    n, d = X.shape
    K = mu.shape[0]
    if var.shape != mu.shape:
        raise ValueError("var must have shape (K,d) matching mu")

    diff = X[:, None, :] - mu[None, :, :]  # (n,K,d)
    quad = np.sum((diff * diff) / var[None, :, :], axis=2)  # (n,K)
    logdet = np.sum(np.log(var), axis=1).reshape(1, K)  # (1,K)

    const = float(d) * np.log(2.0 * np.pi)
    return -0.5 * (const + logdet + quad)


def _log_gaussian_spherical(X: np.ndarray, mu: np.ndarray, var_scalar: np.ndarray) -> np.ndarray:
    """Log density of a spherical Gaussian (one variance per component)."""
    n, d = X.shape
    K = mu.shape[0]

    if var_scalar.shape != (K,):
        raise ValueError("var_scalar must have shape (K,)")

    diff = X[:, None, :] - mu[None, :, :]
    quad = np.sum(diff * diff, axis=2)  # (n,K)

    v = var_scalar.reshape(1, K)
    const = float(d) * np.log(2.0 * np.pi)
    logdet = float(d) * np.log(v)

    return -0.5 * (const + logdet + (quad / v))


def _gating_logit(
    X_beh: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    cfg: RAJCConfig,
) -> np.ndarray:
    """Compute unnormalized log gating scores log pi_k + log p(x|k)."""
    log_pi = _safe_log(pi.reshape(1, -1))

    if cfg.gating_type == "soft_kmeans":
        # Compatibility mode: treat var as not used; use negative squared distance.
        diff = X_beh[:, None, :] - mu[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        return log_pi - sq_dist

    # GMM gating
    if cfg.covariance_type == "diag":
        logpdf = _log_gaussian_diag(X_beh, mu, var)
    elif cfg.covariance_type == "spherical":
        # var is stored as (K,1) or (K,d); convert to (K,)
        if var.ndim == 2:
            var_scalar = np.mean(var, axis=1)
        else:
            var_scalar = var
        logpdf = _log_gaussian_spherical(X_beh, mu, var_scalar)
    else:
        raise ValueError(f"Unsupported covariance_type: {cfg.covariance_type}")

    return log_pi + logpdf


def _gating_probs(
    X_beh: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    cfg: RAJCConfig,
) -> np.ndarray:
    """Inference-time gating probabilities g(z|x_beh)."""
    temp = float(cfg.gating_temperature)
    if temp <= 0:
        raise ValueError("gating_temperature must be > 0")

    logits = _gating_logit(X_beh, pi, mu, var, cfg) / temp
    return _row_softmax(logits)


def _update_gmm_params(
    X_beh: np.ndarray,
    q: np.ndarray,
    cfg: RAJCConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M-step update for gating parameters using responsibilities q.

    Notes
    -----
    - ``q`` is expected to be row-stochastic (each row sums to ~1). When we
      re-initialize tiny components we modify ``q`` *carefully* to keep it
      approximately normalized.
    """
    n, d = X_beh.shape
    K = int(cfg.n_clusters)

    # Effective component masses
    Nk = q.sum(axis=0)  # (K,)
    min_mass = float(cfg.min_component_weight) * float(n)

    # Reinitialize tiny components by stealing a random point.
    # We modify one row to keep row sums stable.
    for k in range(K):
        if Nk[k] <= min_mass:
            idx = int(rng.integers(0, n))
            q[idx, :] = 0.0
            q[idx, k] = 1.0

    # Recompute masses after potential modifications
    Nk = q.sum(axis=0)

    # Mixture weights
    pi = Nk / float(n)
    pi = np.clip(pi, 1e-12, 1.0)
    pi = pi / pi.sum()

    # Means
    mu = np.zeros((K, d), dtype=float)
    for k in range(K):
        wk = q[:, k].reshape(-1, 1)
        denom = float(Nk[k])
        mu[k] = (wk * X_beh).sum(axis=0) / max(denom, 1e-12)

    # Variances
    reg = float(cfg.covariance_reg)
    if cfg.gating_type == "soft_kmeans":
        # In soft-kmeans mode, variance is not used for gating.
        global_var = np.var(X_beh, axis=0) + reg
        var = np.tile(global_var.reshape(1, -1), (K, 1))
        return pi, mu, var

    if cfg.covariance_type == "diag":
        var = np.zeros((K, d), dtype=float)
        for k in range(K):
            wk = q[:, k].reshape(-1, 1)
            denom = float(Nk[k])
            diff = X_beh - mu[k]
            var[k] = (wk * (diff * diff)).sum(axis=0) / max(denom, 1e-12)
        var = var + reg
        var = np.clip(var, reg, None)
        return pi, mu, var

    if cfg.covariance_type == "spherical":
        var = np.zeros((K, d), dtype=float)
        for k in range(K):
            wk = q[:, k].reshape(-1, 1)
            denom = float(Nk[k])
            diff = X_beh - mu[k]
            v = float((wk * (diff * diff)).sum() / max(denom * d, 1e-12))
            v = max(v, reg)
            var[k] = v
        return pi, mu, var

    raise ValueError(f"Unsupported covariance_type: {cfg.covariance_type}")


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class RAJCModel:
    """Unified response-aware joint model.

    See module docstring for usage.
    """

    def __init__(self, config: RAJCConfig):
        self.config = config

        # Shared fitted attributes
        self.centers_: Optional[np.ndarray] = None  # gating means (K,d)
        self.covariances_: Optional[np.ndarray] = None  # diag/spherical variances (K,d)
        self.assignments_: Optional[np.ndarray] = None  # hard assignments (train)
        self.global_response_rate_: Optional[float] = None

        # constant_prob attributes
        self.cluster_probs_: Optional[np.ndarray] = None

        # logreg/ramoe attributes
        self.experts_: Optional[List[object]] = None
        self.pi_: Optional[np.ndarray] = None  # mixture prior
        self.responsibilities_: Optional[np.ndarray] = None  # (n,K)
        self.global_expert_: Optional[object] = None

        # diagnostics
        self.log_likelihood_trace_: List[float] = []

        self._is_fitted: bool = False

    # -------------------- Public API --------------------

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        full_features: Optional[np.ndarray] = None,
    ) -> "RAJCModel":
        """Fit the model.

        Parameters
        ----------
        features:
            Behavior feature matrix X_beh used for clustering/gating.
        labels:
            Binary labels y (0/1).
        full_features:
            Full feature matrix X_full used by experts for response prediction.
            If None, defaults to `features` (backward compatible).
        """
        X_beh = _as_float_array(features)
        y = _as_int_labels(labels)
        X_full = _as_float_array(full_features) if full_features is not None else X_beh

        if X_beh.shape[0] != y.shape[0] or X_full.shape[0] != y.shape[0]:
            raise ValueError("features/full_features/labels must have the same number of rows")

        if self.config.model_type == "constant_prob":
            self._fit_constant_prob_mode(X_beh, y)
        elif self.config.model_type == "logreg":
            self._fit_logreg_mode(X_beh, X_full, y)
        elif self.config.model_type == "ramoe":
            self._fit_ramoe_mode(X_beh, X_full, y)
        else:
            raise ValueError(f"Unsupported model_type: {self.config.model_type}")

        self._is_fitted = True
        return self

    def predict_clusters(self, features: np.ndarray) -> np.ndarray:
        """Hard cluster assignment for new samples."""
        if not self._is_fitted or self.centers_ is None:
            raise RuntimeError("RAJCModel is not fitted yet")

        X_beh = _as_float_array(features)

        if self.config.model_type == "ramoe":
            if self.pi_ is None or self.covariances_ is None:
                raise RuntimeError("ramoe model missing pi_/covariances_")
            g = _gating_probs(X_beh, self.pi_, self.centers_, self.covariances_, self.config)
            return np.argmax(g, axis=1)

        # default: nearest center
        diff = X_beh[:, None, :] - self.centers_[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        return np.argmin(sq_dist, axis=1)

    def predict_response(self, features: np.ndarray, full_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict response probability for new samples."""
        if not self._is_fitted:
            raise RuntimeError("RAJCModel is not fitted yet")

        X_beh = _as_float_array(features)
        X_full = _as_float_array(full_features) if full_features is not None else X_beh

        # constant_prob: p = p_cluster
        if self.config.model_type == "constant_prob":
            if self.cluster_probs_ is None:
                raise RuntimeError("constant_prob model missing cluster_probs_")
            z = self.predict_clusters(X_beh)
            p = self.cluster_probs_[z]
            if self.global_response_rate_ is not None:
                p = np.where(np.isfinite(p), p, float(self.global_response_rate_))
            return np.clip(p, 1e-7, 1.0 - 1e-7)

        # logreg: hard cluster -> expert_k
        if self.config.model_type == "logreg":
            if self.experts_ is None:
                raise RuntimeError("logreg model missing experts_")
            z = self.predict_clusters(X_beh)
            out = np.zeros(X_beh.shape[0], dtype=float)
            for k, ex in enumerate(self.experts_):
                idx = np.where(z == k)[0]
                if idx.size == 0:
                    continue
                out[idx] = ex.predict_proba(X_full[idx])[:, 1]
            if self.global_response_rate_ is not None:
                out = np.where(np.isfinite(out), out, float(self.global_response_rate_))
            return np.clip(out, 1e-7, 1.0 - 1e-7)

        # ramoe: mixture sum_k g_k(x_beh) * p_k(y|x_full)
        if self.config.model_type == "ramoe":
            if self.experts_ is None or self.pi_ is None or self.centers_ is None or self.covariances_ is None:
                raise RuntimeError("ramoe model missing experts_/pi_/centers_/covariances_")

            g = _gating_probs(X_beh, self.pi_, self.centers_, self.covariances_, self.config)
            P = _experts_predict_matrix(self.experts_, X_full)
            p_moe = np.sum(g * P, axis=1)

            # optional global expert
            if self.config.use_global_expert and self.global_expert_ is not None:
                alpha = float(np.clip(self.config.hybrid_alpha, 0.0, 1.0))
                p_global = self.global_expert_.predict_proba(X_full)[:, 1]
                p_global = np.clip(p_global, 1e-7, 1.0 - 1e-7)
                out = (1.0 - alpha) * p_moe + alpha * p_global
            else:
                out = p_moe

            if self.global_response_rate_ is not None:
                out = np.where(np.isfinite(out), out, float(self.global_response_rate_))

            return np.clip(out, 1e-7, 1.0 - 1e-7)

        raise ValueError(f"Unsupported model_type: {self.config.model_type}")

    def get_cluster_stats(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (counts, response_rates) under the fitted hard assignments_ (train)."""
        if not self._is_fitted or self.assignments_ is None:
            raise RuntimeError("RAJCModel is not fitted yet")

        z = self.assignments_
        y = _as_int_labels(labels)

        counts = np.zeros(self.config.n_clusters, dtype=int)
        rates = np.zeros(self.config.n_clusters, dtype=float)

        for k in range(self.config.n_clusters):
            mask = z == k
            n_k = int(mask.sum())
            counts[k] = n_k
            rates[k] = float(y[mask].mean()) if n_k > 0 else float("nan")

        return counts, rates

    # -------------------- Internal: constant_prob --------------------

    def _fit_constant_prob_mode(self, X_beh: np.ndarray, y: np.ndarray) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)

        kmeans = KMeans(
            n_clusters=cfg.n_clusters,
            n_init=cfg.kmeans_n_init,
            random_state=cfg.random_state,
        )
        z = kmeans.fit_predict(X_beh)
        centers = kmeans.cluster_centers_

        global_rate = float(y.mean())
        p_k = _update_cluster_probs_constant(
            labels=y,
            assignments=z,
            n_clusters=cfg.n_clusters,
            smoothing=cfg.smoothing,
            global_rate=global_rate,
        )

        for _ in range(cfg.max_iter):
            cost = _compute_cost_matrix_constant(
                features=X_beh,
                labels=y,
                centers=centers,
                cluster_probs=p_k,
                config=cfg,
            )
            z_new = np.argmin(cost, axis=1)
            change_ratio = float(np.mean(z_new != z))
            z = z_new

            centers = self._update_centers(X_beh, z, cfg.n_clusters, rng)
            p_k = _update_cluster_probs_constant(
                labels=y,
                assignments=z,
                n_clusters=cfg.n_clusters,
                smoothing=cfg.smoothing,
                global_rate=global_rate,
            )

            if change_ratio < cfg.tol:
                break

        self.centers_ = centers
        self.covariances_ = None
        self.assignments_ = z
        self.cluster_probs_ = p_k
        self.global_response_rate_ = global_rate
        self.experts_ = None
        self.pi_ = None
        self.responsibilities_ = None
        self.global_expert_ = None
        self.log_likelihood_trace_ = []

    # -------------------- Internal: logreg (hard experts) --------------------

    def _fit_logreg_mode(self, X_beh: np.ndarray, X_full: np.ndarray, y: np.ndarray) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        n = X_beh.shape[0]

        kmeans = KMeans(
            n_clusters=cfg.n_clusters,
            n_init=cfg.kmeans_n_init,
            random_state=cfg.random_state,
        )
        z = kmeans.fit_predict(X_beh)
        centers = kmeans.cluster_centers_

        global_rate = float(y.mean())

        experts: List[object] = []
        for k in range(cfg.n_clusters):
            w = (z == k).astype(float)
            ex = _fit_weighted_expert(
                X_full,
                y,
                w,
                cfg,
                fallback_p=global_rate,
                expert_type="logreg",
            )
            experts.append(ex)

        for _ in range(cfg.max_iter):
            diff = X_beh[:, None, :] - centers[None, :, :]
            sq_dist = np.sum(diff * diff, axis=2)

            P = _experts_predict_matrix(experts, X_full)
            ycol = y.reshape(-1, 1).astype(float)
            ce = -(ycol * _safe_log(P) + (1.0 - ycol) * _safe_log(1.0 - P))

            cost = sq_dist + float(cfg.lambda_) * ce
            z_new = np.argmin(cost, axis=1)
            change_ratio = float(np.mean(z_new != z))
            z = z_new

            centers = self._update_centers(X_beh, z, cfg.n_clusters, rng)

            experts = []
            for k in range(cfg.n_clusters):
                w = (z == k).astype(float)
                ex = _fit_weighted_expert(
                    X_full,
                    y,
                    w,
                    cfg,
                    fallback_p=global_rate,
                    expert_type="logreg",
                )
                experts.append(ex)

            if change_ratio < cfg.tol:
                break

        self.centers_ = centers
        self.covariances_ = None
        self.assignments_ = z
        self.experts_ = experts
        self.global_response_rate_ = global_rate

        self.cluster_probs_ = np.array(
            [float(y[z == k].mean()) if np.any(z == k) else global_rate for k in range(cfg.n_clusters)],
            dtype=float,
        )
        self.pi_ = np.bincount(z, minlength=cfg.n_clusters).astype(float) / float(n)
        self.responsibilities_ = None
        self.global_expert_ = None
        self.log_likelihood_trace_ = []

    # -------------------- Internal: ramoe (soft gating + experts) --------------------

    def _fit_ramoe_mode(self, X_beh: np.ndarray, X_full: np.ndarray, y: np.ndarray) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        n, d = X_beh.shape
        K = int(cfg.n_clusters)

        if float(cfg.temperature) <= 0:
            raise ValueError("temperature must be > 0")

        # --- init gating via KMeans on behavior ---
        kmeans = KMeans(
            n_clusters=K,
            n_init=cfg.kmeans_n_init,
            random_state=cfg.random_state,
        )
        z0 = kmeans.fit_predict(X_beh)

        global_rate = float(y.mean())

        # initial pi, mu
        pi = np.bincount(z0, minlength=K).astype(float)
        pi = np.clip(pi / float(n), 1e-12, 1.0)
        pi = pi / pi.sum()

        mu = kmeans.cluster_centers_.astype(float)

        # initial var (diag)
        reg = float(cfg.covariance_reg)
        var = np.zeros((K, d), dtype=float)
        global_var = np.var(X_beh, axis=0) + reg
        for k in range(K):
            mask = z0 == k
            if not np.any(mask):
                var[k] = global_var
            else:
                v = np.var(X_beh[mask], axis=0)
                var[k] = v + reg
        var = np.clip(var, reg, None)

        # --- init experts with hard assignment weights ---
        experts: List[object] = []
        for k in range(K):
            w = (z0 == k).astype(float)
            experts.append(_fit_weighted_expert(X_full, y, w, cfg, fallback_p=global_rate))

        # --- optional global expert ---
        if cfg.use_global_expert:
            self.global_expert_ = _fit_weighted_expert(
                X_full,
                y,
                np.ones(n, dtype=float),
                cfg,
                fallback_p=global_rate,
                expert_type=cfg.global_expert_type,
            )
        else:
            self.global_expert_ = None

        responsibilities: Optional[np.ndarray] = None
        prev_z = z0.copy()

        for _ in range(int(cfg.max_iter)):
            # --- E-step ---
            log_g = _gating_logit(X_beh, pi, mu, var, cfg)  # (n,K)

            P = _experts_predict_matrix(experts, X_full)  # (n,K)
            ycol = y.reshape(-1, 1).astype(float)
            log_py = ycol * _safe_log(P) + (1.0 - ycol) * _safe_log(1.0 - P)  # (n,K)

            logits = (log_g + float(cfg.lambda_) * log_py) / float(cfg.temperature)
            q = _row_softmax(logits)
            responsibilities = q

            # diagnostics: joint log-likelihood (up to constant)
            ll = float(np.sum(_logsumexp(logits, axis=1)))
            self.log_likelihood_trace_.append(ll)

            # hard assignments for stopping/profiling
            z = np.argmax(q, axis=1)
            change_ratio = float(np.mean(z != prev_z))
            prev_z = z.copy()

            # --- budget-aware reweighting (optional) ---
            w_budget = np.ones(n, dtype=float)
            if float(cfg.budget_reweight_alpha) > 0:
                g_infer = _row_softmax(log_g / float(cfg.gating_temperature))
                p_moe = np.sum(g_infer * P, axis=1)

                if cfg.use_global_expert and self.global_expert_ is not None:
                    alpha = float(np.clip(cfg.hybrid_alpha, 0.0, 1.0))
                    p_global = self.global_expert_.predict_proba(X_full)[:, 1]
                    p_hat = (1.0 - alpha) * p_moe + alpha * p_global
                else:
                    p_hat = p_moe

                p_hat = np.clip(p_hat, 1e-7, 1.0 - 1e-7)
                top_frac = float(np.clip(cfg.budget_top_frac, 1e-3, 1.0))
                # threshold for top-q
                thr = np.quantile(p_hat, 1.0 - top_frac)
                w_budget = 1.0 + float(cfg.budget_reweight_alpha) * (p_hat >= thr).astype(float)

            # --- M-step: update gating params ---
            pi, mu, var = _update_gmm_params(X_beh, q.copy(), cfg, rng)

            # --- M-step: refit experts using q as weights ---
            experts = []
            for k in range(K):
                sw = q[:, k] * w_budget
                experts.append(_fit_weighted_expert(X_full, y, sw, cfg, fallback_p=global_rate))

            # update global expert (optional)
            if cfg.use_global_expert:
                self.global_expert_ = _fit_weighted_expert(
                    X_full,
                    y,
                    w_budget,
                    cfg,
                    fallback_p=global_rate,
                    expert_type=cfg.global_expert_type,
                )

            if change_ratio < float(cfg.tol):
                break

        # store
        self.centers_ = mu
        self.covariances_ = var
        self.pi_ = pi
        self.experts_ = experts
        self.responsibilities_ = responsibilities
        self.assignments_ = np.argmax(responsibilities, axis=1) if responsibilities is not None else prev_z
        self.global_response_rate_ = global_rate

        if responsibilities is None:
            self.cluster_probs_ = np.array(
                [float(y[self.assignments_ == k].mean()) if np.any(self.assignments_ == k) else global_rate
                 for k in range(K)],
                dtype=float,
            )
        else:
            Nk = responsibilities.sum(axis=0)
            num = (responsibilities * y.reshape(-1, 1)).sum(axis=0)
            pk = num / np.clip(Nk, 1e-12, None)
            self.cluster_probs_ = np.clip(pk, 1e-7, 1.0 - 1e-7)

    # -------------------- Helpers --------------------

    @staticmethod
    def _update_centers(
        X: np.ndarray,
        z: np.ndarray,
        n_clusters: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Update centers by hard assignments; reinit empty clusters."""
        n, d = X.shape
        centers = np.zeros((n_clusters, d), dtype=float)
        for k in range(n_clusters):
            mask = z == k
            if not np.any(mask):
                idx = int(rng.integers(0, n))
                centers[k] = X[idx]
            else:
                centers[k] = X[mask].mean(axis=0)
        return centers
