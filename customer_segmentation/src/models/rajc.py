# src/models/rajc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


ModelType = Literal["ramoe", "logreg", "constant_prob"]


class _ConstantProbClassifier:
    """A robust fallback expert that outputs a fixed probability."""

    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-7, 1.0 - 1e-7))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
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


@dataclass
class RAJCConfig:
    """
    Response-Aware Joint Clustering / Mixture-of-Experts configuration.

    Main upgraded method:
    ---------------------
    - model_type="ramoe" (default): Response-Aware Mixture-of-Experts (soft gating)
      * Gating depends only on behavior features (X_beh)
      * Experts predict response using full features (X_full)
      * EM-like training using responsibilities

    Other supported modes:
    ----------------------
    - model_type="logreg": hard assignment joint optimization with per-cluster LR experts
    - model_type="constant_prob": RAJC-CP++ with cluster-wise constant p_k

    Shared parameters:
    ------------------
    - n_clusters: number of segments K
    - lambda_: controls how strongly the label likelihood influences assignments
              * ramoe: q ∝ g(x_beh) * p(y|x_full)^lambda_
              * logreg/constant_prob: cost includes lambda_ * CE
    - max_iter, tol, random_state, kmeans_n_init
    """

    # Core
    n_clusters: int = 4
    model_type: ModelType = "ramoe"

    # Behavior-vs-response tradeoff
    lambda_: float = 1.0

    # Optimization
    max_iter: int = 30
    tol: float = 1e-3
    random_state: Optional[int] = 42
    kmeans_n_init: int = 10

    # -------- constant_prob (RAJC-CP++) specific --------
    gamma: float = 0.0
    smoothing: float = 1.0

    # -------- logreg / ramoe expert LR settings --------
    logreg_C: float = 1.0
    logreg_max_iter: int = 200
    logreg_solver: str = "lbfgs"
    logreg_class_weight: Optional[object] = "balanced"  # str|dict|None

    # -------- ramoe gating settings --------
    temperature: float = 1.0  # >0, smaller => crisper gating
    min_component_weight: float = 1e-6  # re-init if responsibility sum too small


def _update_cluster_probs_constant(
    labels: np.ndarray,
    assignments: np.ndarray,
    n_clusters: int,
    smoothing: float,
    global_rate: float,
) -> np.ndarray:
    """Compute smoothed cluster response probabilities p_k for constant_prob mode."""
    alpha = float(smoothing)
    probs = np.zeros(n_clusters, dtype=float)

    for k in range(n_clusters):
        mask = assignments == k
        n_k = int(mask.sum())
        if n_k == 0:
            probs[k] = global_rate
            continue

        pos = float(labels[mask].sum())
        p_k = (pos + alpha) / (n_k + 2.0 * alpha)
        probs[k] = p_k

    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return probs


def _compute_cost_matrix_constant(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_probs: np.ndarray,
    config: RAJCConfig,
) -> np.ndarray:
    """
    Compute RAJC-CP++ cost:
        cost_{ik} = ||x_i - mu_k||^2 + lambda * CE(y_i; p_k) + gamma * p_k(1-p_k)

    Note: the polarization term is constant per cluster and broadcast to all samples.
    """
    diff = features[:, None, :] - centers[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)

    y = labels.reshape(-1, 1).astype(float)
    p = cluster_probs.reshape(1, -1).astype(float)
    ce = -(y * _safe_log(p) + (1.0 - y) * _safe_log(1.0 - p))

    polarization = cluster_probs * (1.0 - cluster_probs)  # (K,)
    pol_pen = config.gamma * polarization.reshape(1, -1)

    return sq_dist + config.lambda_ * ce + pol_pen


def _fit_weighted_expert(
    X_full: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    cfg: RAJCConfig,
    fallback_p: float,
) -> object:
    """Fit a weighted LogisticRegression expert, or fall back to constant prob."""
    sw = np.asarray(sample_weight, dtype=float).ravel()
    sw_sum = float(sw.sum())
    if not np.isfinite(sw_sum) or sw_sum <= 1e-12:
        return _ConstantProbClassifier(fallback_p)

    # weighted class mass
    pos_mass = float((sw * y).sum())
    neg_mass = sw_sum - pos_mass
    if pos_mass <= 1e-12 or neg_mass <= 1e-12:
        p = pos_mass / max(sw_sum, 1e-12)
        return _ConstantProbClassifier(p)

    clf = LogisticRegression(
        C=cfg.logreg_C,
        max_iter=cfg.logreg_max_iter,
        solver=cfg.logreg_solver,
        class_weight=cfg.logreg_class_weight,
    )
    clf.fit(X_full, y, sample_weight=sw)
    return clf


def _experts_predict_matrix(experts: List[object], X_full: np.ndarray) -> np.ndarray:
    """Return matrix P where P[i,k] = expert_k.predict_proba(X_full)[i,1]."""
    probs = []
    for ex in experts:
        p = ex.predict_proba(X_full)[:, 1]
        probs.append(p)
    P = np.column_stack(probs)
    return np.clip(P, 1e-7, 1.0 - 1e-7)


def _gating_softmax_from_centers(
    X_beh: np.ndarray,
    centers: np.ndarray,
    pi: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Soft gating probabilities g_k(x) based on distance-to-centers (soft k-means gating)."""
    temp = float(temperature)
    if temp <= 0:
        raise ValueError("temperature must be > 0 for ramoe gating.")

    diff = X_beh[:, None, :] - centers[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)  # (n,K)

    log_pi = _safe_log(pi.reshape(1, -1))
    logits = log_pi - (sq_dist / temp)
    return _row_softmax(logits)


class RAJCModel:
    """
    Unified response-aware joint model with three modes:

    - ramoe (default): Response-Aware Mixture-of-Experts (soft gating + weighted LR experts)
    - logreg: hard assignment joint optimization with LR experts
    - constant_prob: RAJC-CP++ constant response probability per cluster

    API:
        model.fit(X_beh, y, full_features=X_full)
        z = model.predict_clusters(X_beh_new)
        p = model.predict_response(X_beh_new, full_features=X_full_new)
    """

    def __init__(self, config: RAJCConfig):
        self.config = config

        # Shared fitted attributes
        self.centers_: Optional[np.ndarray] = None
        self.assignments_: Optional[np.ndarray] = None  # hard assignments (train)
        self.global_response_rate_: Optional[float] = None

        # constant_prob attributes
        self.cluster_probs_: Optional[np.ndarray] = None

        # logreg/ramoe attributes
        self.experts_: Optional[List[object]] = None
        self.pi_: Optional[np.ndarray] = None  # mixture prior for ramoe gating
        self.responsibilities_: Optional[np.ndarray] = None  # (n,K) for ramoe (train)

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
            If None, it defaults to `features` for backward compatibility,
            but for the upgraded method you should pass X_full explicitly.
        """
        X_beh = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int).ravel()
        X_full = np.asarray(full_features, dtype=float) if full_features is not None else X_beh

        if X_beh.shape[0] != y.shape[0] or X_full.shape[0] != y.shape[0]:
            raise ValueError("features/full_features/labels must have the same number of rows.")

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
        """Hard cluster assignment based on fitted centers (and ramoe gating if enabled)."""
        if not self._is_fitted or self.centers_ is None:
            raise RuntimeError("RAJCModel is not fitted yet.")

        X_beh = np.asarray(features, dtype=float)

        if self.config.model_type == "ramoe" and self.pi_ is not None:
            g = _gating_softmax_from_centers(
                X_beh=X_beh,
                centers=self.centers_,
                pi=self.pi_,
                temperature=self.config.temperature,
            )
            return np.argmax(g, axis=1)

        # default: nearest center (hard)
        diff = X_beh[:, None, :] - self.centers_[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        return np.argmin(sq_dist, axis=1)

    def predict_response(self, features: np.ndarray, full_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict response probability for new samples."""
        if not self._is_fitted:
            raise RuntimeError("RAJCModel is not fitted yet.")

        X_beh = np.asarray(features, dtype=float)
        X_full = np.asarray(full_features, dtype=float) if full_features is not None else X_beh

        # constant_prob: p = p_cluster
        if self.config.model_type == "constant_prob":
            if self.cluster_probs_ is None:
                raise RuntimeError("constant_prob model missing cluster_probs_.")
            z = self.predict_clusters(X_beh)
            p = self.cluster_probs_[z]
            if self.global_response_rate_ is not None:
                p = np.where(np.isfinite(p), p, float(self.global_response_rate_))
            return np.clip(p, 1e-7, 1.0 - 1e-7)

        # logreg: hard cluster -> expert_k
        if self.config.model_type == "logreg":
            if self.experts_ is None:
                raise RuntimeError("logreg model missing experts_.")
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
            if self.experts_ is None or self.pi_ is None or self.centers_ is None:
                raise RuntimeError("ramoe model missing experts_/pi_/centers_.")
            g = _gating_softmax_from_centers(
                X_beh=X_beh,
                centers=self.centers_,
                pi=self.pi_,
                temperature=self.config.temperature,
            )
            P = _experts_predict_matrix(self.experts_, X_full)  # (n,K)
            out = np.sum(g * P, axis=1)
            if self.global_response_rate_ is not None:
                out = np.where(np.isfinite(out), out, float(self.global_response_rate_))
            return np.clip(out, 1e-7, 1.0 - 1e-7)

        raise ValueError(f"Unsupported model_type: {self.config.model_type}")

    def get_cluster_stats(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (counts, response_rates) under the fitted hard assignments_ (train)."""
        if not self._is_fitted or self.assignments_ is None:
            raise RuntimeError("RAJCModel is not fitted yet.")

        z = self.assignments_
        y = np.asarray(labels, dtype=int).ravel()

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

        # KMeans init
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

            # update centers
            centers = self._update_centers(X_beh, z, cfg.n_clusters, rng)

            # update p_k
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
        self.assignments_ = z
        self.cluster_probs_ = p_k
        self.global_response_rate_ = global_rate
        self.experts_ = None
        self.pi_ = None
        self.responsibilities_ = None

    # -------------------- Internal: logreg (hard experts) --------------------

    def _fit_logreg_mode(self, X_beh: np.ndarray, X_full: np.ndarray, y: np.ndarray) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        n = X_beh.shape[0]

        # init by KMeans on behavior
        kmeans = KMeans(
            n_clusters=cfg.n_clusters,
            n_init=cfg.kmeans_n_init,
            random_state=cfg.random_state,
        )
        z = kmeans.fit_predict(X_beh)
        centers = kmeans.cluster_centers_

        global_rate = float(y.mean())

        # initial experts
        experts: List[object] = []
        for k in range(cfg.n_clusters):
            w = (z == k).astype(float)
            ex = _fit_weighted_expert(X_full, y, w, cfg, fallback_p=global_rate)
            experts.append(ex)

        for _ in range(cfg.max_iter):
            # compute distance term
            diff = X_beh[:, None, :] - centers[None, :, :]
            sq_dist = np.sum(diff * diff, axis=2)  # (n,K)

            # compute CE term using experts
            P = _experts_predict_matrix(experts, X_full)  # (n,K)
            ycol = y.reshape(-1, 1).astype(float)
            ce = -(ycol * _safe_log(P) + (1.0 - ycol) * _safe_log(1.0 - P))

            cost = sq_dist + cfg.lambda_ * ce
            z_new = np.argmin(cost, axis=1)
            change_ratio = float(np.mean(z_new != z))
            z = z_new

            # update centers
            centers = self._update_centers(X_beh, z, cfg.n_clusters, rng)

            # refit experts
            experts = []
            for k in range(cfg.n_clusters):
                w = (z == k).astype(float)
                ex = _fit_weighted_expert(X_full, y, w, cfg, fallback_p=global_rate)
                experts.append(ex)

            if change_ratio < cfg.tol:
                break

        # store
        self.centers_ = centers
        self.assignments_ = z
        self.experts_ = experts
        self.global_response_rate_ = global_rate

        # convenience stats
        self.cluster_probs_ = np.array(
            [float(y[z == k].mean()) if np.any(z == k) else global_rate for k in range(cfg.n_clusters)],
            dtype=float,
        )
        self.pi_ = np.bincount(z, minlength=cfg.n_clusters).astype(float) / float(n)
        self.responsibilities_ = None

    # -------------------- Internal: ramoe (soft gating + experts) --------------------

    def _fit_ramoe_mode(self, X_beh: np.ndarray, X_full: np.ndarray, y: np.ndarray) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        n = X_beh.shape[0]

        # init by KMeans on behavior
        kmeans = KMeans(
            n_clusters=cfg.n_clusters,
            n_init=cfg.kmeans_n_init,
            random_state=cfg.random_state,
        )
        z = kmeans.fit_predict(X_beh)
        centers = kmeans.cluster_centers_

        global_rate = float(y.mean())

        # init priors
        pi = np.bincount(z, minlength=cfg.n_clusters).astype(float)
        pi = np.clip(pi / float(n), 1e-12, 1.0)
        pi = pi / pi.sum()

        # init experts using hard assignment weights
        experts: List[object] = []
        for k in range(cfg.n_clusters):
            w = (z == k).astype(float)
            experts.append(_fit_weighted_expert(X_full, y, w, cfg, fallback_p=global_rate))

        responsibilities = None
        prev_z = z.copy()

        for _ in range(cfg.max_iter):
            # E-step: compute soft gating g(x_beh)
            g = _gating_softmax_from_centers(
                X_beh=X_beh,
                centers=centers,
                pi=pi,
                temperature=cfg.temperature,
            )  # (n,K)

            # expert probs
            P = _experts_predict_matrix(experts, X_full)  # (n,K)

            # responsibilities: q ∝ g * p(y|x)^lambda
            ycol = y.reshape(-1, 1).astype(float)
            log_py = ycol * _safe_log(P) + (1.0 - ycol) * _safe_log(1.0 - P)

            logits = _safe_log(g) + cfg.lambda_ * log_py
            q = _row_softmax(logits)  # (n,K)
            responsibilities = q

            # hard assignments for stopping criterion / profiling
            z = np.argmax(q, axis=1)
            change_ratio = float(np.mean(z != prev_z))
            prev_z = z.copy()

            # M-step: update pi and centers using q
            Nk = q.sum(axis=0)  # (K,)
            # reinit tiny components
            for k in range(cfg.n_clusters):
                if Nk[k] <= cfg.min_component_weight * n:
                    idx = int(rng.integers(0, n))
                    centers[k] = X_beh[idx]
                    Nk[k] = max(Nk[k], cfg.min_component_weight * n)

            pi = Nk / float(n)
            pi = np.clip(pi, 1e-12, 1.0)
            pi = pi / pi.sum()

            # update centers
            new_centers = np.zeros_like(centers)
            for k in range(cfg.n_clusters):
                wk = q[:, k].reshape(-1, 1)
                denom = float(Nk[k])
                if denom <= 1e-12:
                    idx = int(rng.integers(0, n))
                    new_centers[k] = X_beh[idx]
                else:
                    new_centers[k] = (wk * X_beh).sum(axis=0) / denom
            centers = new_centers

            # refit experts using q as sample weights
            experts = []
            for k in range(cfg.n_clusters):
                experts.append(_fit_weighted_expert(X_full, y, q[:, k], cfg, fallback_p=global_rate))

            if change_ratio < cfg.tol:
                break

        # store
        self.centers_ = centers
        self.pi_ = pi
        self.experts_ = experts
        self.responsibilities_ = responsibilities
        self.assignments_ = np.argmax(responsibilities, axis=1) if responsibilities is not None else z
        self.global_response_rate_ = global_rate

        # for segmentation tables / interpretability: cluster-wise empirical response under q
        if responsibilities is None:
            self.cluster_probs_ = np.array(
                [float(y[self.assignments_ == k].mean()) if np.any(self.assignments_ == k) else global_rate
                 for k in range(cfg.n_clusters)],
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
