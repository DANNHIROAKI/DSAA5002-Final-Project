# src/models/rajc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class RAJCConfig:
    """
    配置响应感知联合聚类模型（RAJC-CP++）。

    - n_clusters: 簇的个数 K
    - lambda_: 行为距离 vs 响应交叉熵 的权衡系数
    - gamma: 簇级极化正则系数（>0 鼓励高/低响应簇）
    - smoothing: Laplace 平滑系数 alpha
    - max_iter: 最大迭代轮数
    - tol: 簇标签变化比例小于 tol 即停止
    - random_state: 随机种子（用于 KMeans 初始化）
    - kmeans_n_init: KMeans 初始化次数
    - model_type: 当前实现支持 "constant_prob"；其它值会报错
    """

    n_clusters: int = 4
    lambda_: float = 0.3
    gamma: float = 0.0
    smoothing: float = 1.0
    max_iter: int = 30
    tol: float = 1e-3
    random_state: Optional[int] = 42
    kmeans_n_init: int = 10
    logreg_max_iter: int = 200  # 预留给将来的 RAJC-L，当前未使用
    model_type: str = "constant_prob"


def _update_cluster_probs(
    labels: np.ndarray,
    assignments: np.ndarray,
    n_clusters: int,
    smoothing: float,
    global_rate: float,
) -> np.ndarray:
    """
    根据当前簇分配，计算每个簇的平滑后响应概率 p_k。
    空簇回退为全局响应率。
    """
    alpha = smoothing
    probs = np.zeros(n_clusters, dtype=float)

    for k in range(n_clusters):
        mask = assignments == k
        n_k = int(mask.sum())
        if n_k == 0:
            probs[k] = global_rate
            continue

        pos = labels[mask].sum()
        p_k = (pos + alpha) / (n_k + 2.0 * alpha)
        probs[k] = p_k

    # 防止 log(0)
    eps = 1e-6
    probs = np.clip(probs, eps, 1.0 - eps)
    return probs


def _compute_cost_matrix_constant(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_probs: np.ndarray,
    config: RAJCConfig,
) -> np.ndarray:
    """
    计算 RAJC-CP++ 的 cost 矩阵：
    cost_{ik} = ||x_i - mu_k||^2 + lambda * CE(y_i; p_k) + gamma * p_k(1-p_k)
    """
    # 欧式距离平方项 (n, K)
    diff = features[:, None, :] - centers[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)

    # 交叉熵项 (n, K) - 利用广播
    y = labels.reshape(-1, 1)
    p = cluster_probs.reshape(1, -1)
    ce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    # 极化正则项：对每个簇是常数，在样本维上广播
    polarization = cluster_probs * (1.0 - cluster_probs)  # (K,)
    polarization_penalty = config.gamma * polarization.reshape(1, -1)

    cost = sq_dist + config.lambda_ * ce + polarization_penalty
    return cost


class RAJCModel:
    """
    响应感知联合聚类模型 (RAJC-CP++)，簇级常数概率版本。

    使用方法（典型）：
        rajc_cfg = RAJCConfig(...)
        model = RAJCModel(rajc_cfg)
        model.fit(X_behavior, y)
        cluster_labels = model.assignments_
        prob = model.predict_response(X_behavior_new)
    """

    def __init__(self, config: RAJCConfig):
        self.config = config

        # 拟合后填充的属性
        self.centers_: Optional[np.ndarray] = None
        self.cluster_probs_: Optional[np.ndarray] = None
        self.assignments_: Optional[np.ndarray] = None
        self.global_response_rate_: Optional[float] = None

        self._is_fitted: bool = False

    # --------- 对外 API ---------

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        full_features: Optional[np.ndarray] = None,
    ) -> "RAJCModel":
        """
        拟合模型。

        参数：
        - features: 用于聚类的行为特征 (n, d)
        - labels: 二元响应标签 (n,)
        - full_features: 预留给 RAJC-L 的额外特征，目前未使用
        """
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=int).ravel()

        if self.config.model_type != "constant_prob":
            raise ValueError(
                f"Unsupported model_type '{self.config.model_type}'. "
                "Current implementation only supports 'constant_prob'."
            )

        self._fit_constant_prob_mode(features, labels)
        self._is_fitted = True
        return self

    def predict_clusters(self, features: np.ndarray) -> np.ndarray:
        """
        仅根据行为特征和簇中心进行最近簇分配。
        """
        if not self._is_fitted or self.centers_ is None:
            raise RuntimeError("RAJCModel is not fitted yet.")

        X = np.asarray(features, dtype=float)
        diff = X[:, None, :] - self.centers_[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        return np.argmin(sq_dist, axis=1)

    def predict_response(self, features: np.ndarray) -> np.ndarray:
        """
        对新样本预测响应概率：
        - constant_prob: 使用簇级概率 p_k
        """
        if not self._is_fitted or self.cluster_probs_ is None:
            raise RuntimeError("RAJCModel is not fitted yet.")

        clusters = self.predict_clusters(features)
        probs = self.cluster_probs_[clusters]

        # 极少数意外情况（例如 cluster id 超出范围）回退到全局响应率
        if self.global_response_rate_ is not None:
            global_rate = float(self.global_response_rate_)
            probs = np.where(
                np.logical_or(clusters < 0, clusters >= len(self.cluster_probs_)),
                global_rate,
                probs,
            )

        return probs

    # --------- 内部实现：RAJC-CP++ constant probability 模式 ---------

    def _fit_constant_prob_mode(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        n_samples, n_features = features.shape
        cfg = self.config

        # KMeans 初始化
        kmeans = KMeans(
            n_clusters=cfg.n_clusters,
            n_init=cfg.kmeans_n_init,
            random_state=cfg.random_state,
        )
        assignments = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_

        global_rate = float(labels.mean())
        cluster_probs = _update_cluster_probs(
            labels=labels,
            assignments=assignments,
            n_clusters=cfg.n_clusters,
            smoothing=cfg.smoothing,
            global_rate=global_rate,
        )

        for it in range(cfg.max_iter):
            # E-step: 更新簇分配
            cost = _compute_cost_matrix_constant(
                features=features,
                labels=labels,
                centers=centers,
                cluster_probs=cluster_probs,
                config=cfg,
            )
            new_assignments = np.argmin(cost, axis=1)

            change_ratio = np.mean(new_assignments != assignments)
            assignments = new_assignments

            # M-step: 更新中心 & 簇级概率
            centers = self._update_centers(features, assignments, cfg.n_clusters)
            cluster_probs = _update_cluster_probs(
                labels=labels,
                assignments=assignments,
                n_clusters=cfg.n_clusters,
                smoothing=cfg.smoothing,
                global_rate=global_rate,
            )

            if change_ratio < cfg.tol:
                break

        self.centers_ = centers
        self.cluster_probs_ = cluster_probs
        self.assignments_ = assignments
        self.global_response_rate_ = global_rate

    @staticmethod
    def _update_centers(
        features: np.ndarray,
        assignments: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        """
        按当前簇分配更新簇中心，对空簇随机重启。
        """
        n_samples, n_features = features.shape
        centers = np.zeros((n_clusters, n_features), dtype=float)

        rng = np.random.default_rng()

        for k in range(n_clusters):
            mask = assignments == k
            if not np.any(mask):
                # 空簇：随机重启
                idx = rng.integers(0, n_samples)
                centers[k] = features[idx]
            else:
                centers[k] = features[mask].mean(axis=0)

        return centers

    # --------- 一些统计信息接口（可选，用于可视化 / 评估） ---------

    def get_cluster_stats(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回：
        - counts: 每个簇的样本数
        - response_rates: 每个簇的真实响应率（不带平滑）
        """
        if not self._is_fitted or self.assignments_ is None:
            raise RuntimeError("RAJCModel is not fitted yet.")

        assignments = self.assignments_
        labels = np.asarray(labels, dtype=int).ravel()

        counts = np.zeros(self.config.n_clusters, dtype=int)
        rates = np.zeros(self.config.n_clusters, dtype=float)

        for k in range(self.config.n_clusters):
            mask = assignments == k
            n_k = int(mask.sum())
            counts[k] = n_k
            if n_k > 0:
                rates[k] = labels[mask].mean()
            else:
                rates[k] = np.nan

        return counts, rates
