"""Market-orthogonal training losses and calibration metrics.

Implements:
1. Market Logit-Offset Residual Loss (``init_score = z_mkt_shin`` with L2 & logit-variance tether).
2. Hubacek, Sourek & Zelezny (2019, 2021) Decorrelated Brier Loss
   ``(p_hat - y)^2 - c * (p_hat - pi)^2`` for linear solvers and Newton GBDT.
3. Walsh & Joshi (2024) Classwise Expected Calibration Error (ECE).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeRegressor


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0))), dtype=float)


def classwise_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Quantile-binned Expected Calibration Error (Walsh & Joshi 2024)."""
    p = np.asarray(probs, dtype=float)
    yy = np.asarray(y, dtype=float)
    if p.size == 0:
        return 0.0
    edges = np.quantile(p, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ece = 0.0
    n = float(p.size)
    for b in range(int(n_bins)):
        mask = (p > edges[b]) & (p <= edges[b + 1])
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(yy[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def decorrelated_brier_grad_hess(
    u: np.ndarray,
    y: np.ndarray,
    pi: np.ndarray,
    *,
    c: float = 0.5,
    u_anchor: np.ndarray | None = None,
    mu_tether: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Value, gradient, and safeguarded Hessian of Hubacek Decorrelated Brier w.r.t. logit ``u``.

    Loss per row: ``(p - y)^2 - c * (p - pi)^2 + mu_tether * (u - u_anchor)^2``.
    """
    p = sigmoid(u)
    sp = p * (1.0 - p)
    spp = sp * (1.0 - 2.0 * p)
    diff_term = (1.0 - c) * p - (y - c * pi)
    loss_vec = (p - y) ** 2 - c * ((p - pi) ** 2)
    grad = 2.0 * sp * diff_term
    hess = 2.0 * (1.0 - c) * (sp ** 2) + 2.0 * spp * diff_term
    if u_anchor is not None and mu_tether > 0.0:
        du = u - u_anchor
        loss_vec = loss_vec + mu_tether * (du ** 2)
        grad = grad + 2.0 * mu_tether * du
        hess = hess + 2.0 * mu_tether
    hess = np.maximum(hess, 1e-4)
    return float(np.mean(loss_vec)), grad / float(len(y)), hess / float(len(y))


def fit_logit_offset_residual(
    X: np.ndarray,
    y: np.ndarray,
    z_mkt: np.ndarray,
    *,
    l2_lambda: float = 0.05,
    mu_delta: float = 0.50,
) -> np.ndarray:
    """Fit ``p = sigmoid(z_mkt + X1 @ w)`` under NLL + L2(w) + mu * mean(delta^2)."""
    n, d = X.shape
    X1 = np.column_stack([X, np.ones(n, dtype=float)])
    pen = np.full(d + 1, float(l2_lambda), dtype=float)
    pen[-1] = float(l2_lambda) * 0.25

    def obj(w: np.ndarray) -> tuple[float, np.ndarray]:
        delta = X1 @ w
        p = sigmoid(z_mkt + delta)
        eps = 1e-12
        nll = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        reg = 0.5 * float(np.sum(pen * (w ** 2))) + float(mu_delta) * float(np.mean(delta ** 2))
        g_delta = ((p - y) + 2.0 * float(mu_delta) * delta) / float(n)
        grad = X1.T @ g_delta + pen * w
        return float(nll + reg), grad

    res = minimize(obj, np.zeros(d + 1, dtype=float), method="L-BFGS-B", jac=True)
    return res.x


def fit_hubacek_linear(
    X: np.ndarray,
    y: np.ndarray,
    pi_mkt: np.ndarray,
    *,
    c: float = 0.50,
    l2_lambda: float = 0.02,
    z_offset: np.ndarray | None = None,
    mu_tether: float = 0.0,
) -> np.ndarray:
    """Fit linear logit model under Hubacek Decorrelated Brier Loss.

    If ``z_offset`` is None, learns standalone ``u = X1 @ w`` (Hubacek 2019).
    If ``z_offset`` is provided, learns ``u = z_offset + X1 @ w``.
    """
    n, d = X.shape
    X1 = np.column_stack([X, np.ones(n, dtype=float)])
    pen = np.full(d + 1, float(l2_lambda), dtype=float)
    pen[-1] = float(l2_lambda) * 0.1
    base = z_offset if z_offset is not None else np.zeros(n, dtype=float)

    def obj(w: np.ndarray) -> tuple[float, np.ndarray]:
        u = base + X1 @ w
        val, g_u, _ = decorrelated_brier_grad_hess(
            u, y, pi_mkt, c=c, u_anchor=z_offset, mu_tether=mu_tether
        )
        reg = 0.5 * float(np.sum(pen * (w ** 2)))
        grad = X1.T @ g_u + pen * w
        return float(val + reg), grad

    res = minimize(obj, np.zeros(d + 1, dtype=float), method="L-BFGS-B", jac=True, options={"maxiter": 300})
    return res.x


class CustomNewtonGBDT:
    """Shallow Newton-boosted decision trees supporting ``init_score`` and custom ``(g, h)``.

    Supports both ``mode='logit_offset'`` (NLL + mu_tether * delta^2) and
    ``mode='hubacek'`` (Decorrelated Brier with parameter ``c``).
    """

    def __init__(
        self,
        *,
        mode: str = "hubacek",
        c: float = 0.50,
        mu_tether: float = 0.25,
        n_estimators: int = 45,
        learning_rate: float = 0.04,
        max_depth: int = 2,
        min_samples_leaf: int = 80,
        l2_leaf_reg: float = 25.0,
        random_state: int = 42,
    ) -> None:
        self.mode = mode
        self.c = float(c)
        self.mu_tether = float(mu_tether)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.random_state = int(random_state)
        self.trees_: list[tuple[DecisionTreeRegressor, dict[int, float]]] = []
        self.base_bias_: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pi_mkt: np.ndarray,
        *,
        init_score: np.ndarray | None = None,
    ) -> CustomNewtonGBDT:
        n = len(y)
        if init_score is not None:
            u = np.asarray(init_score, dtype=float).copy()
            u_anchor = u.copy()
            self.base_bias_ = 0.0
        else:
            p_bar = float(np.clip(np.mean(y), 0.05, 0.95))
            self.base_bias_ = float(np.log(p_bar / (1.0 - p_bar)))
            u = np.full(n, self.base_bias_, dtype=float)
            u_anchor = None

        self.trees_ = []
        for m in range(self.n_estimators):
            p = sigmoid(u)
            if self.mode == "logit_offset":
                delta = u - (u_anchor if u_anchor is not None else 0.0)
                g = (p - y) + 2.0 * self.mu_tether * delta
                h = np.maximum(p * (1.0 - p) + 2.0 * self.mu_tether, 1e-4)
            else:
                sp = p * (1.0 - p)
                spp = sp * (1.0 - 2.0 * p)
                diff_term = (1.0 - self.c) * p - (y - self.c * pi_mkt)
                g = 2.0 * sp * diff_term
                h = 2.0 * (1.0 - self.c) * (sp ** 2) + 2.0 * spp * diff_term
                if u_anchor is not None and self.mu_tether > 0.0:
                    g = g + 2.0 * self.mu_tether * (u - u_anchor)
                    h = h + 2.0 * self.mu_tether
                h = np.maximum(h, 1e-4)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + m,
            )
            tree.fit(X, -g / h, sample_weight=h)
            leaf_ids = tree.apply(X)
            leaf_vals: dict[int, float] = {}
            step = np.zeros(n, dtype=float)
            for lid in np.unique(leaf_ids):
                mask = leaf_ids == lid
                # Exact regularized Newton leaf step: -sum(g) / (sum(h) + l2_leaf_reg)
                val = -float(np.sum(g[mask])) / (float(np.sum(h[mask])) + self.l2_leaf_reg)
                val = float(np.clip(val, -0.8, 0.8))
                leaf_vals[int(lid)] = val
                step[mask] = val
            u = u + self.learning_rate * step
            self.trees_.append((tree, leaf_vals))
        return self

    def predict_logit(self, X: np.ndarray, *, init_score: np.ndarray | None = None) -> np.ndarray:
        n = X.shape[0]
        u = np.asarray(init_score, dtype=float).copy() if init_score is not None else np.full(n, self.base_bias_, dtype=float)
        for tree, leaf_vals in self.trees_:
            lids = tree.apply(X)
            step = np.array([leaf_vals.get(int(lid), 0.0) for lid in lids], dtype=float)
            u = u + self.learning_rate * step
        return u

    def predict_proba(self, X: np.ndarray, *, init_score: np.ndarray | None = None) -> np.ndarray:
        return sigmoid(self.predict_logit(X, init_score=init_score))
