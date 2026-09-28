from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .attribution import covariate_attribution
from .budget import SoTPlan, allocate_branch_budgets
from .decompose import DecomposabilityReport, evaluate_decomposability
from .economics import SoTEconomicsReport, estimate_economics


@dataclass
class SoTAttributionReport:
    covariate_vimp: np.ndarray
    domain_auc: float
    mmd2: float
    overlap_ok: bool
    overlap_ess: float
    top_features: List[int]
    decomposability: DecomposabilityReport

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["covariate_vimp"] = self.covariate_vimp.tolist()
        d["top_features"] = self.top_features
        return d


class FSDSSoT:
    """
    FSDS-powered Skeleton-of-Thought controller.

    Flow:
      1) covariate attribution on trace features (reference vs live batch)
      2) decomposability gate on skeleton branch embeddings
      3) budget allocation: expansion length + check budget + model tier
      4) economics report (latency/cost ROI)
    """

    def __init__(self, *, seed: int = 2026):
        self.seed = seed
        self._cov_cache: Optional[tuple] = None

    def fit_plan_branches(
        self,
        branch_embeddings: np.ndarray,
        branch_quality: Optional[np.ndarray] = None,
        *,
        total_token_budget: int = 4096,
        latency_cap_tokens: int = 512,
        need_weights: Optional[np.ndarray] = None,
        min_tokens_by_branch: Optional[np.ndarray] = None,
        budget_kappa: float = 1.0,
        entropy_lambda: float = 0.15,
        temperature: float = 1.0,
    ) -> tuple[SoTPlan, DecomposabilityReport]:
        """Budget + topology only (no batch covariate re-fit)."""
        ref = branch_embeddings.mean(axis=0, keepdims=True)
        shift_scores = np.linalg.norm(branch_embeddings - ref, axis=1)
        shift_scores = shift_scores / (shift_scores.max() + 1e-9)
        if branch_quality is None:
            branch_quality = np.ones(branch_embeddings.shape[0]) * 0.7
        decomp = evaluate_decomposability(branch_embeddings, shift_scores)
        plan = allocate_branch_budgets(
            shift_scores,
            branch_quality,
            total_token_budget=total_token_budget,
            latency_cap_tokens=latency_cap_tokens,
            need_weights=need_weights,
            min_tokens_by_branch=min_tokens_by_branch,
            kappa=budget_kappa,
            entropy_lambda=entropy_lambda,
            temperature=temperature,
        )
        plan.use_parallel = decomp.allow_parallel
        plan.clusters = decomp.sequential_clusters
        return plan, decomp

    def fit_plan(
        self,
        X_old: np.ndarray,
        X_new: np.ndarray,
        branch_embeddings: np.ndarray,
        branch_quality: Optional[np.ndarray] = None,
        *,
        total_token_budget: int = 4096,
        latency_cap_tokens: int = 512,
        need_weights: Optional[np.ndarray] = None,
        min_tokens_by_branch: Optional[np.ndarray] = None,
        budget_kappa: float = 1.0,
        entropy_lambda: float = 0.15,
        temperature: float = 1.0,
    ) -> tuple[SoTAttributionReport, SoTPlan, SoTEconomicsReport]:
        cov = covariate_attribution(X_old, X_new, seed=self.seed)
        self._cov_cache = (X_old, X_new, cov)
        top_k = min(10, cov.vimp.size)
        top_features = np.argsort(-cov.vimp)[:top_k].tolist()

        plan, decomp = self.fit_plan_branches(
            branch_embeddings,
            branch_quality,
            total_token_budget=total_token_budget,
            latency_cap_tokens=latency_cap_tokens,
            need_weights=need_weights,
            min_tokens_by_branch=min_tokens_by_branch,
            budget_kappa=budget_kappa,
            entropy_lambda=entropy_lambda,
            temperature=temperature,
        )

        from .incremental_value import success_probability, uniform_plan

        ref = branch_embeddings.mean(axis=0, keepdims=True)
        shift = np.linalg.norm(branch_embeddings - ref, axis=1)
        shift = shift / (shift.max() + 1e-9)
        q = branch_quality if branch_quality is not None else np.ones(branch_embeddings.shape[0]) * 0.7
        uni = uniform_plan(shift, q, latency_cap_tokens=latency_cap_tokens, total_token_budget=total_token_budget)
        need = (
            need_weights / (need_weights.sum() + 1e-9)
            if need_weights is not None
            else np.ones(len(q)) / len(q)
        )
        sr = success_probability(plan, need, q, uniform_L=latency_cap_tokens)
        sr0 = success_probability(uni, need, q, uniform_L=latency_cap_tokens)

        econ = estimate_economics(
            plan,
            branches=branch_embeddings.shape[0],
            uniform_tokens_per_branch=latency_cap_tokens,
            success_rate=sr,
            baseline_success_rate=sr0,
        )

        report = SoTAttributionReport(
            covariate_vimp=cov.vimp,
            domain_auc=cov.domain_auc,
            mmd2=cov.mmd2,
            overlap_ok=cov.overlap_ok,
            overlap_ess=cov.overlap_ess,
            top_features=top_features,
            decomposability=decomp,
        )
        return report, plan, econ
