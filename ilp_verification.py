"""
Verification Protocol
---------------------
Re-run ILP detection after intervention. If reduction >1pp:
attribution was correct. If not: re-attribute with relaxed thresholds.

"""

from __future__ import annotations

from typing import List, Dict
from dataclasses import dataclass, field

from ilp_intervention import InterventionResult


@dataclass
class VerificationReport:
    ilp_before: float
    ilp_after: float
    reduction: float
    reduction_pct: float
    attribution_correct: bool
    interventions_effective: List[str] = field(default_factory=list)
    interventions_ineffective: List[str] = field(default_factory=list)
    recommendation: str = ""


class Verification:
    """Check whether interventions produced a measurable ILP drop."""

    def __init__(self, min_effect: float = 0.01):
        self.min_effect = min_effect

    def verify(
        self,
        ilp_before: float,
        ilp_after: float,
        outcomes: List[InterventionResult],
    ) -> VerificationReport:
        reduction = ilp_before - ilp_after
        reduction_pct = (reduction / ilp_before * 100.0) if ilp_before > 0 else 0.0
        correct = reduction > self.min_effect

        effective = []
        ineffective = []
        for o in outcomes:
            label = f"[{o.root_cause.value}] {o.strategy}"
            (effective if o.improvement > 0 else ineffective).append(label)

        if correct:
            recom = (
                "Attribution correct but residual ILP remains. "
                "Re-attribute remaining samples with relaxed thresholds."
                if reduction_pct <= 50
                else "Attribution correct, interventions highly effective."
            )
        else:
            recom = (
                "No significant reduction — re-check 2×2 matrix thresholds."
            )

        return VerificationReport(
            ilp_before=ilp_before,
            ilp_after=ilp_after,
            reduction=reduction,
            reduction_pct=round(reduction_pct, 1),
            attribution_correct=correct,
            interventions_effective=effective,
            interventions_ineffective=ineffective,
            recommendation=recom,
        )
