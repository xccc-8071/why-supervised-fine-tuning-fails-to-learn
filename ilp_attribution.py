"""
ILP Root-Cause Attribution
---------------------------
Two-step attribution for Incomplete Learning Phenomenon samples:

  Step 1 — 2×2 Attribution Matrix
      Base model knowledge (ZS probing) × SFT label correctness
      → separates knowledge problems (I, II) from data/training (III, IV, V)

  Step 2 — Fine-grained diagnosis
      I   Knowledge Missing      — fact absent from pretraining corpus
      II  Knowledge Conflict     — pretraining answer != SFT label
      III Data Contradiction     — conflicting annotations in training data
      IV  Left-side Forgetting   — recency bias overwrites early samples
      V   Optimization Deficit   — loss plateaus before full convergence

"""

from __future__ import annotations

import numpy as np

from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


class RootCause(Enum):
    I_KNOWLEDGE_MISSING    = "I"
    II_KNOWLEDGE_CONFLICT  = "II"
    III_DATA_CONTRADICTION = "III"
    IV_LEFT_FORGETTING     = "IV"
    V_OPTIMIZATION_DEFICIT = "V"
    UNKNOWN                = "UNKNOWN"


ROOT_CAUSE_DESCRIPTIONS: Dict[RootCause, str] = {
    RootCause.I_KNOWLEDGE_MISSING:
        "Knowledge Missing — fact absent from pretraining (verified via OLMo2 + Dolma 5T).",
    RootCause.II_KNOWLEDGE_CONFLICT:
        "Knowledge Conflict — base model learned a different answer that contradicts SFT.",
    RootCause.III_DATA_CONTRADICTION:
        "Data Contradiction — similar training samples have inconsistent labels.",
    RootCause.IV_LEFT_FORGETTING:
        "Left-side Forgetting — autoregressive recency bias overwrites early samples.",
    RootCause.V_OPTIMIZATION_DEFICIT:
        "Optimization Deficit — loss plateaus before these samples are fully fitted.",
    RootCause.UNKNOWN:
        "Unattributed — no single root cause identified.",
}


@dataclass
class AttributionResult:
    example_id: str
    root_cause: RootCause
    confidence: float
    base_model_knows: bool
    sft_label_correct: bool
    quadrant: str


# ---- 2×2 Matrix -------------------------------------------------------------

class AttributionMatrix:
    """
    Quadrant  │  Base model knows?  │  SFT label correct?
    ──────────┼─────────────────────┼────────────────────
    Q1 (II)   │  Yes                │  Yes
    Q2 (I)    │  No                 │  Yes
    Q3 (—)    │  Yes                │  No   (label error, excluded)
    Q4 (III)  │  No                 │  No
    """

    def __init__(
        self,
        knowledge_threshold: float = 0.5,
        label_quality_threshold: float = 0.8,
    ):
        self.knowledge_threshold = knowledge_threshold
        self.label_quality_threshold = label_quality_threshold

    def classify(
        self,
        example_id: str,
        zero_shot_accuracy: float,
        label_quality: float,
    ) -> Tuple[RootCause, str, float]:
        base_knows = zero_shot_accuracy >= self.knowledge_threshold
        label_ok = label_quality >= self.label_quality_threshold

        if base_knows and label_ok:
            return RootCause.II_KNOWLEDGE_CONFLICT, "Q1", 0.85
        elif not base_knows and label_ok:
            return RootCause.I_KNOWLEDGE_MISSING, "Q2", 0.85
        elif base_knows and not label_ok:
            return RootCause.UNKNOWN, "Q3", 0.3
        else:
            return RootCause.III_DATA_CONTRADICTION, "Q4", 0.80


# ---- Forgetting Detector ----------------------------------------------------

class ForgettingDetector:
    """
    If early-20% ILP rate > late-20% ILP rate by >5pp → left-side forgetting.
    """

    def __init__(self, early_frac: float = 0.2, late_frac: float = 0.8):
        self.early_frac = early_frac
        self.late_frac = late_frac

    def analyze(
        self,
        results: List,          # DetectionResult
        positions: List[int],
    ) -> Dict:
        if len(results) != len(positions):
            raise ValueError("len(results) != len(positions)")

        n = len(results)
        order = np.argsort(positions)

        early_end = int(n * self.early_frac)
        late_start = int(n * self.late_frac)

        early_ilp = 1.0 - sum(
            1 for i in order[:early_end] if results[i].is_learned
        ) / max(1, early_end)

        late_ilp = 1.0 - sum(
            1 for i in order[late_start:] if results[i].is_learned
        ) / max(1, n - late_start)

        gap = early_ilp - late_ilp
        return {
            "early_ilp": early_ilp,
            "late_ilp": late_ilp,
            "forgetting_gap": gap,
            "detected": gap > 0.05,
        }


# ---- Convergence Analyzer ---------------------------------------------------

class ConvergenceAnalyzer:
    """Checks if loss has stabilised (delta < threshold between last two windows)."""

    def __init__(self, threshold: float = 0.001, window: int = 50):
        self.threshold = threshold
        self.window = window

    def analyze(self, loss_history: List[float]) -> Dict:
        if len(loss_history) < 2 * self.window:
            return {"converged": True, "delta": 0.0}

        recent = loss_history[-self.window:]
        prior = loss_history[-2 * self.window:-self.window]
        delta = abs(np.mean(prior) - np.mean(recent))
        return {
            "converged": delta < self.threshold,
            "delta": delta,
            "recent_mean": float(np.mean(recent)),
        }


# ---- Attribution Pipeline ---------------------------------------------------

class ILPAttributionPipeline:
    """Two-step attribution: 2×2 matrix → refine by position / convergence."""

    def __init__(self):
        self.matrix = AttributionMatrix()
        self.forgetting = ForgettingDetector()
        self.convergence = ConvergenceAnalyzer()

    def attribute(
        self,
        detection_results: List,
        zero_shot_accuracies: List[float],
        label_qualities: List[float],
        training_positions: Optional[List[int]] = None,
        loss_history: Optional[List[float]] = None,
    ) -> List[AttributionResult]:
        n = len(detection_results)
        if training_positions is None:
            training_positions = list(range(1, n + 1))

        forgetting_info = self.forgetting.analyze(
            detection_results, training_positions
        )

        conv_info: Dict = {}
        if loss_history:
            conv_info = self.convergence.analyze(loss_history)

        results: List[AttributionResult] = []
        for i, r in enumerate(detection_results):
            if r.is_learned:
                continue

            cause, quad, conf = self.matrix.classify(
                r.example_id,
                zero_shot_accuracies[i],
                label_qualities[i],
            )

            if cause not in (RootCause.I_KNOWLEDGE_MISSING,
                             RootCause.II_KNOWLEDGE_CONFLICT):
                cause = self._refine(
                    training_positions[i], n, forgetting_info, conv_info,
                )

            results.append(AttributionResult(
                example_id=r.example_id,
                root_cause=cause,
                confidence=conf,
                base_model_knows=zero_shot_accuracies[i] >= self.matrix.knowledge_threshold,
                sft_label_correct=label_qualities[i] >= self.matrix.label_quality_threshold,
                quadrant=quad,
            ))
        return results

    def _refine(
        self,
        position: int,
        total: int,
        forgetting_info: Dict,
        conv_info: Dict,
    ) -> RootCause:
        if forgetting_info.get("detected") and (position / total) < 0.3:
            return RootCause.IV_LEFT_FORGETTING
        if conv_info and not conv_info.get("converged", True):
            return RootCause.V_OPTIMIZATION_DEFICIT
        return RootCause.III_DATA_CONTRADICTION

    @staticmethod
    def summarize(results: List[AttributionResult]) -> Dict:
        counter = Counter(r.root_cause for r in results)
        total = len(results)

        distribution = {}
        for cause in RootCause:
            count = counter.get(cause, 0)
            distribution[cause.value] = {
                "count": count,
                "pct": round(count / total * 100, 1) if total else 0,
            }

        matrix_covered = (
            counter.get(RootCause.I_KNOWLEDGE_MISSING, 0)
            + counter.get(RootCause.II_KNOWLEDGE_CONFLICT, 0)
        )
        coverage = round(matrix_covered / total * 100, 1) if total else 0

        return {
            "total": total,
            "distribution": distribution,
            "matrix_coverage_pct": coverage,
        }
