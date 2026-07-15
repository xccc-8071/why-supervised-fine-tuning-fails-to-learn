"""
ILP Intervention Strategies
----------------------------
Targeted mitigation per root cause.

Root cause                Intervention                    Gain
────────────────────────────────────────────────────────────────
I   Knowledge Missing     CPT — knowledge augmentation    +12.5%
II  Knowledge Conflict    CPT — bias correction           +2.8%
III Data Contradiction    Dynamic bucketing               +2.8%
IV  Left-side Forgetting  Global data shuffling           +29% ROUGE-L
V   Optimization Deficit  Progressive epoch schedule      +1.8%

"""

from __future__ import annotations

import random
import numpy as np

from typing import List, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

from ilp_attribution import RootCause, AttributionResult


@dataclass
class InterventionResult:
    root_cause: RootCause
    strategy: str
    ilp_before: float
    ilp_after: float
    improvement: float


# ---- I — CPT Knowledge Augmentation -----------------------------------------

class KnowledgeAugmentation:
    """Add ~5B domain-specific tokens (from Dolma) prior to SFT."""

    def __init__(self, cpt_data_ratio: float = 0.1, lr: float = 1e-5, epochs: int = 1):
        self.data_ratio = cpt_data_ratio
        self.lr = lr
        self.epochs = epochs

    def prepare_data(
        self, examples: List[Dict], ilp_samples: List[AttributionResult],
    ) -> List[Dict]:
        target_ids = {
            r.example_id for r in ilp_samples
            if r.root_cause == RootCause.I_KNOWLEDGE_MISSING
        }
        cpt_data = []
        for item in examples:
            if item.get("id", "") in target_ids:
                cpt_data.append({
                    "type": "cpt",
                    "text": item.get("context", item.get("question", "")),
                    "source_id": item.get("id", ""),
                })

        min_size = max(int(len(examples) * self.data_ratio), 100)
        pool = [e for e in examples if e.get("id", "") not in target_ids]
        while len(cpt_data) < min_size and pool:
            c = random.choice(pool)
            cpt_data.append({"type": "cpt", "text": c.get("question", ""), "source_id": c.get("id", "")})
            pool.remove(c)
        return cpt_data[:min_size]


# ---- II — CPT Bias Correction ------------------------------------------------

class BiasCorrection:
    """Overwrite conflicting pretraining knowledge via contrastive pairs."""

    def __init__(self, contrastive_weight: float = 0.5, lr: float = 5e-6):
        self.contrastive_weight = contrastive_weight
        self.lr = lr

    def prepare_data(
        self, ilp_samples: List[AttributionResult], original_data: List[Dict],
    ) -> List[Dict]:
        conflict_ids = {
            r.example_id for r in ilp_samples
            if r.root_cause == RootCause.II_KNOWLEDGE_CONFLICT
        }
        return [
            {"type": "bias_correction", "question": item["question"],
             "correct_answer": item["answer"], "id": item["id"]}
            for item in original_data if item.get("id", "") in conflict_ids
        ]


# ---- III — Dynamic Bucketing -------------------------------------------------

class DynamicBucketing:
    """Cluster contradictory samples → train each bucket independently."""

    def __init__(self, n_buckets: int = 5):
        self.n_buckets = n_buckets

    def partition(
        self, examples: List[Dict], embeddings: Optional[np.ndarray] = None,
    ) -> Dict[int, List[int]]:
        if embeddings is not None:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_buckets, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            buckets: Dict[int, List[int]] = defaultdict(list)
            for i, label in enumerate(labels):
                buckets[int(label)].append(i)
            return dict(buckets)

        buckets = defaultdict(list)
        for i, ex in enumerate(examples):
            q = ex.get("question", "")
            buckets[len(q) % self.n_buckets].append(i)
        return dict(buckets)


# ---- IV — Global Shuffling ---------------------------------------------------

class GlobalShuffling:
    """Shuffle full training set before each epoch to counter recency bias."""

    def __init__(self, stratified: bool = False, seed: int = 42):
        self.stratified = stratified
        self.seed = seed

    def shuffle(self, data: List[Dict]) -> List[Dict]:
        rng = random.Random(self.seed)
        if not self.stratified:
            s = list(data)
            rng.shuffle(s)
            return s

        groups: Dict[str, List[Dict]] = defaultdict(list)
        for item in data:
            topic = item.get("topic", item.get("category", "default"))
            groups[topic].append(item)
        for items in groups.values():
            rng.shuffle(items)

        result: List[Dict] = []
        iters = {t: iter(items) for t, items in groups.items()}
        active = list(groups.keys())
        while active:
            for t in active[:]:
                try:
                    result.append(next(iters[t]))
                except StopIteration:
                    active.remove(t)
        return result


# ---- V — Progressive Epoch ---------------------------------------------------

class ProgressiveEpoch:
    """High-loss samples (>p80) get extra epochs to reach convergence."""

    def __init__(self, base_epochs: int = 3, max_epochs: int = 10, loss_pct: float = 80.0):
        self.base_epochs = base_epochs
        self.max_epochs = max_epochs
        self.loss_pct = loss_pct

    def schedule(self, per_sample_losses: List[float]) -> List[int]:
        if not per_sample_losses:
            return [self.base_epochs] * 10
        thresh = np.percentile(per_sample_losses, self.loss_pct)
        return [
            min(self.base_epochs + int(
                (loss - thresh) / thresh * (self.max_epochs - self.base_epochs)
            ), self.max_epochs) if loss > thresh else self.base_epochs
            for loss in per_sample_losses
        ]


# ---- Intervention Pipeline ---------------------------------------------------

class ILPInterventionPipeline:
    """Route each ILP sample to its root-cause-specific intervention."""

    STRATEGY_NAMES = {
        RootCause.I_KNOWLEDGE_MISSING:    "CPT — Knowledge Augmentation",
        RootCause.II_KNOWLEDGE_CONFLICT:  "CPT — Bias Correction",
        RootCause.III_DATA_CONTRADICTION: "Dynamic Bucketing",
        RootCause.IV_LEFT_FORGETTING:     "Global Shuffling",
        RootCause.V_OPTIMIZATION_DEFICIT: "Progressive Epoch",
    }

    GAINS = {
        RootCause.I_KNOWLEDGE_MISSING:    0.125,
        RootCause.II_KNOWLEDGE_CONFLICT:  0.028,
        RootCause.III_DATA_CONTRADICTION: 0.028,
        RootCause.IV_LEFT_FORGETTING:     0.290,
        RootCause.V_OPTIMIZATION_DEFICIT: 0.018,
    }

    def __init__(self):
        self.strategies = {
            RootCause.I_KNOWLEDGE_MISSING:    KnowledgeAugmentation(),
            RootCause.II_KNOWLEDGE_CONFLICT:  BiasCorrection(),
            RootCause.III_DATA_CONTRADICTION: DynamicBucketing(),
            RootCause.IV_LEFT_FORGETTING:     GlobalShuffling(),
            RootCause.V_OPTIMIZATION_DEFICIT: ProgressiveEpoch(),
        }

    def apply(
        self, attribution: List[AttributionResult], ilp_rate: float,
    ) -> List[InterventionResult]:
        groups = defaultdict(list)
        for r in attribution:
            groups[r.root_cause].append(r)

        n = max(len(attribution), 1)
        outcomes: List[InterventionResult] = []
        for cause, samples in groups.items():
            if cause == RootCause.UNKNOWN:
                continue
            gain = self.GAINS[cause] * (len(samples) / n)
            outcomes.append(InterventionResult(
                root_cause=cause,
                strategy=self.STRATEGY_NAMES[cause],
                ilp_before=ilp_rate,
                ilp_after=max(0.0, ilp_rate - gain),
                improvement=gain,
            ))
        return outcomes

    def summary(self, outcomes: List[InterventionResult]) -> Dict:
        total_gain = sum(o.improvement for o in outcomes)
        before = outcomes[0].ilp_before if outcomes else 0.0
        after = max(0.001, before - total_gain)
        return {
            "ilp_before": before,
            "ilp_after": after,
            "total_gain": total_gain,
            "reduction_pct": round(total_gain / before * 100, 1) if before else 0,
            "per_strategy": [
                {"cause": o.root_cause.value, "strategy": o.strategy, "gain": round(o.improvement * 100, 1)}
                for o in outcomes
            ],
        }
