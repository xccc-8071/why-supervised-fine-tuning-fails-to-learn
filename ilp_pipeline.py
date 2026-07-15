"""
ILP Analysis Pipeline
---------------------
Detect → Attribute → Intervene → Verify

"""

from __future__ import annotations

import json
from typing import List, Dict, Optional, Callable

from ilp_detector import (
    SFTExample, DetectionResult, compute_ilp_stats,
    ILPDetectionPipeline, MCConverter, PassAtKEvaluator, TripleValidator,
)
from ilp_attribution import (
    AttributionResult, ILPAttributionPipeline, ROOT_CAUSE_DESCRIPTIONS,
)
from ilp_intervention import (
    InterventionResult, ILPInterventionPipeline,
)
from ilp_verification import Verification, VerificationReport


class ILPPipeline:
    """Four-phase ILP analysis: detection → attribution → intervention → verification."""

    def __init__(self, pass_k: int = 5, temperatures: Optional[List[float]] = None):
        self.pass_k = pass_k
        self.temperatures = temperatures or [0.0, 0.3, 0.7, 1.0]
        self.detector = ILPDetectionPipeline(
            converter=MCConverter(),
            evaluator=PassAtKEvaluator(k=pass_k),
            validator=TripleValidator(self.temperatures),
        )
        self.attributor = ILPAttributionPipeline()
        self.intervenor = ILPInterventionPipeline()
        self.verifier = Verification()

    def run(
        self,
        examples: List[SFTExample],
        model_fn: Callable[..., List[str]],
        zero_shot_accuracies: List[float],
        label_qualities: List[float],
        training_positions: Optional[List[int]] = None,
        loss_history: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> Dict:
        """Execute the complete pipeline and return a report dict."""
        def log(msg: str):
            if verbose:
                print(msg)

        log("=" * 60)
        log("ILP Pipeline: Detect → Attribute → Intervene → Verify")
        log("=" * 60)

        # Phase 1
        log("\n[1/4] Detection")
        detection_results, detection_report = self.detector.run(
            examples, model_fn, self.temperatures,
        )
        stats = compute_ilp_stats(detection_results)
        log(f"  Samples: {stats['total']}  |  "
            f"Not learned: {stats['not_learned']}  |  "
            f"ILP rate: {stats['ilp_rate']:.1%}")
        log(f"  Cross-temp max diff: "
            f"{detection_report['cross_temperature']['max_difference']:.4f}")
        log(f"  Test-retest Kappa: "
            f"{detection_report['test_retest']['kappa']:.3f}")

        # Phase 2
        log("\n[2/4] Attribution")
        attr_results = self.attributor.attribute(
            detection_results, zero_shot_accuracies, label_qualities,
            training_positions, loss_history,
        )
        attr_summary = self.attributor.summarize(attr_results)
        log(f"  Attributed: {attr_summary['total']}")
        log(f"  2×2 matrix coverage: {attr_summary['matrix_coverage_pct']:.0f}%")
        for cause, info in attr_summary["distribution"].items():
            if info["count"] > 0:
                log(f"  {cause}: {info['count']} ({info['pct']}%)")

        # Phase 3
        log("\n[3/4] Intervention")
        ilp_before = stats["ilp_rate"]
        interventions = self.intervenor.apply(attr_results, ilp_before)
        interv_summary = self.intervenor.summary(interventions)
        log(f"  Before: {ilp_before:.1%}  →  "
            f"After: {interv_summary['ilp_after']:.1%}")
        for s in interv_summary["per_strategy"]:
            log(f"  [{s['cause']}] {s['strategy']}: +{s['gain']}%")

        # Phase 4
        log("\n[4/4] Verification")
        verification = self.verifier.verify(
            ilp_before, interv_summary["ilp_after"], interventions,
        )
        log(f"  Reduction: {verification.reduction_pct}%")
        log(f"  Attribution correct: {verification.attribution_correct}")
        log(f"  → {verification.recommendation}")

        report = {
            "detection": {
                "ilp_rate": round(ilp_before, 4),
                "n_total": stats["total"],
                "n_not_learned": stats["not_learned"],
                "pass_distribution": stats["pass_distribution"],
                "cross_temp_max_diff": round(
                    detection_report["cross_temperature"]["max_difference"], 4,
                ),
                "test_retest_kappa": round(
                    detection_report["test_retest"]["kappa"], 3,
                ),
            },
            "attribution": {
                "matrix_coverage_pct": attr_summary["matrix_coverage_pct"],
                "distribution": {
                    k: v for k, v in attr_summary["distribution"].items()
                    if v["count"] > 0
                },
            },
            "intervention": {
                "ilp_before": round(ilp_before * 100, 1),
                "ilp_after": round(interv_summary["ilp_after"] * 100, 1),
                "reduction_pct": interv_summary["reduction_pct"],
                "strategies": interv_summary["per_strategy"],
            },
            "verification": {
                "attribution_correct": verification.attribution_correct,
                "effective": verification.interventions_effective,
                "recommendation": verification.recommendation,
            },
        }

        log(f"\nKey result: ILP {ilp_before:.1%} → "
            f"{interv_summary['ilp_after']:.1%} "
            f"(-{interv_summary['reduction_pct']}%)")
        log("=" * 60)
        return report

    def print_summary(self, report: Dict):
        r = report
        print("\n" + "=" * 55)
        print("ILP Analysis Summary")
        print("=" * 55)
        print(f"\nDetection: ILP {r['detection']['ilp_rate']*100:.1f}%  "
              f"(n={r['detection']['n_total']}, "
              f"ILP={r['detection']['n_not_learned']})")
        print(f"  Kappa = {r['detection']['test_retest_kappa']:.3f}")
        print(f"\nAttribution: 2×2 coverage {r['attribution']['matrix_coverage_pct']:.0f}%")
        for cause, info in r["attribution"]["distribution"].items():
            print(f"  {cause}: {info['count']} ({info['pct']}%)")
        i = r["intervention"]
        print(f"\nIntervention: {i['ilp_before']}% → {i['ilp_after']}%  "
              f"(-{i['reduction_pct']}%)")
        for s in i["strategies"]:
            print(f"  [{s['cause']}] {s['strategy']}: +{s['gain']}%")
        print(f"\nVerification: {r['verification']['recommendation']}")
        print("=" * 55)

    def to_json(self, report: Dict, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
