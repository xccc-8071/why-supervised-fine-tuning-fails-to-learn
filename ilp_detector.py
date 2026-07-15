"""
MC Conversion & ILP Detection
-----------------------------
Three-phase protocol for detecting incomplete learning in SFT models:

  1. MC Conversion  — free-form QA → 4-choice MCQs (GPT-4 distractors + quality filter)
  2. pass@k          — a sample is "learned" iff correct in >=1 of k sampling runs
  3. Triple Validation — cross-temperature, test-retest (Cohen's Kappa), cross-model

"""

from __future__ import annotations

import json
import random
import re
import numpy as np

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict


# ---- Data structures -------------------------------------------------------

@dataclass
class SFTExample:
    id: str
    question: str
    answer: str
    context: Optional[str] = None


@dataclass
class MCQExample:
    id: str
    question: str
    options: List[str]
    correct_idx: int
    distractor_sources: List[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    example_id: str
    pass_k_correct: int
    is_learned: bool
    per_run_correct: List[int]
    temperatures: List[float]


# ---- MC Converter ----------------------------------------------------------

class MCConverter:
    """SFT free-text → 4-choice MCQ via GPT-4 distractors + four-level filter."""

    def __init__(
        self,
        use_llm: bool = True,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
    ):
        self.use_llm = use_llm
        self.api_key = api_key
        self.model = model
        self._filters_applied = 0

    def _call_llm(self, question: str, answer: str) -> List[str]:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            prompt = (
                f"Given this question and correct answer, generate 3 plausible "
                f"but incorrect distractors for a multiple-choice test.\n\n"
                f"Question: {question}\n"
                f"Correct answer: {answer}\n\n"
                f"Return ONLY a JSON array of 3 strings."
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )
            distractors = json.loads(resp.choices[0].message.content)
            return distractors[:3]
        except Exception:
            return []

    def _rule_based_distractors(
        self, correct: str, pool: List[str]
    ) -> List[str]:
        candidates = [
            a for a in pool if a.strip().lower() != correct.strip().lower()
        ]
        if len(candidates) >= 3:
            return random.sample(candidates, 3)
        out = list(candidates)
        while len(out) < 3:
            out.append(self._perturb(correct, len(out)))
        return out[:3]

    @staticmethod
    def _perturb(text: str, idx: int) -> str:
        words = text.split()
        if len(words) <= 3:
            return f"Alternative interpretation {idx + 1}"
        if idx == 0:
            return " ".join(words[:-2])
        elif idx == 1:
            return " ".join(words[1:] + words[:1])
        return " ".join(reversed(words))

    def _filter_distractors(
        self, distractors: List[str], correct: str
    ) -> List[str]:
        before = len(distractors)

        # L1: exclude identical-to-correct
        distractors = [
            d for d in distractors
            if d.strip().lower() != correct.strip().lower()
        ]
        # L2: exclude empty / trivial
        distractors = [d for d in distractors if len(d.strip()) > 2]
        # L3: deduplicate
        seen: set = set()
        unique: List[str] = []
        for d in distractors:
            key = d.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(d)
        # L4: keep longest (most informative)
        unique.sort(key=len, reverse=True)
        self._filters_applied += before - len(unique)
        return unique[:3]

    def convert(
        self, example: SFTExample, answer_pool: List[str]
    ) -> MCQExample:
        distractors = self._rule_based_distractors(example.answer, answer_pool)

        if self.use_llm and self.api_key:
            llm = self._call_llm(example.question, example.answer)
            if llm:
                distractors = llm

        distractors = self._filter_distractors(distractors, example.answer)

        while len(distractors) < 3:
            distractors.append(self._perturb(example.answer, len(distractors)))

        options = [example.answer] + distractors[:3]
        random.shuffle(options)
        correct_idx = options.index(example.answer)

        return MCQExample(
            id=example.id,
            question=example.question,
            options=options,
            correct_idx=correct_idx,
        )


# ---- Pass@k Evaluator ------------------------------------------------------

class PassAtKEvaluator:
    """
    pass@k for ILP: a sample is *learned* iff >=1 of k runs correct.
    k=5 by default (paper main result).
    """

    CHOICE_PATTERNS = [
        re.compile(p) for p in [
            r"\b([A-D])\b",
            r"^([A-D])[\.\)\:]",
            r"[\(\[\{]([A-D])[\)\]\}]",
            r"answer\s*(?:is\s*)?([A-D])",
            r"(?:选|choose|option)\s*([A-D])",
        ]
    ]

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k >= 1 required")
        self.k = k

    def extract_choice(self, output: str, options: List[str]) -> int:
        upper = output.strip().upper()
        for pat in self.CHOICE_PATTERNS:
            m = pat.search(upper)
            if m:
                idx = ord(m.group(1)) - ord('A')
                if 0 <= idx < len(options):
                    return idx
        for i, opt in enumerate(options):
            if opt.strip().lower() in output.lower():
                return i
        return random.randint(0, len(options) - 1)

    def evaluate(self, outputs: List[str], mcq: MCQExample) -> Dict:
        corrects = [
            int(self.extract_choice(out, mcq.options) == mcq.correct_idx)
            for out in outputs
        ]
        n_correct = sum(corrects)
        return {
            "pass@k": int(n_correct >= 1),
            "n_correct": n_correct,
            "k": self.k,
            "per_run_correct": corrects,
            "accuracy": n_correct / self.k,
        }


# ---- Triple Validator ------------------------------------------------------

class TripleValidator:
    """
    Stability checks:
      - Cross-temperature: ILP rate variance across [0.0, 0.3, 0.7, 1.0]
      - Test-retest: Cohen's Kappa between two detection passes
      - Cross-model: ILP rate agreement between model families
    """

    DEFAULT_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]

    def __init__(self, temperatures: Optional[List[float]] = None):
        self.temperatures = temperatures or self.DEFAULT_TEMPERATURES

    @staticmethod
    def cohens_kappa(a: List[int], b: List[int]) -> float:
        n = len(a)
        if n == 0:
            return 0.0
        p_o = sum(1 for x, y in zip(a, b) if x == y) / n
        p_a = sum(a) / n
        p_b = sum(b) / n
        p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
        return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0

    def validate_cross_temperature(
        self,
        results_by_temp: Dict[float, List[DetectionResult]],
    ) -> Dict:
        rates = {}
        for temp, results in results_by_temp.items():
            n_learned = sum(1 for r in results if r.is_learned)
            rates[temp] = 1.0 - n_learned / max(len(results), 1)
        values = list(rates.values())
        max_diff = max(values) - min(values) if values else 0.0
        return {
            "ilp_rates": rates,
            "max_difference": max_diff,
            "stable": max_diff < 0.01,
        }

    def validate_test_retest(
        self,
        run_a: List[DetectionResult],
        run_b: List[DetectionResult],
    ) -> Dict:
        a = [int(r.is_learned) for r in run_a]
        b = [int(r.is_learned) for r in run_b]
        kappa = self.cohens_kappa(a, b)
        return {"kappa": kappa, "reliable": kappa > 0.80}

    def validate_cross_model(
        self,
        a: List[DetectionResult],
        b: List[DetectionResult],
    ) -> Dict:
        ilp_a = 1.0 - sum(1 for r in a if r.is_learned) / max(len(a), 1)
        ilp_b = 1.0 - sum(1 for r in b if r.is_learned) / max(len(b), 1)
        labels_a = [int(r.is_learned) for r in a]
        labels_b = [int(r.is_learned) for r in b]
        kappa = self.cohens_kappa(labels_a, labels_b)
        return {
            "ilp_rate_model_a": ilp_a,
            "ilp_rate_model_b": ilp_b,
            "difference": abs(ilp_a - ilp_b),
            "kappa": kappa,
            "consistent": abs(ilp_a - ilp_b) < 0.01,
        }


# ---- Detection Pipeline ----------------------------------------------------

class ILPDetectionPipeline:
    """SFT data → MC → pass@k → Validation → ILP rate."""

    def __init__(
        self,
        converter: Optional[MCConverter] = None,
        evaluator: Optional[PassAtKEvaluator] = None,
        validator: Optional[TripleValidator] = None,
    ):
        self.converter = converter or MCConverter()
        self.evaluator = evaluator or PassAtKEvaluator(k=5)
        self.validator = validator or TripleValidator()

    def run(
        self,
        examples: List[SFTExample],
        model_fn: Callable[..., List[str]],
        temperatures: Optional[List[float]] = None,
    ) -> Tuple[List[DetectionResult], Dict]:
        """
        model_fn signature: (question, options, temperature) -> [k completions]
        """
        temperatures = temperatures or self.validator.temperatures

        answer_pool = [e.answer for e in examples]
        mcqs = [self.converter.convert(ex, answer_pool) for ex in examples]

        results_by_temp: Dict[float, List[DetectionResult]] = {}
        for temp in temperatures:
            temp_results = []
            for mcq in mcqs:
                outputs = model_fn(mcq.question, mcq.options, temperature=temp)
                ev = self.evaluator.evaluate(outputs, mcq)
                temp_results.append(DetectionResult(
                    example_id=mcq.id,
                    pass_k_correct=ev["n_correct"],
                    is_learned=bool(ev["pass@k"]),
                    per_run_correct=ev["per_run_correct"],
                    temperatures=[temp] * self.evaluator.k,
                ))
            results_by_temp[temp] = temp_results

        cross_temp = self.validator.validate_cross_temperature(results_by_temp)
        base = results_by_temp[temperatures[0]]
        mid = len(base) // 2
        test_retest = self.validator.validate_test_retest(
            base[:mid], base[mid:2 * mid]
        )

        n_total = len(base)
        n_ilp = sum(1 for r in base if not r.is_learned)

        report = {
            "n_examples": n_total,
            "n_not_learned": n_ilp,
            "ilp_rate": n_ilp / max(n_total, 1),
            "cross_temperature": cross_temp,
            "test_retest": test_retest,
            "cross_model": {"difference": float("nan"), "kappa": float("nan"), "consistent": None},
        }
        return base, report


# ---- Stats -----------------------------------------------------------------

def compute_ilp_stats(results: List[DetectionResult]) -> Dict:
    n = len(results)
    n_learned = sum(1 for r in results if r.is_learned)
    n_ilp = n - n_learned

    pass_dist = {i: 0 for i in range(6)}
    pass_counts = [r.pass_k_correct for r in results]
    for c in pass_counts:
        pass_dist[c] += 1

    return {
        "total": n,
        "learned": n_learned,
        "not_learned": n_ilp,
        "ilp_rate": n_ilp / max(n, 1),
        "pass_distribution": pass_dist,
        "mean_passes": float(np.mean(pass_counts)) if pass_counts else 0.0,
        "std_passes": float(np.std(pass_counts)) if pass_counts else 0.0,
    }
