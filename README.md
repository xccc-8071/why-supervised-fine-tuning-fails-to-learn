[English](README.md) | [中文](README-zh.md)

---

# *Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models*

[Chao Xue](https://aclanthology.org/people/chao-xue-7974/)
· [Yao Wang](https://aclanthology.org/people/yao-wang/)
· [Mengqiao Liu](https://aclanthology.org/people/mengqiao-liu/)
· [Di Liang](https://aclanthology.org/people/di-liang/)
· [Xingsheng Han](https://aclanthology.org/people/xingsheng-han/)
· [Peiyang Liu](https://aclanthology.org/people/peiyang-liu/)
· [Xianjie Wu](https://aclanthology.org/people/xianjie-wu/)
· [Chenyao Lu](https://aclanthology.org/people/chenyao-lu/)
· [Lei Jiang](https://aclanthology.org/people/lei-jiang-3052/)
· [Yu Lu](https://aclanthology.org/people/yu-lu-7040/)
· [Haibo Shi](https://aclanthology.org/people/haibo-shi/)
· [Shuang Liang](https://aclanthology.org/people/shuang-liang/)
· [Minlong Peng](https://aclanthology.org/people/minlong-peng/)
· [Flora D. Salim](https://aclanthology.org/people/flora-d-salim/)

*ACL 2026* · [arXiv](https://arxiv.org/abs/2604.10079) · [ACL Anthology](https://aclanthology.org/2026.acl-long.1393/)

**Abstract.** Supervised Fine-Tuning (SFT) is the standard approach for adapting
large language models (LLMs) to downstream tasks. However, we observe a
persistent failure mode: even after convergence, models often fail to correctly
reproduce a subset of their own supervised training data. We refer to this
behavior as the Incomplete Learning Phenomenon (ILP). This paper presents the
first systematic study of ILP in LLM fine-tuning. We formalize ILP as
post-training failure to internalize supervised instances and demonstrate its
prevalence across multiple model families, domains, and datasets. Through
controlled analyses, we identify five recurrent sources of incomplete learning:
(1) missing prerequisite knowledge in the pre-trained model, (2) conflicts
between SFT supervision and pre-training knowledge, (3) internal
inconsistencies within SFT data, (4) left-side forgetting during sequential
fine-tuning, and (5) insufficient optimization for rare or complex patterns. We
introduce a diagnostic-first framework that maps unlearned samples to these
causes using observable training and inference signals, and study several
targeted mitigation strategies as causal interventions.

## Framework Overview

```
Detect → Attribute → Intervene → Verify
```

| Phase | Module | Description |
|-------|--------|-------------|
| **Detect** | `ilp_detector.py` | MC conversion + pass@k evaluation + triple validation |
| **Attribute** | `ilp_attribution.py` | 2×2 matrix → five root causes (I–V) |
| **Intervene** | `ilp_intervention.py` | Targeted fix per root cause |
| **Verify** | `ilp_verification.py` | Re-check ILP after intervention |
| Pipeline | `ilp_pipeline.py` | End-to-end orchestrator |

## Five Root Causes

| #  | Root Cause | Dominant in | Mitigation | Gain |
|----|-----------|-------------|------------|------|
| I  | Knowledge Missing | MedQA, MMLU, MATH | CPT — knowledge augmentation | +12.5% |
| II | Knowledge Conflict | General QA, GSM8K | CPT — bias correction | +2.8% |
| III | Data Contradiction | Alpaca, ShareGPT | Dynamic bucketing | +2.8% |
| IV | Left-side Forgetting | Instruction tuning | Global shuffling | +29% ROUGE-L |
| V  | Optimization Deficit | MATH, OpenOrca | Progressive epoch | +1.8% |

## Environment Setup

### 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Linux / Windows / macOS | Linux (Ubuntu 22.04) |
| Python | ≥ 3.9 | 3.10+ |
| GPU | None (dry-run mode) | 1× A100 80GB (full reproduction) |
| VRAM | — | ≥ 40GB (7B model inference) |
| Disk | 1 GB | 100 GB (10 datasets + 5 model weights + Dolma cache) |

### 2. Dependencies

#### Core (required)
```bash
pip install numpy>=1.26,<3.0
```

#### Model inference (pass@k, GPU required)
```bash
pip install torch>=2.3.0,<3.0
pip install transformers>=4.44.0
pip install accelerate>=0.30.0
pip install sentencepiece protobuf  # tokenizer backends
```

#### Dataset loading (required for real data)
```bash
pip install datasets>=2.20.0
```

#### MC conversion distractors (optional, GPT-4 API)
```bash
pip install openai>=1.50.0
# Set API key
export OPENAI_API_KEY="sk-xxx"
```

#### Dynamic bucketing (optional)
```bash
pip install scikit-learn>=1.5.0
```

#### One-line install
```bash
pip install numpy torch transformers accelerate sentencepiece protobuf datasets openai scikit-learn
```

### 3. Dataset Preparation

Place the 10 evaluation datasets as JSONL files under `./data/`:

```
why-supervised-fine-tuning-fails-to-learn/
└── data/
    ├── medqa_train.jsonl
    ├── mmlu_medical_train.jsonl
    ├── mmlu_stem_train.jsonl
    ├── gsm8k_train.jsonl
    ├── math_train.jsonl
    ├── bbh_train.jsonl
    ├── alpaca_train.jsonl
    ├── dolly_train.jsonl
    ├── sharegpt_train.jsonl
    └── openorca_train.jsonl
```

Per-line format:
```json
{"id": "medqa_0001", "question": "...", "answer": "...", "context": "..."}
```

Dataset sources:

| Dataset | Source |
|---------|--------|
| MedQA | <https://github.com/jind11/MedQA> |
| MMLU | <https://github.com/hendrycks/test> |
| GSM8K | <https://github.com/openai/grade-school-math> |
| MATH | <https://github.com/hendrycks/math> |
| Alpaca / Dolly / ShareGPT / OpenOrca | HuggingFace datasets |
| BBH | <https://github.com/suzgunmirac/BIG-Bench-Hard> |

### 4. Model Setup

The paper uses 5 base models. Download manually or pull via Hugging Face:

| Model | HuggingFace ID |
|-------|---------------|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` |
| Qwen2.5-14B | `Qwen/Qwen2.5-14B` |
| LLaMA-3-8B | `meta-llama/Meta-Llama-3-8B` |
| OLMo-2-7B | `allenai/OLMo-2-7B` |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` |

> LLaMA requires Hugging Face login: `huggingface-cli login`

### 5. GPT-4 API (MC distractors)

The paper uses GPT-4 to generate 3 distractors per sample:

```bash
export OPENAI_API_KEY="your-api-key"
```

### 6. Verify Installation

```bash
# Framework self-check (no GPU needed)
python -c "
from ilp_detector import MCConverter, PassAtKEvaluator, TripleValidator
from ilp_attribution import AttributionMatrix
from ilp_pipeline import ILPPipeline
print('OK: All modules loaded')
"

# Dry-run full pipeline (no GPU / real data needed)
python experiments.py --dataset medqa --max-samples 200 --dry-run
```

## Quick Start

### Dry-run (pipeline validation, no GPU)

```bash
python experiments.py --dataset medqa --max-samples 200 --dry-run
```

### Single-dataset detection (GPU + model + data)

```bash
python experiments.py --dataset medqa --model Qwen/Qwen2.5-7B --max-samples 500
```

### Full cross-dataset reproduction (GPU cluster, 5 models × 10 datasets)

```bash
for ds in medqa mmlu_medical mmlu_stem gsm8k math bbh alpaca dolly sharegpt openorca; do
    python experiments.py --dataset $ds --model Qwen/Qwen2.5-7B --max-samples 500
done
```

### Programmatic use

```python
from ilp_detector import SFTExample
from ilp_pipeline import ILPPipeline

examples = [
    SFTExample(id="0", question="...", answer="..."),
    # ...
]

pipeline = ILPPipeline(pass_k=5)
report = pipeline.run(
    examples=examples,
    model_fn=your_model_inference_function,
    zero_shot_accuracies=[...],
    label_qualities=[...],
)
```

## Key Results

- **ILP rate**: 8.6%–21.4% across 10 datasets, mean 15.3% (σ = 2.1%)
- **Detection stability**: cross-temperature diff <0.5%, test-retest Kappa = 0.89
- **CPT vs epochs**: 2× compute (CPT) gives +12.5%; 10× compute (more epochs) gives only +2%
- **Scale independence**: 1.8B → 14B reduces ILP by only 2.1 percentage points
- **2×2 matrix** covers ~80% of ILP cases within minutes
- **Physical traceback**: 19.3% of knowledge absent from Dolma 5T; 14.5% conflicts

## Citation

```bibtex
@inproceedings{xue-etal-2026-supervised,
  title     = {Why Supervised Fine-Tuning Fails to Learn: A Systematic
               Study of Incomplete Learning in Large Language Models},
  author    = {Chao Xue and Yao Wang and Mengqiao Liu and Di Liang and
               Xingsheng Han and Peiyang Liu and Xianjie Wu and Chenyao Lu and
               Lei Jiang and Yu Lu and Haibo Shi and Shuang Liang and
               Minlong Peng and Flora D. Salim},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association
               for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {30186--30213},
  year      = {2026},
  address   = {San Diego, California, United States},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2026.acl-long.1393},
  url       = {https://aclanthology.org/2026.acl-long.1393/},
}
```

## Reproducibility Notes

This repository provides the complete algorithmic framework for all four
stages (Detect → Attribute → Intervene → Verify). Exact numerical reproduction
requires the following external resources:

| Component | Status | Requirement for exact reproduction |
|-----------|--------|------------------------------------|
| ILP Detection (pass@k) | Fully implemented | — |
| Triple Validation (cross-temp, Kappa) | Fully implemented | — |
| 2×2 Attribution Matrix | Fully implemented | — |
| Five intervention strategies | Algorithmic skeleton | Real model training (CPT, SFT) on GPU |
| MC Conversion (GPT-4) | API call ready | OpenAI API key + GPT-4 access |
| VerificationReport | Fully implemented | — |
| Real model inference | HF loading ready | GPU + model weights (7B–14B) |
| ILP rate statistics | Logic complete | 10 real datasets + 5 model weights |
| Physical traceback (Dolma) | Logic described | Dolma 5T corpus (~5 TB disk) |
| Intervention gains | Hardcoded constants in code | CPT / SFT training runs on GPU cluster |

The `--dry-run` mode uses mock model calls and synthetic data for pipeline
validation and API testing. All ILP rates, intervention gains, and zero-shot
accuracies in dry-run are **illustrative**, not paper-accurate.

**To reproduce the paper's exact tables and figures:** a Linux server with
1× NVIDIA A100 (80 GB), the 10 datasets listed above, 5 model checkpoints
(1.8B–14B), GPT-4 API access, and the Dolma 5T corpus for knowledge traceback.

## License

MIT
