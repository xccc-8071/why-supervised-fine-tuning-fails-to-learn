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

**摘要。** 有监督微调（SFT）是将大语言模型（LLM）适配到下游任务的标准方法。
然而，我们观察到一个持续存在的失败模式：即使训练收敛后，模型仍经常无法正确
复现其自身监督训练数据中的一部分样本。我们将此现象称为不完全学习现象（ILP）。
本文首次对 LLM 微调中的 ILP 进行了系统性研究。我们将 ILP 形式定义为训练后
未能内化监督实例，并展示了其在多个模型系列、领域和数据集上的普遍性。通过
控制分析，我们确定了五个反复出现的不完全学习来源：(1) 预训练模型中缺失的
先验知识，(2) SFT 监督与预训练知识之间的冲突，(3) SFT 数据内部的矛盾，
(4) 序列微调中的左侧遗忘，以及 (5) 对罕见或复杂模式的优化不足。我们引入了
一个诊断优先的框架，利用可观测的训练和推理信号将未学会的样本映射到这些原因，
并研究了几种有针对性的缓解策略作为因果干预。

## 框架总览

```
检测（Detect）→ 归因（Attribute）→ 干预（Intervene）→ 验证（Verify）
```

| 阶段 | 模块 | 说明 |
|------|------|------|
| **检测** | `ilp_detector.py` | MC 转换 + pass@k 评估 + 三重验证 |
| **归因** | `ilp_attribution.py` | 2×2 矩阵 → 五大根因（I ~ V） |
| **干预** | `ilp_intervention.py` | 针对每个根因的定向修复 |
| **验证** | `ilp_verification.py` | 干预后重新检测 ILP |
| 编排 | `ilp_pipeline.py` | 端到端流程编排器 |

## 五大根因

| 编号 | 根因 | 高发数据集 | 缓解策略 | 效果 |
|------|------|-----------|----------|------|
| I | 知识缺失 | MedQA, MMLU, MATH | CPT — 知识增强 | +12.5% |
| II | 知识冲突 | 通用问答, GSM8K | CPT — 纠偏训练 | +2.8% |
| III | 数据矛盾 | Alpaca, ShareGPT | 动态分桶 | +2.8% |
| IV | 左侧遗忘 | 指令微调 | 全局打乱 | +29% ROUGE-L |
| V | 优化不足 | MATH, OpenOrca | 渐进式 Epoch | +1.8% |

## 环境配置

### 1. 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Linux / Windows / macOS | Linux (Ubuntu 22.04) |
| Python | ≥ 3.9 | 3.10+ |
| GPU | 无 (dry-run 模式) | 1× A100 80GB (完整复现) |
| 显存 | — | ≥ 40GB (7B 模型推理) |
| 磁盘 | 1 GB | 100 GB (10 数据集 + 5 模型权重 + Dolma 缓存) |

### 2. 依赖包

#### 核心依赖（必须）
```bash
pip install numpy>=1.26,<3.0
```

#### 模型推理（pass@k 检测，需 GPU）
```bash
pip install torch>=2.3.0,<3.0
pip install transformers>=4.44.0
pip install accelerate>=0.30.0
pip install sentencepiece protobuf  # tokenizer 后端
```

#### 数据集加载（真实数据复现需要）
```bash
pip install datasets>=2.20.0
```

#### MC 转换干扰项生成（可选，需 GPT-4 API）
```bash
pip install openai>=1.50.0
# 设置 API key
export OPENAI_API_KEY="sk-xxx"
```

#### 动态分桶聚类（可选）
```bash
pip install scikit-learn>=1.5.0
```

#### 一键安装
```bash
pip install numpy torch transformers accelerate sentencepiece protobuf datasets openai scikit-learn
```

### 3. 数据集准备

将 10 个评估数据集以 JSONL 格式放入 `./data/` 目录：

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

每行格式：
```json
{"id": "medqa_0001", "question": "...", "answer": "...", "context": "..."}
```

数据集来源：

| 数据集 | 来源 |
|--------|------|
| MedQA | <https://github.com/jind11/MedQA> |
| MMLU | <https://github.com/hendrycks/test> |
| GSM8K | <https://github.com/openai/grade-school-math> |
| MATH | <https://github.com/hendrycks/math> |
| Alpaca / Dolly / ShareGPT / OpenOrca | HuggingFace datasets |
| BBH | <https://github.com/suzgunmirac/BIG-Bench-Hard> |

### 4. 模型准备

论文使用 5 个基座模型，通过 Hugging Face 自动下载或手动获取：

| 模型 | HuggingFace ID |
|------|---------------|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` |
| Qwen2.5-14B | `Qwen/Qwen2.5-14B` |
| LLaMA-3-8B | `meta-llama/Meta-Llama-3-8B` |
| OLMo-2-7B | `allenai/OLMo-2-7B` |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` |

> LLaMA 需先登录 Hugging Face：`huggingface-cli login`

### 5. GPT-4 API（MC 转换干扰项）

论文使用 GPT-4 为每个样本生成 3 个干扰项：

```bash
export OPENAI_API_KEY="your-api-key"
```

### 6. 验证安装

```bash
# 框架自检（无需 GPU）
python -c "
from ilp_detector import MCConverter, PassAtKEvaluator, TripleValidator
from ilp_attribution import AttributionMatrix
from ilp_pipeline import ILPPipeline
print('OK: All modules loaded')
"

# dry-run 完整流程（无需 GPU / 真实数据）
python experiments.py --dataset medqa --max-samples 200 --dry-run
```

## 快速开始

### Dry-run 模式（无需 GPU，验证流程）

```bash
python experiments.py --dataset medqa --max-samples 200 --dry-run
```

### 单数据集完整检测（需 GPU + 模型 + 数据）

```bash
python experiments.py --dataset medqa --model Qwen/Qwen2.5-7B --max-samples 500
```

### 跨数据集全量复现（需 GPU 集群，5 模型 × 10 数据集）

```bash
for ds in medqa mmlu_medical mmlu_stem gsm8k math bbh alpaca dolly sharegpt openorca; do
    python experiments.py --dataset $ds --model Qwen/Qwen2.5-7B --max-samples 500
done
```

### 编程方式调用

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

## 核心发现

- **ILP 率**：10 个数据集 8.6%~21.4%，均值 15.3%（标准差 2.1%）
- **检测稳定性**：跨温度差异 <0.5%，重测信度 Kappa = 0.89
- **CPT vs 加 Epoch**：2 倍算力（CPT）提升 +12.5%；10 倍算力（更多 epoch）仅提升 +2%
- **模型规模无关**：1.8B → 14B，ILP 仅下降 2.1 个百分点
- **2×2 矩阵**可覆盖约 80% 的 ILP 样本，分钟级归因
- **物理溯源**：19.3% 的知识在 Dolma 5T 预训练语料中不存在；14.5% 存在冲突

## 引用

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

## 复现说明

本仓库提供了四阶段检测框架的完整算法实现（Detect → Attribute → Intervene → Verify）。
完全复现论文中的数值结果需要以下外部资源：

| 组件 | 状态 | 精确复现所需条件 |
|------|------|------------------|
| ILP 检测 (pass@k) | 完整实现 | — |
| 三重验证（跨温度、Kappa） | 完整实现 | — |
| 2×2 归因矩阵 | 完整实现 | — |
| 五类干预策略 | 算法骨架 | 真实模型训练（CPT、SFT）需 GPU |
| MC 转换（GPT-4） | API 调用就绪 | OpenAI API key + GPT-4 访问 |
| VerificationReport | 完整实现 | — |
| 真实模型推理 | HF 加载就绪 | GPU + 模型权重（7B–14B） |
| ILP 率统计 | 逻辑完成 | 10 个真实数据集 + 5 个模型权重 |
| 物理溯源（Dolma） | 逻辑描述 | Dolma 5T 语料库（~5 TB 磁盘） |
| 干预增益 | 代码中硬编码常量 | CPT / SFT 训练需 GPU 集群 |

`--dry-run` 模式使用 Mock 模型调用和合成数据，用于流程验证和接口测试。
Dry-run 中的所有 ILP 率、干预增益和零样本准确率均为**示意性数值**，不代表论文真实结果。

**要复现论文中的精确表格和图表：** 需要一台搭载 1× NVIDIA A100（80 GB）的 Linux 服务器，
加上上述 10 个数据集、5 个模型检查点（1.8B–14B）、GPT-4 API 访问权限，
以及用于知识溯源的 Dolma 5T 语料库。

## 许可证

MIT
