# Distributed Reasoning Loop

End-to-end pipeline for training reasoning models using synthetic data generation, distributed verification, and reinforcement learning.

## 🎯 Results

| Metric | Base Model | GRPO Trained | Improvement |
|--------|------------|--------------|-------------|
| Pass@1 | 35.0% | 55.0% | **+20.0%** |
| Pass@4 | 65.0% | 75.0% | **+10.0%** |
| Pass@8 | 80.0% | 90.0% | **+10.0%** |

## ⚡ Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **SGLang** | Batched speedup | **2.5x** vs sequential |
| **Ray** | Parallelization efficiency | **99.2%** |
| **GRPO** | Trainable params | **0.07%** (LoRA) |
| **Pipeline** | End-to-end | **12 min** on 1x H100 |
| **Result** | Pass@1 accuracy | **+20%** improvement |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED REASONING LOOP                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   SGLang     │ -> │     Ray      │ -> │    GRPO      │       │
│  │  Generation  │    │ Verification │    │   Training   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  • RadixAttention    • 4 parallel       • No reward model       │
│  • Prefix caching      workers          • Group-relative        │
│  • 10 paths/problem  • Math: SymPy        advantages            │
│  • Batched requests  • Code: Docker     • LoRA fine-tuning      │
│                        sandbox                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install sglang[all] ray[default] transformers peft trl datasets omegaconf accelerate bitsandbytes
apt-get install -y libnuma1
```

### Run Full Pipeline

```bash
# 1. Start SGLang server
nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --port 30000 --host 0.0.0.0 > sglang.log 2>&1 &
sleep 90

# 2. Run pipeline: SGLang → Ray → GRPO
python scripts/run_ray_pipeline.py
```

### Evaluate
```bash
python scripts/eval_pass_at_k.py \
    --model ./outputs/grpo_model \
    --dataset gsm8k \
    --k 1 4 8 \
    --num-problems 100
```

## 📁 Project Structure

```
distributed-reasoning-loop/
├── src/
│   ├── data_generator/
│   │   ├── cot_generator.py          # SGLang/vLLM inference
│   │   ├── synthetic_data_pipeline.py
│   │   └── dataset_loader.py         # GSM8K, HumanEval, MATH, MBPP
│   ├── verifier/
│   │   ├── math_verifier.py          # SymPy symbolic verification
│   │   └── code_verifier.py          # Docker sandbox execution
│   ├── orchestration/
│   │   ├── ray_workers.py            # Distributed processing
│   │   └── kafka_streaming.py        # Streaming pipeline
│   ├── training/
│   │   ├── grpo_trainer.py           # Group Relative Policy Optimization
│   │   ├── dpo_trainer.py            # Direct Preference Optimization
│   │   └── sft_trainer.py            # Supervised Fine-Tuning
│   └── evaluation/
│       ├── benchmarks.py             # GSM8K, HumanEval evaluators
│       └── test_time_compute.py      # Best-of-N, MCTS, Self-Consistency
├── scripts/
│   ├── run_ray_pipeline.py           # Full SGLang → Ray → GRPO pipeline
│   └── eval_pass_at_k.py             # Pass@k evaluation
└── config/
    └── default.yaml                  # Pipeline configuration
```

## 🔧 Key Components

### 1. SGLang Generation
- **RadixAttention**: Automatic prefix caching for shared prompts
- **Batched inference**: 32 concurrent requests via ThreadPoolExecutor
- **Multi-path sampling**: 10 reasoning paths per problem

### 2. Ray Distributed Verification
- **Parallel workers**: 4 actors, 1250 samples each
- **Math verification**: SymPy symbolic comparison
- **Code verification**: Docker sandbox (256MB RAM, 30s timeout)

### 3. GRPO Training (DeepSeek-R1 Approach)
- **No reward model**: Uses group-relative advantages
- **Verification-based**: Correct = positive, incorrect = negative
- **Efficient**: LoRA (r=16, alpha=32), 8-bit quantization

## 📊 Pipeline Stats

```
Dataset:           GSM8K (500 problems)
Paths generated:   5,000 (10 per problem)
Correct paths:     1,154 (23%)
Incorrect paths:   3,846 (77%)
DPO pairs:         2,090
Training epochs:   5
Final loss:        -0.1119
```

## 📚 References

- [DeepSeek-R1](https://arxiv.org/abs/2401.02954) - GRPO algorithm
- [SGLang](https://github.com/sgl-project/sglang) - RadixAttention inference
- [Ray](https://ray.io/) - Distributed computing
- [GSM8K](https://arxiv.org/abs/2110.14168) - Math reasoning benchmark
