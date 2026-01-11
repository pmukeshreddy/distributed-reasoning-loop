# Distributed Reasoning Loop

End-to-end pipeline for training reasoning models using synthetic data generation, distributed verification, and reinforcement learning.

## 🎯 Results

| Metric | Base Model | GRPO Trained | Improvement |
|--------|------------|--------------|-------------|
| Pass@1 | 35.0% | 55.0% | **+20.0%** |
| Pass@4 | 65.0% | 75.0% | **+10.0%** |
| Pass@8 | 80.0% | 90.0% | **+10.0%** |

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

### Run Individual Phases

```bash
# Phase 1: SGLang Generation (5000 reasoning paths)
python main.py generate --dataset gsm8k --num-paths 10 --subset-size 500

# Phase 2: Ray Verification (distributed)
# Automatically runs in pipeline

# Phase 3: GRPO Training
python main.py train-grpo --data-path ./outputs/synthetic_data/dpo_pairs.jsonl --epochs 5
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
│   ├── run_pipeline.py               # Configurable pipeline
│   └── eval_pass_at_k.py             # Pass@k evaluation
├── config/
│   └── default.yaml                  # Pipeline configuration
└── outputs/
    ├── synthetic_data/               # Generated DPO pairs
    └── grpo_model/                   # Trained LoRA weights
```

## 🔧 Key Components

### 1. SGLang Generation
- **RadixAttention**: Automatic prefix caching for shared prompts
- **Batched inference**: 32 concurrent requests via ThreadPoolExecutor
- **Multi-path sampling**: 10 reasoning paths per problem with temperature=0.8

### 2. Ray Distributed Verification
- **Parallel workers**: 4 actors processing 1250 samples each
- **Math verification**: SymPy symbolic comparison, numeric tolerance
- **Code verification**: Docker sandbox with resource limits (256MB RAM, 30s timeout)

### 3. GRPO Training (DeepSeek-R1 Approach)
- **No reward model needed**: Uses group-relative advantages
- **Verification-based labels**: Correct paths = positive, incorrect = negative
- **Efficient training**: LoRA (r=16, alpha=32), 8-bit quantization

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

## 🧠 How GRPO Works

```python
# Group Relative Policy Optimization (no reward model needed)

# 1. For each problem, we have correct and incorrect paths
correct_paths = [p for p in paths if verified(p)]    # reward = +1
incorrect_paths = [p for p in paths if not verified(p)]  # reward = -1

# 2. Compute advantages relative to group mean
advantages = rewards - mean(rewards)  # Group-relative normalization

# 3. Policy gradient with KL penalty
loss = -log_prob(chosen) * advantage + β * KL(policy || reference)
```

## 🔬 Verification Methods

### Math (GSM8K, MATH)
```python
# Extract answer patterns
patterns = [r'####\s*(\d+)', r'\\boxed{(.+)}', r'answer is (\d+)']

# Compare using SymPy
sympy.simplify(predicted - ground_truth) == 0  # Symbolic
abs(float(predicted) - float(ground_truth)) < 1e-5  # Numeric
```

### Code (HumanEval, MBPP)
```python
# Docker sandbox execution
container = docker.run(
    image="python:3.11-slim",
    memory="256m",
    network_disabled=True,
    timeout=30
)
result = container.exec(code + test_cases)
```

## 📈 Test-Time Compute Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Best-of-N | Sample N, pick highest confidence | Simple, fast |
| Majority Vote | Sample N, return most common answer | Robust to outliers |
| MCTS | Tree search with UCB exploration | Complex reasoning |
| Self-Consistency | Sample N, weighted voting | Best accuracy |

```bash
# Run with test-time compute
python scripts/eval_pass_at_k.py --model ./outputs/grpo_model --majority-vote --k 10
```

## ⚙️ Configuration

```yaml
# config/default.yaml
data_generator:
  teacher_model: "Qwen/Qwen2.5-1.5B-Instruct"
  num_cot_paths: 10
  temperature: 0.8
  max_new_tokens: 2048

training:
  method: "grpo"
  batch_size: 2
  learning_rate: 1e-6
  num_epochs: 5
  lora:
    r: 16
    alpha: 32
    dropout: 0.05

ray:
  num_workers: 4
  num_gpus: 1
```

## 🐳 Docker Deployment

```bash
# Build
docker-compose build

# Run full stack (SGLang + Ray + Training)
docker-compose up -d

# Scale Ray workers
docker-compose up -d --scale ray-worker=8
```

## 📚 References

- [DeepSeek-R1](https://arxiv.org/abs/2401.02954) - GRPO algorithm
- [SGLang](https://github.com/sgl-project/sglang) - RadixAttention inference
- [Ray](https://ray.io/) - Distributed computing
- [GSM8K](https://arxiv.org/abs/2110.14168) - Math reasoning benchmark

## 📄 License

MIT License

## 🙏 Acknowledgments

Built for demonstrating distributed ML infrastructure skills for reasoning model training.
