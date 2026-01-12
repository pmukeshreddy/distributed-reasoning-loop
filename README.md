# Distributed Reasoning Loop

End-to-end pipeline for training reasoning models using synthetic data generation, distributed verification, and reinforcement learning.

## 🎯 Results

| Metric | Base Model | GRPO Trained | Improvement |
|--------|------------|--------------|-------------|
| Pass@1 | 35.0% | 55.0% | **+20.0%** |
| Pass@4 | 65.0% | 75.0% | **+10.0%** |
| Pass@8 | 70.0% | 85.0% | **+15.0%** |

## ⚡ Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **SGLang** | Generation | **3.5 min** for 5K samples |
| **Ray** | Workers | **4 parallel**, balanced distribution |
| **GRPO** | Trainable params | **0.07%** (LoRA) |
| **Pipeline** | End-to-end | **~12 min** on 1x H100 |

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
pip install sglang[all] ray[default] transformers peft trl datasets omegaconf accelerate bitsandbytes jinja2 --upgrade
apt-get install -y python-is-python3 python3-pip
```

### Run Full Pipeline
```bash
# 1. Apply fixes
sed -i 's/from verifier import/from src.verifier import/g' src/orchestration/ray_workers.py
sed -i 's/chunk_size = (len(data) + num_workers - 1) \/\/ num_workers/chunk_size = max(1, (len(data) + num_workers - 1) \/\/ num_workers)/' src/orchestration/ray_workers.py

# 2. Start SGLang server (use GPU 1 if multi-GPU)
CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --port 30000 --host 0.0.0.0 > sglang.log 2>&1 &
sleep 45

# 3. Run pipeline: SGLang → Ray → GRPO (use GPU 0)
CUDA_VISIBLE_DEVICES=0 python scripts/run_ray_pipeline.py
```

### Evaluate
```bash
# Serve trained model
pkill -f sglang && sleep 2
CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server \
    --model-path ./outputs/grpo_model \
    --port 30000 --host 0.0.0.0 --trust-remote-code > sglang.log 2>&1 &
sleep 45

# Run Pass@k evaluation
python scripts/eval_pass_at_k.py \
    --model http://localhost:30000 \
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
- **Batched inference**: Concurrent requests via ThreadPoolExecutor
- **Multi-path sampling**: 10 reasoning paths per problem

### 2. Ray Distributed Verification
- **Parallel workers**: 4 actors processing chunks
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
Correct paths:     1,162 (23%)
Incorrect paths:   3,838 (77%)
DPO pairs:         2,085
Training epochs:   5
Final loss:        -0.0249
```

## 📚 References

- [DeepSeek-R1](https://arxiv.org/abs/2401.02954) - GRPO algorithm
- [SGLang](https://github.com/sgl-project/sglang) - RadixAttention inference
- [Ray](https://ray.io/) - Distributed computing
- [GSM8K](https://arxiv.org/abs/2110.14168) - Math reasoning benchmark
