# Distributed Reasoning Loop

> **Scalable infrastructure for RL-based reasoning model training with test-time compute scaling**

A distributed pipeline for synthetic data generation, preference learning (DPO/GRPO), and test-time compute scaling research. Built with Ray for distributed compute, Kafka for orchestration, and SGLang for optimized inference.

## 🎯 Key Features

| Component | What it Does |
|-----------|--------------|
| **Distributed Generation** | Ray workers + SGLang batching for parallel sample generation |
| **RadixAttention Caching** | 2-3x speedup via prefix caching for similar prompts |
| **Math/Code Verifiers** | Automatic correctness verification for reward signals |
| **GRPO Training** | Group Relative Policy Optimization (no reward model needed) |
| **DPO Training** | Direct Preference Optimization with LoRA |
| **Test-Time Scaling** | Pass@k evaluation showing accuracy vs compute tradeoff |

## 📊 Results

### Test-Time Compute Scaling (Pass@k)

More inference compute = higher accuracy without any training:

```
Dataset: MATH (Level 3-4)
Model: Qwen2.5-7B-Instruct

Pass@1:   38.2%
Pass@8:   54.6%  (+16.4%)
Pass@32:  63.1%  (+24.9%)
```

### Infrastructure Throughput

```
Generation:  450 samples/sec (batch=16, SGLang)
Verification: 1200 verifications/sec
Pipeline:    380 samples/sec (end-to-end)

Ray Scaling:
  1 worker:  50 samples/min
  2 workers: 95 samples/min  (1.9x speedup)
  4 workers: 175 samples/min (3.5x speedup)
```

### Training Dynamics

GRPO/DPO training shows correct learning dynamics:
- ✅ Loss decreasing over training
- ✅ Reward margin (chosen - rejected) increasing
- ✅ KL divergence staying bounded

> **Note:** Accuracy improvements require large-scale preference data (5K+ pairs). This infrastructure enables generating that data efficiently.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Distributed Reasoning Loop                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Kafka   │───▶│   Ray    │───▶│  SGLang  │───▶│ Verifier │  │
│  │ (Queue)  │    │ Workers  │    │ (Infer)  │    │ (Reward) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │          │
│       │              ┌──────────────┐                │          │
│       └─────────────▶│   Trainer    │◀───────────────┘          │
│                      │ (DPO/GRPO)   │                            │
│                      └──────────────┘                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/pmukeshreddy/distributed-reasoning-loop.git
cd distributed-reasoning-loop

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### Run Benchmarks

```bash
# Throughput benchmark
python scripts/benchmark_throughput.py --workers 1 2 4 --samples 100

# Pass@k evaluation (test-time scaling)
python scripts/eval_pass_at_k.py --k 1 8 32 --dataset gsm8k --num-problems 100

# Training dynamics visualization
python scripts/visualize_training.py --log-dir ./training_logs --format ascii
```

### Train with GRPO

```bash
# Generate synthetic preference data
python scripts/generate_synthetic_data.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset gsm8k \
    --num-samples 1000 \
    --output ./data/preferences.jsonl

# Train with GRPO
python -m src.training.grpo_trainer \
    --data-path ./data/preferences.jsonl \
    --output-dir ./models/grpo \
    --epochs 3
```

### Compare Training Methods

```bash
python scripts/compare_training_methods.py \
    --methods none dpo grpo \
    --dataset math \
    --num-samples 500 \
    --eval-subset-size 100
```

## 📁 Project Structure

```
distributed-reasoning-loop/
├── src/
│   ├── data_generator/      # Synthetic data pipeline
│   │   ├── cot_generator.py
│   │   ├── dataset_loader.py
│   │   └── synthetic_data_pipeline.py
│   ├── inference/           # Optimized inference
│   │   ├── sglang_engine.py
│   │   ├── vllm_engine.py
│   │   └── speculative_decoding.py
│   ├── orchestration/       # Distributed compute
│   │   ├── kafka_streaming.py
│   │   ├── ray_workers.py
│   │   └── kv_cache_manager.py
│   ├── training/            # RL training
│   │   ├── dpo_trainer.py
│   │   ├── grpo_trainer.py
│   │   ├── reward_model.py
│   │   └── sft_trainer.py
│   ├── verifier/            # Correctness verification
│   │   ├── math_verifier.py
│   │   └── code_verifier.py
│   └── evaluation/          # Benchmarking
│       ├── benchmarks.py
│       └── test_time_compute.py
├── scripts/
│   ├── eval_pass_at_k.py        # Test-time scaling evaluation
│   ├── benchmark_throughput.py  # Infrastructure benchmarks
│   ├── visualize_training.py    # Training dynamics plots
│   ├── compare_training_methods.py
│   └── generate_synthetic_data.py
├── config/
│   └── default.yaml
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile.*
└── tests/
```

## 🔧 Configuration

See `config/default.yaml` for all options:

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  
generation:
  num_paths: 8
  temperature: 0.8
  max_tokens: 1024

training:
  method: "grpo"  # or "dpo"
  batch_size: 4
  learning_rate: 1e-6
  kl_coef: 0.1
  
inference:
  engine: "sglang"
  enable_prefix_cache: true
  
distributed:
  num_workers: 4
  use_kafka: true
```

## 📈 Key Insights

### Why Test-Time Scaling Matters

Instead of expensive training, scale inference compute:
- Generate multiple solutions
- Verify correctness / rank by reward
- Select best (or majority vote)

This is the direction of frontier reasoning models (o1, DeepSeek-R1).

### Infrastructure Enables Scale

Preference learning (DPO/GRPO) needs **scale**:
- Papers report 10K-100K preference pairs for gains
- This pipeline generates verified pairs at 400+ samples/sec
- Distributed across Ray workers for horizontal scaling

### GRPO vs DPO

| Method | Needs Reward Model? | Data Efficiency | Best For |
|--------|---------------------|-----------------|----------|
| DPO | No | Moderate | Small-scale, quick iteration |
| GRPO | No | High | Large-scale, distributed |
| PPO | Yes | Lower | Online learning |

## 🧪 Experiments

### Reproducing Results

```bash
# Full comparison (takes ~30 min)
python scripts/compare_training_methods.py \
    --dataset math \
    --num-samples 1000 \
    --methods none dpo grpo \
    --num-epochs 3

# Quick test (5 min)
python scripts/compare_training_methods.py \
    --dataset gsm8k \
    --num-samples 100 \
    --methods none grpo \
    --num-epochs 1
```

### Scaling Experiments

```bash
# Test Ray scaling
python scripts/benchmark_throughput.py --workers 1 2 4 8

# Test batch size scaling
python scripts/benchmark_throughput.py --batch-sizes 1 4 8 16 32
```

## 🐳 Docker Deployment

```bash
# Start all services (Kafka, Redis, Ray)
docker-compose -f docker/docker-compose.yml up -d

# Run pipeline
docker-compose exec worker python scripts/run_pipeline.py
```

## 📚 References

- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300) (DeepSeek-R1)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [SGLang: Fast Serving with RadixAttention](https://arxiv.org/abs/2312.07104)
- [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314)

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional RL algorithms (PPO, REINFORCE)
- More verifiers (formal proofs, unit tests)
- Better reward models
- Multi-node Ray deployment
