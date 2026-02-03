# Distributed Reasoning Loop

End-to-end GRPO (Group Relative Policy Optimization) pipeline for training reasoning models. Implements DeepSeek-R1's approach: synthetic data generation, distributed verification, and RL training without reward models.

**Now with Continuous Distributed GRPO Training** - parallel rollout generation, LoRA hot-reload without restarts!

## 🎯 Results

Evaluated on full GSM8K test set (1319 problems):

| Metric | Base Model | GRPO Trained | Improvement |
|--------|------------|--------------|-------------|
| Pass@1 | 44.7% | 74.0% | **+29.3%** |
| Pass@4 | 75.1% | 88.0% | **+12.9%** |
| Pass@8 | 84.2% | 92.6% | **+8.4%** |

## ⚡ Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **SGLang** | Generation | **3.5 min** for 5K samples |
| **SGLang** | Throughput | **24K tokens/sec** |
| **Ray** | Workers | **4 parallel**, balanced distribution |
| **GRPO** | Trainable params | **0.07%** (LoRA) |
| **Pipeline** | End-to-end | **~12 min** on 2x H100 |

## 🏗️ Architecture

### Batch Training Pipeline (Original)

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
│  • 10 paths/problem  • SymPy verify       advantages            │
│  • Batched requests  • 24K tok/sec      • LoRA fine-tuning      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Continuous Distributed GRPO (NEW!)

```
┌─────────────────────────────────────────────────────────────────┐
│              CONTINUOUS DISTRIBUTED GRPO                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: Rollout (Parallel across workers)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │ SGLang 0 │  │ SGLang 1 │  │ SGLang 2 │  ← All use LoRA vN   │
│  │ 50 paths │  │ 50 paths │  │ 50 paths │                       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
│       │             │             │                              │
│       └─────────────┼─────────────┘                              │
│                     ↓                                            │
│  STEP 2: Verify (Ray workers)                                   │
│              ┌──────────────┐                                   │
│              │ Ray Verify   │  ← SymPy check correct/incorrect  │
│              └──────┬───────┘                                   │
│                     ↓                                            │
│  STEP 3: Train (GRPO on verified batch)                         │
│              ┌──────────────┐                                   │
│              │ GRPO Trainer │  ← Compute advantages, update     │
│              └──────┬───────┘                                   │
│                     ↓                                            │
│  STEP 4: Broadcast LoRA vN+1 to ALL workers                     │
│              ┌──────────────┐                                   │
│              │ Coordinator  │  ← Hot-reload without restart     │
│              └──────┬───────┘                                   │
│                     ↓                                            │
│  STEP 5: Workers hot-reload (NO RESTART)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │ SGLang 0 │  │ SGLang 1 │  │ SGLang 2 │  ← All now use vN+1  │
│  │    ✓     │  │    ✓     │  │    ✓     │                       │
│  └──────────┘  └──────────┘  └──────────┘                       │
│                     ↓                                            │
│            REPEAT FROM STEP 1                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# System packages (Ubuntu)
apt-get update && apt-get install -y python-is-python3 python3-pip

# Python dependencies
pip install sglang[all] ray[default] transformers peft trl datasets \
    omegaconf accelerate bitsandbytes jsonschema jinja2 aiohttp --upgrade

# Optional: For dashboard
pip install streamlit plotly pandas
```

### Step 1: Clone Repository

```bash
git clone https://github.com/pmukeshreddy/distributed-reasoning-loop.git
cd distributed-reasoning-loop
git checkout extend
```

---

## Option A: Batch Training (Original Pipeline)

### Step 2: Start SGLang Server

```bash
# Start inference server (GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 --port 30000 > sglang.log 2>&1 &

# Wait for server to initialize
sleep 45

# Verify server is running
tail -n 3 sglang.log
```

### Step 3: Run Full Pipeline (SGLang → Ray → GRPO)

```bash
# Run pipeline (GPU 0)
CUDA_VISIBLE_DEVICES=0 python scripts/run_ray_pipeline.py
```

Expected output:
```
Phase 1: SGLang Generation
Generating: 100%|████████████████| 50/50 [03:23<00:00, 4.07s/it]

Phase 2: Ray Verification
Initialized 4 workers of each type
Ray stats: {'total_processed': 5000}

Phase 3: GRPO Training
Training: 100%|████████████████| 210/210 [08:00<00:00]

Done: SGLang -> Ray -> GRPO
```

---

## Option B: Continuous Distributed GRPO (NEW!)

### Step 2: Start Multiple SGLang Workers

```bash
# Start 3 workers with LoRA hot-reload support
chmod +x scripts/start_workers.sh
./scripts/start_workers.sh 3 30001

# Or manually start workers:
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 --port 30001 \
    --max-loras-per-batch 8 --max-lora-rank 64 &

CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 --port 30002 \
    --max-loras-per-batch 8 --max-lora-rank 64 &

CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 --port 30003 \
    --max-loras-per-batch 8 --max-lora-rank 64 &
```

### Step 3: Run Continuous Training

```bash
# Run continuous GRPO training (GPU 0 for training)
CUDA_VISIBLE_DEVICES=0 python scripts/run_continuous_grpo.py \
    --ports 30001 30002 30003 \
    --iterations 100 \
    --target-accuracy 0.9
```

Expected output:
```
============================================================
INITIALIZING CONTINUOUS GRPO PIPELINE
============================================================
Workers ready: 3/3
Loaded 7473 problems

============================================================
ITERATION 1
============================================================
Generating 500 rollouts...
Verification: 234/500 correct (46.8%)
Training on 500 samples...
Broadcasting LoRA v1 to all workers...
All workers updated to LoRA v1

Iteration 1 Summary:
  Rollouts: 500 in 45.2s
  Accuracy: 46.8% (234/500)
  Loss: 0.2341
  LoRA version: v1
  Total time: 52.3s

============================================================
ITERATION 2
============================================================
...
```

### Step 4: Monitor Training (Optional)

```bash
# Start the dashboard
streamlit run scripts/dashboard.py

# Or view metrics in terminal
tail -f outputs/continuous_grpo/training_metrics.jsonl
```

### Step 5: Stop Workers

```bash
./scripts/stop_workers.sh
```

### Step 4: Evaluate Base Model

```bash
# Restart server with base model
pkill -f sglang && sleep 2
CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 --port 30000 \
    --trust-remote-code > sglang.log 2>&1 &
sleep 45

# Evaluate
python scripts/eval_pass_at_k.py \
    --model http://localhost:30000 \
    --dataset gsm8k \
    --k 1 4 8
```

Expected output:
```
============================================================
PASS@K RESULTS
============================================================
k        Accuracy     Tokens/s    
------------------------------------------------------------
1          44.7%       23322
4          75.1%       23322
8          84.2%       23322
```

### Step 5: Evaluate Trained Model

```bash
# Restart server with trained model
pkill -f sglang && sleep 2
CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server \
    --model-path ./outputs/grpo_model \
    --host 0.0.0.0 --port 30000 \
    --trust-remote-code > sglang.log 2>&1 &
sleep 45

# Evaluate
python scripts/eval_pass_at_k.py \
    --model http://localhost:30000 \
    --dataset gsm8k \
    --k 1 4 8
```

Expected output:
```
============================================================
PASS@K RESULTS
============================================================
k        Accuracy     Tokens/s    
------------------------------------------------------------
1          74.0%       24038
4          88.0%       24038
8          92.6%       24038
```

## 📁 Project Structure

```
distributed-reasoning-loop/
├── src/
│   ├── data_generator/
│   │   ├── cot_generator.py           # SGLang inference
│   │   ├── synthetic_data_pipeline.py # Data generation pipeline
│   │   ├── data_preprocessor.py       # Quality filtering, deduplication
│   │   └── dataset_loader.py          # GSM8K, HumanEval loaders
│   ├── verifier/
│   │   ├── math_verifier.py           # SymPy symbolic verification
│   │   └── code_verifier.py           # Docker sandbox execution
│   ├── orchestration/
│   │   ├── ray_workers.py             # Distributed processing
│   │   ├── kafka_streaming.py         # Streaming pipeline
│   │   ├── worker_pool.py             # SGLang worker pool (NEW!)
│   │   └── coordinator.py             # Training coordinator (NEW!)
│   ├── training/
│   │   ├── grpo_trainer.py            # Group Relative Policy Optimization
│   │   ├── dpo_trainer.py             # Direct Preference Optimization
│   │   └── sft_trainer.py             # Supervised Fine-Tuning
│   └── evaluation/
│       ├── benchmarks.py              # Evaluation metrics
│       └── test_time_compute.py       # Pass@k, Best-of-N
├── scripts/
│   ├── run_ray_pipeline.py            # Batch pipeline script
│   ├── run_continuous_grpo.py         # Continuous training (NEW!)
│   ├── start_workers.sh               # Worker launcher (NEW!)
│   ├── stop_workers.sh                # Worker shutdown (NEW!)
│   ├── dashboard.py                   # Training dashboard (NEW!)
│   └── eval_pass_at_k.py              # Evaluation script
├── config/
│   └── default.yaml                   # Configuration
└── outputs/
    ├── synthetic_data/                # Generated data
    ├── grpo_model/                    # Trained model
    └── continuous_grpo/               # Continuous training outputs (NEW!)
        ├── lora_checkpoints/          # Versioned LoRA adapters
        └── logs/                       # Training metrics
```

## 🔧 Key Components

### 1. SGLang Generation
- **RadixAttention**: Automatic prefix caching for shared prompts
- **Batched inference**: Concurrent requests for high throughput
- **Multi-path sampling**: 10 reasoning paths per problem
- **LoRA Hot-Reload**: Dynamic adapter loading without server restart (NEW!)

### 2. Ray Distributed Verification
- **Parallel workers**: 4 actors processing chunks (1250 samples each)
- **Math verification**: SymPy symbolic comparison
- **Balanced distribution**: Even workload across workers

### 3. GRPO Training (DeepSeek-R1 Approach)
- **No reward model**: Uses group-relative advantages
- **Verification-based**: Correct = positive, incorrect = negative
- **Efficient**: LoRA with 0.07% trainable parameters (1,089,536 params)
- **Incremental training**: Single-step training for continuous loop (NEW!)

### 4. Worker Pool & Coordinator (NEW!)
- **Multi-worker pool**: Manages multiple SGLang inference servers
- **Load balancing**: Round-robin or least-pending request distribution
- **LoRA version tracking**: Each worker tracks current LoRA version
- **Broadcast updates**: Atomically update all workers with new LoRA
- **Health monitoring**: Automatic health checks and failure recovery

### 5. Continuous Training Loop (NEW!)
```
For each iteration:
  1. Generate rollouts (parallel across N workers, all using LoRA vK)
  2. Verify rollouts (Ray workers with SymPy)
  3. Train GRPO step (compute advantages, gradient update)
  4. Save LoRA checkpoint (vK+1)
  5. Broadcast to all workers (hot-reload vK+1)
  6. Repeat with updated policy
```

Key benefits:
- **No cold starts**: Workers stay running, only LoRA weights reload
- **Parallel generation**: N workers = Nx throughput
- **Fresh policy**: Every iteration uses the latest trained weights
- **Automatic checkpointing**: Version-tracked LoRA adapters



##  Hardware Requirements

- **GPU**: 2x H100 (80GB) or equivalent
- **RAM**: 256GB+ recommended
- **Storage**: 50GB for models and data

## 📚 References

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - GRPO algorithm
- [SGLang](https://github.com/sgl-project/sglang) - RadixAttention inference
- [Ray](https://ray.io/) - Distributed computing
- [GSM8K](https://arxiv.org/abs/2110.14168) - Math reasoning benchmark
- [Prime Intellect](https://www.primeintellect.ai/) - Distributed RL infrastructure

