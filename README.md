# Distributed Reasoning Loop

End-to-end GRPO (Group Relative Policy Optimization) pipeline for training reasoning models. Implements DeepSeek-R1's approach: synthetic data generation, distributed verification, and RL training without reward models.

## 🎯 Results

Evaluated on GSM8K test problems:

| Metric | Base Model | GRPO Trained | Improvement |
|--------|------------|--------------|-------------|
| Pass@1 | 42.5% | 69.0% | **+26.5%** |
| Pass@4 | 74.5% | 88.0% | **+13.5%** |
| Pass@8 | 82.0% | 92.5% | **+10.5%** |

## ⚡ Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **SGLang** | Generation | **3.5 min** for 5K samples |
| **SGLang** | Throughput | **24K tokens/sec** |
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
│  • 10 paths/problem  • SymPy verify       advantages            │
│  • Batched requests  • 24K tok/sec      • LoRA fine-tuning      │
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
    omegaconf accelerate bitsandbytes jsonschema jinja2 --upgrade
```

### Step 1: Clone Repository

```bash
git clone https://github.com/pmukeshreddy/distributed-reasoning-loop.git
cd distributed-reasoning-loop
```

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
Epoch 1: 100%|████████████████| 213/213 [01:46<00:00, loss=-0.0302]
Epoch 2: 100%|████████████████| 213/213 [01:41<00:00, loss=-0.0345]
Epoch 3: 100%|████████████████| 213/213 [01:41<00:00, loss=-0.0409]
Epoch 4: 100%|████████████████| 213/213 [01:41<00:00, loss=-0.0505]
Epoch 5: 100%|████████████████| 213/213 [01:41<00:00, loss=-0.0723]

Done: SGLang -> Ray -> GRPO
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
    --k 1 4 8 \
    --num-problems 200
```

Expected output:
```
============================================================
PASS@K RESULTS
============================================================
k        Accuracy     Correct      Tokens/s    
------------------------------------------------------------
1          42.5%     85/200         21960
4          74.5%     149/200        21960
8          82.0%     164/200        21960
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
    --k 1 4 8 \
    --num-problems 200
```

Expected output:
```
============================================================
PASS@K RESULTS
============================================================
k        Accuracy     Correct      Tokens/s    
------------------------------------------------------------
1          69.0%     138/200        22432
4          88.0%     176/200        22432
8          92.5%     185/200        22432
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
│   │   └── kafka_streaming.py         # Streaming pipeline
│   ├── training/
│   │   ├── grpo_trainer.py            # Group Relative Policy Optimization
│   │   ├── dpo_trainer.py             # Direct Preference Optimization
│   │   └── sft_trainer.py             # Supervised Fine-Tuning
│   └── evaluation/
│       ├── benchmarks.py              # Evaluation metrics
│       └── test_time_compute.py       # Pass@k, Best-of-N
├── scripts/
│   ├── run_ray_pipeline.py            # Full pipeline script
│   └── eval_pass_at_k.py              # Evaluation script
├── config/
│   └── default.yaml                   # Configuration
└── outputs/
    ├── synthetic_data/                # Generated data
    └── grpo_model/                    # Trained model
```

## 🔧 Key Components

### 1. SGLang Generation
- **RadixAttention**: Automatic prefix caching for shared prompts
- **Batched inference**: Concurrent requests for high throughput
- **Multi-path sampling**: 10 reasoning paths per problem

### 2. Ray Distributed Verification
- **Parallel workers**: 4 actors processing chunks (1250 samples each)
- **Math verification**: SymPy symbolic comparison
- **Balanced distribution**: Even workload across workers

### 3. GRPO Training (DeepSeek-R1 Approach)
- **No reward model**: Uses group-relative advantages
- **Verification-based**: Correct = positive, incorrect = negative
- **Efficient**: LoRA with 0.07% trainable parameters (1,089,536 params)

## 📊 Pipeline Stats

From actual run:

```
Dataset:           GSM8K (500 problems)
Paths generated:   5,000 (10 per problem)
Correct paths:     1,192 (24%)
Incorrect paths:   3,808 (76%)
After preprocessing: 4,877 samples
DPO pairs created: 2,125
Training groups:   425
Training epochs:   5
Loss progression:  -0.0066 → -0.0106 → -0.0125 → -0.0149 → -0.0220
Total time:        ~12 minutes on 1x H100
```

## 🖥️ Hardware Requirements

- **GPU**: 1x H100 (80GB) or equivalent
- **RAM**: 256GB+ recommended
- **Storage**: 50GB for models and data

## 📚 References

- [DeepSeek-R1](https://arxiv.org/abs/2401.02954) - GRPO algorithm
- [SGLang](https://github.com/sgl-project/sglang) - RadixAttention inference
- [Ray](https://ray.io/) - Distributed computing
- [GSM8K](https://arxiv.org/abs/2110.14168) - Math reasoning benchmark
- [Prime Intellect](https://www.primeintellect.ai/) - Distributed RL infrastructure
