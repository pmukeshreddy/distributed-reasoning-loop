# Distributed Reasoning Loop

> **Scalable infrastructure for RL-based reasoning model training with test-time compute scaling**

A high-performance distributed pipeline for synthetic data generation, preference learning (GRPO/DPO), and test-time compute scaling research. This project demonstrates how small, efficient architectures (1.5B) can achieve frontier-level reasoning consistency through verifiable RL post-training.

## 🎯 Key Features

| Component | What it Does |
|-----------|--------------|
| **GRPO Trainer** | Group Relative Policy Optimization implementation that eliminates the need for a separate reward model by using group-relative advantages. |
| **Inter-Problem Batching** | Optimized inference engine achieving ~0.5s per sample throughput by saturating GPU compute across multiple reasoning problems. |
| **Verifiable Rewards** | Automatic correctness verification for Math (GSM8K) and Code, ensuring logic-driven policy improvement. |
| **Test-Time Scaling** | Native Pass@k evaluation suite to measure the accuracy-compute tradeoff. |

## 📊 Performance Benchmarks

### Test-Time Compute Scaling (Pass@k)
Results achieved after **GRPO post-training** on a **Qwen2.5-1.5B-Instruct** base using the distributed reasoning loop.

**Dataset:** GSM8K (Held-out test set)  
**Method:** GRPO (3 Epochs, Verified Rewards)

| Metric | Base Model | Fine-tuned (GRPO) | Improvement (Δ) |
|--------|-----------:|------------------:|----------------:|
| **Pass@1** | 35.0% | **55.0%** | **+20.0%** |
| **Pass@4** | 65.0% | 75.0% | +10.0% |
| **Pass@8** | 80.0% | **90.0%** | +10.0% |

> **Note:** The +20% jump in Pass@1 demonstrates that the RL loop effectively aligned the model's primary reasoning path with correct logical steps.

### Reasoning Quality

| Metric | Base | Fine-tuned |
|--------|-----:|-----------:|
| **Avg Reasoning Steps** | 14.8 | **17.0** |
| **Avg Response Length** | 1151 chars | 1307 chars |

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                   Distributed Reasoning Loop                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Kafka   │───▶│   Ray    │───▶│  SGLang  │───▶│ Verifier │  │
│  │ (Queue)  │    │ Workers  │    │ (Infer)  │    │ (Reward) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│        │                                               │        │
│        │              ┌──────────────┐                 │        │
│        └─────────────▶│   Trainer    │◀────────────────┘        │
│                       │ (GRPO/DPO)   │                          │
│                       └──────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘



## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/pmukeshreddy/distributed-reasoning-loop.git
cd distributed-reasoning-loop
pip install -e .

### Reproduce Evaluation
```bash

python scripts/eval_finetuned.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --finetuned-model ./outputs/grpo_model \
  --num-problems 50

