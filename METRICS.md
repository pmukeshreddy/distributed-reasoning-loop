
## 📊 Evaluation Results

### Model Comparison: Base vs GRPO Fine-tuned

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| Pass@1 | 96.0% | 100.0% | +4.0% |
| Pass@4 | 100.0% | 100.0% | 0.0% |
| Pass@8 | 100.0% | 100.0% | 0.0% |
| Avg Response Length | 97 chars | 272 chars | - |
| Avg Reasoning Steps | 1.7 | 4.8 | - |

**Evaluation Details:**
- Held-out problems: 50
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Fine-tuned model: GRPO with 3 epochs, lr=5e-5
- Timestamp: 2026-01-10

### Training Configuration

```yaml
method: GRPO (Group Relative Policy Optimization)
base_model: Qwen/Qwen2.5-7B-Instruct
epochs: 3
learning_rate: 5e-5
lora_r: 8
lora_alpha: 16
kl_coef: 0.1
clip_range: 0.2
training_samples: 2000
dpo_pairs: 3125
```

### Key Insights

1. **GRPO Training**: Successfully trained without a reward model by using group-relative advantages
2. **Loss Trajectory**: Loss decreased from -0.0017 → -0.0352 (more negative = better preference learning)
3. **Test-Time Scaling**: Pass@k shows 0.0% improvement from k=1 to k=8
