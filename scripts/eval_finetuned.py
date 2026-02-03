#!/usr/bin/env python3
"""
Evaluate Fine-tuned vs Base Model

Compares GRPO fine-tuned model against base model on held-out problems.
Generates metrics for README and validates training effectiveness.

Usage:
    python scripts/eval_finetuned.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --finetuned-model ./outputs/grpo_model \
        --num-problems 50
"""

import argparse
import json
import logging
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    model_name: str
    pass_at_1: float
    pass_at_4: float
    pass_at_8: float
    avg_response_length: float
    avg_reasoning_steps: float
    inference_time_per_problem: float
    total_problems: int
    correct_at_1: int


@dataclass 
class ComparisonResults:
    """Comparison between base and fine-tuned models."""
    base: ModelMetrics
    finetuned: ModelMetrics
    improvement_pass_at_1: float
    improvement_pass_at_4: float
    improvement_pass_at_8: float
    held_out_problems: int
    timestamp: str


class ModelEvaluator:
    """Evaluates a single model on reasoning tasks with batching."""
    
    def __init__(self, model_path: str, device: str = "cuda", batch_size: int = 8):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        # Required for batching: pad on the left so the model generates from the end
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info(f"Model loaded: {self.model_path}")
    
    def generate(
        self,
        prompt: str,
        n_samples: int = 1,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate n responses for a prompt using batching."""
        messages = [{"role": "user", "content": prompt}]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Expand prompt to n_samples for batching
        batch_prompts = [input_text] * n_samples
        
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=2048,
        ).to(self.device)
        
        responses = []
        with torch.no_grad():
            # Process in sub-batches to manage VRAM
            for i in range(0, n_samples, self.batch_size):
                sub_batch = {k: v[i : i + self.batch_size] for k, v in inputs.items()}
                outputs = self.model.generate(
                    **sub_batch,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                # Decode only the generated part
                prompt_len = sub_batch["input_ids"].shape[1]
                decoded = self.tokenizer.batch_decode(
                    outputs[:, prompt_len:], 
                    skip_special_tokens=True
                )
                responses.extend(decoded)
        
        return responses
    
    def check_answer(self, response: str, ground_truth: str) -> bool:
        """Check if response contains correct answer."""
        patterns = [
            r'####\s*(-?\d+\.?\d*)',
            r'\\boxed\{([^}]+)\}',
            r'answer[:\s]+(-?\d+\.?\d*)',
            r'=\s*(-?\d+\.?\d*)\s*$',
        ]
        
        response_answer = None
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                response_answer = matches[-1].strip()
                break
        
        if response_answer is None:
            nums = re.findall(r'-?\d+\.?\d*', response)
            if nums:
                response_answer = nums[-1]
        
        gt_clean = ground_truth.strip().replace(",", "").replace("$", "")
        gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
        gt_answer = gt_nums[-1] if gt_nums else gt_clean
        
        if response_answer is None:
            return False
        
        try:
            return abs(float(response_answer) - float(gt_answer)) < 1e-6
        except ValueError:
            return response_answer.lower() == gt_answer.lower()
    
    def count_reasoning_steps(self, response: str) -> int:
        """Count reasoning steps in response."""
        step_patterns = [
            r'step\s*\d+',
            r'^\d+\.',
            r'first|second|third|then|next|finally',
        ]
        count = 0
        for pattern in step_patterns:
            count += len(re.findall(pattern, response, re.IGNORECASE | re.MULTILINE))
        return max(count, response.count('\n') // 2)

    def evaluate(
        self,
        problems: List[Dict],
        k_values: List[int] = [1, 4, 8],
        temperature: float = 0.7,
    ) -> ModelMetrics:
        """Evaluate model on problems."""
        max_k = max(k_values)
        all_correctness = []
        all_lengths = []
        all_steps = []
        total_time = 0
        
        for prob in tqdm(problems, desc=f"Evaluating {Path(self.model_path).name}"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            start = time.time()
            responses = self.generate(prompt, n_samples=max_k, temperature=temperature)
            total_time += time.time() - start
            
            correctness = [self.check_answer(r, answer) for r in responses]
            all_correctness.append(correctness)
            
            all_lengths.extend([len(r) for r in responses])
            all_steps.extend([self.count_reasoning_steps(r) for r in responses])
        
        pass_at_k = {}
        for k in k_values:
            correct = sum(1 for c in all_correctness if any(c[:k]))
            pass_at_k[k] = correct / len(problems)
        
        return ModelMetrics(
            model_name=self.model_path,
            pass_at_1=pass_at_k.get(1, 0),
            pass_at_4=pass_at_k.get(4, 0),
            pass_at_8=pass_at_k.get(8, 0),
            avg_response_length=sum(all_lengths) / len(all_lengths) if all_lengths else 0,
            avg_reasoning_steps=sum(all_steps) / len(all_steps) if all_steps else 0,
            inference_time_per_problem=total_time / len(problems),
            total_problems=len(problems),
            correct_at_1=sum(1 for c in all_correctness if c[0]),
        )


def load_held_out_problems(num_problems: int = 50) -> List[Dict]:
    """Load held-out evaluation problems from GSM8K."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split=f"test[:{num_problems}]")
        problems = []
        for item in ds:
            answer = item["answer"].split("####")[-1].strip()
            problems.append({
                "prompt": item["question"],
                "answer": answer,
            })
        logger.info(f"Loaded {len(problems)} GSM8K test problems")
        return problems
    except Exception as e:
        logger.warning(f"Could not load GSM8K: {e}")
        return []


def compare_models(
    base_model: str,
    finetuned_model: str,
    num_problems: int = 50,
    output_path: str = "./comparison_results.json",
) -> ComparisonResults:
    """Compare base and fine-tuned models."""
    problems = load_held_out_problems(num_problems)
    
    # Base
    logger.info("=" * 60)
    logger.info("Evaluating BASE model")
    logger.info("=" * 60)
    base_evaluator = ModelEvaluator(base_model)
    base_evaluator.load()
    base_metrics = base_evaluator.evaluate(problems)
    
    del base_evaluator.model
    torch.cuda.empty_cache()
    
    # Fine-tuned
    logger.info("=" * 60)
    logger.info("Evaluating FINE-TUNED model")
    logger.info("=" * 60)
    ft_evaluator = ModelEvaluator(finetuned_model)
    ft_evaluator.load()
    ft_metrics = ft_evaluator.evaluate(problems)
    
    from datetime import datetime
    results = ComparisonResults(
        base=base_metrics,
        finetuned=ft_metrics,
        improvement_pass_at_1=(ft_metrics.pass_at_1 - base_metrics.pass_at_1) * 100,
        improvement_pass_at_4=(ft_metrics.pass_at_4 - base_metrics.pass_at_4) * 100,
        improvement_pass_at_8=(ft_metrics.pass_at_8 - base_metrics.pass_at_8) * 100,
        held_out_problems=num_problems,
        timestamp=datetime.now().isoformat(),
    )
    
    with open(output_path, "w") as f:
        json.dump({
            "base": asdict(base_metrics),
            "finetuned": asdict(ft_metrics),
            "improvements": {
                "pass_at_1": results.improvement_pass_at_1,
                "pass_at_4": results.improvement_pass_at_4,
                "pass_at_8": results.improvement_pass_at_8,
            },
            "held_out_problems": num_problems,
            "timestamp": results.timestamp,
        }, f, indent=2)
    
    return results


def print_comparison_report(results: ComparisonResults):
    """Print formatted comparison report."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT: Base vs GRPO Fine-tuned")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Base':<15} {'Fine-tuned':<15} {'Δ':<10}")
    print("-" * 70)
    
    b, f = results.base, results.finetuned
    fmt_delta = lambda d: f"+{d:.1f}%" if d > 0 else f"{d:.1f}%"
    
    print(f"{'Pass@1':<30} {b.pass_at_1*100:>12.1f}% {f.pass_at_1*100:>12.1f}% {fmt_delta(results.improvement_pass_at_1):>10}")
    print(f"{'Pass@4':<30} {b.pass_at_4*100:>12.1f}% {f.pass_at_4*100:>12.1f}% {fmt_delta(results.improvement_pass_at_4):>10}")
    print(f"{'Pass@8':<30} {b.pass_at_8*100:>12.1f}% {f.pass_at_8*100:>12.1f}% {fmt_delta(results.improvement_pass_at_8):>10}")
    print(f"{'Avg Response Length':<30} {b.avg_response_length:>12.0f} {f.avg_response_length:>12.0f}")
    print(f"{'Avg Reasoning Steps':<30} {b.avg_reasoning_steps:>12.1f} {f.avg_reasoning_steps:>12.1f}")
    print(f"{'Inference Time (s/prob)':<30} {b.inference_time_per_problem:>12.2f} {f.inference_time_per_problem:>12.2f}")
    
    print("\n📊 SUMMARY:")
    print(f"   Pass@1 improvement: {fmt_delta(results.improvement_pass_at_1)}")
    print(f"   Pass@8 improvement: {fmt_delta(results.improvement_pass_at_8)}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--finetuned-model", type=str, default="./outputs/grpo_model")
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--output", type=str, default="./comparison_results.json")
    parser.add_argument("--readme-output", type=str, default="./METRICS.md")
    
    args = parser.parse_args()
    results = compare_models(args.base_model, args.finetuned_model, args.num_problems, args.output)
    print_comparison_report(results)

if __name__ == "__main__":
    main()