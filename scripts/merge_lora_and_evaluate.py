#!/usr/bin/env python3
"""
Merge LoRA weights into base model and evaluate with Pass@k metrics.

Evaluates Pass@1, Pass@4, Pass@8 for both base and trained models.

Usage:
    python scripts/merge_lora_and_evaluate.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --lora-path ./outputs/continuous_grpo/lora_checkpoints/lora_v10 \
        --num-problems 1319 \
        --batch-size 32
"""

import argparse
import json
import logging
import sys
import time
import re
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PassAtKResult:
    """Pass@k evaluation result."""
    model_name: str
    k: int
    accuracy: float
    num_correct: int
    num_total: int
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "k": self.k,
            "accuracy_percent": round(self.accuracy * 100, 2),
            "num_correct": self.num_correct,
            "num_total": self.num_total,
        }


@dataclass
class ModelResults:
    """All results for a model."""
    model_name: str
    pass_at_1: float = 0.0
    pass_at_4: float = 0.0
    pass_at_8: float = 0.0
    total_time: float = 0.0


def merge_lora_to_base(
    base_model_name: str,
    lora_path: str,
    output_path: str,
) -> str:
    """Merge LoRA weights into base model and save."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    
    logger.info(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    del base_model, model, merged_model
    torch.cuda.empty_cache()
    
    logger.info("✓ Merge complete!")
    return output_path


def load_gsm8k(num_samples: int = 1319) -> List[Dict]:
    """Load GSM8K test set."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    
    problems = []
    for item in list(ds)[:num_samples]:
        answer_text = item["answer"]
        if "####" in answer_text:
            answer = answer_text.split("####")[-1].strip()
        else:
            answer = answer_text.strip()
        
        problems.append({
            "prompt": f"Solve this math problem step by step. Show your work and put your final answer after ####.\n\nProblem: {item['question']}",
            "answer": answer,
        })
    
    return problems


def extract_answer(response: str) -> Optional[str]:
    """Extract numerical answer from response."""
    nums = re.findall(r'####\s*(-?\d+[\d,]*\.?\d*)', response)
    if nums:
        return nums[-1].replace(",", "")
    
    nums = re.findall(r'answer is[:\s]*(-?\d+[\d,]*\.?\d*)', response.lower())
    if nums:
        return nums[-1].replace(",", "")
    
    nums = re.findall(r'(-?\d+[\d,]*\.?\d*)', response)
    if nums:
        return nums[-1].replace(",", "")
    
    return None


def check_correctness(response: str, ground_truth: str) -> bool:
    """Check if response is correct."""
    extracted = extract_answer(response)
    if extracted is None:
        return False
    
    gt_clean = ground_truth.strip().replace(",", "")
    gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
    if not gt_nums:
        return False
    
    gt_answer = gt_nums[-1]
    
    try:
        return abs(float(extracted) - float(gt_answer)) < 0.001
    except:
        return extracted == gt_answer


def evaluate_pass_at_k(
    model_path: str,
    problems: List[Dict],
    model_name: str,
    k_values: List[int] = [1, 4, 8],
    batch_size: int = 32,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
) -> ModelResults:
    """
    Evaluate Pass@k for multiple k values.
    
    Pass@k: Generate k samples, problem is correct if ANY sample is correct.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    max_k = max(k_values)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"Pass@k for k={k_values}")
    logger.info(f"Generating {max_k} samples per problem")
    logger.info(f"{'='*70}")
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model.eval()
    
    # Store correctness for each sample of each problem
    # problem_results[i] = [True/False for each of k samples]
    problem_results = []
    
    total_time = 0
    num_batches = (len(problems) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(problems))
        batch_problems = problems[batch_start:batch_end]
        
        # For each problem, generate max_k samples
        batch_correctness = [[] for _ in batch_problems]
        
        # Generate samples in sub-batches to avoid OOM
        samples_per_generation = min(4, max_k)  # Generate 4 samples at a time
        num_generations = (max_k + samples_per_generation - 1) // samples_per_generation
        
        for gen_round in range(num_generations):
            samples_this_round = min(samples_per_generation, max_k - gen_round * samples_per_generation)
            
            # Prepare prompts (repeat each prompt for number of samples)
            prompts = [p["prompt"] for p in batch_problems]
            answers = [p["answer"] for p in batch_problems]
            
            texts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
            
            start_time = time.time()
            
            # Tokenize
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            
            # Generate multiple samples with sampling
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    num_return_sequences=samples_this_round,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Decode and check correctness
            # outputs shape: [batch_size * samples_this_round, seq_len]
            for i, answer in enumerate(answers):
                for s in range(samples_this_round):
                    output_idx = i * samples_this_round + s
                    input_len = inputs.input_ids[i].shape[0]
                    
                    response = tokenizer.decode(
                        outputs[output_idx][input_len:],
                        skip_special_tokens=True
                    )
                    
                    is_correct = check_correctness(response, answer)
                    batch_correctness[i].append(is_correct)
        
        problem_results.extend(batch_correctness)
        
        # Progress update
        if (batch_idx + 1) % 5 == 0:
            # Quick Pass@1 estimate
            p1_correct = sum(1 for r in problem_results if r and r[0])
            p1_acc = p1_correct / len(problem_results) * 100
            speed = len(problem_results) / total_time
            logger.info(f"Progress: {len(problem_results)}/{len(problems)} | Pass@1: {p1_acc:.1f}% | Speed: {speed:.1f} prob/s")
    
    # Calculate Pass@k for each k
    results = ModelResults(model_name=model_name, total_time=total_time)
    
    for k in k_values:
        correct = 0
        for problem_correctness in problem_results:
            # Pass@k: correct if ANY of first k samples is correct
            if any(problem_correctness[:k]):
                correct += 1
        
        accuracy = correct / len(problems) if problems else 0
        
        if k == 1:
            results.pass_at_1 = accuracy
        elif k == 4:
            results.pass_at_4 = accuracy
        elif k == 8:
            results.pass_at_8 = accuracy
        
        logger.info(f"Pass@{k}: {accuracy*100:.1f}% ({correct}/{len(problems)})")
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    return results


def print_comparison_table(base_results: ModelResults, trained_results: ModelResults, num_problems: int):
    """Print comparison table in the requested format."""
    print("\n")
    print("=" * 80)
    print(f"GSM8K EVALUATION RESULTS ({num_problems} problems)")
    print("=" * 80)
    print()
    print(f"{'Metric':<12} {'Base Model':<15} {'GRPO Trained':<15} {'Improvement':<15}")
    print("-" * 80)
    
    # Pass@1
    base_p1 = base_results.pass_at_1 * 100
    trained_p1 = trained_results.pass_at_1 * 100
    imp_p1 = trained_p1 - base_p1
    print(f"{'Pass@1':<12} {base_p1:>6.1f}%         {trained_p1:>6.1f}%         {imp_p1:>+6.1f}%")
    
    # Pass@4
    base_p4 = base_results.pass_at_4 * 100
    trained_p4 = trained_results.pass_at_4 * 100
    imp_p4 = trained_p4 - base_p4
    print(f"{'Pass@4':<12} {base_p4:>6.1f}%         {trained_p4:>6.1f}%         {imp_p4:>+6.1f}%")
    
    # Pass@8
    base_p8 = base_results.pass_at_8 * 100
    trained_p8 = trained_results.pass_at_8 * 100
    imp_p8 = trained_p8 - base_p8
    print(f"{'Pass@8':<12} {base_p8:>6.1f}%         {trained_p8:>6.1f}%         {imp_p8:>+6.1f}%")
    
    print("=" * 80)
    print()
    print(f"📈 Key Improvement: Pass@1 {base_p1:.1f}% → {trained_p1:.1f}% (+{imp_p1:.1f}%)")
    print()
    
    if imp_p1 > 20:
        print("🎉 Excellent! Training achieved significant improvement!")
    elif imp_p1 > 10:
        print("✅ Good improvement from training!")
    elif imp_p1 > 0:
        print("📊 Modest improvement from training.")
    else:
        print("⚠️ No improvement - may need more training iterations.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pass@k for Base vs Trained Model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output-path", type=str, default="./outputs/merged_model")
    parser.add_argument("--num-problems", type=int, default=1319, help="Number of problems")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--output", type=str, default="./eval_pass_at_k_results.json")
    
    args = parser.parse_args()
    
    # Merge LoRA
    if not args.eval_only:
        merged_path = merge_lora_to_base(
            args.base_model,
            args.lora_path,
            args.output_path,
        )
    else:
        merged_path = args.output_path
    
    if args.merge_only:
        logger.info(f"\n✅ Merged model saved to: {merged_path}")
        return
    
    # Load dataset
    logger.info(f"\nLoading GSM8K test set ({args.num_problems} problems)...")
    problems = load_gsm8k(args.num_problems)
    logger.info(f"Loaded {len(problems)} problems")
    
    k_values = [1, 4, 8]
    
    # Evaluate base model
    if not args.skip_base:
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: Evaluating BASE MODEL")
        logger.info("="*70)
        
        base_results = evaluate_pass_at_k(
            model_path=args.base_model,
            problems=problems,
            model_name="Base (Qwen2.5-1.5B-Instruct)",
            k_values=k_values,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        # Use placeholder if skipping
        base_results = ModelResults(
            model_name="Base (Qwen2.5-1.5B-Instruct)",
            pass_at_1=0.447,  # Previous result
            pass_at_4=0.751,
            pass_at_8=0.842,
        )
    
    # Evaluate trained model
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: Evaluating TRAINED MODEL (GRPO)")
    logger.info("="*70)
    
    trained_results = evaluate_pass_at_k(
        model_path=merged_path,
        problems=problems,
        model_name="Trained (GRPO LoRA)",
        k_values=k_values,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    # Print comparison table
    print_comparison_table(base_results, trained_results, len(problems))
    
    # Save results
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "num_problems": args.num_problems,
            "k_values": k_values,
            "temperature": args.temperature,
        },
        "base_model": {
            "pass_at_1": round(base_results.pass_at_1 * 100, 2),
            "pass_at_4": round(base_results.pass_at_4 * 100, 2),
            "pass_at_8": round(base_results.pass_at_8 * 100, 2),
        },
        "trained_model": {
            "pass_at_1": round(trained_results.pass_at_1 * 100, 2),
            "pass_at_4": round(trained_results.pass_at_4 * 100, 2),
            "pass_at_8": round(trained_results.pass_at_8 * 100, 2),
        },
        "improvement": {
            "pass_at_1": round((trained_results.pass_at_1 - base_results.pass_at_1) * 100, 2),
            "pass_at_4": round((trained_results.pass_at_4 - base_results.pass_at_4) * 100, 2),
            "pass_at_8": round((trained_results.pass_at_8 - base_results.pass_at_8) * 100, 2),
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
