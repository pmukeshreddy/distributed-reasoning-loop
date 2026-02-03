#!/usr/bin/env python3
"""
Merge LoRA weights into base model and evaluate with BATCHED inference.

This is the most reliable way to evaluate a LoRA-trained model.
Uses batched inference for 5-10x speedup!

Usage:
    # Full evaluation with batching
    python scripts/merge_lora_and_evaluate.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --lora-path ./outputs/continuous_grpo/lora_checkpoints/lora_v10 \
        --num-problems 1319 \
        --batch-size 16
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
from dataclasses import dataclass
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result."""
    model_name: str
    accuracy: float
    num_correct: int
    num_total: int
    avg_time_per_problem: float
    total_time: float
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "accuracy_percent": round(self.accuracy * 100, 2),
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "avg_time_per_problem": round(self.avg_time_per_problem, 3),
            "total_time": round(self.total_time, 2),
        }


def merge_lora_to_base(
    base_model_name: str,
    lora_path: str,
    output_path: str,
    device: str = "cuda",
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
    
    # Free memory
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


def evaluate_model_batched(
    model_path: str,
    problems: List[Dict],
    model_name: str,
    batch_size: int = 16,
    max_new_tokens: int = 1024,
) -> EvalResult:
    """Evaluate a model using BATCHED inference for speed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"{'='*60}")
    
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
    tokenizer.padding_side = "left"  # Important for batch generation
    
    model.eval()
    
    correct = 0
    total_time = 0
    all_responses = []
    
    # Process in batches
    num_batches = (len(problems) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(problems))
        batch_problems = problems[batch_start:batch_end]
        
        # Prepare batch
        prompts = [p["prompt"] for p in batch_problems]
        answers = [p["answer"] for p in batch_problems]
        
        # Format as chat messages
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
        
        # Tokenize batch
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        
        # Generate batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.eos_token_id,
            )
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Decode responses
        for i, (output, answer) in enumerate(zip(outputs, answers)):
            # Get only the generated part (after input)
            input_len = inputs.input_ids[i].shape[0]
            response = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            )
            all_responses.append(response)
            
            if check_correctness(response, answer):
                correct += 1
        
        # Progress update every 10 batches
        if (batch_idx + 1) % 10 == 0:
            current_acc = correct / (batch_end) * 100
            speed = batch_end / total_time
            logger.info(f"Progress: {batch_end}/{len(problems)} | Acc: {current_acc:.1f}% | Speed: {speed:.1f} problems/s")
    
    accuracy = correct / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    result = EvalResult(
        model_name=model_name,
        accuracy=accuracy,
        num_correct=correct,
        num_total=len(problems),
        avg_time_per_problem=avg_time,
        total_time=total_time,
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} FINAL RESULTS:")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  Correct: {correct}/{len(problems)}")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Speed: {len(problems)/total_time:.1f} problems/s")
    logger.info(f"{'='*60}")
    
    return result


def print_comparison(results: List[EvalResult]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - GSM8K FULL TEST SET")
    print("=" * 80)
    print(f"{'Model':<40} {'Accuracy':<15} {'Correct':<15} {'Time'}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.model_name:<40} {r.accuracy*100:>6.2f}%        {r.num_correct:>4}/{r.num_total:<4}       {r.total_time:.1f}s")
    
    print("=" * 80)
    
    if len(results) >= 2:
        base_acc = results[0].accuracy
        trained_acc = results[-1].accuracy
        improvement = trained_acc - base_acc
        
        print(f"\n📈 IMPROVEMENT: {improvement*100:+.2f}%")
        print(f"   Base: {base_acc*100:.2f}% → Trained: {trained_acc*100:.2f}%")
        
        if improvement > 0:
            print(f"   ✅ Training improved accuracy by {improvement*100:.1f} percentage points!")
        elif improvement < 0:
            print(f"   ⚠️ Training decreased accuracy")
        else:
            print(f"   ➡️ No change in accuracy")
    print()


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and Evaluate (Batched)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output-path", type=str, default="./outputs/merged_model")
    parser.add_argument("--num-problems", type=int, default=1319, help="Number of problems (1319 = full GSM8K test)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--merge-only", action="store_true", help="Only merge, don't evaluate")
    parser.add_argument("--eval-only", action="store_true", help="Skip merge, just evaluate existing merged model")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model evaluation")
    parser.add_argument("--output", type=str, default="./eval_full_results.json")
    
    args = parser.parse_args()
    
    # Merge LoRA into base model
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
    
    results = []
    
    # Evaluate base model
    if not args.skip_base:
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Evaluating BASE MODEL")
        logger.info("="*60)
        
        base_result = evaluate_model_batched(
            model_path=args.base_model,
            problems=problems,
            model_name="Base (Qwen2.5-1.5B-Instruct)",
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
        )
        results.append(base_result)
    
    # Evaluate merged model
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Evaluating TRAINED MODEL (Merged)")
    logger.info("="*60)
    
    trained_result = evaluate_model_batched(
        model_path=merged_path,
        problems=problems,
        model_name=f"Trained (GRPO LoRA)",
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
    )
    results.append(trained_result)
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "merged_path": merged_path,
            "num_problems": args.num_problems,
            "batch_size": args.batch_size,
        },
        "results": [r.to_dict() for r in results],
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
