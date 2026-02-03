#!/usr/bin/env python3
"""
Comprehensive Evaluation: Base Model vs Trained (LoRA) Model

This script properly evaluates:
1. Base model (Qwen/Qwen2.5-1.5B-Instruct) without any fine-tuning
2. Trained model (Base + LoRA adapter from continuous GRPO)

Usage:
    # Evaluate both models
    python scripts/eval_trained_vs_base.py --port 30001 --lora-path ./outputs/continuous_grpo/lora_checkpoints/lora_v9
    
    # Base model only
    python scripts/eval_trained_vs_base.py --port 30001 --base-only
    
    # Full evaluation with all methods
    python scripts/eval_trained_vs_base.py --port 30001 --lora-path ./outputs/continuous_grpo/lora_checkpoints/lora_v9 --all-methods
"""

import argparse
import json
import logging
import sys
import time
import re
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result for a model."""
    model_name: str
    accuracy: float
    num_correct: int
    num_total: int
    avg_time_per_problem: float
    total_time: float
    method: str = "greedy"
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "accuracy": round(self.accuracy * 100, 2),
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "avg_time_per_problem": round(self.avg_time_per_problem, 3),
            "total_time": round(self.total_time, 2),
            "method": self.method,
        }


class ModelEvaluator:
    """Evaluator for comparing base vs trained models."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:30001",
        temperature: float = 0.0,  # Greedy for fair comparison
        max_tokens: int = 1024,
    ):
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verifier = None
        
    def setup(self):
        """Initialize verifier."""
        try:
            from verifier import MathVerifier
            self.verifier = MathVerifier()
            logger.info("Math verifier initialized")
        except Exception as e:
            logger.warning(f"Could not load verifier: {e}")
            self.verifier = None
    
    def check_server_health(self) -> bool:
        """Check if SGLang server is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False
    
    def load_lora_adapter(self, lora_path: str, lora_name: str = "trained") -> bool:
        """Load LoRA adapter on the server."""
        logger.info(f"Loading LoRA adapter from {lora_path}...")
        
        # Try multiple endpoints
        endpoints = [
            "/update_lora",
            "/add_lora", 
            "/v1/load_lora_adapter",
            "/v1/lora/load",
        ]
        
        for endpoint in endpoints:
            try:
                payload = {
                    "lora_name": lora_name,
                    "lora_path": lora_path,
                }
                resp = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=60
                )
                if resp.status_code == 200:
                    logger.info(f"✓ LoRA loaded successfully via {endpoint}")
                    return True
                else:
                    logger.debug(f"Endpoint {endpoint} returned {resp.status_code}")
            except Exception as e:
                logger.debug(f"Endpoint {endpoint} failed: {e}")
        
        logger.warning("Could not load LoRA via any endpoint - will evaluate base model only")
        return False
    
    def generate(
        self,
        prompt: str,
        lora_name: Optional[str] = None,
        n_samples: int = 1,
    ) -> Tuple[List[str], float]:
        """Generate response(s) from the model."""
        start_time = time.time()
        
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": n_samples,
        }
        
        # Add LoRA name if specified
        if lora_name:
            payload["lora_name"] = lora_name
        
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            
            if resp.status_code == 200:
                data = resp.json()
                responses = [c["message"]["content"] for c in data["choices"]]
                elapsed = time.time() - start_time
                return responses, elapsed
            else:
                logger.warning(f"API error {resp.status_code}: {resp.text[:100]}")
                return [""] * n_samples, time.time() - start_time
                
        except Exception as e:
            logger.warning(f"Request failed: {e}")
            return [""] * n_samples, time.time() - start_time
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract numerical answer from response."""
        # Look for #### format (GSM8K style)
        nums = re.findall(r'####\s*(-?\d+[\d,]*\.?\d*)', response)
        if nums:
            return nums[-1].replace(",", "")
        
        # Look for "answer is X" format
        nums = re.findall(r'answer is[:\s]*(-?\d+[\d,]*\.?\d*)', response.lower())
        if nums:
            return nums[-1].replace(",", "")
        
        # Look for boxed answer (MATH style)
        boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
        if boxed:
            return boxed[-1].strip()
        
        # Last number in response
        nums = re.findall(r'(-?\d+[\d,]*\.?\d*)', response)
        if nums:
            return nums[-1].replace(",", "")
        
        return None
    
    def check_correctness(self, response: str, ground_truth: str) -> bool:
        """Check if response is correct."""
        # Try verifier first
        if self.verifier:
            try:
                from verifier import VerificationStatus
                result = self.verifier.verify_reasoning_path(response, ground_truth)
                return result.status == VerificationStatus.CORRECT
            except:
                pass
        
        # Fallback to string matching
        extracted = self.extract_answer(response)
        if extracted is None:
            return False
        
        # Clean ground truth
        gt_clean = ground_truth.strip().replace(",", "").replace("$", "")
        gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
        if not gt_nums:
            return extracted.lower() == gt_clean.lower()
        
        gt_answer = gt_nums[-1]
        
        # Compare numerically
        try:
            return abs(float(extracted) - float(gt_answer)) < 0.001
        except:
            return extracted == gt_answer
    
    def evaluate_model(
        self,
        problems: List[Dict],
        model_name: str,
        lora_name: Optional[str] = None,
        method: str = "greedy",
        n_samples: int = 1,
        use_majority_vote: bool = False,
    ) -> EvalResult:
        """Evaluate a model on the given problems."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"Method: {method}, Samples per problem: {n_samples}")
        logger.info(f"{'='*60}")
        
        correct = 0
        total_time = 0
        
        for prob in tqdm(problems, desc=f"Evaluating {model_name}"):
            prompt = prob.get("prompt", prob.get("question", prob.get("problem", "")))
            answer = prob.get("answer", "")
            
            responses, elapsed = self.generate(prompt, lora_name, n_samples)
            total_time += elapsed
            
            if use_majority_vote and n_samples > 1:
                # Majority voting
                extracted = [self.extract_answer(r) for r in responses]
                extracted = [e for e in extracted if e is not None]
                if extracted:
                    counter = Counter(extracted)
                    majority_answer = counter.most_common(1)[0][0]
                    # Create fake response for checking
                    is_correct = self.check_correctness(f"#### {majority_answer}", answer)
                else:
                    is_correct = False
            else:
                # Pass@k - any correct counts
                is_correct = any(self.check_correctness(r, answer) for r in responses)
            
            if is_correct:
                correct += 1
        
        accuracy = correct / len(problems) if problems else 0
        avg_time = total_time / len(problems) if problems else 0
        
        result = EvalResult(
            model_name=model_name,
            accuracy=accuracy,
            num_correct=correct,
            num_total=len(problems),
            avg_time_per_problem=avg_time,
            total_time=total_time,
            method=method,
        )
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy: {accuracy*100:.1f}%")
        logger.info(f"  Correct: {correct}/{len(problems)}")
        logger.info(f"  Avg time: {avg_time:.2f}s per problem")
        
        return result


def load_gsm8k(num_samples: int = 100) -> List[Dict]:
    """Load GSM8K test set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        
        problems = []
        for item in list(ds)[:num_samples]:
            # Extract answer from "#### X" format
            answer_text = item["answer"]
            if "####" in answer_text:
                answer = answer_text.split("####")[-1].strip()
            else:
                answer = answer_text.strip()
            
            problems.append({
                "prompt": f"Solve this math problem step by step. Show your work and put your final answer after ####.\n\nProblem: {item['question']}",
                "question": item["question"],
                "answer": answer,
                "full_answer": answer_text,
            })
        
        return problems
    except Exception as e:
        logger.error(f"Failed to load GSM8K: {e}")
        raise


def print_comparison_table(results: List[EvalResult]):
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Model':<35} {'Method':<15} {'Accuracy':<12} {'Correct':<12} {'Time/prob'}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.model_name:<35} {r.method:<15} {r.accuracy*100:>6.1f}%     {r.num_correct:>4}/{r.num_total:<4}    {r.avg_time_per_problem:>6.2f}s")
    
    print("=" * 80)
    
    # Calculate improvement if we have base and trained
    base_results = [r for r in results if "base" in r.model_name.lower()]
    trained_results = [r for r in results if "trained" in r.model_name.lower() or "lora" in r.model_name.lower()]
    
    if base_results and trained_results:
        base_acc = base_results[0].accuracy
        trained_acc = trained_results[0].accuracy
        improvement = trained_acc - base_acc
        
        print(f"\n📈 IMPROVEMENT: {improvement*100:+.1f}% ({base_acc*100:.1f}% → {trained_acc*100:.1f}%)")
        
        if improvement > 0:
            print("✅ Training improved the model!")
        elif improvement < 0:
            print("⚠️ Training decreased accuracy (may need more iterations or different hyperparameters)")
        else:
            print("➡️ No change in accuracy")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs Trained Model")
    parser.add_argument("--port", type=int, default=30001, help="SGLang server port")
    parser.add_argument("--host", type=str, default="localhost", help="SGLang server host")
    parser.add_argument("--lora-path", type=str, help="Path to LoRA checkpoint")
    parser.add_argument("--lora-name", type=str, default="trained", help="Name for loaded LoRA")
    parser.add_argument("--num-problems", type=int, default=100, help="Number of problems")
    parser.add_argument("--output", type=str, default="./eval_comparison.json", help="Output file")
    
    # Evaluation modes
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--trained-only", action="store_true", help="Only evaluate trained model")
    
    # Test-time compute methods
    parser.add_argument("--all-methods", action="store_true", help="Run all evaluation methods")
    parser.add_argument("--pass-at-k", type=int, nargs="+", default=None, help="Evaluate Pass@k")
    parser.add_argument("--majority-vote", type=int, default=None, help="Majority voting with N samples")
    
    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (0=greedy)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens")
    
    args = parser.parse_args()
    
    # Setup evaluator
    base_url = f"http://{args.host}:{args.port}"
    evaluator = ModelEvaluator(
        base_url=base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    evaluator.setup()
    
    # Check server health
    logger.info(f"Checking server at {base_url}...")
    if not evaluator.check_server_health():
        logger.error(f"❌ Server not responding at {base_url}")
        logger.error("Please start SGLang server first:")
        logger.error(f"  CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \\")
        logger.error(f"      --model-path Qwen/Qwen2.5-1.5B-Instruct \\")
        logger.error(f"      --host 0.0.0.0 --port {args.port} \\")
        logger.error(f"      --enable-lora --max-loras-per-batch 8 --max-lora-rank 64")
        sys.exit(1)
    
    logger.info("✓ Server is healthy")
    
    # Load dataset
    logger.info(f"Loading GSM8K test set ({args.num_problems} problems)...")
    problems = load_gsm8k(args.num_problems)
    logger.info(f"Loaded {len(problems)} problems")
    
    all_results = []
    
    # Evaluate Base Model
    if not args.trained_only:
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Evaluating BASE MODEL (no LoRA)")
        logger.info("="*60)
        
        base_result = evaluator.evaluate_model(
            problems=problems,
            model_name="Base (Qwen2.5-1.5B-Instruct)",
            lora_name=None,
            method="greedy",
        )
        all_results.append(base_result)
        
        # Additional methods for base
        if args.all_methods or args.pass_at_k:
            k_values = args.pass_at_k or [4, 8]
            for k in k_values:
                evaluator.temperature = 0.8  # Need sampling for Pass@k
                result = evaluator.evaluate_model(
                    problems=problems,
                    model_name=f"Base Model",
                    lora_name=None,
                    method=f"Pass@{k}",
                    n_samples=k,
                )
                all_results.append(result)
            evaluator.temperature = args.temperature
        
        if args.all_methods or args.majority_vote:
            n_samples = args.majority_vote or 8
            evaluator.temperature = 0.8
            result = evaluator.evaluate_model(
                problems=problems,
                model_name=f"Base Model",
                lora_name=None,
                method=f"Majority@{n_samples}",
                n_samples=n_samples,
                use_majority_vote=True,
            )
            all_results.append(result)
            evaluator.temperature = args.temperature
    
    # Evaluate Trained Model (with LoRA)
    if not args.base_only and args.lora_path:
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Evaluating TRAINED MODEL (with LoRA)")
        logger.info("="*60)
        
        # Load LoRA
        lora_loaded = evaluator.load_lora_adapter(args.lora_path, args.lora_name)
        
        if lora_loaded:
            trained_result = evaluator.evaluate_model(
                problems=problems,
                model_name=f"Trained (LoRA: {Path(args.lora_path).name})",
                lora_name=args.lora_name,
                method="greedy",
            )
            all_results.append(trained_result)
            
            # Additional methods for trained
            if args.all_methods or args.pass_at_k:
                k_values = args.pass_at_k or [4, 8]
                for k in k_values:
                    evaluator.temperature = 0.8
                    result = evaluator.evaluate_model(
                        problems=problems,
                        model_name=f"Trained (LoRA)",
                        lora_name=args.lora_name,
                        method=f"Pass@{k}",
                        n_samples=k,
                    )
                    all_results.append(result)
                evaluator.temperature = args.temperature
            
            if args.all_methods or args.majority_vote:
                n_samples = args.majority_vote or 8
                evaluator.temperature = 0.8
                result = evaluator.evaluate_model(
                    problems=problems,
                    model_name=f"Trained (LoRA)",
                    lora_name=args.lora_name,
                    method=f"Majority@{n_samples}",
                    n_samples=n_samples,
                    use_majority_vote=True,
                )
                all_results.append(result)
                evaluator.temperature = args.temperature
        else:
            logger.warning("⚠️ Could not load LoRA - skipping trained model evaluation")
            logger.warning("Your SGLang version may not support dynamic LoRA loading.")
            logger.warning("Alternative: Merge LoRA into base model and serve the merged model.")
    
    # Print comparison
    print_comparison_table(all_results)
    
    # Save results
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "server": base_url,
            "lora_path": args.lora_path,
            "num_problems": args.num_problems,
            "temperature": args.temperature,
        },
        "results": [r.to_dict() for r in all_results],
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {args.output}")
    
    return all_results


if __name__ == "__main__":
    main()
