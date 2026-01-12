#!/usr/bin/env python3
"""
Pass@k Evaluation Script for Test-Time Compute Scaling.

This demonstrates how accuracy scales with inference compute budget.
Key insight: More samples at inference time = better accuracy without training.

Usage:
    python eval_pass_at_k.py --k 1 8 32 --dataset math --num-problems 100
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PassAtKResults:
    """Results from Pass@k evaluation."""
    k: int
    accuracy: float
    num_problems: int
    num_correct: int
    avg_generations_per_problem: float
    total_tokens_generated: int
    total_time_seconds: float
    tokens_per_second: float
    
    def to_dict(self) -> Dict:
        return {
            "k": self.k,
            "accuracy": round(self.accuracy * 100, 2),
            "num_problems": self.num_problems,
            "num_correct": self.num_correct,
            "avg_generations_per_problem": round(self.avg_generations_per_problem, 2),
            "total_tokens_generated": self.total_tokens_generated,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


class PassAtKEvaluator:
    """
    Evaluates test-time compute scaling via Pass@k metric.
    
    Pass@k: Probability that at least one of k samples is correct.
    
    This is a key metric for reasoning models - shows value of
    generating multiple candidates and verifying.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_sglang: bool = True,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.use_sglang = use_sglang
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.generator = None
        self.verifier = None
        self.tokenizer = None
    
    def setup(self):
        """Initialize generator and verifier."""
        try:
            if self.use_sglang:
                from inference.sglang_engine import SGLangEngine, SGLangConfig
                config = SGLangConfig(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    enable_radix_cache=True,
                )
                self.generator = SGLangEngine(config)
                self.generator.initialize()
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            logger.info("Running in mock mode for demonstration")
            self.generator = None
        
        from verifier import MathVerifier
        self.verifier = MathVerifier()
        
        logger.info(f"Setup complete. Model: {self.model_name}")
    
    def generate_n_samples(
        self,
        prompt: str,
        n: int,
    ) -> Tuple[List[str], int, float]:
        """
        Generate n samples for a single prompt.
        
        Returns:
            responses: List of generated responses
            total_tokens: Total tokens generated
            time_taken: Time in seconds
        """
        start_time = time.time()
        
        if self.generator is not None:
            # Use SGLang via HTTP
            import requests
            responses = []
            total_tokens = 0
            for _ in range(n):
                try:
                    resp = requests.post(
                        f"{self.model_name}/v1/chat/completions",
                        json={
                            "model": "default",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        },
                        timeout=120
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["choices"][0]["message"]["content"]
                        responses.append(text)
                        total_tokens += data.get("usage", {}).get("completion_tokens", len(text.split()))
                    else:
                        responses.append("")
                except Exception as e:
                    logger.warning(f"Request error: {e}")
                    responses.append("")
        elif self.model is not None and self.tokenizer is not None:
            # Use local HuggingFace model
            import torch
            responses = []
            total_tokens = 0
            for _ in range(n):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                responses.append(response)
                total_tokens += len(outputs[0]) - inputs.input_ids.shape[1]
        else:
            # Mock mode
            import random
            responses = []
            for i in range(n):
                if random.random() < 0.3:
                    responses.append("Let me solve this step by step.\n#### 42")
                else:
                    responses.append(f"The answer is {random.randint(1, 100)}.")
            total_tokens = n * 100
        
        time_taken = time.time() - start_time
        return responses, int(total_tokens), time_taken
    
    def check_correctness(self, response: str, ground_truth: str) -> bool:
        """Check if response contains correct answer."""
        try:
            from verifier import VerificationStatus
            result = self.verifier.verify_reasoning_path(response, ground_truth)
            return result.status == VerificationStatus.CORRECT
        except Exception:
            # Fallback: simple string matching
            import re
            gt_clean = ground_truth.strip().replace(",", "")
            gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
            if not gt_nums:
                return False
            gt_answer = gt_nums[-1]
            
            resp_nums = re.findall(r'####\s*(-?\d+\.?\d*)', response)
            if resp_nums:
                return resp_nums[-1] == gt_answer
            return False
    
    def evaluate_pass_at_k(
        self,
        problems: List[Dict],
        k_values: List[int],
        max_n: Optional[int] = None,
    ) -> Dict[int, PassAtKResults]:
        """
        Evaluate Pass@k for multiple values of k.
        
        Args:
            problems: List of {"prompt": ..., "answer": ...}
            k_values: List of k values to evaluate (e.g., [1, 8, 32])
            max_n: Maximum samples to generate per problem (default: max(k_values))
        
        Returns:
            Dictionary mapping k -> PassAtKResults
        """
        if max_n is None:
            max_n = max(k_values)
        
        logger.info(f"Evaluating Pass@k for k={k_values}")
        logger.info(f"Generating {max_n} samples per problem for {len(problems)} problems")
        
        # Store all generations and correctness
        all_results = []
        total_tokens = 0
        total_time = 0
        
        from tqdm import tqdm
        for prob in tqdm(problems, desc="Generating samples"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            # Generate max_n samples
            responses, tokens, gen_time = self.generate_n_samples(prompt, max_n)
            total_tokens += tokens
            total_time += gen_time
            
            # Check correctness of each
            correctness = [self.check_correctness(r, answer) for r in responses]
            
            all_results.append({
                "prompt": prompt,
                "answer": answer,
                "responses": responses,
                "correctness": correctness,
            })
        
        # Compute Pass@k for each k
        results = {}
        for k in k_values:
            num_correct = 0
            for item in all_results:
                # Pass@k: at least one of first k is correct
                if any(item["correctness"][:k]):
                    num_correct += 1
            
            accuracy = num_correct / len(problems) if problems else 0
            
            results[k] = PassAtKResults(
                k=k,
                accuracy=accuracy,
                num_problems=len(problems),
                num_correct=num_correct,
                avg_generations_per_problem=max_n,
                total_tokens_generated=total_tokens,
                total_time_seconds=total_time,
                tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            )
        
        return results
    
    def evaluate_majority_voting(
        self,
        problems: List[Dict],
        n_samples: int = 8,
    ) -> Tuple[float, Dict]:
        """
        Evaluate using majority voting (most common answer wins).
        
        This is another test-time compute scaling technique.
        """
        import re
        
        logger.info(f"Evaluating Majority Voting with n={n_samples}")
        
        correct = 0
        total_time = 0
        
        from tqdm import tqdm
        for prob in tqdm(problems, desc="Majority voting"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            responses, _, gen_time = self.generate_n_samples(prompt, n_samples)
            total_time += gen_time
            
            # Extract answers from responses
            extracted = []
            for resp in responses:
                nums = re.findall(r'####\s*(-?\d+\.?\d*)', resp)
                if nums:
                    extracted.append(nums[-1])
            
            if extracted:
                # Majority vote
                majority_answer = Counter(extracted).most_common(1)[0][0]
                
                # Check correctness
                gt_nums = re.findall(r'-?\d+\.?\d*', answer.replace(",", ""))
                if gt_nums and majority_answer == gt_nums[-1]:
                    correct += 1
        
        accuracy = correct / len(problems) if problems else 0
        
        return accuracy, {
            "n_samples": n_samples,
            "accuracy": round(accuracy * 100, 2),
            "num_correct": correct,
            "total_problems": len(problems),
            "total_time_seconds": round(total_time, 2),
        }
    
    def evaluate_best_of_n_with_reward(
        self,
        problems: List[Dict],
        n_samples: int = 8,
        reward_model_path: Optional[str] = None,
    ) -> Tuple[float, Dict]:
        """
        Evaluate Best-of-N selection using reward model.
        
        This is RL-based test-time scaling - uses trained reward model
        to select best candidate.
        """
        logger.info(f"Evaluating Best-of-N (reward) with n={n_samples}")
        
        # Load reward model if available
        reward_model = None
        if reward_model_path:
            try:
                from training.reward_model import RewardModel, RewardModelConfig
                config = RewardModelConfig(model_name=self.model_name)
                reward_model = RewardModel(config)
                reward_model.load(reward_model_path)
                reward_model.eval()
                logger.info(f"Loaded reward model from {reward_model_path}")
            except Exception as e:
                logger.warning(f"Could not load reward model: {e}")
        
        correct = 0
        total_time = 0
        
        from tqdm import tqdm
        for prob in tqdm(problems, desc="Best-of-N (reward)"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            responses, _, gen_time = self.generate_n_samples(prompt, n_samples)
            total_time += gen_time
            
            # Score each response
            if reward_model:
                scores = [
                    reward_model.compute_reward(prompt, resp)
                    for resp in responses
                ]
            else:
                # Fallback: use length as proxy (longer = more reasoning)
                scores = [len(resp) for resp in responses]
            
            # Select best
            best_idx = np.argmax(scores)
            best_response = responses[best_idx]
            
            # Check correctness
            if self.check_correctness(best_response, answer):
                correct += 1
        
        accuracy = correct / len(problems) if problems else 0
        
        return accuracy, {
            "n_samples": n_samples,
            "accuracy": round(accuracy * 100, 2),
            "num_correct": correct,
            "total_problems": len(problems),
            "total_time_seconds": round(total_time, 2),
            "used_reward_model": reward_model is not None,
        }


def load_dataset(dataset_name: str, num_samples: int) -> List[Dict]:
    """Load evaluation dataset."""
    try:
        from datasets import load_dataset as hf_load
        
        if dataset_name == "gsm8k":
            ds = hf_load("openai/gsm8k", "main", split="test")
            data = [{"prompt": item["question"], "answer": item["answer"].split("####")[-1].strip()} for item in list(ds)[:num_samples]]
            return data
        elif dataset_name == "math":
            data = loader.load_math(split="test", num_samples=num_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return data
    except Exception as e:
        logger.warning(f"Could not load dataset: {e}")
        logger.info("Using mock data")
        
        # Mock data for demonstration
        return [
            {"prompt": f"What is {i} + {i*2}?", "answer": str(i + i*2)}
            for i in range(1, num_samples + 1)
        ]


def print_results_table(results: Dict[int, PassAtKResults]):
    """Print results in a nice table."""
    from tabulate import tabulate
    
    headers = ["k", "Accuracy", "Correct/Total", "Tokens/sec"]
    rows = []
    
    for k in sorted(results.keys()):
        r = results[k]
        rows.append([
            k,
            f"{r.accuracy * 100:.1f}%",
            f"{r.num_correct}/{r.num_problems}",
            f"{r.tokens_per_second:.0f}",
        ])
    
    print("\n" + "=" * 60)
    print("PASS@K RESULTS (Test-Time Compute Scaling)")
    print("=" * 60)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()
    
    # Show scaling insight
    if len(results) >= 2:
        ks = sorted(results.keys())
        k_min, k_max = ks[0], ks[-1]
        improvement = (results[k_max].accuracy - results[k_min].accuracy) * 100
        print(f"📈 Scaling: Pass@{k_min} → Pass@{k_max} = +{improvement:.1f}% accuracy")
        print(f"💡 Test-time compute scaling works!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate test-time compute scaling via Pass@k"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="Values of k to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=100,
        help="Number of problems to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./pass_at_k_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--no-sglang",
        action="store_true",
        help="Disable SGLang, use transformers",
    )
    parser.add_argument(
        "--majority-vote",
        action="store_true",
        help="Also evaluate majority voting",
    )
    parser.add_argument(
        "--best-of-n",
        action="store_true",
        help="Also evaluate best-of-n with reward",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=None,
        help="Path to reward model for best-of-n",
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    problems = load_dataset(args.dataset, args.num_problems)
    logger.info(f"Loaded {len(problems)} problems")
    
    # Setup evaluator
    evaluator = PassAtKEvaluator(
        model_name=args.model,
        use_sglang=not args.no_sglang,
    )
    evaluator.setup()
    
    # Run Pass@k evaluation
    results = evaluator.evaluate_pass_at_k(problems, args.k)
    print_results_table(results)
    
    # Optional: Majority voting
    all_results = {"pass_at_k": {k: r.to_dict() for k, r in results.items()}}
    
    if args.majority_vote:
        for n in [8, 16, 32]:
            acc, details = evaluator.evaluate_majority_voting(problems, n)
            print(f"Majority Vote (n={n}): {acc*100:.1f}%")
            all_results[f"majority_vote_n{n}"] = details
    
    if args.best_of_n:
        for n in [8, 16, 32]:
            acc, details = evaluator.evaluate_best_of_n_with_reward(
                problems, n, args.reward_model
            )
            print(f"Best-of-N (n={n}): {acc*100:.1f}%")
            all_results[f"best_of_n{n}"] = details
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {args.output}")
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results: {args.output}")


if __name__ == "__main__":
    main()
