#!/usr/bin/env python3
"""
Pass@k Evaluation Script for Test-Time Compute Scaling.

This demonstrates how accuracy scales with inference compute budget.
Key insight: More samples at inference time = better accuracy without training.

Usage:
    python eval_pass_at_k.py --k 1 8 32 --dataset gsm8k --num-problems 100
    python eval_pass_at_k.py --model http://localhost:30000 --all-methods
"""

import argparse
import json
import logging
import sys
import time
import re
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


@dataclass 
class TTCResults:
    """Results from test-time compute methods."""
    method: str
    accuracy: float
    num_problems: int
    num_correct: int
    avg_samples: float
    total_time_seconds: float
    
    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "accuracy": round(self.accuracy * 100, 2),
            "num_problems": self.num_problems,
            "num_correct": self.num_correct,
            "avg_samples": round(self.avg_samples, 1),
            "total_time_seconds": round(self.total_time_seconds, 2),
        }


class PassAtKEvaluator:
    """
    Evaluates test-time compute scaling via Pass@k metric.
    
    Pass@k: Probability that at least one of k samples is correct.
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
        self.model = None
    
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
        """Generate n samples for a single prompt."""
        start_time = time.time()
        
        if self.model_name.startswith("http"):
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
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract numerical answer from response."""
        nums = re.findall(r'####\s*(-?\d+\.?\d*)', response)
        if nums:
            return nums[-1]
        nums = re.findall(r'answer is[:\s]*(-?\d+\.?\d*)', response.lower())
        if nums:
            return nums[-1]
        return None
    
    def check_correctness(self, response: str, ground_truth: str) -> bool:
        """Check if response contains correct answer."""
        try:
            from verifier import VerificationStatus
            result = self.verifier.verify_reasoning_path(response, ground_truth)
            return result.status == VerificationStatus.CORRECT
        except Exception:
            gt_clean = ground_truth.strip().replace(",", "")
            gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
            if not gt_nums:
                return False
            gt_answer = gt_nums[-1]
            
            extracted = self.extract_answer(response)
            if extracted:
                return extracted == gt_answer
            return False
    
    def evaluate_pass_at_k(
        self,
        problems: List[Dict],
        k_values: List[int],
        max_n: Optional[int] = None,
    ) -> Dict[int, PassAtKResults]:
        """Evaluate Pass@k for multiple values of k."""
        if max_n is None:
            max_n = max(k_values)
        
        logger.info(f"Evaluating Pass@k for k={k_values}")
        logger.info(f"Generating {max_n} samples per problem for {len(problems)} problems")
        
        all_results = []
        total_tokens = 0
        total_time = 0
        
        from tqdm import tqdm
        for prob in tqdm(problems, desc="Pass@k"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            responses, tokens, gen_time = self.generate_n_samples(prompt, max_n)
            total_tokens += tokens
            total_time += gen_time
            
            correctness = [self.check_correctness(r, answer) for r in responses]
            
            all_results.append({
                "prompt": prompt,
                "answer": answer,
                "responses": responses,
                "correctness": correctness,
            })
        
        results = {}
        for k in k_values:
            num_correct = 0
            for item in all_results:
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
        """Evaluate using majority voting."""
        logger.info(f"Evaluating Majority Voting with n={n_samples}")
        
        correct = 0
        total_time = 0
        
        from tqdm import tqdm
        for prob in tqdm(problems, desc="Majority Vote"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            responses, _, gen_time = self.generate_n_samples(prompt, n_samples)
            total_time += gen_time
            
            extracted = [self.extract_answer(r) for r in responses]
            extracted = [e for e in extracted if e is not None]
            
            if extracted:
                majority_answer = Counter(extracted).most_common(1)[0][0]
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
    
    def evaluate_self_consistency(
        self,
        problems: List[Dict],
        n_samples: int = 40,
    ) -> TTCResults:
        """
        Self-Consistency: Sample many paths, majority vote on final answer.
        More robust than single-sample majority voting.
        """
        logger.info(f"Evaluating Self-Consistency with n={n_samples}")
        
        correct = 0
        total_time = 0
        
        from tqdm import tqdm
        for prob in tqdm(problems, desc="Self-Consistency"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            responses, _, gen_time = self.generate_n_samples(prompt, n_samples)
            total_time += gen_time
            
            # Extract all answers
            extracted = [self.extract_answer(r) for r in responses]
            extracted = [e for e in extracted if e is not None]
            
            if extracted:
                # Majority vote with confidence
                counts = Counter(extracted)
                majority_answer, count = counts.most_common(1)[0]
                confidence = count / len(extracted)
                
                gt_nums = re.findall(r'-?\d+\.?\d*', answer.replace(",", ""))
                if gt_nums and majority_answer == gt_nums[-1]:
                    correct += 1
        
        accuracy = correct / len(problems) if problems else 0
        
        return TTCResults(
            method="Self-Consistency",
            accuracy=accuracy,
            num_problems=len(problems),
            num_correct=correct,
            avg_samples=n_samples,
            total_time_seconds=total_time,
        )
    
    def evaluate_beam_search(
        self,
        problems: List[Dict],
        beam_width: int = 4,
        max_steps: int = 5,
    ) -> TTCResults:
        """
        Beam Search: Keep top-k partial solutions at each step.
        Prunes bad reasoning paths early.
        """
        logger.info(f"Evaluating Beam Search with width={beam_width}, steps={max_steps}")
        
        correct = 0
        total_time = 0
        total_samples = 0
        
        from tqdm import tqdm
        for prob in tqdm(problems, desc="Beam Search"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            start_time = time.time()
            
            # Initialize beam with empty continuations
            beam = [(0.0, "")]  # (score, partial_response)
            samples_used = 0
            
            for step in range(max_steps):
                candidates = []
                
                for score, partial in beam:
                    # Generate continuations from current partial
                    full_prompt = f"{prompt}\n{partial}" if partial else prompt
                    responses, _, _ = self.generate_n_samples(full_prompt, beam_width)
                    samples_used += beam_width
                    
                    for resp in responses:
                        # Score based on answer presence and length
                        new_partial = partial + resp if partial else resp
                        new_score = score
                        
                        # Boost if answer found
                        if "####" in resp or "answer is" in resp.lower():
                            new_score += 10.0
                        
                        # Small penalty for length (prefer concise)
                        new_score -= len(resp) * 0.001
                        
                        candidates.append((new_score, new_partial))
                
                # Keep top beam_width
                candidates.sort(key=lambda x: -x[0])
                beam = candidates[:beam_width]
                
                # Early stop if best has answer
                if beam and ("####" in beam[0][1] or "answer is" in beam[0][1].lower()):
                    break
            
            total_time += time.time() - start_time
            total_samples += samples_used
            
            # Check best beam
            if beam:
                best_response = beam[0][1]
                if self.check_correctness(best_response, answer):
                    correct += 1
        
        accuracy = correct / len(problems) if problems else 0
        avg_samples = total_samples / len(problems) if problems else 0
        
        return TTCResults(
            method="Beam Search",
            accuracy=accuracy,
            num_problems=len(problems),
            num_correct=correct,
            avg_samples=avg_samples,
            total_time_seconds=total_time,
        )
    
    def evaluate_mcts(
        self,
        problems: List[Dict],
        num_simulations: int = 50,
        exploration: float = 1.41,
    ) -> TTCResults:
        """
        MCTS: Monte Carlo Tree Search for reasoning.
        Balances exploration vs exploitation via UCB.
        """
        logger.info(f"Evaluating MCTS with simulations={num_simulations}")
        
        correct = 0
        total_time = 0
        total_samples = 0
        
        from tqdm import tqdm
        for prob in tqdm(problems, desc="MCTS"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            start_time = time.time()
            
            # Simple MCTS node structure
            class Node:
                def __init__(self, state, parent=None):
                    self.state = state
                    self.parent = parent
                    self.children = []
                    self.visits = 0
                    self.value = 0.0
                
                def ucb(self):
                    if self.visits == 0:
                        return float('inf')
                    exploit = self.value / self.visits
                    explore = exploration * np.sqrt(np.log(self.parent.visits) / self.visits)
                    return exploit + explore
            
            root = Node(prompt)
            samples_used = 0
            best_response = ""
            best_score = -float('inf')
            
            for sim in range(num_simulations):
                # Selection: traverse to leaf using UCB
                node = root
                while node.children:
                    node = max(node.children, key=lambda n: n.ucb())
                
                # Expansion: generate children
                if node.visits > 0 or node == root:
                    responses, _, _ = self.generate_n_samples(node.state, 4)
                    samples_used += 4
                    
                    for resp in responses:
                        child_state = f"{node.state}\n{resp}" if node != root else resp
                        child = Node(child_state, parent=node)
                        node.children.append(child)
                    
                    if node.children:
                        node = node.children[0]
                
                # Simulation: evaluate leaf
                response = node.state if node != root else ""
                if response:
                    # Score: 1 if correct, 0.5 if has answer format, 0 otherwise
                    if self.check_correctness(response, answer):
                        reward = 1.0
                        if best_score < 1.0:
                            best_response = response
                            best_score = 1.0
                    elif "####" in response or "answer is" in response.lower():
                        reward = 0.3
                        if best_score < 0.3:
                            best_response = response
                            best_score = 0.3
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
                
                # Backpropagation
                while node:
                    node.visits += 1
                    node.value += reward
                    node = node.parent
            
            total_time += time.time() - start_time
            total_samples += samples_used
            
            # Final check
            if best_score >= 1.0:
                correct += 1
        
        accuracy = correct / len(problems) if problems else 0
        avg_samples = total_samples / len(problems) if problems else 0
        
        return TTCResults(
            method="MCTS",
            accuracy=accuracy,
            num_problems=len(problems),
            num_correct=correct,
            avg_samples=avg_samples,
            total_time_seconds=total_time,
        )
    
    def evaluate_verifier_guided(
        self,
        problems: List[Dict],
        max_samples: int = 64,
    ) -> TTCResults:
        """
        Verifier-Guided: Generate until verifier confirms correct.
        Early stopping = efficient. This is the key insight.
        """
        logger.info(f"Evaluating Verifier-Guided with max_samples={max_samples}")
        
        correct = 0
        total_time = 0
        total_samples = 0
        
        from tqdm import tqdm
        for prob in tqdm(problems, desc="Verifier-Guided"):
            prompt = prob.get("prompt", prob.get("problem", ""))
            answer = prob.get("answer", "")
            
            start_time = time.time()
            samples_used = 0
            found_correct = False
            
            # Generate in batches until correct or max reached
            batch_size = 8
            all_responses = []
            
            while samples_used < max_samples and not found_correct:
                responses, _, _ = self.generate_n_samples(prompt, batch_size)
                samples_used += batch_size
                all_responses.extend(responses)
                
                # Check each response
                for resp in responses:
                    if self.check_correctness(resp, answer):
                        found_correct = True
                        break
            
            total_time += time.time() - start_time
            total_samples += samples_used
            
            if found_correct:
                correct += 1
        
        accuracy = correct / len(problems) if problems else 0
        avg_samples = total_samples / len(problems) if problems else 0
        
        return TTCResults(
            method="Verifier-Guided",
            accuracy=accuracy,
            num_problems=len(problems),
            num_correct=correct,
            avg_samples=avg_samples,
            total_time_seconds=total_time,
        )


def load_dataset(dataset_name: str, num_samples: int) -> List[Dict]:
    """Load evaluation dataset."""
    try:
        from datasets import load_dataset as hf_load
        
        if dataset_name == "gsm8k":
            ds = hf_load("openai/gsm8k", "main", split="test")
            data = [{"prompt": item["question"], "answer": item["answer"].split("####")[-1].strip()} for item in list(ds)[:num_samples]]
            return data
        elif dataset_name == "math":
            ds = hf_load("hendrycks/competition_math", split="test")
            data = [{"prompt": item["problem"], "answer": item["solution"]} for item in list(ds)[:num_samples]]
            return data
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        logger.warning(f"Could not load dataset: {e}")
        logger.info("Using mock data")
        return [
            {"prompt": f"What is {i} + {i*2}?", "answer": str(i + i*2)}
            for i in range(1, num_samples + 1)
        ]


def print_results_table(results: Dict[int, PassAtKResults]):
    """Print Pass@k results table."""
    print("\n" + "=" * 60)
    print("PASS@K RESULTS")
    print("=" * 60)
    print(f"{'k':<8} {'Accuracy':<12} {'Correct':<12} {'Tokens/s':<12}")
    print("-" * 60)
    
    for k in sorted(results.keys()):
        r = results[k]
        print(f"{k:<8} {r.accuracy * 100:>6.1f}%     {r.num_correct}/{r.num_problems:<6}   {r.tokens_per_second:>8.0f}")
    print()


def print_ttc_results_table(results: List[TTCResults]):
    """Print test-time compute results table."""
    print("\n" + "=" * 70)
    print("TEST-TIME COMPUTE SCALING RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':<12} {'Correct':<12} {'Avg Samples':<12} {'Time(s)':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.method:<20} {r.accuracy * 100:>6.1f}%     {r.num_correct}/{r.num_problems:<6}   {r.avg_samples:>8.1f}     {r.total_time_seconds:>8.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate test-time compute scaling"
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 8], help="Values of k for Pass@k")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"])
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--model", type=str, default="http://localhost:30000")
    parser.add_argument("--output", type=str, default="./ttc_results.json")
    parser.add_argument("--no-sglang", action="store_true")
    
    # Test-time compute methods
    parser.add_argument("--majority-vote", action="store_true", help="Evaluate majority voting")
    parser.add_argument("--self-consistency", action="store_true", help="Evaluate self-consistency (n=40)")
    parser.add_argument("--beam-search", action="store_true", help="Evaluate beam search")
    parser.add_argument("--mcts", action="store_true", help="Evaluate MCTS")
    parser.add_argument("--verifier-guided", action="store_true", help="Evaluate verifier-guided search")
    parser.add_argument("--all-methods", action="store_true", help="Run all test-time compute methods")
    
    # Method parameters
    parser.add_argument("--sc-samples", type=int, default=40, help="Samples for self-consistency")
    parser.add_argument("--mcts-sims", type=int, default=50, help="Simulations for MCTS")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width")
    parser.add_argument("--vg-max", type=int, default=64, help="Max samples for verifier-guided")
    
    args = parser.parse_args()
    
    # Enable all if --all-methods
    if args.all_methods:
        args.self_consistency = True
        args.beam_search = True
        args.mcts = True
        args.verifier_guided = True
    
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
    
    all_results = {}
    ttc_results = []
    
    # Pass@k (baseline)
    results = evaluator.evaluate_pass_at_k(problems, args.k)
    print_results_table(results)
    all_results["pass_at_k"] = {k: r.to_dict() for k, r in results.items()}
    
    # Convert Pass@k to TTCResults for combined table
    for k, r in results.items():
        ttc_results.append(TTCResults(
            method=f"Pass@{k}",
            accuracy=r.accuracy,
            num_problems=r.num_problems,
            num_correct=r.num_correct,
            avg_samples=k,
            total_time_seconds=r.total_time_seconds,
        ))
    
    # Self-Consistency
    if args.self_consistency:
        sc_result = evaluator.evaluate_self_consistency(problems, args.sc_samples)
        ttc_results.append(sc_result)
        all_results["self_consistency"] = sc_result.to_dict()
    
    # Beam Search
    if args.beam_search:
        beam_result = evaluator.evaluate_beam_search(problems, args.beam_width)
        ttc_results.append(beam_result)
        all_results["beam_search"] = beam_result.to_dict()
    
    # MCTS
    if args.mcts:
        mcts_result = evaluator.evaluate_mcts(problems, args.mcts_sims)
        ttc_results.append(mcts_result)
        all_results["mcts"] = mcts_result.to_dict()
    
    # Verifier-Guided
    if args.verifier_guided:
        vg_result = evaluator.evaluate_verifier_guided(problems, args.vg_max)
        ttc_results.append(vg_result)
        all_results["verifier_guided"] = vg_result.to_dict()
    
    # Print combined table
    if len(ttc_results) > 2:
        print_ttc_results_table(ttc_results)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {args.output}")
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results: {args.output}")


if __name__ == "__main__":
    main()
