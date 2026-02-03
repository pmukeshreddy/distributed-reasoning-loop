#!/usr/bin/env python3
"""
Compare DPO, GRPO, and No Training (Baseline) on standard metrics.
Prints rewards, accuracy, and other metrics for comparison.
"""

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from tabulate import tabulate
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MethodMetrics:
    """Metrics for a single training method."""
    method_name: str
    accuracy: float = 0.0
    avg_reward: float = 0.0
    avg_chosen_reward: float = 0.0
    avg_rejected_reward: float = 0.0
    reward_margin: float = 0.0  # chosen - rejected
    correct: int = 0
    total: int = 0
    avg_response_length: float = 0.0
    avg_inference_time: float = 0.0
    training_time: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "accuracy": self.accuracy,
            "avg_reward": self.avg_reward,
            "avg_chosen_reward": self.avg_chosen_reward,
            "avg_rejected_reward": self.avg_rejected_reward,
            "reward_margin": self.reward_margin,
            "correct": self.correct,
            "total": self.total,
            "avg_response_length": self.avg_response_length,
            "avg_inference_time": self.avg_inference_time,
            "training_time": self.training_time,
        }


class TrainingMethodComparison:
    """
    Compare DPO, GRPO, and baseline (no training) on standard metrics.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",  # Small model for visible improvements
        output_dir: str = "./comparison_results",
        eval_subset_size: int = 100,
        use_lora: bool = True,
        num_epochs: int = 1,
        batch_size: int = 2,
        use_sglang: bool = True,  # Use SGLang for faster inference
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_subset_size = eval_subset_size
        self.use_lora = use_lora
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.use_sglang = use_sglang
        
        self.results: Dict[str, MethodMetrics] = {}
        self.reward_model = None
        self.tokenizer = None
        self.sglang_engine = None
    
    def load_data(self, data_path: str) -> List[Dict]:
        """Load training data from JSONL file."""
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} samples from {data_path}")
        return data
    
    def train_reward_model(self, train_data: List[Dict]) -> None:
        """
        Train the reward model on preference pairs BEFORE using it.
        This gives meaningful learned reward scores instead of random values.
        
        Args:
            train_data: List of dicts with 'prompt', 'chosen', 'rejected' keys
        """
        from training.reward_model import RewardModel, RewardModelConfig
        
        logger.info("\n" + "="*60)
        logger.info("Training Reward Model on Preference Pairs...")
        logger.info("="*60)
        
        config = RewardModelConfig(
            model_name=self.model_name,
            num_epochs=3,  # Train for a few epochs
            batch_size=2,  # Small batch for memory
            learning_rate=1e-5,
            max_length=1024,
            output_dir=str(self.output_dir / "reward_model"),
        )
        
        self.reward_model = RewardModel(config)
        self.reward_model.setup()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_model.to(device)
        
        # Train on preference pairs (chosen > rejected)
        start_time = time.time()
        self.reward_model.train_model(train_data)
        train_time = time.time() - start_time
        
        self.reward_model.eval()
        logger.info(f"Reward model trained in {train_time:.1f}s")
        logger.info("="*60 + "\n")
    
    def compute_learned_reward(self, prompt: str, response: str) -> float:
        """
        Compute reward using the TRAINED reward model.
        """
        if self.reward_model is None:
            logger.warning("Reward model not trained! Returning 0.0")
            return 0.0
        
        try:
            reward = self.reward_model.compute_reward(prompt, response)
            return reward
        except Exception as e:
            logger.warning(f"Error computing learned reward: {e}")
            return 0.0
    
    def compute_simple_reward(self, response: str, ground_truth: str) -> float:
        """
        Compute a simple correctness-based reward (as backup).
        Returns: 1.0 if correct, 0.0 if incorrect, with partial credit for close answers.
        """
        import re
        
        # Extract ground truth number
        gt_clean = ground_truth.strip().replace(",", "").replace("$", "")
        gt_numbers = re.findall(r'-?\d+\.?\d*', gt_clean)
        if not gt_numbers:
            return 0.0
        gt_answer = float(gt_numbers[-1])
        
        # Extract predicted answer using same logic as _check_answer_correct
        response_clean = response.replace(",", "").replace("$", "")
        
        answer_patterns = [
            r'####\s*(-?\d+\.?\d*)',
            r'\\boxed\{(-?\d+\.?\d*)\}',
            r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(-?\d+\.?\d*)',
            r'[Aa]nswer[:\s]+(-?\d+\.?\d*)\s*$',
            r'=\s*(-?\d+\.?\d*)\s*(?:dollars?|cents?|units?)?\s*$',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if matches:
                try:
                    pred_answer = float(matches[-1])
                    # Exact match = 1.0, close = partial credit
                    if abs(pred_answer - gt_answer) < 0.01:
                        return 1.0
                    elif abs(pred_answer - gt_answer) < 1.0:
                        return 0.5  # Close but not exact
                except ValueError:
                    continue
        
        return 0.0  # No valid answer found
    
    def setup_sglang_engine(self, model_path: Optional[str] = None):
        """Setup SGLang engine for fast inference with RadixAttention."""
        try:
            from inference.sglang_engine import SGLangEngine, SGLangConfig
            
            model = model_path if model_path else self.model_name
            config = SGLangConfig(
                model_name=model,
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                enable_radix_cache=True,  # Enable prefix caching
            )
            self.sglang_engine = SGLangEngine(config)
            self.sglang_engine.initialize()
            logger.info(f"SGLang engine initialized with {model}")
            return True
        except Exception as e:
            logger.warning(f"SGLang not available, falling back to transformers: {e}")
            self.use_sglang = False
            return False
    
    def compute_rewards(
        self,
        model_path: Optional[str],
        eval_data: List[Dict],
        method_name: str,
    ) -> MethodMetrics:
        """
        Compute rewards and metrics for a model.
        Uses SGLang for fast inference with RadixAttention prefix caching.
        
        Args:
            model_path: Path to trained model (None for baseline)
            eval_data: Evaluation data
            method_name: Name of the method
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {method_name}")
        logger.info(f"{'='*60}")
        
        load_path = model_path if model_path else self.model_name
        logger.info(f"Loading model from: {load_path}")
        
        # Try to use SGLang for faster inference
        use_sglang_for_this = self.use_sglang
        if use_sglang_for_this:
            use_sglang_for_this = self.setup_sglang_engine(load_path)
        
        # Fallback to transformers if SGLang not available
        model = None
        tokenizer = None
        if not use_sglang_for_this:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model.eval()
        
        metrics = MethodMetrics(method_name=method_name)
        
        generated_rewards = []  # Rewards for model-generated responses
        response_lengths = []
        inference_times = []
        correct_count = 0
        
        from tqdm import tqdm
        
        for item in tqdm(eval_data[:self.eval_subset_size], desc=f"Evaluating {method_name}"):
            prompt = item.get("prompt", item.get("problem", ""))
            ground_truth = item.get("answer", "")
            
            # Generate a response and measure time
            start_time = time.time()
            try:
                if use_sglang_for_this and self.sglang_engine:
                    # Use SGLang with RadixAttention for fast inference
                    response = self.sglang_engine.generate(
                        problem=prompt,
                        problem_type="math",  # Default to math
                    )
                else:
                    # Fallback to transformers
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                response_lengths.append(len(response))
                
                # Compute reward for generated response (correctness-based)
                gen_reward = self.compute_simple_reward(response, ground_truth)
                generated_rewards.append(gen_reward)
                
                # Check if generated answer is CORRECT (compare to ground truth)
                if ground_truth:
                    is_correct = self._check_answer_correct(response, ground_truth)
                    if is_correct:
                        correct_count += 1
                        
            except Exception as e:
                logger.warning(f"Error generating response: {e}")
                continue
        
        # Compute metrics
        metrics.total = min(len(eval_data), self.eval_subset_size)
        metrics.correct = correct_count
        metrics.accuracy = correct_count / metrics.total if metrics.total > 0 else 0
        
        # Reward is now based on model-generated responses (correctness-based)
        if generated_rewards:
            metrics.avg_reward = sum(generated_rewards) / len(generated_rewards)
        
        # Compute chosen/rejected rewards using TRAINED reward model
        if self.reward_model is not None:
            chosen_rewards = []
            rejected_rewards = []
            
            for item in eval_data[:self.eval_subset_size]:
                prompt = item.get("prompt", item.get("problem", ""))
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")
                
                if chosen and rejected:
                    c_reward = self.compute_learned_reward(prompt, chosen)
                    r_reward = self.compute_learned_reward(prompt, rejected)
                    chosen_rewards.append(c_reward)
                    rejected_rewards.append(r_reward)
            
            if chosen_rewards:
                metrics.avg_chosen_reward = sum(chosen_rewards) / len(chosen_rewards)
                metrics.avg_rejected_reward = sum(rejected_rewards) / len(rejected_rewards)
                metrics.reward_margin = metrics.avg_chosen_reward - metrics.avg_rejected_reward
        else:
            metrics.avg_chosen_reward = 0.0
            metrics.avg_rejected_reward = 0.0
            metrics.reward_margin = 0.0
        
        if response_lengths:
            metrics.avg_response_length = sum(response_lengths) / len(response_lengths)
        if inference_times:
            metrics.avg_inference_time = sum(inference_times) / len(inference_times)
        
        # Print detailed results
        self._print_method_results(metrics)
        
        # Clean up
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics
    
    def _check_answer_correct(self, response: str, ground_truth: str) -> bool:
        """
        Check if the model's generated response contains the correct FINAL answer.
        Only looks at specific answer patterns, not random occurrences.
        """
        import re
        
        # Clean ground truth
        gt_clean = ground_truth.strip().replace(",", "").replace("$", "").replace(" ", "")
        
        # Try to extract number from ground truth
        gt_numbers = re.findall(r'-?\d+\.?\d*', gt_clean)
        if not gt_numbers:
            return False
        gt_answer = float(gt_numbers[-1])  # Take last number as answer
        
        # Clean response
        response_clean = response.replace(",", "").replace("$", "")
        
        # Only check EXPLICIT answer patterns (strict matching)
        answer_patterns = [
            r'####\s*(-?\d+\.?\d*)',                          # GSM8K format
            r'\\boxed\{(-?\d+\.?\d*)\}',                      # LaTeX boxed
            r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(-?\d+\.?\d*)',  # "The answer is X"
            r'[Aa]nswer[:\s]+(-?\d+\.?\d*)\s*$',              # "Answer: X" at end
            r'=\s*(-?\d+\.?\d*)\s*(?:dollars?|cents?|units?)?\s*$',  # "= X" at very end
            r'[Tt]herefore[,:\s]+.*?(-?\d+\.?\d*)\s*$',       # "Therefore... X"
            r'[Ss]o[,:\s]+.*?(-?\d+\.?\d*)\s*$',              # "So... X" at end
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if matches:
                try:
                    pred_answer = float(matches[-1])
                    # Compare with small tolerance
                    if abs(pred_answer - gt_answer) < 0.01:
                        return True
                except ValueError:
                    continue
        
        # Check the LAST line specifically for a standalone number
        lines = response_clean.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # Only if last line is mostly just a number
            last_numbers = re.findall(r'^[^\d]*(-?\d+\.?\d*)[^\d]*$', last_line)
            if last_numbers:
                try:
                    pred_answer = float(last_numbers[0])
                    if abs(pred_answer - gt_answer) < 0.01:
                        return True
                except ValueError:
                    pass
        
        # NO FALLBACK - must match explicit answer pattern
        return False
    
    def _print_method_results(self, metrics: MethodMetrics):
        """Print detailed results for a method."""
        print(f"\n{'─'*50}")
        print(f"📊 Results for {metrics.method_name}")
        print(f"{'─'*50}")
        print(f"  Accuracy:            {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total})")
        print(f"  Avg Reward:          {metrics.avg_reward:.4f} (correctness-based)")
        if metrics.avg_chosen_reward != 0.0 or metrics.avg_rejected_reward != 0.0:
            print(f"  Avg Chosen Reward:   {metrics.avg_chosen_reward:.4f} (learned)")
            print(f"  Avg Rejected Reward: {metrics.avg_rejected_reward:.4f} (learned)")
            print(f"  Reward Margin:       {metrics.reward_margin:.4f} (chosen - rejected)")
        print(f"  Avg Response Length: {metrics.avg_response_length:.1f} chars")
        print(f"  Avg Inference Time:  {metrics.avg_inference_time:.3f}s")
        if metrics.training_time > 0:
            print(f"  Training Time:       {metrics.training_time:.1f}s")
        print(f"{'─'*50}\n")
    
    def train_dpo(self, train_data: List[Dict], eval_data: List[Dict]) -> str:
        """Train using DPO and return model path."""
        from training import DPOTrainerConfig, ReasoningDPOTrainer
        
        logger.info("\n" + "="*60)
        logger.info("Training with DPO...")
        logger.info("="*60)
        
        output_dir = str(self.output_dir / "dpo_model")
        
        config = DPOTrainerConfig(
            model_name=self.model_name,
            output_dir=output_dir,
            use_lora=self.use_lora,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            logging_steps=5,
        )
        
        trainer = ReasoningDPOTrainer(config)
        
        start_time = time.time()
        trainer.train(train_data, eval_data)
        training_time = time.time() - start_time
        
        logger.info(f"DPO training completed in {training_time:.1f}s")
        
        return output_dir, training_time
    
    def train_grpo(self, train_data: List[Dict]) -> str:
        """Train using GRPO and return model path."""
        from training.grpo_trainer import GRPOConfig, ReasoningGRPOTrainer
        
        logger.info("\n" + "="*60)
        logger.info("Training with GRPO...")
        logger.info("="*60)
        
        output_dir = str(self.output_dir / "grpo_model")
        
        config = GRPOConfig(
            model_name=self.model_name,
            output_dir=output_dir,
            use_lora=self.use_lora,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            logging_steps=5,
        )
        
        trainer = ReasoningGRPOTrainer(config)
        
        start_time = time.time()
        trainer.train(train_data)
        training_time = time.time() - start_time
        
        logger.info(f"GRPO training completed in {training_time:.1f}s")
        
        return output_dir, training_time
    
    def run_comparison(
        self,
        train_data: List[Dict],
        eval_data: List[Dict],
        methods: List[str] = ["none", "dpo", "grpo"],
    ) -> Dict[str, MethodMetrics]:
        """
        Run full comparison of training methods.
        
        Args:
            train_data: Training data
            eval_data: Evaluation data
            methods: List of methods to compare ("none", "dpo", "grpo")
        """
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " TRAINING METHOD COMPARISON ".center(58) + "║")
        print("║" + f" Model: {self.model_name[:40]}...".ljust(58) + "║")
        print("╚" + "═"*58 + "╝\n")
        
        # 0. Train Reward Model FIRST on preference pairs
        # This gives us meaningful reward scores for evaluation
        logger.info("Step 0: Training reward model on preference pairs...")
        self.train_reward_model(train_data)
        
        # 1. Baseline (no training)
        if "none" in methods:
            logger.info("Evaluating baseline (no training)...")
            baseline_metrics = self.compute_rewards(None, eval_data, "None (Baseline)")
            baseline_metrics.training_time = 0
            self.results["none"] = baseline_metrics
        
        # 2. DPO Training
        if "dpo" in methods:
            dpo_path, dpo_time = self.train_dpo(train_data, eval_data)
            dpo_metrics = self.compute_rewards(dpo_path, eval_data, "DPO")
            dpo_metrics.training_time = dpo_time
            self.results["dpo"] = dpo_metrics
        
        # 3. GRPO Training
        if "grpo" in methods:
            grpo_path, grpo_time = self.train_grpo(train_data)
            grpo_metrics = self.compute_rewards(grpo_path, eval_data, "GRPO")
            grpo_metrics.training_time = grpo_time
            self.results["grpo"] = grpo_metrics
        
        # Print comparison table
        self._print_comparison_table()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _print_comparison_table(self):
        """Print a formatted comparison table."""
        print("\n" + "╔" + "═"*70 + "╗")
        print("║" + " COMPARISON RESULTS ".center(70) + "║")
        print("╚" + "═"*70 + "╝\n")
        
        # Prepare table data with learned reward columns
        headers = [
            "Method",
            "Accuracy",
            "Correct Reward",
            "Chosen (learned)",
            "Rejected (learned)",
            "Margin",
            "Train Time"
        ]
        
        rows = []
        for method, metrics in self.results.items():
            rows.append([
                metrics.method_name,
                f"{metrics.accuracy:.4f}",
                f"{metrics.avg_reward:.4f}",
                f"{metrics.avg_chosen_reward:.4f}",
                f"{metrics.avg_rejected_reward:.4f}",
                f"{metrics.reward_margin:+.4f}",
                f"{metrics.training_time:.1f}s" if metrics.training_time > 0 else "N/A"
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Print insights
        print("\n📈 Key Insights:")
        print("─" * 50)
        
        if len(self.results) > 1:
            # Find best method by accuracy
            best_acc = max(self.results.items(), key=lambda x: x[1].accuracy)
            print(f"  ✅ Best Accuracy: {best_acc[1].method_name} ({best_acc[1].accuracy:.4f})")
            
            # Find best by reward (correctness-based)
            best_reward = max(self.results.items(), key=lambda x: x[1].avg_reward)
            print(f"  ✅ Best Correctness Reward: {best_reward[1].method_name} ({best_reward[1].avg_reward:.4f})")
            
            # Find best by learned reward margin
            best_margin = max(self.results.items(), key=lambda x: x[1].reward_margin)
            if best_margin[1].reward_margin != 0:
                print(f"  ✅ Best Learned Reward Margin: {best_margin[1].method_name} ({best_margin[1].reward_margin:+.4f})")
            
            # Compare DPO vs GRPO if both exist
            if "dpo" in self.results and "grpo" in self.results:
                dpo = self.results["dpo"]
                grpo = self.results["grpo"]
                
                print(f"\n  📊 DPO vs GRPO:")
                acc_diff = dpo.accuracy - grpo.accuracy
                
                if acc_diff > 0.001:
                    print(f"     DPO wins on accuracy by {abs(acc_diff):.4f}")
                elif acc_diff < -0.001:
                    print(f"     GRPO wins on accuracy by {abs(acc_diff):.4f}")
                else:
                    print(f"     Tied on accuracy")
                
                time_diff = dpo.training_time - grpo.training_time
                if time_diff > 1:
                    print(f"     GRPO faster by {abs(time_diff):.1f}s")
                elif time_diff < -1:
                    print(f"     DPO faster by {abs(time_diff):.1f}s")
            
            # Compare with baseline
            if "none" in self.results:
                baseline = self.results["none"]
                for method, metrics in self.results.items():
                    if method != "none":
                        improvement = metrics.accuracy - baseline.accuracy
                        print(f"\n  📈 {metrics.method_name} vs Baseline:")
                        print(f"     Accuracy improvement: {improvement:+.4f} ({improvement*100:+.1f}%)")
        
        print("\n")
    
    def _save_results(self):
        """Save comparison results to file."""
        output_file = self.output_dir / "comparison_results.json"
        
        results_dict = {
            method: metrics.to_dict()
            for method, metrics in self.results.items()
        }
        
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")


def load_standard_dataset(
    dataset_name: str = "gsm8k",
    num_samples: int = 100,
    split: str = "train",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    generate_rejected: bool = True,
    cache_dir: str = "./preference_cache",
    difficulty: Optional[str] = None,  # For MATH dataset: "Level 1" to "Level 5"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load data from standard benchmark datasets (GSM8K, MATH, HumanEval).
    Creates preference pairs by generating real incorrect solutions using the model.
    Caches generated pairs to avoid regenerating.
    
    Args:
        dataset_name: One of "gsm8k", "math", "humaneval", "mbpp"
        num_samples: Number of samples to load
        split: Dataset split to use
        model_name: Model to use for generating rejected responses
        generate_rejected: If True, generate real rejected responses using model
        cache_dir: Directory to cache generated preference pairs
        difficulty: For MATH dataset, filter by difficulty level
    
    Returns:
        Tuple of (train_data, eval_data) with preference pairs
    """
    import os
    from data_generator.dataset_loader import get_loader
    
    # Check for cached data
    diff_str = f"_{difficulty}" if difficulty else ""
    cache_file = f"{cache_dir}/{dataset_name}{diff_str}_{num_samples}_{split}_preferences.json"
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(cache_file):
        logger.info(f"Loading cached preference pairs from {cache_file}")
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        train_data = cached["train"]
        eval_data = cached["eval"]
        logger.info(f"Loaded {len(train_data)} train and {len(eval_data)} eval samples from cache")
        return train_data, eval_data
    
    logger.info(f"Loading {dataset_name} dataset ({split} split, difficulty={difficulty})...")
    
    # Pass difficulty to loader if MATH dataset
    if dataset_name == "math" and difficulty:
        loader = get_loader(dataset_name, split=split, subset_size=num_samples, difficulty=difficulty)
    else:
        loader = get_loader(dataset_name, split=split, subset_size=num_samples)
    problems = loader.load()
    
    logger.info(f"Loaded {len(problems)} problems from {dataset_name}")
    
    data = []
    
    if generate_rejected:
        # Generate REAL rejected responses using the model
        logger.info("Generating real rejected responses using model...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        
        from tqdm import tqdm
        import re
        
        for problem in tqdm(problems, desc="Generating preference pairs"):
            full_solution = problem.metadata.get("full_solution", problem.answer or "")
            gt_answer = problem.answer
            
            # Generate multiple responses and find incorrect ones
            prompt = problem.problem
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            rejected_responses = []
            
            # Generate 3 responses with high temperature to get variety
            for temp in [1.0, 1.2, 0.9]:
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=temp,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()
                    
                    # Check if this response is WRONG
                    gt_clean = gt_answer.strip().replace(",", "").replace("$", "")
                    gt_numbers = re.findall(r'-?\d+\.?\d*', gt_clean)
                    
                    if gt_numbers:
                        gt_num = float(gt_numbers[-1])
                        
                        # Check if response answer is different from ground truth
                        resp_numbers = re.findall(r'####\s*(-?\d+\.?\d*)|[Aa]nswer.*?(-?\d+\.?\d*)', response)
                        resp_numbers = [n for group in resp_numbers for n in group if n]
                        
                        if resp_numbers:
                            resp_num = float(resp_numbers[-1])
                            if abs(resp_num - gt_num) > 0.01:  # Different answer = rejected
                                rejected_responses.append(response)
                except Exception as e:
                    continue
            
            # Use the first wrong response, or create a fallback
            if rejected_responses:
                rejected = rejected_responses[0]
            else:
                # Fallback: modify the correct answer slightly
                rejected = f"Let me work through this step by step...\n\nAfter careful calculation, the answer is {int(float(gt_numbers[-1]) + 7) if gt_numbers else 999}."
            
            data.append({
                "prompt": problem.problem,
                "chosen": full_solution,
                "rejected": rejected,
                "answer": problem.answer,
                "problem_id": problem.id,
                "dataset": dataset_name,
            })
        
        # Clean up model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Simple fallback (not recommended)
        for problem in problems:
            full_solution = problem.metadata.get("full_solution", problem.answer or "")
            
            data.append({
                "prompt": problem.problem,
                "chosen": full_solution,
                "rejected": f"I'm not sure, but the answer might be {hash(problem.problem) % 100}.",
                "answer": problem.answer,
                "problem_id": problem.id,
                "dataset": dataset_name,
            })
    
    # Split into train/eval (80/20)
    import random
    random.shuffle(data)
    train_size = int(0.8 * len(data))
    
    train_data = data[:train_size]
    eval_data = data[train_size:]
    
    # Save to cache for future runs
    logger.info(f"Saving preference pairs to cache: {cache_file}")
    with open(cache_file, 'w') as f:
        json.dump({"train": train_data, "eval": eval_data}, f)
    
    logger.info(f"Created {len(train_data)} train and {len(eval_data)} eval samples")
    
    return train_data, eval_data


def generate_preference_pairs_with_model(
    problems: List[Dict],
    model_name: str,
    num_samples_per_problem: int = 4,
    verifier_type: str = "math",
) -> List[Dict]:
    """
    Generate preference pairs by sampling multiple solutions and verifying them.
    Uses the verifier to identify correct vs incorrect solutions.
    
    Args:
        problems: List of problems with 'prompt' and 'answer' keys
        model_name: Model to use for generation
        num_samples_per_problem: Number of solutions to generate per problem
        verifier_type: Type of verifier ("math" or "code")
    
    Returns:
        List of preference pairs with 'prompt', 'chosen', 'rejected'
    """
    from data_generator import CoTGenerator, GenerationConfig
    
    if verifier_type == "math":
        from verifier import MathVerifier, VerificationStatus
        verifier = MathVerifier()
    else:
        from verifier import CodeVerifier, ExecutionStatus
        verifier = CodeVerifier()
    
    # Setup generator
    config = GenerationConfig(
        model_name=model_name,
        num_paths=num_samples_per_problem,
        temperature=0.8,
    )
    generator = CoTGenerator(config)
    generator.initialize()
    
    preference_pairs = []
    
    from tqdm import tqdm
    for prob in tqdm(problems, desc="Generating preference pairs"):
        prompt = prob.get("prompt", prob.get("problem", ""))
        answer = prob.get("answer", "")
        
        # Generate multiple solutions
        try:
            paths = generator.generate_single(
                problem=prompt,
                problem_id=prob.get("problem_id", "unknown"),
                problem_type=verifier_type,
            )
        except Exception as e:
            logger.warning(f"Error generating for problem: {e}")
            continue
        
        correct_paths = []
        incorrect_paths = []
        
        for path in paths:
            if verifier_type == "math":
                result = verifier.verify_reasoning_path(path.reasoning, answer)
                is_correct = result.status == VerificationStatus.CORRECT
            else:
                code = verifier.extract_code(path.reasoning)
                result = verifier.verify_function_output(
                    code, "python", prob.get("function_call", ""), answer
                )
                is_correct = result.status == ExecutionStatus.SUCCESS
            
            if is_correct:
                correct_paths.append(path.reasoning)
            else:
                incorrect_paths.append(path.reasoning)
        
        # Create pairs
        if correct_paths and incorrect_paths:
            for chosen in correct_paths[:2]:  # Limit pairs
                for rejected in incorrect_paths[:2]:
                    preference_pairs.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "answer": answer,
                    })
    
    logger.info(f"Generated {len(preference_pairs)} preference pairs")
    return preference_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Compare DPO, GRPO, and baseline training methods on standard benchmarks"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data (JSONL with prompt/chosen/rejected)",
    )
    parser.add_argument(
        "--eval-data-path",
        type=str,
        default=None,
        help="Path to evaluation data (JSONL)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math",  # MATH is harder - better for seeing improvements
        choices=["gsm8k", "math", "humaneval", "mbpp"],
        help="Standard dataset to use (math=hard, gsm8k=easy)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
        help="MATH difficulty level (Level 5 = hardest)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to load from dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model = lower baseline = room to improve!
        help="Base model to use (smaller models recommended for seeing improvements)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./comparison_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["none", "dpo", "grpo"],
        choices=["none", "dpo", "grpo"],
        help="Training methods to compare",
    )
    parser.add_argument(
        "--eval-subset-size",
        type=int,
        default=50,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--generate-pairs",
        action="store_true",
        help="Generate preference pairs using model (slower but more accurate)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (use full fine-tuning)",
    )
    parser.add_argument(
        "--use-sglang",
        action="store_true",
        default=True,
        help="Use SGLang for fast inference with RadixAttention (default: True)",
    )
    parser.add_argument(
        "--no-sglang",
        action="store_true",
        help="Disable SGLang, use transformers instead",
    )
    
    args = parser.parse_args()
    
    # Load data from standard dataset or custom path
    if args.data_path:
        # Load from custom JSONL file
        comparison = TrainingMethodComparison(model_name=args.model)
        train_data = comparison.load_data(args.data_path)
        eval_data = comparison.load_data(args.eval_data_path) if args.eval_data_path else train_data[:50]
    else:
        # Load from standard benchmark dataset
        logger.info(f"Loading standard dataset: {args.dataset}")
        train_data, eval_data = load_standard_dataset(
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            split="train",
            model_name=args.model,
            generate_rejected=True,  # Generate REAL rejected responses
            difficulty=getattr(args, 'difficulty', None),  # For MATH dataset
        )
        
        # Optionally generate real preference pairs using model
        if args.generate_pairs:
            logger.info("Generating preference pairs with model verification...")
            verifier_type = "code" if args.dataset in ["humaneval", "mbpp"] else "math"
            train_data = generate_preference_pairs_with_model(
                train_data,
                model_name=args.model,
                verifier_type=verifier_type,
            )
    
    print(f"\n📊 Dataset: {args.dataset.upper()}")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Evaluation samples: {len(eval_data)}")
    
    # Run comparison
    comparison = TrainingMethodComparison(
        model_name=args.model,
        output_dir=args.output_dir,
        eval_subset_size=args.eval_subset_size,
        use_lora=not args.no_lora,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        use_sglang=args.use_sglang and not args.no_sglang,
    )
    
    results = comparison.run_comparison(
        train_data=train_data,
        eval_data=eval_data,
        methods=args.methods,
    )
    
    print("\n✅ Comparison complete!")
    print(f"📁 Results saved to: {args.output_dir}/comparison_results.json")


if __name__ == "__main__":
    main()
