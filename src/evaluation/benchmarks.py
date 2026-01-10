"""
Benchmark evaluators for reasoning models.
Supports GSM8K, HumanEval, and MATH benchmarks.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from benchmark evaluation."""
    benchmark_name: str
    total_problems: int
    correct: int
    incorrect: int
    errors: int
    accuracy: float
    avg_time_per_problem: float
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "total": self.total_problems,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "errors": self.errors,
            "accuracy": self.accuracy,
            "avg_time": self.avg_time_per_problem,
        }
    
    def save(self, path: str):
        """Save results to file."""
        with open(path, "w") as f:
            json.dump({
                "summary": self.to_dict(),
                "details": self.details,
            }, f, indent=2)


class BaseEvaluator:
    """Base class for benchmark evaluators."""
    
    def __init__(
        self,
        model_name: str,
        num_samples: int = 1,
        use_test_time_compute: bool = False,
        ttc_samples: int = 16,
    ):
        self.model_name = model_name
        self.num_samples = num_samples
        self.use_test_time_compute = use_test_time_compute
        self.ttc_samples = ttc_samples
        
        self.generator = None
        self.ttc = None
        self.verifier = None
    
    def setup(self):
        """Initialize components."""
        try:
            from data_generator import CoTGenerator, GenerationConfig
        except ImportError:
            from ..data_generator import CoTGenerator, GenerationConfig
        
        config = GenerationConfig(
            model_name=self.model_name,
            num_paths=self.num_samples,
        )
        self.generator = CoTGenerator(config)
        self.generator.initialize()
        
        if self.use_test_time_compute:
            from .test_time_compute import TestTimeCompute, TestTimeComputeConfig
            ttc_config = TestTimeComputeConfig(num_samples=self.ttc_samples)
            self.ttc = TestTimeCompute(self.model_name, ttc_config)
    
    def evaluate(self, problems: List[Dict[str, Any]]) -> BenchmarkResult:
        """Evaluate on a list of problems."""
        raise NotImplementedError


class GSM8KEvaluator(BaseEvaluator):
    """Evaluator for GSM8K math reasoning benchmark."""
    
    BENCHMARK_NAME = "GSM8K"
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
    
    def setup(self):
        """Setup with math verifier."""
        super().setup()
        try:
            from verifier import GSM8KVerifier
        except ImportError:
            from ..verifier import GSM8KVerifier
        self.verifier = GSM8KVerifier()
    
    def evaluate(
        self,
        split: str = "test",
        subset_size: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Evaluate on GSM8K dataset.
        
        Args:
            split: Dataset split (train/test)
            subset_size: Number of problems to evaluate
        """
        try:
            from data_generator import GSM8KLoader
        except ImportError:
            from ..data_generator import GSM8KLoader
        
        self.setup()
        
        # Load dataset
        loader = GSM8KLoader(split=split, subset_size=subset_size)
        problems = loader.load()
        
        results = []
        correct = 0
        errors = 0
        total_time = 0
        
        for problem in tqdm(problems, desc=f"Evaluating {self.BENCHMARK_NAME}"):
            start_time = time.time()
            
            try:
                if self.use_test_time_compute:
                    best, _ = self.ttc.solve(problem.problem, problem.answer)
                    predicted = best.final_answer
                    reasoning = best.reasoning
                else:
                    paths = self.generator.generate_single(
                        problem=problem.problem,
                        problem_id=problem.id,
                        problem_type="math",
                    )
                    if paths:
                        reasoning = paths[0].reasoning
                        result = self.verifier.verify_reasoning_path(
                            reasoning, problem.answer
                        )
                        predicted = result.predicted
                    else:
                        predicted = None
                        reasoning = ""
                
                # Verify
                try:
                    from verifier import VerificationStatus
                except ImportError:
                    from ..verifier import VerificationStatus
                result = self.verifier.verify(predicted or "", problem.answer)
                is_correct = result.status == VerificationStatus.CORRECT
                
                if is_correct:
                    correct += 1
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                results.append({
                    "problem_id": problem.id,
                    "problem": problem.problem,
                    "expected": problem.answer,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "reasoning": reasoning[:500],  # Truncate for storage
                    "time": elapsed,
                })
                
            except Exception as e:
                logger.error(f"Error on problem {problem.id}: {e}")
                errors += 1
                results.append({
                    "problem_id": problem.id,
                    "error": str(e),
                })
        
        num_problems = len(problems)
        return BenchmarkResult(
            benchmark_name=self.BENCHMARK_NAME,
            total_problems=num_problems,
            correct=correct,
            incorrect=num_problems - correct - errors,
            errors=errors,
            accuracy=correct / num_problems if num_problems > 0 else 0,
            avg_time_per_problem=total_time / num_problems if num_problems > 0 else 0,
            details=results,
        )


class HumanEvalEvaluator(BaseEvaluator):
    """Evaluator for HumanEval code generation benchmark."""
    
    BENCHMARK_NAME = "HumanEval"
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
    
    def setup(self):
        """Setup with code verifier."""
        super().setup()
        try:
            from verifier import HumanEvalVerifier
        except ImportError:
            from ..verifier import HumanEvalVerifier
        self.verifier = HumanEvalVerifier()
    
    def evaluate(
        self,
        subset_size: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Evaluate on HumanEval dataset.
        
        Args:
            subset_size: Number of problems to evaluate
        """
        try:
            from data_generator import HumanEvalLoader
        except ImportError:
            from ..data_generator import HumanEvalLoader
        
        self.setup()
        
        # Load dataset
        loader = HumanEvalLoader(subset_size=subset_size)
        problems = loader.load()
        
        results = []
        correct = 0
        errors = 0
        total_time = 0
        
        for problem in tqdm(problems, desc=f"Evaluating {self.BENCHMARK_NAME}"):
            start_time = time.time()
            
            try:
                # Generate solution
                paths = self.generator.generate_single(
                    problem=problem.problem,
                    problem_id=problem.id,
                    problem_type="code",
                )
                
                if not paths:
                    errors += 1
                    continue
                
                # Extract code
                code = self.verifier.extract_code(paths[0].reasoning)
                
                # Verify
                try:
                    from verifier import ExecutionStatus
                except ImportError:
                    from ..verifier import ExecutionStatus
                result = self.verifier.verify_problem(
                    task_id=problem.id,
                    prompt=problem.problem,
                    completion=code,
                    test=problem.metadata["test"],
                    entry_point=problem.metadata["entry_point"],
                )
                
                is_correct = result.status == ExecutionStatus.SUCCESS
                
                if is_correct:
                    correct += 1
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                results.append({
                    "task_id": problem.id,
                    "prompt": problem.problem[:200],
                    "completion": code[:500],
                    "is_correct": is_correct,
                    "status": result.status.value,
                    "time": elapsed,
                })
                
            except Exception as e:
                logger.error(f"Error on problem {problem.id}: {e}")
                errors += 1
                results.append({
                    "task_id": problem.id,
                    "error": str(e),
                })
        
        num_problems = len(problems)
        return BenchmarkResult(
            benchmark_name=self.BENCHMARK_NAME,
            total_problems=num_problems,
            correct=correct,
            incorrect=num_problems - correct - errors,
            errors=errors,
            accuracy=correct / num_problems if num_problems > 0 else 0,
            avg_time_per_problem=total_time / num_problems if num_problems > 0 else 0,
            details=results,
        )


class MATHEvaluator(BaseEvaluator):
    """Evaluator for MATH competition-level math benchmark."""
    
    BENCHMARK_NAME = "MATH"
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
    
    def setup(self):
        """Setup with math verifier."""
        super().setup()
        try:
            from verifier import MathVerifier
        except ImportError:
            from ..verifier import MathVerifier
        self.verifier = MathVerifier()
    
    def evaluate(
        self,
        split: str = "test",
        difficulty: Optional[str] = None,
        subject: Optional[str] = None,
        subset_size: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Evaluate on MATH dataset.
        
        Args:
            split: Dataset split
            difficulty: Filter by difficulty level
            subject: Filter by math subject
            subset_size: Number of problems to evaluate
        """
        try:
            from data_generator import MATHLoader
        except ImportError:
            from ..data_generator import MATHLoader
        
        self.setup()
        
        # Load dataset
        loader = MATHLoader(
            split=split,
            difficulty=difficulty,
            subject=subject,
            subset_size=subset_size,
        )
        problems = loader.load()
        
        results = []
        correct = 0
        errors = 0
        total_time = 0
        
        for problem in tqdm(problems, desc=f"Evaluating {self.BENCHMARK_NAME}"):
            start_time = time.time()
            
            try:
                if self.use_test_time_compute:
                    best, _ = self.ttc.solve(problem.problem, problem.answer)
                    predicted = best.final_answer
                    reasoning = best.reasoning
                else:
                    paths = self.generator.generate_single(
                        problem=problem.problem,
                        problem_id=problem.id,
                        problem_type="math",
                    )
                    if paths:
                        reasoning = paths[0].reasoning
                        result = self.verifier.verify_reasoning_path(
                            reasoning, problem.answer
                        )
                        predicted = result.predicted
                    else:
                        predicted = None
                        reasoning = ""
                
                # Verify
                try:
                    from verifier import VerificationStatus
                except ImportError:
                    from ..verifier import VerificationStatus
                result = self.verifier.verify(predicted or "", problem.answer)
                is_correct = result.status == VerificationStatus.CORRECT
                
                if is_correct:
                    correct += 1
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                results.append({
                    "problem_id": problem.id,
                    "difficulty": problem.metadata.get("level"),
                    "subject": problem.metadata.get("type"),
                    "expected": problem.answer,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "time": elapsed,
                })
                
            except Exception as e:
                logger.error(f"Error on problem {problem.id}: {e}")
                errors += 1
                results.append({
                    "problem_id": problem.id,
                    "error": str(e),
                })
        
        num_problems = len(problems)
        return BenchmarkResult(
            benchmark_name=self.BENCHMARK_NAME,
            total_problems=num_problems,
            correct=correct,
            incorrect=num_problems - correct - errors,
            errors=errors,
            accuracy=correct / num_problems if num_problems > 0 else 0,
            avg_time_per_problem=total_time / num_problems if num_problems > 0 else 0,
            details=results,
        )


def run_all_benchmarks(
    model_name: str,
    output_dir: str = "./benchmark_results",
    use_ttc: bool = False,
    gsm8k_subset_size: Optional[int] = None,
    humaneval_subset_size: Optional[int] = None,
    ttc_samples: int = 16,
) -> Dict[str, BenchmarkResult]:
    """
    Run all benchmarks and save results.
    
    Args:
        model_name: Model to evaluate
        output_dir: Directory to save results
        use_ttc: Whether to use test-time compute
        gsm8k_subset_size: Number of GSM8K problems (None for all)
        humaneval_subset_size: Number of HumanEval problems (None for all)
        ttc_samples: Number of samples for test-time compute
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # GSM8K
    logger.info("Running GSM8K evaluation...")
    gsm8k = GSM8KEvaluator(model_name, use_test_time_compute=use_ttc, ttc_samples=ttc_samples)
    gsm8k_result = gsm8k.evaluate(subset_size=gsm8k_subset_size)
    gsm8k_result.save(f"{output_dir}/gsm8k_results.json")
    results["gsm8k"] = gsm8k_result
    
    # HumanEval
    logger.info("Running HumanEval evaluation...")
    humaneval = HumanEvalEvaluator(model_name)
    humaneval_result = humaneval.evaluate(subset_size=humaneval_subset_size)
    humaneval_result.save(f"{output_dir}/humaneval_results.json")
    results["humaneval"] = humaneval_result
    
    # Summary
    summary = {
        name: result.to_dict()
        for name, result in results.items()
    }
    
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_dir}")
    
    return results
