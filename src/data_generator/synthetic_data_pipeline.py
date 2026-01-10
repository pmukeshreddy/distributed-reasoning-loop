"""
Synthetic Data Pipeline for generating and filtering reasoning paths.
Creates positive/negative pairs for RL training.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import hashlib

from .cot_generator import CoTGenerator, ReasoningPath, GenerationConfig
from .dataset_loader import DatasetLoader, Problem, get_loader

# Import verifiers - handle both package and direct import
import sys
from pathlib import Path
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from verifier import (
    MathVerifier,
    GSM8KVerifier,
    CodeVerifier,
    HumanEvalVerifier,
    VerificationStatus,
    ExecutionStatus,
)

# Import preprocessor
try:
    from .data_preprocessor import DataPreprocessor, PreprocessConfig
except ImportError:
    from data_preprocessor import DataPreprocessor, PreprocessConfig

logger = logging.getLogger(__name__)


@dataclass
class FilteredSample:
    """A verified and filtered reasoning sample."""
    problem_id: str
    problem: str
    reasoning: str
    final_answer: str
    expected_answer: str
    is_correct: bool
    verification_confidence: float
    path_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem": self.problem,
            "reasoning": self.reasoning,
            "final_answer": self.final_answer,
            "expected_answer": self.expected_answer,
            "is_correct": self.is_correct,
            "verification_confidence": self.verification_confidence,
            "path_hash": self.path_hash,
            "metadata": self.metadata,
        }


@dataclass
class SamplePair:
    """A positive/negative pair for DPO training."""
    problem_id: str
    problem: str
    chosen: str  # Correct reasoning path
    rejected: str  # Incorrect reasoning path
    chosen_answer: str
    rejected_answer: str
    expected_answer: str
    diversity_score: float = 0.0  # How different the paths are
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem": self.problem,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_answer": self.chosen_answer,
            "rejected_answer": self.rejected_answer,
            "expected_answer": self.expected_answer,
            "diversity_score": self.diversity_score,
        }
    
    def to_dpo_format(self) -> Dict[str, str]:
        """Format for TRL DPOTrainer."""
        return {
            "prompt": self.problem,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


class SyntheticDataPipeline:
    """
    End-to-end pipeline for synthetic reasoning data generation.
    
    Pipeline stages:
    1. Load problems from dataset
    2. Generate multiple CoT paths per problem
    3. Verify each path using appropriate verifier
    4. Filter and create positive/negative pairs
    5. Output for RL training
    """
    
    def __init__(
        self,
        generator_config: GenerationConfig,
        dataset_name: str = "gsm8k",
        output_dir: str = "./synthetic_data",
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
    ):
        self.generator_config = generator_config
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        
        # Initialize components
        self.generator = CoTGenerator(generator_config)
        self.dataset_loader = get_loader(dataset_name, cache_dir=cache_dir)
        
        # Select appropriate verifier
        if dataset_name in ["gsm8k", "math"]:
            self.verifier = GSM8KVerifier() if dataset_name == "gsm8k" else MathVerifier()
            self.problem_type = "math"
        else:
            self.verifier = HumanEvalVerifier() if dataset_name == "humaneval" else CodeVerifier()
            self.problem_type = "code"
        
        # Statistics
        self.stats = {
            "total_problems": 0,
            "total_paths_generated": 0,
            "correct_paths": 0,
            "incorrect_paths": 0,
            "pairs_created": 0,
        }
    
    def generate_for_problem(
        self,
        problem: Problem,
    ) -> List[FilteredSample]:
        """Generate and verify reasoning paths for a single problem."""
        # Generate multiple paths
        paths = self.generator.generate_single(
            problem=problem.problem,
            problem_id=problem.id,
            problem_type=self.problem_type,
        )
        
        filtered_samples = []
        
        for path in paths:
            # Verify the path
            if self.problem_type == "math":
                result = self.verifier.verify_reasoning_path(
                    path.reasoning,
                    problem.answer,
                )
                is_correct = result.status == VerificationStatus.CORRECT
                confidence = result.confidence
                final_answer = result.predicted or ""
            else:
                # Code verification
                code = self.verifier.extract_code(path.reasoning)
                if problem.metadata.get("test"):
                    result = self.verifier.verify_humaneval(
                        code,
                        problem.metadata["entry_point"],
                        problem.metadata["test"],
                    )
                    is_correct = result.status == ExecutionStatus.SUCCESS
                    confidence = 1.0 if is_correct else 0.0
                    final_answer = code
                else:
                    is_correct = False
                    confidence = 0.0
                    final_answer = code
            
            sample = FilteredSample(
                problem_id=problem.id,
                problem=problem.problem,
                reasoning=path.reasoning,
                final_answer=final_answer,
                expected_answer=problem.answer,
                is_correct=is_correct,
                verification_confidence=confidence,
                path_hash=path.path_hash,
                metadata={
                    "generation_config": {
                        "model": self.generator_config.model_name,
                        "temperature": self.generator_config.temperature,
                    }
                }
            )
            filtered_samples.append(sample)
        
        return filtered_samples
    
    def create_pairs(
        self,
        samples: List[FilteredSample],
        min_diversity: float = 0.1,
    ) -> List[SamplePair]:
        """
        Create positive/negative pairs from filtered samples.
        Only creates pairs where paths have sufficient diversity.
        """
        # Group by problem
        by_problem: Dict[str, List[FilteredSample]] = {}
        for sample in samples:
            if sample.problem_id not in by_problem:
                by_problem[sample.problem_id] = []
            by_problem[sample.problem_id].append(sample)
        
        pairs = []
        
        for problem_id, problem_samples in by_problem.items():
            correct = [s for s in problem_samples if s.is_correct]
            incorrect = [s for s in problem_samples if not s.is_correct]
            
            if not correct or not incorrect:
                continue
            
            # Create pairs with diversity filtering
            for pos in correct:
                for neg in incorrect:
                    diversity = self._calculate_diversity(pos.reasoning, neg.reasoning)
                    
                    if diversity >= min_diversity:
                        pair = SamplePair(
                            problem_id=problem_id,
                            problem=pos.problem,
                            chosen=pos.reasoning,
                            rejected=neg.reasoning,
                            chosen_answer=pos.final_answer,
                            rejected_answer=neg.final_answer,
                            expected_answer=pos.expected_answer,
                            diversity_score=diversity,
                        )
                        pairs.append(pair)
        
        return pairs
    
    def _calculate_diversity(self, text1: str, text2: str) -> float:
        """Calculate diversity score between two texts using Jaccard distance."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        jaccard = intersection / union if union > 0 else 0
        return 1 - jaccard  # Diversity is inverse of similarity
    
    def run(
        self,
        subset_size: Optional[int] = None,
        batch_size: int = 10,
        save_intermediate: bool = True,
    ) -> Tuple[List[FilteredSample], List[SamplePair]]:
        """
        Run the full synthetic data pipeline.
        
        Args:
            subset_size: Number of problems to process (None for all)
            batch_size: Number of problems to process in parallel
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Tuple of (all_samples, all_pairs)
        """
        # Load problems
        problems = self.dataset_loader.load()
        if subset_size:
            problems = problems[:subset_size]
        
        self.stats["total_problems"] = len(problems)
        logger.info(f"Processing {len(problems)} problems from {self.dataset_name}")
        
        # Initialize generator
        self.generator.initialize()
        
        all_samples = []
        all_pairs = []
        
        # Process in batches
        for i in tqdm(range(0, len(problems), batch_size), desc="Generating"):
            batch = problems[i:i + batch_size]
            
            # Generate for batch
            batch_problems = [{"id": p.id, "problem": p.problem} for p in batch]
            batch_paths = self.generator.generate(batch_problems, self.problem_type)
            
            # Verify and filter
            for prob, paths in zip(batch, batch_paths):
                self.stats["total_paths_generated"] += len(paths)
                
                samples = []
                for path in paths:
                    # Verify
                    if self.problem_type == "math":
                        result = self.verifier.verify_reasoning_path(
                            path.reasoning,
                            prob.answer,
                        )
                        is_correct = result.status == VerificationStatus.CORRECT
                        confidence = result.confidence
                        final_answer = result.predicted or ""
                    else:
                        code = self.verifier.extract_code(path.reasoning)
                        if prob.metadata.get("test"):
                            result = self.verifier.verify_humaneval(
                                code,
                                prob.metadata["entry_point"],
                                prob.metadata["test"],
                            )
                            is_correct = result.status == ExecutionStatus.SUCCESS
                            confidence = 1.0 if is_correct else 0.0
                            final_answer = code
                        else:
                            is_correct = False
                            confidence = 0.0
                            final_answer = code
                    
                    if is_correct:
                        self.stats["correct_paths"] += 1
                    else:
                        self.stats["incorrect_paths"] += 1
                    
                    sample = FilteredSample(
                        problem_id=prob.id,
                        problem=prob.problem,
                        reasoning=path.reasoning,
                        final_answer=final_answer,
                        expected_answer=prob.answer,
                        is_correct=is_correct,
                        verification_confidence=confidence,
                        path_hash=path.path_hash,
                    )
                    samples.append(sample)
                
                all_samples.extend(samples)
                
                # Create pairs for this problem
                pairs = self.create_pairs(samples)
                all_pairs.extend(pairs)
                self.stats["pairs_created"] += len(pairs)
            
            # Save intermediate results
            if save_intermediate and (i + batch_size) % (batch_size * 10) == 0:
                self._save_checkpoint(all_samples, all_pairs, i + batch_size)
        
        # Final preprocessing and save
        logger.info("Running data preprocessing...")
        preprocessor = DataPreprocessor(PreprocessConfig(
            min_response_length=100,
            max_response_length=8000,
            dedup_threshold=0.85,
            min_pair_diversity=0.2,
            max_pairs_per_problem=5,
        ))
        
        # Convert to dicts for preprocessing
        sample_dicts = [s.to_dict() for s in all_samples]
        filtered_samples, smart_pairs = preprocessor.preprocess(sample_dicts, create_pairs=True)
        
        # Convert back and update pairs
        logger.info(f"Preprocessing: {len(all_samples)} -> {len(filtered_samples)} samples")
        logger.info(f"Smart pairs created: {len(smart_pairs)}")
        
        # Save with preprocessed data
        self._save_results(all_samples, all_pairs, filtered_samples, smart_pairs)
        
        logger.info(f"Pipeline complete. Stats: {self.stats}")
        
        return all_samples, all_pairs
    
    def _save_checkpoint(
        self,
        samples: List[FilteredSample],
        pairs: List[SamplePair],
        step: int,
    ):
        """Save intermediate checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        with open(checkpoint_dir / f"samples_{step}.jsonl", "w") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
        
        with open(checkpoint_dir / f"pairs_{step}.jsonl", "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict()) + "\n")
    
    def _save_results(
        self,
        samples: List[FilteredSample],
        pairs: List[SamplePair],
        filtered_samples: List[Dict] = None,
        smart_pairs: List[Dict] = None,
    ):
        """Save final results."""
        # Save all samples (raw)
        with open(self.output_dir / "all_samples.jsonl", "w") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
        
        # Save preprocessed/filtered samples
        if filtered_samples:
            with open(self.output_dir / "filtered_samples.jsonl", "w") as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample) + "\n")
        
        # Save pairs in DPO format (use smart pairs if available)
        final_pairs = smart_pairs if smart_pairs else [p.to_dpo_format() for p in pairs]
        with open(self.output_dir / "dpo_pairs.jsonl", "w") as f:
            for pair in final_pairs:
                # Ensure DPO format
                dpo_pair = {
                    "prompt": pair.get("prompt", pair.get("problem", "")),
                    "chosen": pair.get("chosen", ""),
                    "rejected": pair.get("rejected", ""),
                }
                f.write(json.dumps(dpo_pair) + "\n")
        
        # Save full pairs with metadata
        with open(self.output_dir / "full_pairs.jsonl", "w") as f:
            for pair in (smart_pairs if smart_pairs else pairs):
                if hasattr(pair, 'to_dict'):
                    f.write(json.dumps(pair.to_dict()) + "\n")
                else:
                    f.write(json.dumps(pair) + "\n")
        
        # Save statistics
        with open(self.output_dir / "stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)
        
        # Save correct/incorrect samples separately
        correct = [s for s in samples if s.is_correct]
        incorrect = [s for s in samples if not s.is_correct]
        
        with open(self.output_dir / "correct_samples.jsonl", "w") as f:
            for sample in correct:
                f.write(json.dumps(sample.to_dict()) + "\n")
        
        with open(self.output_dir / "incorrect_samples.jsonl", "w") as f:
            for sample in incorrect:
                f.write(json.dumps(sample.to_dict()) + "\n")
        
        logger.info(f"Results saved to {self.output_dir}")


def create_pipeline_from_config(config: Dict[str, Any]) -> SyntheticDataPipeline:
    """Create pipeline from configuration dictionary."""
    gen_config = GenerationConfig(
        model_name=config.get("teacher_model", "meta-llama/Llama-3-70B-Instruct"),
        num_paths=config.get("num_cot_paths", 10),
        max_new_tokens=config.get("max_new_tokens", 2048),
        temperature=config.get("temperature", 0.8),
    )
    
    return SyntheticDataPipeline(
        generator_config=gen_config,
        dataset_name=config.get("dataset", "gsm8k"),
        output_dir=config.get("output_dir", "./synthetic_data"),
        cache_dir=config.get("cache_dir"),
    )