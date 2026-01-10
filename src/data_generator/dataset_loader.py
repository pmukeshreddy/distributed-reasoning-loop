"""
Dataset loaders for GSM8K and HumanEval.
Provides unified interface for loading and preprocessing problems.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """A single problem from a dataset."""
    id: str
    problem: str
    answer: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "problem": self.problem,
            "answer": self.answer,
            "metadata": self.metadata,
        }


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load(self) -> List[Problem]:
        """Load and return all problems."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of problems."""
        pass
    
    def __iter__(self) -> Iterator[Problem]:
        """Iterate over problems."""
        return iter(self.load())
    
    def get_splits(self) -> Dict[str, List[Problem]]:
        """Return train/test/validation splits if available."""
        return {"all": self.load()}


class GSM8KLoader(DatasetLoader):
    """
    Loader for GSM8K (Grade School Math 8K) dataset.
    https://github.com/openai/grade-school-math
    """
    
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        subset_size: Optional[int] = None,
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.subset_size = subset_size
        self._problems = None
        
    def load(self) -> List[Problem]:
        """Load GSM8K dataset from HuggingFace."""
        if self._problems is not None:
            return self._problems
            
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                "gsm8k",
                "main",
                split=self.split,
                cache_dir=self.cache_dir,
            )
            
            problems = []
            for idx, item in enumerate(dataset):
                if self.subset_size and idx >= self.subset_size:
                    break
                    
                # Extract answer from GSM8K format
                answer = self._extract_answer(item["answer"])
                
                problem = Problem(
                    id=f"gsm8k_{self.split}_{idx}",
                    problem=item["question"],
                    answer=answer,
                    metadata={
                        "full_solution": item["answer"],
                        "dataset": "gsm8k",
                        "split": self.split,
                    }
                )
                problems.append(problem)
            
            self._problems = problems
            logger.info(f"Loaded {len(problems)} problems from GSM8K {self.split}")
            return problems
            
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
    
    def _extract_answer(self, solution: str) -> str:
        """Extract numeric answer from GSM8K solution format."""
        import re
        # GSM8K uses #### to mark the final answer
        match = re.search(r'####\s*(.+?)(?:\n|$)', solution)
        if match:
            return match.group(1).strip().replace(',', '')
        return solution.split('\n')[-1].strip()
    
    def __len__(self) -> int:
        if self._problems is None:
            self.load()
        return len(self._problems)
    
    def get_splits(self) -> Dict[str, List[Problem]]:
        """Return train and test splits."""
        train_loader = GSM8KLoader(split="train", cache_dir=self.cache_dir)
        test_loader = GSM8KLoader(split="test", cache_dir=self.cache_dir)
        return {
            "train": train_loader.load(),
            "test": test_loader.load(),
        }


class HumanEvalLoader(DatasetLoader):
    """
    Loader for HumanEval dataset.
    https://github.com/openai/human-eval
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        subset_size: Optional[int] = None,
    ):
        self.cache_dir = cache_dir
        self.subset_size = subset_size
        self._problems = None
        
    def load(self) -> List[Problem]:
        """Load HumanEval dataset."""
        if self._problems is not None:
            return self._problems
            
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                "openai_humaneval",
                split="test",
                cache_dir=self.cache_dir,
            )
            
            problems = []
            for idx, item in enumerate(dataset):
                if self.subset_size and idx >= self.subset_size:
                    break
                
                problem = Problem(
                    id=item["task_id"],
                    problem=item["prompt"],
                    answer=item["canonical_solution"],
                    metadata={
                        "entry_point": item["entry_point"],
                        "test": item["test"],
                        "dataset": "humaneval",
                    }
                )
                problems.append(problem)
            
            self._problems = problems
            logger.info(f"Loaded {len(problems)} problems from HumanEval")
            return problems
            
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
    
    def __len__(self) -> int:
        if self._problems is None:
            self.load()
        return len(self._problems)


class MBPPLoader(DatasetLoader):
    """
    Loader for MBPP (Mostly Basic Programming Problems) dataset.
    """
    
    def __init__(
        self,
        split: str = "test",
        cache_dir: Optional[str] = None,
        subset_size: Optional[int] = None,
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.subset_size = subset_size
        self._problems = None
        
    def load(self) -> List[Problem]:
        """Load MBPP dataset."""
        if self._problems is not None:
            return self._problems
            
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                "mbpp",
                split=self.split,
                cache_dir=self.cache_dir,
            )
            
            problems = []
            for idx, item in enumerate(dataset):
                if self.subset_size and idx >= self.subset_size:
                    break
                
                # Format test cases
                test_code = "\n".join(item["test_list"])
                
                problem = Problem(
                    id=f"mbpp_{item['task_id']}",
                    problem=item["text"],
                    answer=item["code"],
                    metadata={
                        "test_list": item["test_list"],
                        "test_code": test_code,
                        "challenge_test_list": item.get("challenge_test_list", []),
                        "dataset": "mbpp",
                        "split": self.split,
                    }
                )
                problems.append(problem)
            
            self._problems = problems
            logger.info(f"Loaded {len(problems)} problems from MBPP {self.split}")
            return problems
            
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
    
    def __len__(self) -> int:
        if self._problems is None:
            self.load()
        return len(self._problems)


class MATHLoader(DatasetLoader):
    """
    Loader for MATH dataset (competition mathematics).
    """
    
    def __init__(
        self,
        split: str = "test",
        difficulty: Optional[str] = None,
        subject: Optional[str] = None,
        cache_dir: Optional[str] = None,
        subset_size: Optional[int] = None,
    ):
        self.split = split
        self.difficulty = difficulty
        self.subject = subject
        self.cache_dir = cache_dir
        self.subset_size = subset_size
        self._problems = None
        
    def load(self) -> List[Problem]:
        """Load MATH dataset."""
        if self._problems is not None:
            return self._problems
            
        try:
            from datasets import load_dataset
            
            # Try multiple MATH dataset sources
            dataset = None
            dataset_sources = [
                ("hendrycks/competition_math", None),  # Original
                ("lighteval/MATH", None),              # LightEval mirror  
                ("EleutherAI/hendrycks_math", self.split),  # EleutherAI mirror
            ]
            
            for source, config in dataset_sources:
                try:
                    logger.info(f"Trying MATH dataset from: {source}")
                    if config:
                        dataset = load_dataset(
                            source,
                            config,
                            split=self.split,
                            cache_dir=self.cache_dir,
                            trust_remote_code=True,
                        )
                    else:
                        dataset = load_dataset(
                            source,
                            split=self.split,
                            cache_dir=self.cache_dir,
                            trust_remote_code=True,
                        )
                    logger.info(f"Successfully loaded MATH from {source}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load from {source}: {e}")
                    continue
            
            if dataset is None:
                raise RuntimeError("Could not load MATH dataset from any source")
            
            problems = []
            for idx, item in enumerate(dataset):
                if self.subset_size and idx >= self.subset_size:
                    break
                
                # Filter by difficulty/subject if specified
                if self.difficulty and item["level"] != self.difficulty:
                    continue
                if self.subject and item["type"] != self.subject:
                    continue
                
                # Extract boxed answer
                answer = self._extract_boxed_answer(item["solution"])
                
                problem = Problem(
                    id=f"math_{self.split}_{idx}",
                    problem=item["problem"],
                    answer=answer,
                    metadata={
                        "full_solution": item["solution"],
                        "level": item["level"],
                        "type": item["type"],
                        "dataset": "math",
                        "split": self.split,
                    }
                )
                problems.append(problem)
            
            self._problems = problems
            logger.info(f"Loaded {len(problems)} problems from MATH {self.split}")
            return problems
            
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
    
    def _extract_boxed_answer(self, solution: str) -> str:
        """Extract answer from LaTeX \\boxed{} format."""
        import re
        match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if match:
            return match.group(1)
        return ""
    
    def __len__(self) -> int:
        if self._problems is None:
            self.load()
        return len(self._problems)


def get_loader(dataset_name: str, **kwargs) -> DatasetLoader:
    """Factory function to get the appropriate loader."""
    loaders = {
        "gsm8k": GSM8KLoader,
        "humaneval": HumanEvalLoader,
        "mbpp": MBPPLoader,
        "math": MATHLoader,
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name.lower()](**kwargs)
