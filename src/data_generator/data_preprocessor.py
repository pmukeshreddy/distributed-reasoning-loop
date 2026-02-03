"""
Data Preprocessor for reasoning data.
Handles deduplication, quality filtering, length filtering, and smart pair selection.
"""

import re
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""
    # Length filtering
    min_reasoning_tokens: int = 50
    max_reasoning_tokens: int = 2048
    min_response_length: int = 100  # characters
    max_response_length: int = 8000  # characters
    
    # Quality filtering
    min_step_count: int = 2  # Minimum reasoning steps
    require_final_answer: bool = True
    filter_repetitive: bool = True
    max_repetition_ratio: float = 0.3
    
    # Deduplication
    dedup_threshold: float = 0.85  # Jaccard similarity threshold
    use_semantic_dedup: bool = False
    
    # Pair selection
    min_pair_diversity: float = 0.2
    max_pairs_per_problem: int = 5
    prefer_high_confidence: bool = True
    
    # Normalization
    normalize_whitespace: bool = True
    normalize_math: bool = True
    strip_system_artifacts: bool = True


class DataPreprocessor:
    """
    Preprocesses reasoning data for RL training.
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.stats = defaultdict(int)
    
    def preprocess(
        self,
        samples: List[Dict[str, Any]],
        create_pairs: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Full preprocessing pipeline.
        
        Returns:
            (filtered_samples, dpo_pairs)
        """
        logger.info(f"Starting preprocessing of {len(samples)} samples")
        self.stats = defaultdict(int)
        self.stats["input_samples"] = len(samples)
        
        # Step 1: Normalize
        samples = [self._normalize(s) for s in samples]
        
        # Step 2: Quality filter
        samples = self._quality_filter(samples)
        logger.info(f"After quality filter: {len(samples)}")
        
        # Step 3: Length filter
        samples = self._length_filter(samples)
        logger.info(f"After length filter: {len(samples)}")
        
        # Step 4: Deduplicate
        samples = self._deduplicate(samples)
        logger.info(f"After deduplication: {len(samples)}")
        
        self.stats["output_samples"] = len(samples)
        
        # Step 5: Create pairs
        pairs = []
        if create_pairs:
            pairs = self._create_smart_pairs(samples)
            self.stats["pairs_created"] = len(pairs)
            logger.info(f"Created {len(pairs)} DPO pairs")
        
        logger.info(f"Preprocessing stats: {dict(self.stats)}")
        return samples, pairs
    
    def _normalize(self, sample: Dict) -> Dict:
        """Normalize a single sample."""
        reasoning = sample.get("reasoning", sample.get("response", ""))
        
        if self.config.normalize_whitespace:
            # Collapse multiple newlines
            reasoning = re.sub(r'\n{3,}', '\n\n', reasoning)
            # Collapse multiple spaces
            reasoning = re.sub(r' {2,}', ' ', reasoning)
            reasoning = reasoning.strip()
        
        if self.config.normalize_math:
            # Normalize common math formatting
            reasoning = re.sub(r'\$\$\s*', '$$', reasoning)
            reasoning = re.sub(r'\s*\$\$', '$$', reasoning)
        
        if self.config.strip_system_artifacts:
            # Remove common artifacts
            reasoning = re.sub(r'^(Assistant:|AI:|Response:)\s*', '', reasoning, flags=re.IGNORECASE)
            reasoning = re.sub(r'<\|.*?\|>', '', reasoning)  # Remove special tokens
            reasoning = re.sub(r'\[INST\].*?\[/INST\]', '', reasoning, flags=re.DOTALL)
        
        sample["reasoning"] = reasoning
        return sample
    
    def _quality_filter(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples based on quality criteria."""
        filtered = []
        
        for sample in samples:
            reasoning = sample.get("reasoning", "")
            
            # Check minimum steps
            if self.config.min_step_count > 0:
                step_patterns = [
                    r'step\s*\d+', r'\d+\)', r'\d+\.', 
                    r'first|second|third|then|next|finally',
                    r'let\'s|we can|we need|we have'
                ]
                step_count = sum(
                    len(re.findall(p, reasoning, re.IGNORECASE)) 
                    for p in step_patterns
                )
                if step_count < self.config.min_step_count:
                    self.stats["filtered_no_steps"] += 1
                    continue
            
            # Check for final answer
            if self.config.require_final_answer:
                answer_patterns = [
                    r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)',
                    r'(?:therefore|thus|so|hence)',
                    r'=\s*[\d.]+\s*$',
                    r'\\boxed\{',
                    r'####',
                ]
                has_answer = any(
                    re.search(p, reasoning, re.IGNORECASE | re.MULTILINE)
                    for p in answer_patterns
                )
                if not has_answer:
                    self.stats["filtered_no_answer"] += 1
                    continue
            
            # Check for repetition
            if self.config.filter_repetitive:
                if self._is_repetitive(reasoning):
                    self.stats["filtered_repetitive"] += 1
                    continue
            
            filtered.append(sample)
        
        return filtered
    
    def _is_repetitive(self, text: str) -> bool:
        """Check if text has too much repetition."""
        sentences = re.split(r'[.!?\n]', text)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 3:
            return False
        
        # Check for duplicate sentences
        unique = set(sentences)
        if len(unique) / len(sentences) < (1 - self.config.max_repetition_ratio):
            return True
        
        # Check for repeated phrases
        words = text.lower().split()
        if len(words) < 20:
            return False
        
        # N-gram repetition check
        ngram_size = 5
        ngrams = [' '.join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size)]
        unique_ngrams = set(ngrams)
        
        if len(ngrams) > 0 and len(unique_ngrams) / len(ngrams) < 0.5:
            return True
        
        return False
    
    def _length_filter(self, samples: List[Dict]) -> List[Dict]:
        """Filter by length."""
        filtered = []
        
        for sample in samples:
            reasoning = sample.get("reasoning", "")
            
            # Character length
            if len(reasoning) < self.config.min_response_length:
                self.stats["filtered_too_short"] += 1
                continue
            
            if len(reasoning) > self.config.max_response_length:
                self.stats["filtered_too_long"] += 1
                continue
            
            # Token count (approximate)
            token_count = len(reasoning.split())
            if token_count < self.config.min_reasoning_tokens:
                self.stats["filtered_too_few_tokens"] += 1
                continue
            
            if token_count > self.config.max_reasoning_tokens:
                self.stats["filtered_too_many_tokens"] += 1
                continue
            
            filtered.append(sample)
        
        return filtered
    
    def _deduplicate(self, samples: List[Dict]) -> List[Dict]:
        """Remove near-duplicate samples."""
        if not samples:
            return samples
        
        # Group by problem
        by_problem = defaultdict(list)
        for sample in samples:
            problem_id = sample.get("problem_id", sample.get("id", ""))
            by_problem[problem_id].append(sample)
        
        deduplicated = []
        
        for problem_id, problem_samples in by_problem.items():
            # Keep track of unique samples for this problem
            unique_samples = []
            
            for sample in problem_samples:
                reasoning = sample.get("reasoning", "")
                is_duplicate = False
                
                for existing in unique_samples:
                    existing_reasoning = existing.get("reasoning", "")
                    similarity = self._jaccard_similarity(reasoning, existing_reasoning)
                    
                    if similarity > self.config.dedup_threshold:
                        is_duplicate = True
                        self.stats["dedup_removed"] += 1
                        # Keep the one with higher confidence if available
                        if sample.get("verification_confidence", 0) > existing.get("verification_confidence", 0):
                            unique_samples.remove(existing)
                            unique_samples.append(sample)
                        break
                
                if not is_duplicate:
                    unique_samples.append(sample)
            
            deduplicated.extend(unique_samples)
        
        return deduplicated
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _create_smart_pairs(self, samples: List[Dict]) -> List[Dict]:
        """Create DPO pairs with smart selection."""
        # Group by problem
        by_problem = defaultdict(lambda: {"correct": [], "incorrect": []})
        
        for sample in samples:
            problem_id = sample.get("problem_id", sample.get("id", ""))
            is_correct = sample.get("is_correct", False)
            
            if is_correct:
                by_problem[problem_id]["correct"].append(sample)
            else:
                by_problem[problem_id]["incorrect"].append(sample)
        
        pairs = []
        
        for problem_id, groups in by_problem.items():
            correct = groups["correct"]
            incorrect = groups["incorrect"]
            
            if not correct or not incorrect:
                continue
            
            # Sort by confidence if available
            if self.config.prefer_high_confidence:
                correct.sort(key=lambda x: x.get("verification_confidence", 0), reverse=True)
                incorrect.sort(key=lambda x: x.get("verification_confidence", 0), reverse=True)
            
            # Create pairs with diversity filtering
            problem_pairs = []
            
            for pos in correct:
                for neg in incorrect:
                    pos_reasoning = pos.get("reasoning", "")
                    neg_reasoning = neg.get("reasoning", "")
                    
                    diversity = 1 - self._jaccard_similarity(pos_reasoning, neg_reasoning)
                    
                    if diversity >= self.config.min_pair_diversity:
                        pair = {
                            "prompt": pos.get("problem", ""),
                            "chosen": pos_reasoning,
                            "rejected": neg_reasoning,
                            "problem_id": problem_id,
                            "diversity_score": diversity,
                            "chosen_confidence": pos.get("verification_confidence", 0),
                            "rejected_confidence": neg.get("verification_confidence", 0),
                        }
                        problem_pairs.append(pair)
            
            # Sort by diversity and take top N
            problem_pairs.sort(key=lambda x: x["diversity_score"], reverse=True)
            pairs.extend(problem_pairs[:self.config.max_pairs_per_problem])
        
        return pairs
    
    def get_stats(self) -> Dict[str, int]:
        """Return preprocessing statistics."""
        return dict(self.stats)


def preprocess_jsonl(
    input_path: str,
    output_path: str,
    pairs_output_path: Optional[str] = None,
    config: Optional[PreprocessConfig] = None,
) -> Dict[str, int]:
    """
    Preprocess a JSONL file.
    
    Args:
        input_path: Path to input JSONL
        output_path: Path for filtered samples output
        pairs_output_path: Path for DPO pairs output (optional)
        config: Preprocessing configuration
    
    Returns:
        Statistics dictionary
    """
    import json
    
    # Load samples
    samples = []
    with open(input_path) as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    filtered_samples, pairs = preprocessor.preprocess(
        samples, 
        create_pairs=pairs_output_path is not None
    )
    
    # Save filtered samples
    with open(output_path, 'w') as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Save pairs if requested
    if pairs_output_path and pairs:
        with open(pairs_output_path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
    
    return preprocessor.get_stats()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess reasoning data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--pairs-output", type=str, help="Output path for DPO pairs")
    parser.add_argument("--min-length", type=int, default=100, help="Min response length")
    parser.add_argument("--max-length", type=int, default=8000, help="Max response length")
    parser.add_argument("--dedup-threshold", type=float, default=0.85, help="Dedup threshold")
    
    args = parser.parse_args()
    
    config = PreprocessConfig(
        min_response_length=args.min_length,
        max_response_length=args.max_length,
        dedup_threshold=args.dedup_threshold,
    )
    
    logging.basicConfig(level=logging.INFO)
    stats = preprocess_jsonl(args.input, args.output, args.pairs_output, config)
    print(f"Stats: {stats}")
