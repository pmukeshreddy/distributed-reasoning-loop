"""
Speculative Decoding for accelerated inference.
Uses a small draft model to speed up generation from a larger target model.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    draft_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    target_model: str = "meta-llama/Llama-3-70B-Instruct"
    
    # Speculation parameters
    max_speculative_tokens: int = 5  # K tokens to speculate
    temperature: float = 0.8
    top_p: float = 0.95
    
    # Generation
    max_new_tokens: int = 2048
    
    # Acceptance
    acceptance_threshold: float = 0.0  # Minimum probability ratio
    use_tree_attention: bool = False  # Tree-based speculation


@dataclass
class DraftTargetPair:
    """Pair of draft and target model outputs."""
    draft_tokens: List[int]
    draft_logprobs: torch.Tensor
    target_logprobs: torch.Tensor
    accepted_tokens: List[int]
    acceptance_rate: float


class SpeculativeDecoder:
    """
    Speculative Decoding implementation.
    
    Algorithm:
    1. Draft model generates K speculative tokens
    2. Target model verifies all K+1 positions in parallel
    3. Accept tokens where target agrees with draft
    4. Resample from target at first rejection point
    """
    
    def __init__(self, config: SpeculativeConfig):
        self.config = config
        self.draft_model = None
        self.target_model = None
        self.tokenizer = None
        self._initialized = False
        
        # Statistics
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
    
    def setup(self):
        """Initialize both models."""
        if self._initialized:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading draft model: {self.config.draft_model}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.config.draft_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        logger.info(f"Loading target model: {self.config.target_model}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.config.target_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.target_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._initialized = True
        logger.info("Speculative decoder initialized")
    
    def _sample_with_temperature(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[int, float]:
        """Sample token from logits with temperature."""
        if temperature == 0:
            token = logits.argmax().item()
            prob = F.softmax(logits, dim=-1)[token].item()
            return token, prob
        
        probs = F.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probs, 1).item()
        return token, probs[token].item()
    
    def _draft_step(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> Tuple[List[int], torch.Tensor]:
        """Generate speculative tokens from draft model."""
        draft_tokens = []
        draft_logprobs = []
        
        current_ids = input_ids
        
        with torch.no_grad():
            for _ in range(num_tokens):
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[:, -1, :]
                
                # Sample token
                token, prob = self._sample_with_temperature(
                    logits[0], self.config.temperature
                )
                
                draft_tokens.append(token)
                draft_logprobs.append(F.log_softmax(logits, dim=-1))
                
                # Append to sequence
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[token]], device=current_ids.device)
                ], dim=1)
        
        return draft_tokens, torch.stack(draft_logprobs, dim=1)
    
    def _verify_step(
        self,
        input_ids: torch.Tensor,
        draft_tokens: List[int],
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens with target model."""
        # Append all draft tokens
        draft_tensor = torch.tensor([draft_tokens], device=input_ids.device)
        full_ids = torch.cat([input_ids, draft_tensor], dim=1)
        
        with torch.no_grad():
            outputs = self.target_model(full_ids)
            # Get logprobs for all positions after input
            logits = outputs.logits[:, input_ids.shape[1]-1:-1, :]
            target_logprobs = F.log_softmax(logits, dim=-1)
        
        return target_logprobs
    
    def _acceptance_sampling(
        self,
        draft_tokens: List[int],
        draft_logprobs: torch.Tensor,
        target_logprobs: torch.Tensor,
    ) -> Tuple[List[int], int]:
        """
        Perform acceptance-rejection sampling.
        Returns accepted tokens and the number accepted.
        """
        accepted = []
        
        for i, draft_token in enumerate(draft_tokens):
            # Get probabilities
            draft_prob = torch.exp(draft_logprobs[0, i, draft_token]).item()
            target_prob = torch.exp(target_logprobs[0, i, draft_token]).item()
            
            # Acceptance criterion
            if draft_prob > 0:
                ratio = target_prob / draft_prob
                accept = np.random.random() < min(1.0, ratio)
            else:
                accept = target_prob > self.config.acceptance_threshold
            
            if accept:
                accepted.append(draft_token)
            else:
                # Rejection - sample new token from adjusted distribution
                # p_target - p_draft (clamped to positive)
                adjusted_probs = F.softmax(target_logprobs[0, i], dim=-1) - \
                                F.softmax(draft_logprobs[0, i], dim=-1)
                adjusted_probs = F.relu(adjusted_probs)
                
                if adjusted_probs.sum() > 0:
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    new_token = torch.multinomial(adjusted_probs, 1).item()
                else:
                    new_token = torch.multinomial(
                        F.softmax(target_logprobs[0, i], dim=-1), 1
                    ).item()
                
                accepted.append(new_token)
                break  # Stop at first rejection
        
        return accepted, len(accepted)
    
    def generate_step(
        self,
        input_ids: torch.Tensor,
    ) -> DraftTargetPair:
        """
        Perform one step of speculative decoding.
        
        Returns:
            DraftTargetPair with results and statistics
        """
        # Draft phase
        draft_tokens, draft_logprobs = self._draft_step(
            input_ids,
            self.config.max_speculative_tokens,
        )
        
        # Verify phase
        target_logprobs = self._verify_step(input_ids, draft_tokens)
        
        # Acceptance sampling
        accepted_tokens, num_accepted = self._acceptance_sampling(
            draft_tokens,
            draft_logprobs,
            target_logprobs,
        )
        
        # Update statistics
        self.total_draft_tokens += len(draft_tokens)
        self.total_accepted_tokens += num_accepted
        
        return DraftTargetPair(
            draft_tokens=draft_tokens,
            draft_logprobs=draft_logprobs,
            target_logprobs=target_logprobs,
            accepted_tokens=accepted_tokens,
            acceptance_rate=num_accepted / len(draft_tokens) if draft_tokens else 0,
        )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        self.setup()
        
        max_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Tokenize prompt
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to(self.target_model.device)
        
        generated_tokens = []
        
        while len(generated_tokens) < max_tokens:
            # Speculative generation step
            result = self.generate_step(input_ids)
            
            # Add accepted tokens
            generated_tokens.extend(result.accepted_tokens)
            
            # Update input_ids
            new_tokens = torch.tensor(
                [result.accepted_tokens],
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            
            # Check for EOS
            if self.tokenizer.eos_token_id in result.accepted_tokens:
                break
        
        # Decode
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return output_text
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        acceptance_rate = (
            self.total_accepted_tokens / self.total_draft_tokens
            if self.total_draft_tokens > 0 else 0
        )
        
        # Theoretical speedup
        # If K tokens are drafted and acceptance rate is r,
        # speedup ≈ (K * r + 1) / 1 for each step
        k = self.config.max_speculative_tokens
        theoretical_speedup = k * acceptance_rate + 1
        
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "acceptance_rate": acceptance_rate,
            "theoretical_speedup": theoretical_speedup,
        }


class SpeculativeCoTGenerator:
    """
    Chain-of-Thought generator using speculative decoding.
    Optimized for reasoning tasks.
    """
    
    def __init__(self, config: SpeculativeConfig):
        self.config = config
        self.decoder = SpeculativeDecoder(config)
    
    MATH_PROMPT_TEMPLATE = """You are a helpful math tutor. Solve the following problem step by step.
Show your work clearly, explaining each step of your reasoning.
At the end, provide your final answer after '#### '.

Problem: {problem}

Solution:"""

    CODE_PROMPT_TEMPLATE = """You are an expert programmer. Solve the following coding problem.
Think through the problem step by step before writing code.
Explain your approach, then provide the complete solution in a Python code block.

Problem: {problem}

Solution:"""

    def generate_cot(
        self,
        problem: str,
        problem_type: str = "math",
        max_new_tokens: int = 2048,
    ) -> str:
        """
        Generate Chain-of-Thought solution with speculative decoding.
        
        Args:
            problem: The problem to solve
            problem_type: Either 'math' or 'code'
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated reasoning path
        """
        if problem_type == "math":
            prompt = self.MATH_PROMPT_TEMPLATE.format(problem=problem)
        else:
            prompt = self.CODE_PROMPT_TEMPLATE.format(problem=problem)
        
        output = self.decoder.generate(prompt, max_new_tokens)
        
        return output
    
    def generate_multiple(
        self,
        problem: str,
        num_paths: int = 4,
        problem_type: str = "math",
    ) -> List[str]:
        """Generate multiple reasoning paths."""
        paths = []
        for _ in range(num_paths):
            path = self.generate_cot(problem, problem_type)
            paths.append(path)
        return paths


class TreeSpeculativeDecoder(SpeculativeDecoder):
    """
    Tree-based speculative decoding.
    Explores multiple draft branches for better acceptance.
    """
    
    def __init__(self, config: SpeculativeConfig, num_branches: int = 3):
        super().__init__(config)
        self.num_branches = num_branches
    
    def _draft_tree(
        self,
        input_ids: torch.Tensor,
        depth: int,
    ) -> List[Tuple[List[int], torch.Tensor]]:
        """Generate a tree of speculative tokens."""
        branches = []
        
        with torch.no_grad():
            outputs = self.draft_model(input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Sample multiple starting tokens
            probs = F.softmax(logits / self.config.temperature, dim=-1)
            top_tokens = torch.multinomial(probs, self.num_branches)
            
            for token in top_tokens[0]:
                token = token.item()
                token_logprob = F.log_softmax(logits, dim=-1)
                
                # Continue each branch
                branch_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token]], device=input_ids.device)
                ], dim=1)
                
                remaining_tokens, remaining_logprobs = self._draft_step(
                    branch_ids,
                    depth - 1,
                )
                
                branches.append((
                    [token] + remaining_tokens,
                    torch.cat([token_logprob.unsqueeze(1), remaining_logprobs], dim=1),
                ))
        
        return branches
    
    def generate_step(
        self,
        input_ids: torch.Tensor,
    ) -> DraftTargetPair:
        """Generate with tree-based speculation."""
        if not self.config.use_tree_attention:
            return super().generate_step(input_ids)
        
        # Generate tree of drafts
        branches = self._draft_tree(
            input_ids,
            self.config.max_speculative_tokens,
        )
        
        best_accepted = []
        best_rate = 0.0
        best_draft = None
        best_draft_logprobs = None
        best_target_logprobs = None
        
        # Verify each branch and pick best
        for draft_tokens, draft_logprobs in branches:
            target_logprobs = self._verify_step(input_ids, draft_tokens)
            accepted, num_accepted = self._acceptance_sampling(
                draft_tokens,
                draft_logprobs,
                target_logprobs,
            )
            
            rate = num_accepted / len(draft_tokens) if draft_tokens else 0
            
            if num_accepted > len(best_accepted):
                best_accepted = accepted
                best_rate = rate
                best_draft = draft_tokens
                best_draft_logprobs = draft_logprobs
                best_target_logprobs = target_logprobs
        
        self.total_draft_tokens += len(best_draft) if best_draft else 0
        self.total_accepted_tokens += len(best_accepted)
        
        return DraftTargetPair(
            draft_tokens=best_draft or [],
            draft_logprobs=best_draft_logprobs,
            target_logprobs=best_target_logprobs,
            accepted_tokens=best_accepted,
            acceptance_rate=best_rate,
        )
