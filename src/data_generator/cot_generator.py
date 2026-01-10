"""
Chain-of-Thought Generator using vLLM/SGLang for high-throughput inference.
Generates multiple reasoning paths for each problem.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json
import hashlib
import asyncio

logger = logging.getLogger(__name__)


class InferenceBackend(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TRANSFORMERS = "transformers"


@dataclass
class GenerationConfig:
    """Configuration for CoT generation."""
    model_name: str = "meta-llama/Llama-3-70B-Instruct"
    backend: InferenceBackend = InferenceBackend.VLLM
    num_paths: int = 10
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    
    # Batching
    batch_size: int = 8
    
    # vLLM specific
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    
    # SGLang specific
    enable_radix_cache: bool = True
    
    
@dataclass
class ReasoningPath:
    """A single reasoning path with metadata."""
    problem_id: str
    problem: str
    reasoning: str
    final_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    confidence: float = 0.0
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def path_hash(self) -> str:
        """Unique hash for this reasoning path."""
        content = f"{self.problem_id}:{self.reasoning}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem": self.problem,
            "reasoning": self.reasoning,
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "path_hash": self.path_hash,
            "generation_metadata": self.generation_metadata,
        }


class CoTGenerator:
    """
    High-throughput Chain-of-Thought generator.
    Supports vLLM and SGLang backends for efficient inference.
    """
    
    MATH_SYSTEM_PROMPT = """You are a helpful math tutor. Solve the following problem step by step.
Show your work clearly, explaining each step of your reasoning.
At the end, provide your final answer after '#### '."""

    CODE_SYSTEM_PROMPT = """You are an expert programmer. Solve the following coding problem.
Think through the problem step by step before writing code.
Explain your approach, then provide the complete solution in a Python code block."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def _init_vllm(self):
        """Initialize vLLM backend."""
        try:
            from vllm import LLM, SamplingParams
            
            self.model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
            )
            self.sampling_params = SamplingParams(
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
            )
            self._initialized = True
            logger.info(f"Initialized vLLM with model {self.config.model_name}")
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
    
    def _init_sglang(self):
        """Initialize SGLang backend."""
        try:
            import sglang as sgl
            
            # SGLang uses a runtime that we'll configure
            self.sgl = sgl
            self._initialized = True
            logger.info(f"Initialized SGLang with model {self.config.model_name}")
        except ImportError:
            raise ImportError("SGLang not installed. Install with: pip install sglang")
    
    def _init_transformers(self):
        """Initialize HuggingFace Transformers backend (fallback)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._initialized = True
            logger.info(f"Initialized Transformers with model {self.config.model_name}")
        except ImportError:
            raise ImportError("Transformers not installed. Install with: pip install transformers")
    
    def initialize(self):
        """Initialize the inference backend."""
        if self._initialized:
            return
            
        if self.config.backend == InferenceBackend.VLLM:
            self._init_vllm()
        elif self.config.backend == InferenceBackend.SGLANG:
            self._init_sglang()
        else:
            self._init_transformers()
    
    def _format_prompt(
        self,
        problem: str,
        problem_type: str = "math",
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Format the prompt with system instruction and optional few-shot examples."""
        if problem_type == "math":
            system = self.MATH_SYSTEM_PROMPT
        else:
            system = self.CODE_SYSTEM_PROMPT
        
        messages = [{"role": "system", "content": system}]
        
        if few_shot_examples:
            for example in few_shot_examples:
                messages.append({"role": "user", "content": example["problem"]})
                messages.append({"role": "assistant", "content": example["solution"]})
        
        messages.append({"role": "user", "content": problem})
        
        # Format as chat template
        if self.config.backend == InferenceBackend.TRANSFORMERS and self.tokenizer:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        # Default formatting for vLLM/SGLang
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
        formatted += "<|assistant|>\n"
        
        return formatted
    
    def generate_paths_vllm(
        self,
        problems: List[Dict[str, str]],
        problem_type: str = "math",
    ) -> List[List[ReasoningPath]]:
        """Generate multiple reasoning paths using vLLM."""
        from vllm import SamplingParams
        
        all_prompts = []
        prompt_to_problem = {}
        
        # Create prompts for each problem with num_paths copies
        for prob in problems:
            prompt = self._format_prompt(prob["problem"], problem_type)
            for _ in range(self.config.num_paths):
                all_prompts.append(prompt)
                prompt_to_problem[len(all_prompts) - 1] = prob
        
        # Generate all at once
        outputs = self.model.generate(all_prompts, self.sampling_params)
        
        # Organize results by problem
        results = {prob["id"]: [] for prob in problems}
        
        for idx, output in enumerate(outputs):
            prob = prompt_to_problem[idx]
            generated_text = output.outputs[0].text
            
            path = ReasoningPath(
                problem_id=prob["id"],
                problem=prob["problem"],
                reasoning=generated_text,
                generation_metadata={
                    "model": self.config.model_name,
                    "temperature": self.config.temperature,
                    "backend": "vllm",
                }
            )
            results[prob["id"]].append(path)
        
        return [results[prob["id"]] for prob in problems]
    
    def generate_paths_sglang(
        self,
        problems: List[Dict[str, str]],
        problem_type: str = "math",
    ) -> List[List[ReasoningPath]]:
        """Generate multiple reasoning paths using SGLang with RadixAttention."""
        import sglang as sgl
        
        @sgl.function
        def cot_generation(s, problem, system_prompt):
            s += sgl.system(system_prompt)
            s += sgl.user(problem)
            s += sgl.assistant(sgl.gen("reasoning", max_tokens=self.config.max_new_tokens))
        
        system_prompt = self.MATH_SYSTEM_PROMPT if problem_type == "math" else self.CODE_SYSTEM_PROMPT
        
        results = {prob["id"]: [] for prob in problems}
        
        # SGLang batch processing with prefix caching
        for prob in problems:
            # Generate multiple paths - SGLang will cache the prefix
            states = []
            for _ in range(self.config.num_paths):
                state = cot_generation.run(
                    problem=prob["problem"],
                    system_prompt=system_prompt,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                states.append(state)
            
            for state in states:
                path = ReasoningPath(
                    problem_id=prob["id"],
                    problem=prob["problem"],
                    reasoning=state["reasoning"],
                    generation_metadata={
                        "model": self.config.model_name,
                        "temperature": self.config.temperature,
                        "backend": "sglang",
                        "radix_cache_enabled": self.config.enable_radix_cache,
                    }
                )
                results[prob["id"]].append(path)
        
        return [results[prob["id"]] for prob in problems]
    
    def generate_paths_transformers(
        self,
        problems: List[Dict[str, str]],
        problem_type: str = "math",
    ) -> List[List[ReasoningPath]]:
        """Generate reasoning paths using HuggingFace Transformers."""
        import torch
        
        results = {prob["id"]: [] for prob in problems}
        
        for prob in problems:
            prompt = self._format_prompt(prob["problem"], problem_type)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            for _ in range(self.config.num_paths):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=self.config.do_sample,
                        repetition_penalty=self.config.repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                
                path = ReasoningPath(
                    problem_id=prob["id"],
                    problem=prob["problem"],
                    reasoning=generated,
                    generation_metadata={
                        "model": self.config.model_name,
                        "temperature": self.config.temperature,
                        "backend": "transformers",
                    }
                )
                results[prob["id"]].append(path)
        
        return [results[prob["id"]] for prob in problems]
    
    def generate(
        self,
        problems: List[Dict[str, str]],
        problem_type: str = "math",
    ) -> List[List[ReasoningPath]]:
        """
        Generate multiple reasoning paths for each problem.
        
        Args:
            problems: List of dicts with 'id' and 'problem' keys
            problem_type: Either 'math' or 'code'
            
        Returns:
            List of lists, each inner list contains num_paths ReasoningPath objects
        """
        self.initialize()
        
        if self.config.backend == InferenceBackend.VLLM:
            return self.generate_paths_vllm(problems, problem_type)
        elif self.config.backend == InferenceBackend.SGLANG:
            return self.generate_paths_sglang(problems, problem_type)
        else:
            return self.generate_paths_transformers(problems, problem_type)
    
    def generate_single(
        self,
        problem: str,
        problem_id: str = "single",
        problem_type: str = "math",
    ) -> List[ReasoningPath]:
        """Generate reasoning paths for a single problem."""
        problems = [{"id": problem_id, "problem": problem}]
        results = self.generate(problems, problem_type)
        return results[0] if results else []


class SpeculativeCoTGenerator:
    """
    Speculative Decoding enhanced CoT generator.
    Uses a small draft model to speed up generation from a large target model.
    """
    
    def __init__(
        self,
        draft_config: GenerationConfig,
        target_config: GenerationConfig,
        max_speculative_tokens: int = 5,
    ):
        self.draft_config = draft_config
        self.target_config = target_config
        self.max_speculative_tokens = max_speculative_tokens
        
        self.draft_generator = CoTGenerator(draft_config)
        self.target_generator = CoTGenerator(target_config)
        
    def initialize(self):
        """Initialize both models."""
        self.draft_generator.initialize()
        self.target_generator.initialize()
    
    def generate_with_speculation(
        self,
        problems: List[Dict[str, str]],
        problem_type: str = "math",
    ) -> List[List[ReasoningPath]]:
        """
        Generate reasoning paths using speculative decoding.
        
        The draft model (e.g., 7B) generates candidate tokens quickly,
        which are then verified by the target model (e.g., 70B).
        """
        self.initialize()
        
        # For now, implement a simplified version
        # Full speculative decoding requires lower-level access
        logger.info("Using draft model for initial generation")
        draft_paths = self.draft_generator.generate(problems, problem_type)
        
        # In a full implementation, we would:
        # 1. Generate k tokens with draft model
        # 2. Verify with target model in parallel
        # 3. Accept matching tokens, regenerate from mismatch
        
        return draft_paths
