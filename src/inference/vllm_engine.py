"""
vLLM inference engine wrapper.
Provides high-throughput inference for reasoning models.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for vLLM engine."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    
    # Sampling
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 2048
    
    # Batching
    max_num_seqs: int = 256


class VLLMEngine:
    """
    vLLM inference engine for high-throughput generation.
    """
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.llm = None
        self.sampling_params = None
        self._initialized = False
    
    def initialize(self):
        """Initialize vLLM engine."""
        if self._initialized:
            return
        
        try:
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                trust_remote_code=True,
            )
            
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_tokens=self.config.max_tokens,
            )
            
            self._initialized = True
            logger.info(f"vLLM engine initialized with {self.config.model_name}")
            
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate completions for multiple prompts.
        
        Args:
            prompts: List of prompts
            sampling_params: Optional override sampling parameters
            
        Returns:
            List of generated texts
        """
        self.initialize()
        
        from vllm import SamplingParams
        
        if sampling_params:
            params = SamplingParams(**sampling_params)
        else:
            params = self.sampling_params
        
        outputs = self.llm.generate(prompts, params)
        
        results = []
        for output in outputs:
            results.append(output.outputs[0].text)
        
        return results
    
    def generate_with_logprobs(
        self,
        prompts: List[str],
        num_logprobs: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate with log probabilities."""
        self.initialize()
        
        from vllm import SamplingParams
        
        params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            logprobs=num_logprobs,
        )
        
        outputs = self.llm.generate(prompts, params)
        
        results = []
        for output in outputs:
            result = {
                "text": output.outputs[0].text,
                "logprobs": [],
            }
            
            if output.outputs[0].logprobs:
                for token_logprobs in output.outputs[0].logprobs:
                    result["logprobs"].append({
                        token: logprob
                        for token, logprob in token_logprobs.items()
                    })
            
            results.append(result)
        
        return results
    
    def batch_generate_cot(
        self,
        problems: List[str],
        problem_type: str = "math",
        num_paths_per_problem: int = 1,
    ) -> List[List[str]]:
        """
        Generate Chain-of-Thought solutions for multiple problems.
        
        Args:
            problems: List of problems
            problem_type: 'math' or 'code'
            num_paths_per_problem: Paths to generate per problem
            
        Returns:
            List of lists of reasoning paths
        """
        self.initialize()
        
        # Format prompts
        if problem_type == "math":
            template = """Solve this math problem step by step. Show your work and end with #### followed by the answer.

Problem: {problem}

Solution:"""
        else:
            template = """Solve this coding problem. Explain your approach and provide the solution.

Problem: {problem}

Solution:"""
        
        # Create all prompts
        all_prompts = []
        prompt_mapping = []
        
        for i, problem in enumerate(problems):
            for _ in range(num_paths_per_problem):
                all_prompts.append(template.format(problem=problem))
                prompt_mapping.append(i)
        
        # Generate
        outputs = self.generate(all_prompts)
        
        # Organize results
        results = [[] for _ in problems]
        for output, problem_idx in zip(outputs, prompt_mapping):
            results[problem_idx].append(output)
        
        return results


class AsyncVLLMEngine:
    """
    Async vLLM engine for streaming and async generation.
    """
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.engine = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize async engine."""
        if self._initialized:
            return
        
        try:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs
            
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._initialized = True
            
        except ImportError:
            raise ImportError("vLLM async engine not available")
    
    async def generate_streaming(
        self,
        prompt: str,
        request_id: str,
    ):
        """
        Generate with streaming output.
        
        Yields tokens as they are generated.
        """
        await self.initialize()
        
        from vllm import SamplingParams
        
        params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        async for output in self.engine.generate(prompt, params, request_id):
            yield output.outputs[0].text
