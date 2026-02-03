"""
SGLang inference engine wrapper.
Provides RadixAttention-optimized inference for reasoning models.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SGLangConfig:
    """Configuration for SGLang engine."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # RadixAttention settings
    enable_radix_cache: bool = True
    max_radix_cache_size: int = 16 * 1024 * 1024 * 1024  # 16GB
    
    # Sampling
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 2048
    
    # Parallel sampling
    parallel_sample_num: int = 1


class SGLangEngine:
    """
    SGLang inference engine with RadixAttention support.
    Optimized for reasoning tasks with prefix caching.
    """
    
    def __init__(self, config: SGLangConfig):
        self.config = config
        self.runtime = None
        self._initialized = False
    
    def initialize(self):
        """Initialize SGLang runtime."""
        if self._initialized:
            return
        
        try:
            import sglang as sgl
            
            self.sgl = sgl
            
            # Set default model
            sgl.set_default_backend(sgl.RuntimeEndpoint(self.config.model_name))
            
            self._initialized = True
            logger.info(f"SGLang engine initialized with {self.config.model_name}")
            
        except ImportError:
            raise ImportError("SGLang not installed. Install with: pip install sglang")
    
    def create_cot_program(self, problem_type: str = "math"):
        """Create SGLang program for Chain-of-Thought generation."""
        self.initialize()
        sgl = self.sgl
        
        if problem_type == "math":
            @sgl.function
            def math_cot(s, problem):
                s += sgl.system("You are a helpful math tutor. Solve problems step by step.")
                s += sgl.user(f"Solve this problem:\n{problem}")
                s += sgl.assistant(sgl.gen("reasoning", max_tokens=self.config.max_tokens))
            
            return math_cot
        else:
            @sgl.function
            def code_cot(s, problem):
                s += sgl.system("You are an expert programmer. Solve coding problems step by step.")
                s += sgl.user(f"Solve this problem:\n{problem}")
                s += sgl.assistant(sgl.gen("solution", max_tokens=self.config.max_tokens))
            
            return code_cot
    
    def generate(
        self,
        problem: str,
        problem_type: str = "math",
    ) -> str:
        """
        Generate reasoning path for a single problem.
        Uses RadixAttention for efficient prefix caching.
        """
        program = self.create_cot_program(problem_type)
        
        state = program.run(
            problem=problem,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        if problem_type == "math":
            return state["reasoning"]
        return state["solution"]
    
    def generate_multiple(
        self,
        problem: str,
        num_paths: int = 4,
        problem_type: str = "math",
    ) -> List[str]:
        """
        Generate multiple reasoning paths.
        RadixAttention caches the shared prefix automatically.
        """
        program = self.create_cot_program(problem_type)
        
        paths = []
        for _ in range(num_paths):
            state = program.run(
                problem=problem,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            if problem_type == "math":
                paths.append(state["reasoning"])
            else:
                paths.append(state["solution"])
        
        return paths
    
    def batch_generate(
        self,
        problems: List[str],
        problem_type: str = "math",
    ) -> List[str]:
        """
        Generate for multiple problems in batch.
        """
        program = self.create_cot_program(problem_type)
        
        # Run in parallel
        states = program.run_batch(
            [{"problem": p} for p in problems],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        key = "reasoning" if problem_type == "math" else "solution"
        return [s[key] for s in states]
    
    def create_custom_program(
        self,
        program_fn: Callable,
    ) -> Callable:
        """
        Create a custom SGLang program.
        
        Args:
            program_fn: Function decorated with @sgl.function
            
        Returns:
            Compiled SGLang program
        """
        self.initialize()
        return self.sgl.function(program_fn)


class SGLangReasoningChain:
    """
    Multi-step reasoning chain using SGLang.
    Each step can branch and backtrack.
    """
    
    def __init__(self, config: SGLangConfig):
        self.config = config
        self.engine = SGLangEngine(config)
    
    def solve_with_verification(
        self,
        problem: str,
        verifier: Callable[[str, str], bool],
        max_attempts: int = 5,
    ) -> Optional[str]:
        """
        Generate solutions and verify until correct.
        
        Args:
            problem: The problem to solve
            verifier: Function that takes (solution, problem) and returns bool
            max_attempts: Maximum generation attempts
            
        Returns:
            Verified solution or None
        """
        for _ in range(max_attempts):
            solution = self.engine.generate(problem)
            
            if verifier(solution, problem):
                return solution
        
        return None
    
    def multi_turn_reasoning(
        self,
        problem: str,
        num_turns: int = 3,
    ) -> str:
        """
        Multi-turn reasoning where model refines its solution.
        """
        self.engine.initialize()
        sgl = self.engine.sgl
        
        @sgl.function
        def multi_turn(s, problem):
            s += sgl.system("You are a helpful assistant. Think step by step.")
            s += sgl.user(f"Problem: {problem}\n\nFirst, understand the problem:")
            s += sgl.assistant(sgl.gen("understanding", max_tokens=256))
            
            s += sgl.user("Now, outline your approach:")
            s += sgl.assistant(sgl.gen("approach", max_tokens=256))
            
            s += sgl.user("Finally, solve the problem and provide the answer:")
            s += sgl.assistant(sgl.gen("solution", max_tokens=512))
        
        state = multi_turn.run(
            problem=problem,
            temperature=self.config.temperature,
        )
        
        return f"""Understanding: {state['understanding']}

Approach: {state['approach']}

Solution: {state['solution']}"""
