"""
Test-Time Compute module for enhanced inference.
Implements Best-of-N sampling, Beam Search, and MCTS for reasoning.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import heapq

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class GeneratedPath:
    """A generated reasoning path with metadata."""
    reasoning: str
    score: float = 0.0
    is_correct: Optional[bool] = None
    final_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestTimeComputeConfig:
    """Configuration for test-time compute."""
    num_samples: int = 16
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 2048
    
    # Scoring
    use_reward_model: bool = True
    reward_model_path: Optional[str] = None
    use_verifier: bool = True
    
    # Aggregation
    aggregation_method: str = "best"  # best, majority_vote, weighted_vote
    
    # Search
    beam_width: int = 4
    max_depth: int = 10


class TestTimeCompute:
    """
    Test-Time Compute harness for reasoning models.
    Generates multiple paths and selects the best one.
    """
    
    def __init__(
        self,
        model_name: str,
        config: TestTimeComputeConfig,
        verifier_type: str = "math",
    ):
        self.model_name = model_name
        self.config = config
        self.verifier_type = verifier_type
        
        self.generator = None
        self.reward_model = None
        self.verifier = None
    
    def setup(self):
        """Initialize all components."""
        try:
            from data_generator import CoTGenerator, GenerationConfig
        except ImportError:
            from ..data_generator import CoTGenerator, GenerationConfig
        
        gen_config = GenerationConfig(
            model_name=self.model_name,
            num_paths=self.config.num_samples,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        self.generator = CoTGenerator(gen_config)
        self.generator.initialize()
        
        if self.config.use_reward_model and self.config.reward_model_path:
            try:
                from training import RewardModel, RewardModelConfig
            except ImportError:
                from ..training import RewardModel, RewardModelConfig
            rm_config = RewardModelConfig(model_name=self.model_name)
            self.reward_model = RewardModel(rm_config)
            self.reward_model.load(self.config.reward_model_path)
        
        if self.config.use_verifier:
            if self.verifier_type == "math":
                try:
                    from verifier import GSM8KVerifier
                except ImportError:
                    from ..verifier import GSM8KVerifier
                self.verifier = GSM8KVerifier()
            else:
                try:
                    from verifier import HumanEvalVerifier
                except ImportError:
                    from ..verifier import HumanEvalVerifier
                self.verifier = HumanEvalVerifier()
        
        logger.info("Test-time compute initialized")
    
    def generate_paths(self, problem: str) -> List[GeneratedPath]:
        """Generate multiple reasoning paths."""
        if self.generator is None:
            self.setup()
        
        paths = self.generator.generate_single(
            problem=problem,
            problem_id="ttc",
            problem_type=self.verifier_type,
        )
        
        generated = []
        for path in paths:
            generated.append(GeneratedPath(
                reasoning=path.reasoning,
                final_answer=path.final_answer,
            ))
        
        return generated
    
    def score_paths(
        self,
        problem: str,
        paths: List[GeneratedPath],
        expected_answer: Optional[str] = None,
    ) -> List[GeneratedPath]:
        """Score paths using reward model and/or verifier."""
        for path in paths:
            score = 0.0
            
            # Reward model scoring
            if self.reward_model is not None:
                rm_score = self.reward_model.compute_reward(problem, path.reasoning)
                score += rm_score
                path.metadata["reward_model_score"] = rm_score
            
            # Verifier scoring
            if self.verifier is not None and expected_answer is not None:
                if self.verifier_type == "math":
                    try:
                        from verifier import VerificationStatus
                    except ImportError:
                        from ..verifier import VerificationStatus
                    result = self.verifier.verify_reasoning_path(
                        path.reasoning,
                        expected_answer,
                    )
                    path.is_correct = result.status == VerificationStatus.CORRECT
                    path.final_answer = result.predicted
                    # Boost score for correct answers
                    if path.is_correct:
                        score += 10.0
                    path.metadata["verification_confidence"] = result.confidence
                else:
                    try:
                        from verifier import ExecutionStatus
                    except ImportError:
                        from ..verifier import ExecutionStatus
                    code = self.verifier.extract_code(path.reasoning)
                    path.final_answer = code
            
            path.score = score
        
        return paths
    
    def select_best(
        self,
        paths: List[GeneratedPath],
        method: Optional[str] = None,
    ) -> GeneratedPath:
        """Select the best path based on aggregation method."""
        method = method or self.config.aggregation_method
        
        if method == "best":
            return max(paths, key=lambda p: p.score)
        
        elif method == "majority_vote":
            # Vote by final answer
            from collections import Counter
            answers = [p.final_answer for p in paths if p.final_answer]
            if not answers:
                return max(paths, key=lambda p: p.score)
            
            most_common = Counter(answers).most_common(1)[0][0]
            for path in paths:
                if path.final_answer == most_common:
                    return path
            return paths[0]
        
        elif method == "weighted_vote":
            # Weighted vote by score
            from collections import defaultdict
            answer_scores = defaultdict(float)
            answer_paths = {}
            
            for path in paths:
                if path.final_answer:
                    answer_scores[path.final_answer] += path.score
                    if path.final_answer not in answer_paths or \
                       path.score > answer_paths[path.final_answer].score:
                        answer_paths[path.final_answer] = path
            
            if not answer_scores:
                return max(paths, key=lambda p: p.score)
            
            best_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
            return answer_paths[best_answer]
        
        else:
            return max(paths, key=lambda p: p.score)
    
    def solve(
        self,
        problem: str,
        expected_answer: Optional[str] = None,
    ) -> Tuple[GeneratedPath, List[GeneratedPath]]:
        """
        Solve a problem using test-time compute.
        
        Args:
            problem: The problem to solve
            expected_answer: Optional ground truth for verification
            
        Returns:
            Tuple of (best_path, all_paths)
        """
        paths = self.generate_paths(problem)
        paths = self.score_paths(problem, paths, expected_answer)
        best = self.select_best(paths)
        
        return best, paths


class BestOfNSampler:
    """
    Simple Best-of-N sampling strategy.
    Generates N samples and picks the best one.
    """
    
    def __init__(
        self,
        model_name: str,
        n: int = 16,
        reward_model_path: Optional[str] = None,
    ):
        config = TestTimeComputeConfig(
            num_samples=n,
            use_reward_model=reward_model_path is not None,
            reward_model_path=reward_model_path,
        )
        self.ttc = TestTimeCompute(model_name, config)
    
    def sample(
        self,
        problem: str,
        expected_answer: Optional[str] = None,
    ) -> GeneratedPath:
        """Sample N paths and return the best."""
        best, _ = self.ttc.solve(problem, expected_answer)
        return best


class BeamSearchReasoner:
    """
    Beam search for step-by-step reasoning.
    Maintains top-k partial solutions at each step.
    """
    
    def __init__(
        self,
        model_name: str,
        beam_width: int = 4,
        max_steps: int = 10,
        step_tokens: int = 128,
    ):
        self.model_name = model_name
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.step_tokens = step_tokens
        
        self.model = None
        self.tokenizer = None
        self.reward_model = None
    
    def setup(self):
        """Initialize model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    def generate_step(
        self,
        prefix: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        """Generate next step candidates."""
        if self.model is None:
            self.setup()
        
        inputs = self.tokenizer(
            prefix,
            return_tensors="pt",
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.step_tokens,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        candidates = []
        for i, seq in enumerate(outputs.sequences):
            text = self.tokenizer.decode(seq[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            # Simple score based on sequence probability
            score = 0.0
            if hasattr(outputs, 'scores') and outputs.scores:
                scores = torch.stack(outputs.scores, dim=1)
                score = scores[i].mean().item()
            candidates.append((text, score))
        
        return candidates
    
    def search(
        self,
        problem: str,
        scorer: Optional[Callable[[str, str], float]] = None,
    ) -> GeneratedPath:
        """
        Perform beam search to find best reasoning path.
        
        Args:
            problem: The problem to solve
            scorer: Optional function to score partial solutions
        """
        # Initialize beam with problem prompt
        beam = [(0.0, problem + "\n")]  # (negative_score, prefix)
        heapq.heapify(beam)
        
        for step in range(self.max_steps):
            candidates = []
            
            for neg_score, prefix in beam:
                # Generate continuations
                steps = self.generate_step(prefix, self.beam_width)
                
                for step_text, step_score in steps:
                    new_prefix = prefix + step_text
                    new_score = -neg_score + step_score
                    
                    if scorer:
                        new_score += scorer(problem, new_prefix)
                    
                    # Check for completion
                    if "####" in step_text or "The answer is" in step_text:
                        return GeneratedPath(
                            reasoning=new_prefix[len(problem)+1:],
                            score=new_score,
                        )
                    
                    candidates.append((-new_score, new_prefix))
            
            # Keep top beam_width candidates
            beam = heapq.nsmallest(self.beam_width, candidates)
        
        # Return best from final beam
        best_score, best_prefix = heapq.heappop(beam)
        return GeneratedPath(
            reasoning=best_prefix[len(problem)+1:],
            score=-best_score,
        )


class MCTSNode:
    """Node in Monte Carlo Tree Search."""
    
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False
    
    def ucb_score(self, exploration: float = 1.41) -> float:
        """Calculate UCB score for selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration_term = exploration * np.sqrt(np.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration_term
    
    def best_child(self) -> 'MCTSNode':
        """Select child with highest UCB score."""
        return max(self.children, key=lambda c: c.ucb_score())
    
    def expand(self, actions: List[str]):
        """Expand node with new children."""
        for action in actions:
            child = MCTSNode(
                state=self.state + action,
                parent=self,
            )
            self.children.append(child)
    
    def backpropagate(self, value: float):
        """Backpropagate value to ancestors."""
        node = self
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent


class MCTSReasoner:
    """
    Monte Carlo Tree Search for reasoning.
    Explores reasoning tree with guided search.
    """
    
    def __init__(
        self,
        model_name: str,
        num_simulations: int = 100,
        max_depth: int = 10,
        exploration: float = 1.41,
    ):
        self.model_name = model_name
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration = exploration
        
        self.generator = None
        self.reward_model = None
    
    def setup(self):
        """Initialize components."""
        try:
            from data_generator import CoTGenerator, GenerationConfig
        except ImportError:
            from ..data_generator import CoTGenerator, GenerationConfig
        
        config = GenerationConfig(
            model_name=self.model_name,
            num_paths=4,
            max_new_tokens=256,
            temperature=0.8,
        )
        self.generator = CoTGenerator(config)
        self.generator.initialize()
    
    def simulate(self, node: MCTSNode, problem: str) -> float:
        """Simulate from node to terminal state."""
        if self.generator is None:
            self.setup()
        
        # Generate completion from current state
        current_state = node.state
        
        paths = self.generator.generate_single(
            problem=current_state,
            problem_id="mcts",
            problem_type="math",
        )
        
        if not paths:
            return 0.0
        
        # Simple reward based on whether answer is found
        reasoning = paths[0].reasoning
        if "####" in reasoning or "answer is" in reasoning.lower():
            return 1.0
        return 0.0
    
    def expand_node(self, node: MCTSNode) -> List[str]:
        """Generate possible next steps."""
        if self.generator is None:
            self.setup()
        
        paths = self.generator.generate_single(
            problem=node.state,
            problem_id="expand",
            problem_type="math",
        )
        
        # Extract first step from each path
        steps = []
        for path in paths:
            lines = path.reasoning.split('\n')
            if lines:
                steps.append(lines[0] + '\n')
        
        return list(set(steps))[:4]  # Limit branching factor
    
    def search(
        self,
        problem: str,
        verifier=None,
        expected_answer: Optional[str] = None,
    ) -> GeneratedPath:
        """
        Perform MCTS to find best reasoning path.
        
        Args:
            problem: The problem to solve
            verifier: Optional verifier for terminal reward
            expected_answer: Ground truth for verification
        """
        root = MCTSNode(state=problem + "\n")
        
        for _ in range(self.num_simulations):
            # Selection
            node = root
            while node.children and not node.is_terminal:
                node = node.best_child()
            
            # Expansion
            if not node.is_terminal and node.visits > 0:
                actions = self.expand_node(node)
                node.expand(actions)
                if node.children:
                    node = node.children[0]
            
            # Simulation
            value = self.simulate(node, problem)
            
            # Backpropagation
            node.backpropagate(value)
        
        # Extract best path
        path_nodes = [root]
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            path_nodes.append(node)
        
        reasoning = "".join(n.state for n in path_nodes[1:])
        
        return GeneratedPath(
            reasoning=reasoning,
            score=root.value / max(root.visits, 1),
        )


class SelfConsistency:
    """
    Self-consistency decoding for reasoning.
    Samples multiple solutions and uses majority voting.
    """
    
    def __init__(
        self,
        model_name: str,
        num_samples: int = 40,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.num_samples = num_samples
        self.temperature = temperature
        
        config = TestTimeComputeConfig(
            num_samples=num_samples,
            temperature=temperature,
            aggregation_method="majority_vote",
        )
        self.ttc = TestTimeCompute(model_name, config)
    
    def solve(
        self,
        problem: str,
        expected_answer: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Solve using self-consistency.
        
        Returns:
            Tuple of (answer, confidence)
        """
        best, paths = self.ttc.solve(problem, expected_answer)
        
        # Calculate confidence as fraction of paths with same answer
        from collections import Counter
        answers = [p.final_answer for p in paths if p.final_answer]
        if not answers:
            return best.final_answer or "", 0.0
        
        counts = Counter(answers)
        most_common, count = counts.most_common(1)[0]
        confidence = count / len(answers)
        
        return most_common, confidence
