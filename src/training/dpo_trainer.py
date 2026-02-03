"""
DPO (Direct Preference Optimization) Trainer for reasoning models.
Implements preference learning from verified reasoning paths.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainerConfig:
    """Configuration for DPO training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    ref_model_name: Optional[str] = None  # If None, uses same as model_name
    
    # DPO parameters
    beta: float = 0.1
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    label_smoothing: float = 0.0
    
    # Training
    learning_rate: float = 1e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Sequence lengths
    max_length: int = 2048
    max_prompt_length: int = 512
    
    # LoRA (optional)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./dpo_output"
    
    # Hardware
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True


class DPODataset(Dataset):
    """Dataset for DPO training with preference pairs."""
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=True,
        )
        
        # Tokenize chosen response
        chosen_tokens = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.max_length - len(prompt_tokens["input_ids"]),
            add_special_tokens=False,
        )
        
        # Tokenize rejected response
        rejected_tokens = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.max_length - len(prompt_tokens["input_ids"]),
            add_special_tokens=False,
        )
        
        # Combine prompt + response
        chosen_input_ids = prompt_tokens["input_ids"] + chosen_tokens["input_ids"]
        rejected_input_ids = prompt_tokens["input_ids"] + rejected_tokens["input_ids"]
        
        # Create attention masks
        chosen_attention_mask = [1] * len(chosen_input_ids)
        rejected_attention_mask = [1] * len(rejected_input_ids)
        
        # Create labels (mask prompt portion)
        prompt_len = len(prompt_tokens["input_ids"])
        chosen_labels = [-100] * prompt_len + chosen_tokens["input_ids"]
        rejected_labels = [-100] * prompt_len + rejected_tokens["input_ids"]
        
        return {
            "prompt_input_ids": torch.tensor(prompt_tokens["input_ids"]),
            "chosen_input_ids": torch.tensor(chosen_input_ids),
            "chosen_attention_mask": torch.tensor(chosen_attention_mask),
            "chosen_labels": torch.tensor(chosen_labels),
            "rejected_input_ids": torch.tensor(rejected_input_ids),
            "rejected_attention_mask": torch.tensor(rejected_attention_mask),
            "rejected_labels": torch.tensor(rejected_labels),
        }


class ReasoningDPOTrainer:
    """
    DPO Trainer specialized for reasoning model training.
    Uses TRL's DPOTrainer under the hood with custom configurations.
    """
    
    def __init__(self, config: DPOTrainerConfig):
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup(self):
        """Setup model, tokenizer, and trainer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
        }
        
        if self.config.use_lora:
            # Load in 8-bit for LoRA
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        
        # Apply LoRA if configured
        if self.config.use_lora:
            self._apply_lora()
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded: {self.config.model_name}")
    
    def _apply_lora(self):
        """Apply LoRA adapters to model."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        except ImportError:
            raise ImportError("PEFT required for LoRA. Install with: pip install peft")
    
    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Train the model using DPO.
        
        Args:
            train_data: List of dicts with 'prompt', 'chosen', 'rejected' keys
            eval_data: Optional evaluation data in same format
        """
        try:
            from trl import DPOTrainer, DPOConfig
            from datasets import Dataset as HFDataset
            
            self.setup()
            
            # Create HuggingFace datasets
            train_dataset = HFDataset.from_list(train_data)
            eval_dataset = HFDataset.from_list(eval_data) if eval_data else None
            
            # DPO training config
            training_args = DPOConfig(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                max_grad_norm=self.config.max_grad_norm,
                logging_steps=self.config.logging_steps,
                eval_steps=self.config.eval_steps if eval_data else None,
                save_steps=self.config.save_steps,
                eval_strategy="steps" if eval_data else "no",
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                beta=self.config.beta,
                loss_type=self.config.loss_type,
                max_length=self.config.max_length,
                max_prompt_length=self.config.max_prompt_length,
                remove_unused_columns=False,
            )
            
            # Create DPO trainer
            self.trainer = DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self.tokenizer,
            )
            
            # Train
            logger.info("Starting DPO training...")
            self.trainer.train()
            
            # Save final model
            self.save()
            
            logger.info(f"Training complete. Model saved to {self.config.output_dir}")
            
        except ImportError:
            raise ImportError("TRL required for DPO training. Install with: pip install trl")
    
    def save(self, path: Optional[str] = None):
        """Save the trained model."""
        save_path = path or self.config.output_dir
        
        if self.config.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(save_path)
        else:
            # Save full model
            self.model.save_pretrained(save_path)
        
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: str):
        """Load a trained model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        if self.config.use_lora:
            from peft import PeftModel
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base_model, path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                device_map="auto",
            )


class RejectionSamplingDPO:
    """
    Rejection Sampling + DPO training pipeline.
    Generates samples, filters by reward, then trains with DPO.
    """
    
    def __init__(
        self,
        dpo_config: DPOTrainerConfig,
        generator_config: Optional[Dict[str, Any]] = None,
        verifier_type: str = "math",
    ):
        self.dpo_config = dpo_config
        self.generator_config = generator_config or {}
        self.verifier_type = verifier_type
        
        self.dpo_trainer = ReasoningDPOTrainer(dpo_config)
        self.generator = None
        self.verifier = None
    
    def setup(self):
        """Setup all components."""
        try:
            from data_generator import CoTGenerator, GenerationConfig
            from verifier import MathVerifier, CodeVerifier
        except ImportError:
            from ..data_generator import CoTGenerator, GenerationConfig
            from ..verifier import MathVerifier, CodeVerifier
        
        # Setup generator
        gen_config = GenerationConfig(
            model_name=self.generator_config.get("model_name", self.dpo_config.model_name),
            num_paths=self.generator_config.get("num_paths", 8),
            temperature=self.generator_config.get("temperature", 0.8),
        )
        self.generator = CoTGenerator(gen_config)
        
        # Setup verifier
        if self.verifier_type == "math":
            self.verifier = MathVerifier()
        else:
            self.verifier = CodeVerifier()
        
        logger.info("Rejection Sampling DPO setup complete")
    
    def generate_preference_data(
        self,
        problems: List[Dict[str, str]],
        num_samples_per_problem: int = 8,
    ) -> List[Dict[str, str]]:
        """
        Generate preference pairs through rejection sampling.
        
        Args:
            problems: List of problems with 'problem' and 'answer' keys
            num_samples_per_problem: Number of samples to generate per problem
            
        Returns:
            List of preference pairs for DPO training
        """
        self.setup()
        self.generator.initialize()
        
        preference_pairs = []
        
        for prob in problems:
            # Generate multiple solutions
            paths = self.generator.generate_single(
                problem=prob["problem"],
                problem_id=prob.get("id", "unknown"),
                problem_type=self.verifier_type,
            )
            
            # Verify each path
            correct_paths = []
            incorrect_paths = []
            
            for path in paths:
                if self.verifier_type == "math":
                    try:
                        from verifier import VerificationStatus
                    except ImportError:
                        from ..verifier import VerificationStatus
                    result = self.verifier.verify_reasoning_path(
                        path.reasoning,
                        prob["answer"],
                    )
                    is_correct = result.status == VerificationStatus.CORRECT
                else:
                    try:
                        from verifier import ExecutionStatus
                    except ImportError:
                        from ..verifier import ExecutionStatus
                    code = self.verifier.extract_code(path.reasoning)
                    result = self.verifier.verify_function_output(
                        code, "python", prob.get("function_call", ""),
                        prob["answer"],
                    )
                    is_correct = result.status == ExecutionStatus.SUCCESS
                
                if is_correct:
                    correct_paths.append(path.reasoning)
                else:
                    incorrect_paths.append(path.reasoning)
            
            # Create pairs
            if correct_paths and incorrect_paths:
                for chosen in correct_paths:
                    for rejected in incorrect_paths:
                        preference_pairs.append({
                            "prompt": prob["problem"],
                            "chosen": chosen,
                            "rejected": rejected,
                        })
        
        logger.info(f"Generated {len(preference_pairs)} preference pairs")
        return preference_pairs
    
    def train(
        self,
        problems: List[Dict[str, str]],
        eval_problems: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Full rejection sampling + DPO training pipeline.
        
        Args:
            problems: Training problems
            eval_problems: Optional evaluation problems
        """
        # Generate training data
        train_data = self.generate_preference_data(problems)
        
        # Generate eval data if provided
        eval_data = None
        if eval_problems:
            eval_data = self.generate_preference_data(eval_problems)
        
        # Train with DPO
        self.dpo_trainer.train(train_data, eval_data)
    
    def iterative_train(
        self,
        problems: List[Dict[str, str]],
        num_iterations: int = 3,
        samples_per_iteration: int = 1000,
    ):
        """
        Iterative rejection sampling + DPO training.
        Each iteration uses the current model to generate new data.
        
        Args:
            problems: All available problems
            num_iterations: Number of training iterations
            samples_per_iteration: Problems to sample each iteration
        """
        import random
        
        for iteration in range(num_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")
            
            # Sample problems for this iteration
            sampled = random.sample(problems, min(samples_per_iteration, len(problems)))
            
            # Generate preference data with current model
            if iteration > 0:
                # Update generator to use current model
                self.generator.config.model_name = self.dpo_config.output_dir
                self.generator._initialized = False
            
            train_data = self.generate_preference_data(sampled)
            
            # Train
            self.dpo_trainer.train(train_data)
            
            # Update output dir for next iteration
            self.dpo_config.output_dir = f"{self.dpo_config.output_dir}_iter{iteration + 1}"
            self.dpo_trainer = ReasoningDPOTrainer(self.dpo_config)
            
            logger.info(f"Iteration {iteration + 1} complete")
