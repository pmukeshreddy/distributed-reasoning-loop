"""
Supervised Fine-Tuning (SFT) Trainer for initial model training.
Trains on verified correct reasoning paths before DPO.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainerConfig:
    """Configuration for SFT training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Training
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Sequence length
    max_length: int = 2048
    
    # LoRA (optional)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./sft_output"
    
    # Hardware
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    # Data
    packing: bool = False  # Pack multiple samples into one sequence


class ReasoningSFTTrainer:
    """
    SFT Trainer for training on correct reasoning paths.
    Can be used as initial training before DPO/PPO.
    """
    
    def __init__(self, config: SFTTrainerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup(self):
        """Setup model and tokenizer."""
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
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        
        if self.config.use_lora:
            self._apply_lora()
        
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded: {self.config.model_name}")
    
    def _apply_lora(self):
        """Apply LoRA adapters."""
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
            raise ImportError("PEFT required for LoRA")
    
    def format_example(self, example: Dict[str, str]) -> str:
        """Format a single example for training."""
        prompt = example.get("prompt", example.get("problem", ""))
        response = example.get("response", example.get("reasoning", ""))
        
        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        
        return f"User: {prompt}\n\nAssistant: {response}"
    
    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Train the model with SFT.
        
        Args:
            train_data: List of dicts with 'prompt' and 'response' keys
            eval_data: Optional evaluation data
        """
        try:
            from trl import SFTTrainer, SFTConfig
            from datasets import Dataset as HFDataset
            
            self.setup()
            
            # Format data
            def format_func(examples):
                return {
                    "text": [
                        self.format_example({"prompt": p, "response": r})
                        for p, r in zip(examples["prompt"], examples["response"])
                    ]
                }
            
            # Convert to list of dicts with consistent keys
            formatted_train = []
            for item in train_data:
                formatted_train.append({
                    "prompt": item.get("prompt", item.get("problem", "")),
                    "completion": item.get("response", item.get("reasoning", "")),
                })
            
            train_dataset = HFDataset.from_list(formatted_train)
            
            eval_dataset = None
            if eval_data:
                formatted_eval = []
                for item in eval_data:
                    formatted_eval.append({
                        "prompt": item.get("prompt", item.get("problem", "")),
                        "completion": item.get("response", item.get("reasoning", "")),
                    })
                eval_dataset = HFDataset.from_list(formatted_eval)
            
            # Training config
            training_args = SFTConfig(
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
                fp16=False,
                bf16=True,
                max_length=self.config.max_length,
                packing=self.config.packing,
            )
            
            # Create trainer
            self.trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self.tokenizer,
                
            )
            
            # Train
            logger.info("Starting SFT training...")
            self.trainer.train()
            
            # Save
            self.save()
            
            logger.info(f"Training complete. Model saved to {self.config.output_dir}")
            
        except ImportError:
            raise ImportError("TRL required for SFT training")
    
    def save(self, path: Optional[str] = None):
        """Save the model."""
        save_path = path or self.config.output_dir
        
        if self.config.use_lora:
            self.model.save_pretrained(save_path)
        else:
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


class SFTFromSyntheticData:
    """
    Pipeline for SFT training from synthetic reasoning data.
    Filters correct paths and trains the model.
    """
    
    def __init__(
        self,
        sft_config: SFTTrainerConfig,
        data_path: str,
    ):
        self.sft_config = sft_config
        self.data_path = data_path
        self.trainer = ReasoningSFTTrainer(sft_config)
    
    def load_data(self) -> List[Dict[str, str]]:
        """Load and filter correct reasoning paths."""
        import json
        
        data = []
        data_file = Path(self.data_path)
        
        if data_file.suffix == ".jsonl":
            with open(data_file) as f:
                for line in f:
                    item = json.loads(line)
                    if item.get("is_correct", True):
                        data.append({
                            "prompt": item.get("problem", item.get("prompt", "")),
                            "completion": item.get("reasoning", item.get("response", "")),
                        })
        elif data_file.suffix == ".json":
            with open(data_file) as f:
                items = json.load(f)
                for item in items:
                    if item.get("is_correct", True):
                        data.append({
                            "prompt": item.get("problem", item.get("prompt", "")),
                            "completion": item.get("reasoning", item.get("response", "")),
                        })
        
        logger.info(f"Loaded {len(data)} correct reasoning paths")
        return data
    
    def train(self, eval_split: float = 0.1):
        """Train on the synthetic data."""
        data = self.load_data()
        
        # Split train/eval
        split_idx = int(len(data) * (1 - eval_split))
        train_data = data[:split_idx]
        eval_data = data[split_idx:] if eval_split > 0 else None
        
        self.trainer.train(train_data, eval_data)
