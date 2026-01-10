"""
Group Relative Policy Optimization (GRPO) Trainer
Based on DeepSeek-R1 approach - no reward model needed.

Key insight: Sample multiple responses per prompt, use relative ranking
within the group to compute advantages. No separate reward model.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # GRPO specific
    group_size: int = 2  # Number of responses per prompt to sample
    kl_coef: float = 0.1  # KL penalty coefficient
    clip_range: float = 0.2  # PPO-style clipping
    
    # Training
    learning_rate: float = 1e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Sequence length
    max_length: int = 1024
    max_prompt_length: int = 256
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Logging
    logging_steps: int = 10
    output_dir: str = "./grpo_output"
    
    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True


class GRPODataset(Dataset):
    """Dataset for GRPO training with pre-computed groups."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        
        # Group data by prompt
        self.groups = self._create_groups(data)
    
    def _create_groups(self, data: List[Dict]) -> List[Dict]:
        """Group responses by prompt for GRPO."""
        prompt_to_responses = {}
        
        for item in data:
            # Handle DPO format (prompt, chosen, rejected)
            if "chosen" in item and "rejected" in item:
                prompt = item.get("prompt", "")
                if prompt not in prompt_to_responses:
                    prompt_to_responses[prompt] = {
                        "prompt": prompt,
                        "chosen": [],
                        "rejected": [],
                    }
                prompt_to_responses[prompt]["chosen"].append(item["chosen"])
                prompt_to_responses[prompt]["rejected"].append(item["rejected"])
            else:
                # Handle raw sample format (is_correct field)
                prompt = item.get("prompt", item.get("problem", ""))
                
                if prompt not in prompt_to_responses:
                    prompt_to_responses[prompt] = {
                        "prompt": prompt,
                        "chosen": [],
                        "rejected": [],
                    }
                
                # Add to appropriate list based on correctness
                if item.get("is_correct", False):
                    prompt_to_responses[prompt]["chosen"].append(
                        item.get("reasoning", item.get("response", ""))
                    )
                else:
                    prompt_to_responses[prompt]["rejected"].append(
                        item.get("reasoning", item.get("response", ""))
                    )
        
        # Convert to list and filter groups with both chosen and rejected
        groups = []
        for prompt, responses in prompt_to_responses.items():
            if responses["chosen"] and responses["rejected"]:
                groups.append(responses)
        
        logger.info(f"Created {len(groups)} training groups from {len(prompt_to_responses)} prompts")
        return groups
    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        group = self.groups[idx]
        return {
            "prompt": group["prompt"],
            "chosen": group["chosen"],
            "rejected": group["rejected"],
        }


class ReasoningGRPOTrainer:
    """
    GRPO Trainer for reasoning tasks.
    
    Uses group-relative advantages instead of a reward model.
    Responses within a group are ranked, and advantages are computed
    based on relative position.
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
    
    def setup(self):
        """Setup model, tokenizer, and optimizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model: {self.config.model_name}")
        
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
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        self.model.to("cuda")
        
        # Apply LoRA if configured
        if self.config.use_lora:
            self._apply_lora()
        else:
            # Enable gradients for all parameters if not using LoRA
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Create reference model (frozen copy for KL) - load in 8-bit to save memory
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            device_map="auto",
        )
        self.ref_model.eval()
        
        # Setup optimizer - only optimize trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
        )
        
        logger.info(f"Model loaded: {self.config.model_name}")
    
    def _apply_lora(self):
        """Apply LoRA adapters."""
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        self.model.print_trainable_parameters()
    
    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for sequences."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        
        # Gather log probs for actual tokens
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        # Mask padding
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        seq_log_probs = (gathered_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        
        return seq_log_probs
    
    def compute_grpo_loss(
        self,
        prompt: str,
        chosen_responses: List[str],
        rejected_responses: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss for a group of responses.
        
        1. Sample responses from both chosen and rejected
        2. Compute advantages based on correctness (chosen=+1, rejected=-1)
        3. Apply PPO-style clipped objective with KL penalty
        """
        device = next(self.model.parameters()).device
        
        # Combine responses with labels
        all_responses = []
        advantages = []
        
        # Take up to group_size/2 from each
        n_each = self.config.group_size // 2
        
        for resp in chosen_responses[:n_each]:
            all_responses.append(resp)
            advantages.append(1.0)  # Positive advantage for correct
        
        for resp in rejected_responses[:n_each]:
            all_responses.append(resp)
            advantages.append(-1.0)  # Negative advantage for incorrect
        
        if not all_responses:
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        
        advantages = torch.tensor(advantages, device=device)
        
        # Normalize advantages (GRPO key insight)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Tokenize all sequences
        sequences = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for resp in all_responses
        ]
        
        encodings = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(device)
        
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        # Compute current policy log probs
        policy_log_probs = self.compute_log_probs(
            self.model,
            input_ids,
            attention_mask,
            input_ids,
        )
        
        # Compute reference log probs
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(
                self.ref_model,
                input_ids,
                attention_mask,
                input_ids,
            )
        
        # Compute ratio and clipped objective
        log_ratio = policy_log_probs - ref_log_probs
        ratio = torch.exp(log_ratio)
        
        # PPO clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # KL penalty
        kl_div = (ref_log_probs - policy_log_probs).mean()
        
        # Total loss
        loss = policy_loss + self.config.kl_coef * kl_div
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "ratio_mean": ratio.mean().item(),
            "advantage_mean": advantages.mean().item(),
        }
        
        return loss, metrics
    
    def train(self, data: List[Dict]):
        """
        Train with GRPO.
        
        Args:
            data: List of dicts with reasoning paths (correct and incorrect)
        """
        self.setup()
        
        # Create dataset
        dataset = GRPODataset(
            data,
            self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        
        logger.info(f"Training on {len(dataset)} prompt groups")
        
        if len(dataset) == 0:
            logger.warning("No training data! Check data format.")
            return
        
        # Training loop
        self.model.train()
        self.optimizer.zero_grad()
        global_step = 0
        total_loss = 0
        
        # Setup training logger for dynamics visualization
        from pathlib import Path as PathLib
        log_dir = PathLib(self.config.output_dir) / "training_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = log_dir / "training_metrics.jsonl"
        
        from tqdm import tqdm
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            epoch_loss = 0
            
            pbar = tqdm(range(0, len(dataset), self.config.batch_size), desc=f"Epoch {epoch+1}")
            for batch_idx in pbar:
                batch_loss = 0
                batch_metrics = {"policy_loss": 0, "kl_div": 0, "ratio_mean": 0, "advantage_mean": 0}
                
                for i in range(min(self.config.batch_size, len(dataset) - batch_idx)):
                    group = dataset[batch_idx + i]
                    
                    loss, metrics = self.compute_grpo_loss(
                        group["prompt"],
                        group["chosen"],
                        group["rejected"],
                    )
                    
                    # Accumulate gradients
                    scaled_loss = loss / self.config.gradient_accumulation_steps
                    scaled_loss.backward()
                    
                    batch_loss += loss.item()
                    for k, v in metrics.items():
                        batch_metrics[k] = batch_metrics.get(k, 0) + v
                
                epoch_loss += batch_loss
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})
                
                # Gradient step
                if (batch_idx // self.config.batch_size + 1) % self.config.gradient_accumulation_steps == 0:
                    # Compute gradient norm before clipping
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = total_norm ** 0.5
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Log training metrics
                    num_items = min(self.config.batch_size, len(dataset) - batch_idx)
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + (batch_idx / len(dataset)),
                        "loss": batch_loss / max(num_items, 1),
                        "policy_loss": batch_metrics.get("policy_loss", 0) / max(num_items, 1),
                        "kl_divergence": batch_metrics.get("kl_div", 0) / max(num_items, 1),
                        "reward_margin": batch_metrics.get("advantage_mean", 0) / max(num_items, 1),
                        "gradient_norm": grad_norm,
                        "ratio_mean": batch_metrics.get("ratio_mean", 1.0) / max(num_items, 1),
                        "learning_rate": self.config.learning_rate,
                    }
                    
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                
                total_loss += batch_loss
            
            avg_epoch_loss = epoch_loss / len(dataset)
            logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        logger.info(f"Training logs saved to: {log_dir}")
        
        # Save model
        self.save()
        
        avg_loss = total_loss / max(len(dataset), 1)
        logger.info(f"Training complete. Average loss: {avg_loss:.4f}")
    
    def save(self, path: Optional[str] = None):
        """Save the model."""
        save_path = path or self.config.output_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        if self.config.use_lora:
            # Merge LoRA weights and save full model
            logger.info("Merging LoRA weights...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")


def train_grpo_from_synthetic_data(
    data_path: str,
    output_dir: str = "./grpo_output",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_epochs: int = 1,
    batch_size: int = 2,
):
    """
    Convenience function to train GRPO from synthetic data file.
    """
    # Load data
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} samples from {data_path}")
    
    # Setup config
    config = GRPOConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    
    # Train
    trainer = ReasoningGRPOTrainer(config)
    trainer.train(data)
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with GRPO")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./grpo_output")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    train_grpo_from_synthetic_data(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )