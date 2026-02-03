"""
Reward Models for reasoning evaluation.
Supports both outcome reward models and process reward models.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration for reward model."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for faster training
    
    # Architecture
    num_labels: int = 1
    hidden_size: int = 4096
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Sequence length
    max_length: int = 2048
    
    # Output
    output_dir: str = "./reward_model"


class RewardHead(nn.Module):
    """Reward prediction head."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use last token representation
        hidden_states = hidden_states[:, -1, :]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        reward = self.out_proj(hidden_states)
        return reward.squeeze(-1)


class RewardModel(nn.Module):
    """
    Outcome Reward Model for scoring complete reasoning paths.
    Predicts a scalar reward for the entire response.
    """
    
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        self.backbone = None
        self.reward_head = None
        self.tokenizer = None
        
    def setup(self):
        """Initialize model components."""
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.backbone = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        self.reward_head = RewardHead(
            hidden_size=self.backbone.config.hidden_size,
            dropout=self.config.dropout,
        )
        # Match reward head dtype to backbone
        self.reward_head = self.reward_head.half()
        
        logger.info(f"Reward model initialized with {self.config.model_name}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to compute reward."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        reward = self.reward_head(hidden_states)
        return reward
    
    def compute_reward(
        self,
        prompt: str,
        response: str,
    ) -> float:
        """
        Compute reward for a prompt-response pair.
        
        Args:
            prompt: The input prompt
            response: The generated response
            
        Returns:
            Scalar reward value
        """
        if self.backbone is None:
            self.setup()
        
        text = f"{prompt}\n{response}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            reward = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
            )
        
        return reward.item()
    
    def score_batch(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """Score a batch of responses."""
        if self.backbone is None:
            self.setup()
        
        texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        )
        
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            rewards = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
            )
        
        return rewards.tolist()
    
    def train_model(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Train the reward model on preference data.
        
        Args:
            train_data: List of dicts with 'prompt', 'chosen', 'rejected'
            eval_data: Optional evaluation data
        """
        if self.backbone is None:
            self.setup()
        
        from transformers import Trainer, TrainingArguments
        
        # Create dataset
        class RewardDataset(Dataset):
            def __init__(inner_self, data, tokenizer, max_length):
                inner_self.data = data
                inner_self.tokenizer = tokenizer
                inner_self.max_length = max_length
            
            def __len__(inner_self):
                return len(inner_self.data)
            
            def __getitem__(inner_self, idx):
                item = inner_self.data[idx]
                
                chosen_text = f"{item['prompt']}\n{item['chosen']}"
                rejected_text = f"{item['prompt']}\n{item['rejected']}"
                
                chosen_tokens = inner_self.tokenizer(
                    chosen_text,
                    truncation=True,
                    max_length=inner_self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                rejected_tokens = inner_self.tokenizer(
                    rejected_text,
                    truncation=True,
                    max_length=inner_self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                return {
                    "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
                    "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
                    "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
                    "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
                }
        
        train_dataset = RewardDataset(train_data, self.tokenizer, self.config.max_length)
        eval_dataset = RewardDataset(eval_data, self.tokenizer, self.config.max_length) if eval_data else None
        
        # Custom training loop with pairwise ranking loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        for epoch in range(self.config.num_epochs):
            self.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Compute rewards for chosen and rejected
                chosen_rewards = self.forward(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                )
                rejected_rewards = self.forward(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                )
                
                # Pairwise ranking loss
                loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")
        
        # Save model
        self.save(self.config.output_dir)
    
    def save(self, path: str):
        """Save the reward model."""
        import os
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.state_dict(), f"{path}/reward_model.pt")
        self.tokenizer.save_pretrained(path)
        
        # Save config
        import json
        with open(f"{path}/config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        logger.info(f"Reward model saved to {path}")
    
    def load(self, path: str):
        """Load a saved reward model."""
        self.setup()
        self.load_state_dict(torch.load(f"{path}/reward_model.pt"))
        logger.info(f"Reward model loaded from {path}")


class ProcessRewardModel(RewardModel):
    """
    Process Reward Model (PRM) for scoring intermediate reasoning steps.
    Provides step-level rewards instead of just final outcome.
    """
    
    STEP_SEPARATORS = ["\n", "Step", "Therefore", "So,", "Thus,"]
    
    def __init__(self, config: RewardModelConfig):
        super().__init__(config)
        self.step_reward_head = None
    
    def setup(self):
        """Initialize with additional step-level head."""
        super().setup()
        
        self.step_reward_head = RewardHead(
            hidden_size=self.backbone.config.hidden_size,
            dropout=self.config.dropout,
        )
    
    def forward_steps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_positions: List[List[int]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with step-level rewards.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            step_positions: List of step end positions for each sample
            
        Returns:
            Tuple of (final_reward, list of step_rewards per sample)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        
        # Final reward
        final_reward = self.reward_head(hidden_states)
        
        # Step rewards
        batch_step_rewards = []
        for i, positions in enumerate(step_positions):
            step_rewards = []
            for pos in positions:
                if pos < hidden_states.shape[1]:
                    step_hidden = hidden_states[i, pos, :].unsqueeze(0).unsqueeze(0)
                    step_reward = self.step_reward_head(step_hidden)
                    step_rewards.append(step_reward.item())
            batch_step_rewards.append(step_rewards)
        
        return final_reward, batch_step_rewards
    
    def identify_steps(self, text: str) -> List[int]:
        """
        Identify step boundaries in reasoning text.
        Returns token positions where steps end.
        """
        # Simple heuristic: split by newlines or step markers
        import re
        
        positions = []
        current_pos = 0
        
        for separator in self.STEP_SEPARATORS:
            for match in re.finditer(re.escape(separator), text):
                positions.append(match.start())
        
        positions = sorted(set(positions))
        return positions
    
    def compute_step_rewards(
        self,
        prompt: str,
        response: str,
    ) -> Tuple[float, List[float]]:
        """
        Compute rewards for each step in the response.
        
        Args:
            prompt: The input prompt
            response: The generated response with steps
            
        Returns:
            Tuple of (final_reward, list of step_rewards)
        """
        if self.backbone is None:
            self.setup()
        
        text = f"{prompt}\n{response}"
        
        # Identify step boundaries
        step_char_positions = self.identify_steps(response)
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        
        # Convert character positions to token positions
        prompt_len = len(prompt) + 1  # +1 for newline
        step_token_positions = []
        
        for char_pos in step_char_positions:
            adjusted_pos = prompt_len + char_pos
            # Find token position (approximate)
            token_pos = len(self.tokenizer.encode(text[:adjusted_pos]))
            step_token_positions.append(token_pos)
        
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            final_reward, step_rewards = self.forward_steps(
                inputs["input_ids"],
                inputs["attention_mask"],
                [step_token_positions],
            )
        
        return final_reward.item(), step_rewards[0]
    
    def get_best_prefix(
        self,
        prompt: str,
        response: str,
        threshold: float = 0.5,
    ) -> Tuple[str, float]:
        """
        Find the best prefix of the response (where to continue from).
        Useful for guiding search during generation.
        
        Args:
            prompt: The input prompt
            response: The generated response
            threshold: Minimum reward threshold
            
        Returns:
            Tuple of (best_prefix, reward_at_prefix)
        """
        final_reward, step_rewards = self.compute_step_rewards(prompt, response)
        
        if not step_rewards:
            return response, final_reward
        
        # Find last step above threshold
        step_positions = self.identify_steps(response)
        
        best_idx = -1
        best_reward = float('-inf')
        
        for i, (pos, reward) in enumerate(zip(step_positions, step_rewards)):
            if reward >= threshold and reward > best_reward:
                best_idx = i
                best_reward = reward
        
        if best_idx >= 0 and best_idx < len(step_positions):
            return response[:step_positions[best_idx]], best_reward
        
        return response, final_reward
