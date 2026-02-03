"""
LoRA Format Converter for SGLang Hot-Reload.

PEFT saves LoRA adapters with names like:
    base_model.model.model.layers.X.self_attn.q_proj.lora_A.weight

SGLang's /update_weights_from_disk expects delta weights with names like:
    model.layers.X.self_attn.q_proj.weight

This module converts between formats for true hot-reload.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def convert_peft_to_sglang_delta(
    peft_lora_path: str,
    output_path: str,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> str:
    """
    Convert PEFT LoRA adapter to SGLang-compatible delta weights.
    
    SGLang's /update_weights_from_disk expects the weights to be in the same
    format as the base model, with the LoRA matrices already multiplied together
    (delta = lora_B @ lora_A * scaling).
    
    Args:
        peft_lora_path: Path to PEFT LoRA checkpoint
        output_path: Where to save converted weights
        base_model_name: Base model for reference
        
    Returns:
        Path to converted checkpoint
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    
    logger.info(f"Converting PEFT LoRA from {peft_lora_path} to SGLang format...")
    
    # Load adapter config
    config_path = os.path.join(peft_lora_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"No adapter_config.json found in {peft_lora_path}")
    
    with open(config_path) as f:
        adapter_config = json.load(f)
    
    # Get LoRA parameters
    lora_alpha = adapter_config.get("lora_alpha", 16)
    lora_r = adapter_config.get("r", 8)
    scaling = lora_alpha / lora_r
    
    logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, scaling={scaling}")
    
    # Load PEFT weights
    peft_weights = {}
    for filename in os.listdir(peft_lora_path):
        if filename.endswith(".safetensors"):
            filepath = os.path.join(peft_lora_path, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    peft_weights[key] = f.get_tensor(key)
    
    if not peft_weights:
        # Try .bin format
        for filename in os.listdir(peft_lora_path):
            if filename.endswith(".bin"):
                filepath = os.path.join(peft_lora_path, filename)
                peft_weights.update(torch.load(filepath, map_location="cpu"))
    
    logger.info(f"Loaded {len(peft_weights)} PEFT weight tensors")
    
    # Group lora_A and lora_B pairs
    lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    
    for key, tensor in peft_weights.items():
        # PEFT format: base_model.model.model.layers.X.self_attn.q_proj.lora_A.weight
        # We need to extract: model.layers.X.self_attn.q_proj
        
        if ".lora_A." in key:
            # Extract the base parameter name
            base_key = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["A"] = tensor
            
        elif ".lora_B." in key:
            base_key = key.replace("base_model.model.", "").replace(".lora_B.weight", ".weight")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["B"] = tensor
    
    logger.info(f"Found {len(lora_pairs)} LoRA parameter pairs")
    
    # Compute delta weights: delta = lora_B @ lora_A * scaling
    delta_weights = {}
    
    for param_name, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            logger.warning(f"Incomplete pair for {param_name}, skipping")
            continue
        
        lora_A = pair["A"]  # Shape: (r, in_features)
        lora_B = pair["B"]  # Shape: (out_features, r)
        
        # Compute delta: (out_features, in_features)
        delta = (lora_B @ lora_A) * scaling
        delta_weights[param_name] = delta
        
        logger.debug(f"  {param_name}: delta shape {delta.shape}")
    
    logger.info(f"Computed {len(delta_weights)} delta weight matrices")
    
    # Save in safetensors format
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_path, "model.safetensors")
    
    save_file(delta_weights, output_file)
    logger.info(f"Saved delta weights to {output_file}")
    
    # Also save a marker file so SGLang knows this is a delta update
    marker = {
        "format": "sglang_delta",
        "source": "peft_lora",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_parameters": len(delta_weights),
    }
    with open(os.path.join(output_path, "delta_config.json"), "w") as f:
        json.dump(marker, f, indent=2)
    
    return output_path


def convert_peft_to_merged_checkpoint(
    peft_lora_path: str,
    output_path: str,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> str:
    """
    Merge PEFT LoRA into base model and save full checkpoint.
    
    This is the most compatible approach for SGLang's /update_weights_from_disk,
    but creates larger files and is slower.
    
    Args:
        peft_lora_path: Path to PEFT LoRA checkpoint
        output_path: Where to save merged model
        base_model_name: Base model to merge into
        
    Returns:
        Path to merged checkpoint
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    logger.info(f"Merging LoRA from {peft_lora_path} into {base_model_name}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load on CPU for merging
        trust_remote_code=True,
    )
    
    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, peft_lora_path)
    merged_model = model.merge_and_unload()
    
    # Save merged model
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Merged model saved to {output_path}")
    
    # Cleanup
    del base_model, model, merged_model
    torch.cuda.empty_cache()
    
    return output_path


class LoRAFormatConverter:
    """
    Converts between LoRA formats for different inference backends.
    
    Supports:
    - PEFT (Hugging Face) format
    - SGLang delta weights format
    - Full merged checkpoint format
    """
    
    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.base_model_name = base_model_name
        self._cache_dir = Path("./cache/lora_converted")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def peft_to_sglang(
        self,
        peft_path: str,
        version: int = 0,
    ) -> str:
        """
        Convert PEFT LoRA to SGLang-compatible format.
        
        Args:
            peft_path: Path to PEFT checkpoint
            version: Version number for caching
            
        Returns:
            Path to converted checkpoint
        """
        output_path = self._cache_dir / f"sglang_v{version}"
        
        # Check if already converted
        if output_path.exists():
            config_file = output_path / "delta_config.json"
            if config_file.exists():
                logger.info(f"Using cached conversion at {output_path}")
                return str(output_path)
        
        return convert_peft_to_sglang_delta(
            peft_path,
            str(output_path),
            self.base_model_name,
        )
    
    def peft_to_merged(
        self,
        peft_path: str,
        version: int = 0,
    ) -> str:
        """
        Merge PEFT LoRA into base model.
        
        Args:
            peft_path: Path to PEFT checkpoint
            version: Version number for caching
            
        Returns:
            Path to merged checkpoint
        """
        output_path = self._cache_dir / f"merged_v{version}"
        
        return convert_peft_to_merged_checkpoint(
            peft_path,
            str(output_path),
            self.base_model_name,
        )
    
    def cleanup_old_versions(self, keep_last: int = 3):
        """Remove old converted checkpoints."""
        versions = []
        for path in self._cache_dir.iterdir():
            if path.is_dir() and path.name.startswith(("sglang_v", "merged_v")):
                try:
                    v = int(path.name.split("_v")[1])
                    versions.append((v, path))
                except ValueError:
                    continue
        
        versions.sort(key=lambda x: x[0], reverse=True)
        
        for v, path in versions[keep_last:]:
            logger.info(f"Removing old conversion: {path}")
            shutil.rmtree(path)
