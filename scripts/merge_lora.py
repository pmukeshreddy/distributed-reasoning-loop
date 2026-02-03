#!/usr/bin/env python3
"""
Merge LoRA adapter into base model.

Usage:
    python scripts/merge_lora.py --lora-path ./outputs/continuous_grpo/lora_checkpoints/lora_v10
    python scripts/merge_lora.py --lora-path ./outputs/continuous_grpo/lora_checkpoints/lora_v10 --output-path ./outputs/merged_model
"""

import argparse
import torch
import os
from pathlib import Path


def merge_lora(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    lora_path: str = "./outputs/continuous_grpo/lora_checkpoints/lora_v10",
    output_path: str = "./outputs/merged_model",
):
    """Merge LoRA weights into base model and save."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path)
    
    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Free memory
    del base, model, merged
    torch.cuda.empty_cache()
    
    print("✓ Merge complete!")
    print(f"Merged model saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output-path", type=str, default="./outputs/merged_model")
    
    args = parser.parse_args()
    
    merge_lora(
        base_model=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
