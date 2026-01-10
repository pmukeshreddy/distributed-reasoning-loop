#!/usr/bin/env python3
"""
Training Dynamics Visualization for RL Training.

Logs and visualizes:
- Loss curves (policy loss, KL divergence)
- Reward margins (chosen - rejected)
- Gradient norms
- Learning rate schedules

Shows that RL is working mechanically even if accuracy gains need more data/compute.

Usage:
    python visualize_training.py --log-dir ./training_logs
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics logged during training."""
    step: int
    epoch: float
    loss: float
    policy_loss: float = 0.0
    kl_divergence: float = 0.0
    reward_chosen: float = 0.0
    reward_rejected: float = 0.0
    reward_margin: float = 0.0
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    ratio_mean: float = 1.0
    clip_fraction: float = 0.0
    entropy: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class TrainingLogger:
    """
    Logger for training dynamics.
    
    Tracks metrics that show RL is working:
    - Loss decreasing
    - Reward margin increasing (chosen > rejected)
    - KL staying bounded
    - Gradients not exploding
    """
    
    def __init__(self, log_dir: str = "./training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[TrainingMetrics] = []
        self.log_file = self.log_dir / "training_metrics.jsonl"
    
    def log(self, metrics: TrainingMetrics):
        """Log a single training step."""
        self.metrics.append(metrics)
        
        # Append to JSONL file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")
    
    def log_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        **kwargs,
    ):
        """Convenience method to log a step."""
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            **kwargs,
        )
        self.log(metrics)
    
    def load_metrics(self) -> List[TrainingMetrics]:
        """Load metrics from log file."""
        metrics = []
        if self.log_file.exists():
            with open(self.log_file) as f:
                for line in f:
                    data = json.loads(line)
                    metrics.append(TrainingMetrics(**data))
        self.metrics = metrics
        return metrics
    
    def get_summary(self) -> Dict:
        """Get summary statistics of training."""
        if not self.metrics:
            return {}
        
        losses = [m.loss for m in self.metrics]
        margins = [m.reward_margin for m in self.metrics if m.reward_margin != 0]
        kls = [m.kl_divergence for m in self.metrics if m.kl_divergence != 0]
        
        return {
            "total_steps": len(self.metrics),
            "final_loss": losses[-1] if losses else 0,
            "loss_reduction": losses[0] - losses[-1] if len(losses) > 1 else 0,
            "loss_reduction_pct": ((losses[0] - losses[-1]) / losses[0] * 100) if len(losses) > 1 and losses[0] != 0 else 0,
            "final_reward_margin": margins[-1] if margins else 0,
            "margin_improvement": margins[-1] - margins[0] if len(margins) > 1 else 0,
            "avg_kl_divergence": sum(kls) / len(kls) if kls else 0,
            "max_kl_divergence": max(kls) if kls else 0,
        }


def create_ascii_plot(
    values: List[float],
    title: str,
    width: int = 60,
    height: int = 15,
) -> str:
    """Create ASCII line plot."""
    if not values:
        return f"{title}: No data"
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        max_val = min_val + 1
    
    # Normalize values to height
    normalized = [
        int((v - min_val) / (max_val - min_val) * (height - 1))
        for v in values
    ]
    
    # Sample if too many points
    if len(normalized) > width:
        step = len(normalized) // width
        normalized = normalized[::step][:width]
        values_sampled = values[::step][:width]
    else:
        values_sampled = values
    
    # Create plot
    lines = []
    lines.append(f"  {title}")
    lines.append(f"  {max_val:.4f} ┤")
    
    for row in range(height - 1, -1, -1):
        line = "         │"
        for col, norm_val in enumerate(normalized):
            if norm_val == row:
                line += "●"
            elif norm_val > row and col > 0 and normalized[col-1] < row:
                line += "│"
            elif norm_val < row and col > 0 and normalized[col-1] > row:
                line += "│"
            else:
                line += " "
        lines.append(line)
    
    lines.append(f"  {min_val:.4f} ┤" + "─" * len(normalized))
    lines.append(f"         └{'─' * len(normalized)}→ steps")
    
    return "\n".join(lines)


def visualize_training(log_dir: str, output_format: str = "ascii"):
    """Visualize training dynamics."""
    logger_obj = TrainingLogger(log_dir)
    metrics = logger_obj.load_metrics()
    
    if not metrics:
        print("No training metrics found!")
        return
    
    print("\n" + "=" * 70)
    print("TRAINING DYNAMICS VISUALIZATION")
    print("=" * 70)
    
    # Summary stats
    summary = logger_obj.get_summary()
    print("\n📊 Training Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    # Check if GRPO-style (negative loss)
    is_grpo = metrics[0].loss < 0 or summary['final_loss'] < 0
    if is_grpo:
        delta = summary['final_loss'] - metrics[0].loss
        direction = "↓ more negative" if delta < 0 else "↑ less negative"
        print(f"  Loss: {metrics[0].loss:.4f} → {summary['final_loss']:.4f} ({direction}, Δ={abs(delta):.4f})")
    else:
        print(f"  Loss: {metrics[0].loss:.4f} → {summary['final_loss']:.4f} ({summary['loss_reduction_pct']:.1f}% reduction)")
    if summary['final_reward_margin'] != 0:
        print(f"  Reward margin: {summary['final_reward_margin']:.4f} (improvement: {summary['margin_improvement']:.4f})")
    if summary['avg_kl_divergence'] != 0:
        print(f"  KL divergence: avg={summary['avg_kl_divergence']:.4f}, max={summary['max_kl_divergence']:.4f}")
    
    if output_format == "ascii":
        # Loss curve
        losses = [m.loss for m in metrics]
        print("\n" + create_ascii_plot(losses, "Loss Curve"))
        
        # Reward margin
        margins = [m.reward_margin for m in metrics if m.reward_margin != 0]
        if margins:
            print("\n" + create_ascii_plot(margins, "Reward Margin (chosen - rejected)"))
        
        # KL divergence
        kls = [m.kl_divergence for m in metrics if m.kl_divergence != 0]
        if kls:
            print("\n" + create_ascii_plot(kls, "KL Divergence"))
    
    elif output_format == "matplotlib":
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Loss
            steps = [m.step for m in metrics]
            losses = [m.loss for m in metrics]
            axes[0, 0].plot(steps, losses, 'b-')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Reward margin
            margins = [m.reward_margin for m in metrics]
            axes[0, 1].plot(steps, margins, 'g-')
            axes[0, 1].set_title('Reward Margin (chosen - rejected)')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Margin')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].grid(True, alpha=0.3)
            
            # KL divergence
            kls = [m.kl_divergence for m in metrics]
            axes[1, 0].plot(steps, kls, 'r-')
            axes[1, 0].set_title('KL Divergence')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('KL')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Gradient norm
            grads = [m.gradient_norm for m in metrics]
            axes[1, 1].plot(steps, grads, 'm-')
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Norm')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_path = Path(log_dir) / "training_curves.png"
            plt.savefig(output_path, dpi=150)
            print(f"\n📈 Plot saved to: {output_path}")
            plt.close()
            
        except ImportError:
            print("matplotlib not available, using ASCII plots")
            visualize_training(log_dir, "ascii")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    # Analysis
    insights = []
    
    # For GRPO/DPO, loss goes MORE NEGATIVE when training works
    # Check if this looks like GRPO (loss starts negative or becomes negative)
    is_grpo_style = summary['initial_loss'] < 0 or summary['final_loss'] < 0
    
    if is_grpo_style:
        # GRPO: more negative = better
        loss_improved = summary['final_loss'] < summary['initial_loss']
        if loss_improved:
            improvement = abs(summary['final_loss'] - summary['initial_loss'])
            insights.append(f"✅ GRPO loss improving (more negative) - Δ={improvement:.4f}")
        else:
            insights.append("⚠️ GRPO loss not improving - check learning rate")
    elif summary['loss_reduction_pct'] > 10:
        insights.append("✅ Loss decreasing significantly - training is working")
    elif summary['loss_reduction_pct'] > 0:
        insights.append("⚠️ Loss decreasing slowly - may need more epochs")
    else:
        insights.append("❌ Loss not decreasing - check learning rate")
    
    if summary['final_reward_margin'] > 0:
        insights.append("✅ Positive reward margin - model prefers chosen over rejected")
    elif summary['final_reward_margin'] < 0:
        insights.append("❌ Negative reward margin - check preference data quality")
    
    if summary['max_kl_divergence'] < 0.5:
        insights.append("✅ KL bounded - policy staying close to reference")
    elif summary['max_kl_divergence'] < 1.0:
        insights.append("⚠️ KL moderate - consider reducing learning rate")
    else:
        insights.append("❌ KL too high - policy diverging, reduce LR or increase KL coef")
    
    for insight in insights:
        print(f"  {insight}")
    
    print()


def generate_mock_training_logs(log_dir: str, num_steps: int = 100):
    """Generate mock training logs for demonstration."""
    logger_obj = TrainingLogger(log_dir)
    
    # Clear existing
    if logger_obj.log_file.exists():
        logger_obj.log_file.unlink()
    
    import random
    import math
    
    for step in range(num_steps):
        epoch = step / (num_steps / 3)
        
        # Simulate decreasing loss
        base_loss = 2.0 * math.exp(-step / 30) + 0.5
        loss = base_loss + random.gauss(0, 0.05)
        
        # Simulate increasing reward margin
        margin = 0.1 * (1 - math.exp(-step / 20)) + random.gauss(0, 0.02)
        
        # Simulate bounded KL
        kl = 0.1 + 0.05 * math.sin(step / 10) + random.gauss(0, 0.01)
        
        # Gradient norm
        grad_norm = 1.0 + random.gauss(0, 0.2)
        
        logger_obj.log_step(
            step=step,
            epoch=epoch,
            loss=max(0.1, loss),
            policy_loss=max(0.1, loss * 0.8),
            kl_divergence=max(0, kl),
            reward_chosen=0.5 + margin / 2,
            reward_rejected=0.5 - margin / 2,
            reward_margin=margin,
            gradient_norm=max(0, grad_norm),
            learning_rate=1e-5 * (1 - step / num_steps),
            ratio_mean=1.0 + random.gauss(0, 0.1),
        )
    
    print(f"Generated {num_steps} mock training steps in {log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RL training dynamics"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./training_logs",
        help="Directory containing training logs",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["ascii", "matplotlib"],
        default="ascii",
        help="Output format",
    )
    parser.add_argument(
        "--generate-mock",
        action="store_true",
        help="Generate mock training logs for demo",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of steps for mock generation",
    )
    
    args = parser.parse_args()
    
    if args.generate_mock:
        generate_mock_training_logs(args.log_dir, args.num_steps)
    
    visualize_training(args.log_dir, args.format)


if __name__ == "__main__":
    main()