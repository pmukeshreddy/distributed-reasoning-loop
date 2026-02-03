#!/usr/bin/env python3
"""
Continuous GRPO Training Dashboard

Real-time monitoring dashboard for continuous distributed GRPO training.
Displays:
- Training metrics (loss, accuracy)
- Worker status and throughput
- LoRA version history
- Live log streaming

Usage:
    streamlit run scripts/dashboard.py

Or with custom metrics directory:
    streamlit run scripts/dashboard.py -- --metrics-dir ./outputs/continuous_grpo/logs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

# Check for streamlit
try:
    import streamlit as st
    import pandas as pd
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Check for plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def load_metrics(metrics_file: str) -> List[Dict[str, Any]]:
    """Load metrics from JSONL file."""
    metrics = []
    if not os.path.exists(metrics_file):
        return metrics
    
    with open(metrics_file, "r") as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return metrics


def create_training_plots(metrics: List[Dict[str, Any]]):
    """Create training metrics plots."""
    if not metrics:
        st.warning("No training metrics available yet.")
        return
    
    df = pd.DataFrame(metrics)
    
    # Extract nested metrics
    df["accuracy"] = df["verification"].apply(lambda x: x.get("accuracy", 0) if isinstance(x, dict) else 0)
    df["loss"] = df["training"].apply(lambda x: x.get("loss", 0) if isinstance(x, dict) else 0)
    df["rollouts"] = df["rollouts"].apply(lambda x: x.get("total", 0) if isinstance(x, dict) else 0)
    df["correct"] = df["verification"].apply(lambda x: x.get("correct", 0) if isinstance(x, dict) else 0)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Verification Accuracy", "Training Loss", "Rollouts per Iteration", "Correct Predictions"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=df["iteration"], y=df["accuracy"], mode="lines+markers", name="Accuracy",
                   line=dict(color="#00CC96", width=2)),
        row=1, col=1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=df["iteration"], y=df["loss"], mode="lines+markers", name="Loss",
                   line=dict(color="#EF553B", width=2)),
        row=1, col=2
    )
    
    # Rollouts plot
    fig.add_trace(
        go.Bar(x=df["iteration"], y=df["rollouts"], name="Rollouts",
               marker_color="#636EFA"),
        row=2, col=1
    )
    
    # Correct predictions plot
    fig.add_trace(
        go.Bar(x=df["iteration"], y=df["correct"], name="Correct",
               marker_color="#00CC96"),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Training Progress",
        title_x=0.5,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_worker_status(metrics: List[Dict[str, Any]]):
    """Display worker status."""
    if not metrics:
        st.info("Waiting for worker data...")
        return
    
    latest = metrics[-1]
    worker_stats = latest.get("worker_stats", {})
    workers = worker_stats.get("workers", [])
    
    if not workers:
        st.info("No worker data available.")
        return
    
    # Worker status cards
    cols = st.columns(len(workers))
    for col, worker in zip(cols, workers):
        with col:
            status = worker.get("status", "unknown")
            status_color = "🟢" if status == "ready" else "🟡" if status == "busy" else "🔴"
            
            st.metric(
                label=f"Worker {worker.get('worker_id', '?')} {status_color}",
                value=f"v{worker.get('lora_version', 0)}",
                delta=f"{worker.get('requests_served', 0)} requests"
            )


def create_lora_history(metrics: List[Dict[str, Any]]):
    """Display LoRA version history."""
    if not metrics:
        return
    
    # Extract LoRA versions
    versions = []
    for m in metrics:
        versions.append({
            "iteration": m.get("iteration", 0),
            "version": m.get("lora_version", 0),
            "accuracy": m.get("verification", {}).get("accuracy", 0),
            "loss": m.get("training", {}).get("loss", 0),
        })
    
    df = pd.DataFrame(versions)
    
    st.subheader("LoRA Version History")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["iteration"],
        y=df["version"],
        mode="lines+markers",
        name="LoRA Version",
        marker=dict(size=10, color=df["accuracy"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Accuracy")),
    ))
    
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="LoRA Version",
        height=300,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_time_breakdown(metrics: List[Dict[str, Any]]):
    """Display time breakdown per iteration."""
    if not metrics:
        return
    
    latest = metrics[-1]
    
    times = {
        "Rollout": latest.get("rollouts", {}).get("time_seconds", 0),
        "Verification": latest.get("verification", {}).get("time_seconds", 0),
        "Training": latest.get("training", {}).get("time_seconds", 0),
        "Broadcast": latest.get("broadcast", {}).get("time_seconds", 0),
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(times.keys()),
        values=list(times.values()),
        hole=0.4,
        marker_colors=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
    )])
    
    fig.update_layout(
        title="Time Breakdown (Last Iteration)",
        height=300,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_terminal_logs(log_dir: str, num_lines: int = 50):
    """Display recent log entries."""
    log_file = Path(log_dir) / "training_metrics.jsonl"
    
    if not log_file.exists():
        st.info("No logs available yet.")
        return
    
    # Read last N lines
    with open(log_file, "r") as f:
        lines = f.readlines()[-num_lines:]
    
    # Format as log entries
    log_text = ""
    for line in lines:
        try:
            m = json.loads(line)
            log_text += f"[Iter {m.get('iteration', '?')}] "
            log_text += f"Acc: {m.get('verification', {}).get('accuracy', 0):.1%} | "
            log_text += f"Loss: {m.get('training', {}).get('loss', 0):.4f} | "
            log_text += f"LoRA: v{m.get('lora_version', 0)}\n"
        except:
            continue
    
    st.code(log_text, language="")


def main_dashboard(metrics_dir: str):
    """Main dashboard layout."""
    st.set_page_config(
        page_title="Continuous GRPO Dashboard",
        page_icon="🧠",
        layout="wide",
    )
    
    st.title("🧠 Continuous Distributed GRPO Training")
    
    # Sidebar
    st.sidebar.header("Settings")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
    
    # Auto-refresh
    placeholder = st.empty()
    
    while True:
        with placeholder.container():
            # Load metrics
            metrics_file = Path(metrics_dir) / "training_metrics.jsonl"
            metrics = load_metrics(str(metrics_file))
            
            if metrics:
                latest = metrics[-1]
                
                # Header metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Iteration", latest.get("iteration", 0))
                with col2:
                    acc = latest.get("verification", {}).get("accuracy", 0)
                    st.metric("Current Accuracy", f"{acc:.1%}")
                with col3:
                    loss = latest.get("training", {}).get("loss", 0)
                    st.metric("Current Loss", f"{loss:.4f}")
                with col4:
                    st.metric("LoRA Version", f"v{latest.get('lora_version', 0)}")
            
            st.divider()
            
            # Main content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Training Progress")
                create_training_plots(metrics)
            
            with col2:
                st.subheader("Worker Status")
                create_worker_status(metrics)
                
                st.divider()
                create_time_breakdown(metrics)
            
            # Bottom section
            col1, col2 = st.columns(2)
            
            with col1:
                create_lora_history(metrics)
            
            with col2:
                st.subheader("Recent Logs")
                display_terminal_logs(metrics_dir, num_lines=20)
            
            # Last updated
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        time.sleep(refresh_rate)
        st.rerun()


def fallback_dashboard(metrics_dir: str):
    """Fallback terminal-based dashboard when Streamlit not available."""
    print("=" * 60)
    print("CONTINUOUS GRPO TRAINING MONITOR")
    print("=" * 60)
    print(f"\nMetrics directory: {metrics_dir}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            metrics_file = Path(metrics_dir) / "training_metrics.jsonl"
            metrics = load_metrics(str(metrics_file))
            
            # Clear screen (works on most terminals)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 60)
            print("CONTINUOUS GRPO TRAINING MONITOR")
            print("=" * 60)
            
            if metrics:
                latest = metrics[-1]
                
                print(f"\nIteration: {latest.get('iteration', 0)}")
                print(f"LoRA Version: v{latest.get('lora_version', 0)}")
                print(f"Accuracy: {latest.get('verification', {}).get('accuracy', 0):.1%}")
                print(f"Loss: {latest.get('training', {}).get('loss', 0):.4f}")
                print(f"Rollouts: {latest.get('rollouts', {}).get('total', 0)}")
                print(f"Correct: {latest.get('verification', {}).get('correct', 0)}")
                
                print("\n--- Recent History ---")
                for m in metrics[-5:]:
                    acc = m.get('verification', {}).get('accuracy', 0)
                    loss = m.get('training', {}).get('loss', 0)
                    print(f"  Iter {m.get('iteration', '?'):3d}: Acc={acc:.1%}, Loss={loss:.4f}")
            else:
                print("\nWaiting for training data...")
            
            print(f"\nLast updated: {datetime.now().strftime('%H:%M:%S')}")
            print("\nPress Ctrl+C to stop")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous GRPO Training Dashboard")
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="./outputs/continuous_grpo",
        help="Directory containing training metrics",
    )
    
    # Parse args (handling Streamlit's arg injection)
    args, _ = parser.parse_known_args()
    
    if STREAMLIT_AVAILABLE and PLOTLY_AVAILABLE:
        main_dashboard(args.metrics_dir)
    else:
        if not STREAMLIT_AVAILABLE:
            print("Streamlit not installed. Install with: pip install streamlit")
        if not PLOTLY_AVAILABLE:
            print("Plotly not installed. Install with: pip install plotly")
        print("\nFalling back to terminal monitor...\n")
        fallback_dashboard(args.metrics_dir)
