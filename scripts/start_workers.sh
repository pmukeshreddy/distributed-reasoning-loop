#!/bin/bash
# =============================================================================
# SGLang Worker Launcher for Continuous Distributed GRPO
# =============================================================================
#
# This script starts multiple SGLang inference workers on different GPUs/ports.
# Each worker can dynamically load/unload LoRA adapters for hot-reloading.
#
# Usage:
#   ./scripts/start_workers.sh [NUM_WORKERS] [BASE_PORT] [MODEL]
#
# Examples:
#   # Start 3 workers on ports 30001, 30002, 30003 (default)
#   ./scripts/start_workers.sh
#
#   # Start 2 workers on ports 30001, 30002
#   ./scripts/start_workers.sh 2
#
#   # Start 4 workers starting from port 31000
#   ./scripts/start_workers.sh 4 31000
#
#   # Start workers with a specific model
#   ./scripts/start_workers.sh 3 30001 "Qwen/Qwen2.5-7B-Instruct"
#
# =============================================================================

set -e

# Configuration
NUM_WORKERS=${1:-3}
BASE_PORT=${2:-30001}
MODEL=${3:-"Qwen/Qwen2.5-1.5B-Instruct"}
MAX_LORAS=${4:-8}
LORA_RANK=${5:-64}
LOG_DIR="./logs/workers"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  SGLang Worker Launcher                   ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "  Workers:    ${GREEN}${NUM_WORKERS}${NC}"
echo -e "  Base Port:  ${GREEN}${BASE_PORT}${NC}"
echo -e "  Model:      ${GREEN}${MODEL}${NC}"
echo -e "  Max LoRAs:  ${GREEN}${MAX_LORAS}${NC}"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Function to get available GPUs
get_available_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
    else
        echo "0"
    fi
}

NUM_GPUS=$(get_available_gpus)
echo -e "  GPUs:       ${GREEN}${NUM_GPUS}${NC}"
echo ""

if [ "$NUM_GPUS" -eq 0 ]; then
    echo -e "${YELLOW}Warning: No GPUs detected. Workers will run on CPU.${NC}"
    echo ""
fi

# Kill existing workers on these ports
echo -e "${YELLOW}Checking for existing workers...${NC}"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    PID=$(lsof -t -i:$PORT 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo -e "  Killing existing process on port $PORT (PID: $PID)"
        kill $PID 2>/dev/null || true
        sleep 1
    fi
done
echo ""

# Start workers
echo -e "${GREEN}Starting ${NUM_WORKERS} workers...${NC}"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    
    # Assign GPU (cycle through available GPUs)
    if [ "$NUM_GPUS" -gt 0 ]; then
        GPU_ID=$((i % NUM_GPUS))
        GPU_FLAG="CUDA_VISIBLE_DEVICES=$GPU_ID"
    else
        GPU_FLAG=""
    fi
    
    LOG_FILE="$LOG_DIR/worker_${i}_port_${PORT}.log"
    
    echo -e "  Worker $i:"
    echo -e "    Port: ${GREEN}$PORT${NC}"
    if [ -n "$GPU_FLAG" ]; then
        echo -e "    GPU:  ${GREEN}$GPU_ID${NC}"
    fi
    echo -e "    Log:  ${BLUE}$LOG_FILE${NC}"
    
    # Start worker in background
    if [ -n "$GPU_FLAG" ]; then
        env $GPU_FLAG nohup python -m sglang.launch_server \
            --model-path "$MODEL" \
            --host 0.0.0.0 \
            --port $PORT \
            --max-loras-per-batch $MAX_LORAS \
            --max-lora-rank $LORA_RANK \
            --trust-remote-code \
            > "$LOG_FILE" 2>&1 &
    else
        nohup python -m sglang.launch_server \
            --model-path "$MODEL" \
            --host 0.0.0.0 \
            --port $PORT \
            --max-loras-per-batch $MAX_LORAS \
            --max-lora-rank $LORA_RANK \
            --trust-remote-code \
            > "$LOG_FILE" 2>&1 &
    fi
    
    PIDS+=($!)
    echo ""
done

# Save PIDs for later cleanup
PID_FILE="$LOG_DIR/worker_pids.txt"
echo "${PIDS[@]}" > "$PID_FILE"
echo -e "  PIDs saved to: ${BLUE}$PID_FILE${NC}"
echo ""

# Wait for workers to start
echo -e "${YELLOW}Waiting for workers to initialize (60s)...${NC}"
sleep 30

# Health check function
check_worker() {
    local port=$1
    local response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null || echo "000")
    if [ "$response" = "200" ]; then
        return 0
    else
        return 1
    fi
}

# Check health with retries
echo ""
echo -e "${YELLOW}Checking worker health...${NC}"
MAX_RETRIES=6
RETRY_DELAY=10

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    
    for retry in $(seq 1 $MAX_RETRIES); do
        if check_worker $PORT; then
            echo -e "  Worker $i (port $PORT): ${GREEN}✓ Ready${NC}"
            break
        else
            if [ $retry -lt $MAX_RETRIES ]; then
                echo -e "  Worker $i (port $PORT): ${YELLOW}Starting... (attempt $retry/$MAX_RETRIES)${NC}"
                sleep $RETRY_DELAY
            else
                echo -e "  Worker $i (port $PORT): ${RED}✗ Failed to start${NC}"
                echo -e "    Check log: $LOG_DIR/worker_${i}_port_${PORT}.log"
            fi
        fi
    done
done

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}Workers started!${NC}"
echo ""
echo -e "To run continuous GRPO training:"
echo -e "  ${BLUE}python scripts/run_continuous_grpo.py --ports $(seq -s ' ' $BASE_PORT $((BASE_PORT + NUM_WORKERS - 1)))${NC}"
echo ""
echo -e "To stop workers:"
echo -e "  ${BLUE}./scripts/stop_workers.sh${NC}"
echo ""
echo -e "To view logs:"
echo -e "  ${BLUE}tail -f $LOG_DIR/worker_*.log${NC}"
echo ""
