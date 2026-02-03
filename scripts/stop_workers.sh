#!/bin/bash
# =============================================================================
# Stop SGLang Workers
# =============================================================================

set -e

LOG_DIR="./logs/workers"
PID_FILE="$LOG_DIR/worker_pids.txt"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Stopping SGLang workers..."

# Method 1: Kill by saved PIDs
if [ -f "$PID_FILE" ]; then
    PIDS=$(cat "$PID_FILE")
    for PID in $PIDS; do
        if kill -0 $PID 2>/dev/null; then
            echo -e "  Killing PID $PID"
            kill $PID 2>/dev/null || true
        fi
    done
    rm -f "$PID_FILE"
fi

# Method 2: Kill any remaining sglang processes
SGLANG_PIDS=$(pgrep -f "sglang.launch_server" 2>/dev/null || true)
if [ -n "$SGLANG_PIDS" ]; then
    echo -e "  Killing remaining SGLang processes..."
    echo $SGLANG_PIDS | xargs kill 2>/dev/null || true
fi

sleep 2

# Verify all stopped
REMAINING=$(pgrep -f "sglang.launch_server" 2>/dev/null || true)
if [ -z "$REMAINING" ]; then
    echo -e "${GREEN}All workers stopped.${NC}"
else
    echo -e "${RED}Warning: Some workers may still be running.${NC}"
    echo -e "  Remaining PIDs: $REMAINING"
    echo -e "  Try: kill -9 $REMAINING"
fi
