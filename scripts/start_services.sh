#!/bin/bash
# Start all services for the distributed reasoning loop

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Distributed Reasoning Loop Services${NC}"
echo "=============================================="

# Check for required tools
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker is required but not installed.${NC}" >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo -e "${RED}Docker Compose is required but not installed.${NC}" >&2; exit 1; }

# Parse arguments
MODE="dev"
GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --prod)
            MODE="prod"
            shift
            ;;
        --gpu)
            GPU=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR/docker"

if [ "$MODE" == "dev" ]; then
    echo -e "${YELLOW}Starting in development mode...${NC}"
    docker-compose -f docker-compose.dev.yml up -d
    
    echo -e "${GREEN}Services started:${NC}"
    echo "  - Kafka: localhost:29092"
    echo "  - Kafka UI: http://localhost:8080"
    echo "  - Zookeeper: localhost:2181"
else
    echo -e "${YELLOW}Starting in production mode...${NC}"
    
    if [ "$GPU" == true ]; then
        echo "  - GPU support enabled"
        docker-compose up -d
    else
        echo "  - CPU only mode"
        docker-compose up -d --scale inference=0
    fi
    
    echo -e "${GREEN}Services started:${NC}"
    echo "  - Kafka: localhost:29092"
    echo "  - Kafka UI: http://localhost:8080"
    echo "  - Ray Dashboard: http://localhost:8265"
    if [ "$GPU" == true ]; then
        echo "  - Inference Server: http://localhost:8000"
    fi
fi

echo ""
echo -e "${GREEN}To view logs: docker-compose logs -f${NC}"
echo -e "${GREEN}To stop services: docker-compose down${NC}"
