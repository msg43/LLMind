#!/bin/bash

# LLMind Start Script
# ====================
# This script starts the LLMind application

set -e

echo "🦙 Starting LLMind..."
echo "======================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./scripts/setup.sh first to set up the project."
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating default configuration..."
    ./scripts/setup.sh
fi

# Create data directories if they don't exist
echo "📁 Ensuring data directories exist..."
mkdir -p data/{documents,vector_store,models,audio,backups,logs,cache}
mkdir -p logs

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found!"
    echo "Please ensure you're in the correct directory."
    exit 1
fi

# Get configuration
HOST=${HOST:-localhost}
PORT=${PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-info}

echo "🚀 Starting FastAPI server..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Log Level: $LOG_LEVEL"
echo ""

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use!"
    echo "Please stop the existing service or change the PORT in your .env file."
    exit 1
fi

echo "📱 Access the application at:"
echo "   Web Interface: http://$HOST:$PORT"
echo "   API Documentation: http://$HOST:$PORT/docs"
echo "   OpenAPI Schema: http://$HOST:$PORT/openapi.json"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the application
uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --access-log \
    --loop uvloop
