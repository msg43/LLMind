#!/bin/bash

# LLMind Clean Startup Script
# This script ensures a clean start by killing any existing processes

echo "🦙 Starting LLMind..."

# Kill any existing LLMind processes
echo "🧹 Cleaning up any existing processes..."
pkill -f "python main.py" 2>/dev/null || true
pkill -f uvicorn 2>/dev/null || true

# Kill anything using port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Wait a moment for cleanup
sleep 2

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
fi

# Start LLMind
echo "🚀 Starting LLMind..."
python main.py
