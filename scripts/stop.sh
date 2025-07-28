#!/bin/bash

# LLMind Stop Script
# ==================
# This script stops the LLMind application if it's running

echo "🛑 Stopping LLMind..."
echo "======================="

# Default port
PORT=${PORT:-8000}

# Check if application is running on the specified port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "🔍 Found LLMind running on port $PORT"
    
    # Get the process ID
    PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
    
    echo "📋 Process ID: $PID"
    
    # Try graceful shutdown first
    echo "🤝 Attempting graceful shutdown..."
    kill -TERM $PID
    
    # Wait a bit for graceful shutdown
    sleep 3
    
    # Check if it's still running
    if kill -0 $PID 2>/dev/null; then
        echo "⚠️  Process still running, forcing shutdown..."
        kill -KILL $PID
        sleep 1
    fi
    
    # Verify it's stopped
    if ! kill -0 $PID 2>/dev/null; then
        echo "✅ LLMind stopped successfully"
    else
        echo "❌ Failed to stop LLMind"
        exit 1
    fi
else
    echo "ℹ️  LLMind is not running on port $PORT"
fi

# Also check for any python processes that might be running the app
echo "🔍 Checking for any remaining LLMind processes..."

# Look for python processes running main.py or uvicorn with our app
PIDS=$(pgrep -f "main:app\|uvicorn.*main" 2>/dev/null || true)

if [ -n "$PIDS" ]; then
    echo "🎯 Found additional LLMind processes: $PIDS"
    echo "🛑 Stopping additional processes..."
    
    for pid in $PIDS; do
        if kill -0 $pid 2>/dev/null; then
            echo "  Stopping process $pid..."
            kill -TERM $pid
            sleep 1
            
            # Force kill if still running
            if kill -0 $pid 2>/dev/null; then
                kill -KILL $pid
            fi
        fi
    done
    
            echo "✅ All LLMind processes stopped"
    else
        echo "ℹ️  No additional LLMind processes found"
fi

echo ""
echo "🔒 LLMind shutdown complete" 