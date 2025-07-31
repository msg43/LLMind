#!/bin/bash

# LLMind Development Script
# ==========================
# This script starts the LLMind application in development mode with hot reload

set -e

echo "🔧 Starting LLMind in Development Mode..."
echo "============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./scripts/setup.sh first to set up the project."
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists and set development defaults
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating default development configuration..."
    ./scripts/setup.sh
fi

# Override settings for development
export DEBUG=true
export LOG_LEVEL=debug
export HOT_RELOAD=true
export ENABLE_DOCS=true

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

echo "🚀 Starting FastAPI development server..."
echo "   Mode: Development (Hot Reload Enabled)"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Debug: $DEBUG"
echo ""

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use!"
    echo "Please stop the existing service or change the PORT in your .env file."
    exit 1
fi

echo "📱 Development URLs:"
echo "   Web Interface: http://$HOST:$PORT"
echo "   API Documentation: http://$HOST:$PORT/docs"
echo "   ReDoc Documentation: http://$HOST:$PORT/redoc"
echo "   OpenAPI Schema: http://$HOST:$PORT/openapi.json"
echo ""
echo "🔥 Hot reload is enabled - files will auto-reload on changes"
echo "📝 Debug logging is enabled for detailed output"
echo "Press Ctrl+C to stop the development server"
echo ""

# Start the application with hot reload
uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --reload-dir . \
    --reload-exclude "data/*" \
    --reload-exclude "logs/*" \
    --reload-exclude "*.log" \
    --reload-exclude "__pycache__" \
    --log-level debug \
    --access-log \
    --loop uvloop
