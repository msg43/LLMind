#!/bin/bash

# LLMind Virtual Environment Activation Script
# This script activates the virtual environment for the LLMind project

echo "🔧 Activating LLMind virtual environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run the setup script first."
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

# Verify activation
if [ $? -eq 0 ]; then
    echo "✅ Virtual environment activated successfully!"
    echo "🐍 Python version: $(python --version)"
    echo "📦 Virtual environment: $(which python)"
    echo ""
    echo "To deactivate, run: deactivate"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
