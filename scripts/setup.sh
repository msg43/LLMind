#!/bin/bash

# LLMind Setup Script
# ====================
# This script helps you set up the LLMind application for the first time.

set -e  # Exit on error

echo "🚀 Welcome to LLMind Setup!"
echo "==============================="

# Check if we're on macOS (required for MLX and TTS)
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: LLMind is optimized for macOS with Apple Silicon."
    echo "Some features like MLX acceleration and built-in TTS may not work."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python 3.9+ is required. Found: $python_version"
    echo "Please install Python 3.9 or later and try again."
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📁 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/{documents,vector_store,models,audio,backups,logs,cache}
mkdir -p logs
echo "✅ Directories created"

# Create environment file from template
echo "⚙️  Setting up environment configuration..."
cat > .env << 'EOF'
# LLMind Environment Configuration
# Copy and customize these settings as needed

# Application Settings
HOST=localhost
PORT=8000
ENVIRONMENT=development
DEBUG=true

# Directories
DATA_DIR=./data
DOCUMENTS_DIR=./data/documents
VECTOR_STORE_DIR=./data/vector_store
MODELS_DIR=./data/models
AUDIO_DIR=./data/audio

# Model Settings
DEFAULT_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit
TEMPERATURE=0.7
MAX_TOKENS=1000
MODEL_CONTEXT_LENGTH=4096

# Vector Store Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Audio Settings
ENABLE_VOICE=true
WHISPER_MODEL=base
TTS_VOICE=Samantha
TTS_RATE=200

# Performance Settings
ENABLE_MPS=true
MLX_MEMORY_FRACTION=0.8
CPU_THREADS=8

# Security
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
MAX_REQUEST_SIZE_MB=100

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/llmind.log
EOF

echo "✅ Environment configuration created (.env)"

# Set execute permissions on scripts
echo "🔑 Setting script permissions..."
chmod +x scripts/*.sh
echo "✅ Script permissions set"

# Test imports
echo "🧪 Testing critical imports..."
python3 -c "
import sys
try:
    import fastapi
    print('✅ FastAPI imported successfully')
except ImportError as e:
    print(f'❌ FastAPI import failed: {e}')
    sys.exit(1)

try:
    import mlx.core as mx
    print('✅ MLX imported successfully')
except ImportError as e:
    print(f'⚠️  MLX import failed: {e}')
    print('   MLX is required for optimal Apple Silicon performance')

try:
    import faiss
    print('✅ FAISS imported successfully')
except ImportError as e:
    print(f'❌ FAISS import failed: {e}')
    sys.exit(1)

try:
    import sentence_transformers
    print('✅ SentenceTransformers imported successfully')
except ImportError as e:
    print(f'❌ SentenceTransformers import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Some critical imports failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Customize your .env file if needed"
echo "3. Start the application: ./scripts/start.sh"
echo "4. Or use Docker: docker-compose up"
echo ""
echo "For development with hot reload: uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "📚 Documentation: Open http://localhost:8000/docs after starting"
echo "🎯 Web Interface: Open http://localhost:8000 after starting"
echo ""
echo "Happy chatting with LLMind! 🦙⚡" 