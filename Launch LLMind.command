#!/bin/bash
# LLMind Launcher for macOS
# Double-click this file in Finder to start LLMind

# Change to the directory containing this script
cd "$(dirname "$0")"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "🦙 LLMind Launcher"
echo "=================="

# Check if Python 3 is installed
if ! command_exists python3; then
    echo "❌ Python 3 is not installed."
    echo "Please install Python 3 from https://www.python.org/"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ This doesn't appear to be a git repository."
    echo "Please download the complete LLMind project."
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if virtual environment exists, create if it doesn't
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment."
        read -p "Press Enter to exit..."
        exit 1
    fi

    echo "Setting up LLMind for the first time..."
    echo "This may take a few minutes..."

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "❌ Failed to install requirements."
            read -p "Press Enter to exit..."
            exit 1
        fi
    else
        echo "❌ requirements.txt not found."
        echo "Please ensure you're in the correct LLMind directory."
        read -p "Press Enter to exit..."
        exit 1
    fi

    echo "✅ Setup complete!"
else
    # Activate existing virtual environment
    source venv/bin/activate
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory."
    echo "Please ensure you're in the correct LLMind directory."
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is already in use."
    echo "LLMind may already be running."
    echo ""
    echo "Options:"
    echo "1. Open LLMind in browser (if already running)"
    echo "2. Kill existing process and restart"
    echo "3. Exit"
    echo ""
    read -p "Choose option (1-3): " choice

    case $choice in
        1)
            echo "🌐 Opening LLMind in browser..."
            open "http://localhost:8000"
            echo "If LLMind doesn't load, try option 2 to restart."
            read -p "Press Enter to exit..."
            exit 0
            ;;
        2)
            echo "🔄 Killing existing process..."
            # Kill process using port 8000
            lsof -ti:8000 | xargs kill -9 2>/dev/null
            # Also kill any remaining uvicorn processes
            pkill -f "uvicorn.*main:app" 2>/dev/null || true
            # Kill any Python processes running LLMind
            pkill -f "python.*main.py" 2>/dev/null || true
            sleep 3
            ;;
        3)
            echo "👋 Goodbye!"
            read -p "Press Enter to exit..."
            exit 0
            ;;
        *)
            echo "Invalid option. Exiting..."
            read -p "Press Enter to exit..."
            exit 1
            ;;
    esac
fi

echo "🚀 Starting LLMind..."
echo ""
echo "📋 Important Notes:"
echo "   • LLMind will be available at: http://localhost:8000"
echo "   • Leave this terminal window open while using LLMind"
echo "   • To stop LLMind, press Ctrl+C in this window"
echo ""

# Start the application
python main.py

# This will run when the script exits (including Ctrl+C)
trap 'echo ""; echo "🛑 LLMind stopped"; echo "You can close this window now."; read -p "Press Enter to exit..."' EXIT
