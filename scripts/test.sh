#!/bin/bash

# LLMind Test Script
# ===================
# This script runs the comprehensive test suite for LLMind

set -e

echo "🧪 Running LLMind Test Suite..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./scripts/setup.sh first to set up the project."
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check if tests directory exists
if [ ! -d "tests" ]; then
    echo "❌ Tests directory not found!"
    echo "Please ensure the test suite is properly set up."
    exit 1
fi

# Create test output directory
mkdir -p test_output

# Set test environment variables
export TESTING=true
export LOG_LEVEL=error
export DATA_DIR=./test_output/data

# Create test data directories
echo "📁 Setting up test environment..."
mkdir -p test_output/data/{documents,vector_store,models,audio,backups,logs,cache}

# Parse command line arguments
COVERAGE=false
VERBOSE=false
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --test|-t)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --coverage    Run tests with coverage report"
            echo "  -v, --verbose     Run tests with verbose output"
            echo "  -t, --test TEST   Run specific test file or function"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run all tests"
            echo "  $0 --coverage              # Run with coverage"
            echo "  $0 --test test_api.py      # Run specific test file"
            echo "  $0 -v -c                   # Verbose with coverage"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=html:test_output/htmlcov --cov-report=term-missing"
fi

if [ -n "$SPECIFIC_TEST" ]; then
    PYTEST_CMD="$PYTEST_CMD tests/$SPECIFIC_TEST"
else
    PYTEST_CMD="$PYTEST_CMD tests/"
fi

# Add additional pytest options
PYTEST_CMD="$PYTEST_CMD --tb=short --strict-markers"

echo "🚀 Running tests..."
echo "Command: $PYTEST_CMD"
echo ""

# Run the tests
if $PYTEST_CMD; then
    echo ""
    echo "✅ All tests passed!"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo "📊 Coverage report generated:"
        echo "   HTML Report: test_output/htmlcov/index.html"
        
        # Try to open coverage report on macOS
        if [[ "$OSTYPE" == "darwin"* ]] && command -v open >/dev/null 2>&1; then
            echo "   Opening coverage report..."
            open test_output/htmlcov/index.html
        fi
    fi
    
    echo ""
    echo "🧹 Cleaning up test data..."
    rm -rf test_output/data
    
    echo "✨ Test run completed successfully!"
    exit 0
else
    echo ""
    echo "❌ Some tests failed!"
    echo ""
    echo "📝 Test artifacts available in test_output/"
    if [ "$COVERAGE" = true ]; then
        echo "📊 Coverage report: test_output/htmlcov/index.html"
    fi
    
    echo ""
    echo "💡 Tips for debugging:"
    echo "   • Run with --verbose for more detailed output"
    echo "   • Use --test <filename> to run specific tests"
    echo "   • Check logs in test_output/data/logs/"
    
    exit 1
fi 