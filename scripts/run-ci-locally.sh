#!/bin/bash
# Run CI checks locally before pushing to GitHub
# This mimics what GitHub Actions will run

set -e

echo "🚀 Running local CI checks for LLMind..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Virtual environment not detected. Activating..."
    if [ -f "activate_venv.sh" ]; then
        source activate_venv.sh
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        print_error "No virtual environment found. Please create and activate one first."
        exit 1
    fi
fi

# Install CI dependencies if not present
print_status "Ensuring CI dependencies are installed..."
pip install -q \
    black \
    isort \
    flake8 \
    flake8-docstrings \
    flake8-bugbear \
    flake8-comprehensions \
    pylint \
    mypy \
    bandit \
    detect-secrets \
    pydocstyle \
    safety \
    pytest \
    pytest-cov \
    types-requests \
    types-PyYAML \
    pip-audit

# Track failures
FAILURES=0

echo ""
echo "🔍 Running Code Quality Checks..."
echo "=================================="

# Black formatting check
print_status "Checking code formatting with Black..."
if black --check --diff . ; then
    print_success "Black formatting check passed"
else
    print_error "Black formatting check failed"
    print_status "Run 'black .' to fix formatting issues"
    ((FAILURES++))
fi

# isort import sorting check
print_status "Checking import sorting with isort..."
if isort --check-only --diff . ; then
    print_success "isort check passed"
else
    print_error "isort check failed"
    print_status "Run 'isort .' to fix import sorting"
    ((FAILURES++))
fi

# flake8 linting
print_status "Running flake8 linting..."
if flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics ; then
    print_success "flake8 critical errors check passed"
else
    print_error "flake8 found critical errors"
    ((FAILURES++))
fi

# Extended flake8 check (warnings only)
print_status "Running extended flake8 check..."
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

# pylint check
print_status "Running pylint analysis..."
if pylint --fail-under=7.0 core/ config.py main.py ; then
    print_success "pylint check passed"
else
    print_warning "pylint found issues (continuing...)"
fi

# mypy type checking
print_status "Running mypy type checking..."
if mypy . --ignore-missing-imports ; then
    print_success "mypy type checking passed"
else
    print_warning "mypy found type issues (continuing...)"
fi

echo ""
echo "🔒 Running Security Checks..."
echo "============================="

# bandit security check
print_status "Running bandit security scan..."
if bandit -r . -x tests/ -ll ; then
    print_success "bandit security scan passed"
else
    print_warning "bandit found potential security issues (continuing...)"
fi

# safety dependency check
print_status "Running safety dependency check..."
if safety check --ignore 70612 ; then
    print_success "safety dependency check passed"
else
    print_warning "safety found vulnerable dependencies (continuing...)"
fi

# detect-secrets check
print_status "Running secret detection..."
if detect-secrets scan --baseline .secrets.baseline ; then
    print_success "secret detection passed"
else
    print_error "detect-secrets found potential secrets"
    ((FAILURES++))
fi

# pip-audit dependency audit
print_status "Running pip-audit dependency audit..."
if pip-audit --desc ; then
    print_success "pip-audit passed"
else
    print_warning "pip-audit found issues (continuing...)"
fi

echo ""
echo "🧪 Running Tests..."
echo "=================="

# Run tests
print_status "Running pytest unit tests..."
if pytest tests/ -v --cov=core --cov=config --cov-report=term-missing ; then
    print_success "All tests passed"
else
    print_error "Some tests failed"
    ((FAILURES++))
fi

echo ""
echo "📚 Running Documentation Checks..."
echo "=================================="

# pydocstyle documentation check
print_status "Running pydocstyle documentation check..."
if pydocstyle --convention=google --add-ignore=D100,D101,D102,D103,D104,D107 . ; then
    print_success "Documentation style check passed"
else
    print_warning "Documentation style issues found (continuing...)"
fi

echo ""
echo "🔧 Running Build Check..."
echo "========================="

# Check if application can start
print_status "Testing application startup..."
if timeout 10s python -c "
import sys
sys.path.append('.')
try:
    from main import app
    print('✅ Application imports successfully')
except Exception as e:
    print(f'❌ Application import failed: {e}')
    sys.exit(1)
" ; then
    print_success "Application startup check passed"
else
    print_error "Application startup check failed"
    ((FAILURES++))
fi

# Check if prettier is available and run it
if command -v prettier &> /dev/null; then
    print_status "Checking JavaScript/CSS/HTML formatting with prettier..."
    if prettier --check "static/**/*.{js,css,html}" "templates/**/*.html" 2>/dev/null ; then
        print_success "prettier formatting check passed"
    else
        print_warning "prettier found formatting issues (run 'prettier --write' to fix)"
    fi
else
    print_warning "prettier not found, skipping frontend formatting check"
fi

echo ""
echo "📊 CI Results Summary"
echo "===================="

if [ $FAILURES -eq 0 ]; then
    print_success "🎉 All critical CI checks passed! Ready to push."
    echo ""
    print_status "To push your changes:"
    echo "  git add ."
    echo "  git commit -m 'your commit message'"
    echo "  git push"
else
    print_error "❌ $FAILURES critical check(s) failed. Please fix before pushing."
    echo ""
    print_status "Common fixes:"
    echo "  • Run 'black .' to fix formatting"
    echo "  • Run 'isort .' to fix imports"
    echo "  • Check and fix any test failures"
    echo "  • Review security and secret detection issues"
    exit 1
fi

echo ""
print_status "Optional: Run 'pre-commit run --all-files' for additional checks"
