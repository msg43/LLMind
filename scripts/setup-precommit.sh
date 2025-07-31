#!/bin/bash
# Setup script for pre-commit hooks in LLMind project

set -e

echo "🔧 Setting up pre-commit hooks for LLMind..."

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

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "This script must be run from the root of a git repository."
    exit 1
fi

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

print_status "Virtual environment: $VIRTUAL_ENV"

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    print_status "Installing pre-commit..."
    pip install pre-commit
else
    print_status "pre-commit is already installed"
fi

# Install development dependencies
print_status "Installing development dependencies..."
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
    types-PyYAML

# Install Node.js dependencies for prettier (if Node.js is available)
if command -v npm &> /dev/null; then
    print_status "Installing prettier for JavaScript/HTML/CSS formatting..."
    npm install -g prettier
else
    print_warning "Node.js not found. Prettier will be handled by pre-commit."
fi

# Install pre-commit hooks
print_status "Installing pre-commit hooks..."
pre-commit install

# Install commit message hooks
print_status "Installing commit message hooks..."
pre-commit install --hook-type commit-msg

# Generate initial secrets baseline
if [ ! -f ".secrets.baseline" ]; then
    print_status "Generating initial secrets baseline..."
    detect-secrets scan --baseline .secrets.baseline
else
    print_status "Secrets baseline already exists"
fi

# Run pre-commit on all files to check setup
print_status "Running pre-commit checks on all files (this may take a while)..."
if pre-commit run --all-files; then
    print_success "All pre-commit checks passed!"
else
    print_warning "Some pre-commit checks failed. This is normal on first setup."
    print_status "Running auto-fixes..."

    # Run auto-fixing tools
    print_status "Running black (Python formatter)..."
    black . 2>/dev/null || true

    print_status "Running isort (import sorter)..."
    isort . 2>/dev/null || true

    print_status "Running prettier (JS/HTML/CSS formatter)..."
    if command -v prettier &> /dev/null; then
        prettier --write "static/**/*.{js,css,html}" 2>/dev/null || true
        prettier --write "templates/**/*.html" 2>/dev/null || true
    fi

    print_status "Re-running pre-commit checks..."
    pre-commit run --all-files || true
fi

# Create a simple commit message template
if [ ! -f ".gitmessage" ]; then
    print_status "Creating commit message template..."
    cat > .gitmessage << 'EOF'
# Type: Brief description (50 chars max)
#
# Longer explanation if needed (wrap at 72 chars)
#
# Types:
# feat: New feature
# fix: Bug fix
# docs: Documentation changes
# style: Code style changes (formatting, etc.)
# refactor: Code refactoring
# test: Adding or updating tests
# chore: Maintenance tasks
# perf: Performance improvements
# ci: CI/CD changes
EOF

    git config commit.template .gitmessage
    print_success "Commit message template created and configured"
fi

# Create pre-commit configuration backup
cp .pre-commit-config.yaml .pre-commit-config.yaml.backup 2>/dev/null || true

print_success "Pre-commit setup complete!"
echo ""
print_status "Next steps:"
echo "  1. Make some changes to your code"
echo "  2. Run 'git add .' to stage changes"
echo "  3. Run 'git commit' to trigger pre-commit hooks"
echo ""
print_status "Useful commands:"
echo "  • pre-commit run --all-files    # Run hooks on all files"
echo "  • pre-commit run <hook-name>    # Run specific hook"
echo "  • pre-commit autoupdate        # Update hook versions"
echo "  • pre-commit uninstall         # Remove hooks"
echo ""
print_status "Configuration files created:"
echo "  • .pre-commit-config.yaml      # Pre-commit configuration"
echo "  • .secrets.baseline            # Secrets detection baseline"
echo "  • pyproject.toml               # Python tool configuration"
echo "  • .prettierrc                  # Prettier configuration"
echo "  • .gitmessage                  # Commit message template"
