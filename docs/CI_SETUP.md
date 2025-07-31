# Continuous Integration Setup for LLMind

This document describes the comprehensive CI/CD setup for the LLMind project, including pre-commit hooks, GitHub Actions, and local development tools.

## Overview

The CI system includes:
- **Pre-commit hooks** for immediate feedback during development
- **GitHub Actions** for comprehensive CI/CD pipeline
- **Local CI scripts** for testing before pushing
- **Security scanning** and **code quality** checks
- **Multi-version Python testing**
- **Performance benchmarking**

## Quick Start

### 1. Setup Pre-commit Hooks

```bash
# Run the automated setup script
./scripts/setup-precommit.sh

# Or manually install
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

### 2. Run Local CI Checks

```bash
# Run all CI checks locally before pushing
./scripts/run-ci-locally.sh

# Run specific checks
pre-commit run --all-files
black .
pytest tests/
```

## Pre-commit Hooks

The pre-commit configuration (`.pre-commit-config.yaml`) includes:

### Code Formatting
- **Black** - Python code formatting
- **isort** - Import sorting
- **Prettier** - JavaScript/HTML/CSS formatting

### Code Quality
- **flake8** - Python linting with extensions
- **pylint** - Advanced Python analysis
- **mypy** - Type checking

### Security
- **bandit** - Security vulnerability scanning
- **detect-secrets** - Secret detection
- **safety** - Dependency vulnerability checking

### General Checks
- **trailing-whitespace** - Remove trailing spaces
- **end-of-file-fixer** - Ensure files end with newline
- **check-yaml/json/toml** - Syntax validation
- **check-merge-conflict** - Detect merge conflict markers
- **debug-statements** - Find debug prints

### Documentation
- **pydocstyle** - Documentation style checking

### Shell Scripts
- **shellcheck** - Shell script linting

## GitHub Actions CI Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) includes multiple jobs:

### 1. Code Quality & Security
- Python formatting and linting checks
- Security vulnerability scanning
- Secret detection
- Frontend code formatting

### 2. Testing
- Multi-version Python testing (3.9-3.12)
- Unit tests with coverage reporting
- Coverage upload to Codecov

### 3. Build & Integration
- Application startup verification
- API endpoint testing

### 4. Documentation
- Documentation style checking

### 5. Security Audit
- Comprehensive security scanning
- Dependency vulnerability assessment

### 6. Performance Benchmarks
- MLX performance profiling (main branch only)

### 7. Dependency Check
- pip-audit for dependency vulnerabilities

## Configuration Files

### `.pre-commit-config.yaml`
Main pre-commit configuration with all hooks and their settings.

### `pyproject.toml`
Central configuration for Python tools:
- Black formatting settings
- isort import sorting
- pylint rules
- mypy type checking
- pytest configuration
- Coverage settings

### `.prettierrc`
Prettier configuration for JavaScript/HTML/CSS formatting.

### `.secrets.baseline`
Baseline for detect-secrets to prevent false positives.

### `.gitignore`
Updated to exclude CI artifacts and development files.

## Local Development Workflow

### Daily Development
1. **Make changes** to your code
2. **Stage changes**: `git add .`
3. **Commit**: `git commit -m "your message"`
   - Pre-commit hooks run automatically
   - Auto-fix issues when possible
4. **Push**: `git push`

### Before Major Changes
1. **Run local CI**: `./scripts/run-ci-locally.sh`
2. **Fix any issues** identified
3. **Test thoroughly**: `pytest tests/`
4. **Commit and push**

### Code Quality Fixes
```bash
# Auto-fix formatting
black .
isort .

# Check for issues
flake8 .
pylint core/ config.py main.py
mypy .

# Run security checks
bandit -r . -x tests/
safety check
```

## CI Pipeline Triggers

### Automatic Triggers
- **Push to main/develop** - Full CI pipeline
- **Pull requests** - Full CI pipeline
- **Manual trigger** - GitHub Actions workflow_dispatch

### Performance Benchmarks
- Only run on **main branch pushes**
- Generate performance reports
- Upload as artifacts

## Customization

### Adding New Hooks
Edit `.pre-commit-config.yaml`:
```yaml
- repo: https://github.com/new-tool/repo
  rev: v1.0.0
  hooks:
    - id: new-hook
      args: [--option=value]
```

### Modifying Python Tool Settings
Edit `pyproject.toml`:
```toml
[tool.black]
line-length = 88

[tool.pylint.messages_control]
disable = ["C0114", "C0115"]
```

### Excluding Files
Add to the `exclude` section in `.pre-commit-config.yaml`:
```yaml
exclude: |
  (?x)^(
    new_pattern_to_exclude|
    another_pattern
  )$
```

## Troubleshooting

### Pre-commit Hook Failures
```bash
# Skip hooks for emergency commits
git commit --no-verify -m "emergency fix"

# Fix formatting issues
black .
isort .

# Update hooks
pre-commit autoupdate
```

### CI Failures
1. **Check the specific failing job** in GitHub Actions
2. **Run the same check locally**:
   ```bash
   ./scripts/run-ci-locally.sh
   ```
3. **Fix issues** and push again

### Common Issues

#### Black/isort Conflicts
```bash
# Run in order
isort .
black .
```

#### Secrets Detection False Positives
```bash
# Update baseline
detect-secrets scan --baseline .secrets.baseline
```

#### Test Failures
```bash
# Run tests with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/test_specific.py::test_function -v
```

## Best Practices

### Commit Messages
Use the conventional commit format:
```
type: brief description (50 chars max)

Longer explanation if needed (wrap at 72 chars)

Types: feat, fix, docs, style, refactor, test, chore, perf, ci
```

### Code Quality
- **Write tests** for new features
- **Keep functions small** and focused
- **Add docstrings** for public APIs
- **Use type hints** where appropriate

### Security
- **Never commit secrets** or credentials
- **Review dependency updates** for security
- **Use environment variables** for configuration

### Performance
- **Profile performance-critical code**
- **Monitor CI pipeline duration**
- **Use caching** for dependencies

## Maintenance

### Regular Tasks
- **Update pre-commit hooks**: `pre-commit autoupdate`
- **Update GitHub Actions**: Check for newer action versions
- **Review security reports**: Monitor bandit and safety outputs
- **Update dependencies**: Keep requirements.txt current

### Monthly Reviews
- **Analyze CI metrics**: Pipeline success rates and duration
- **Review security baseline**: Update `.secrets.baseline` if needed
- **Update tool configurations**: Adjust rules based on project evolution

## Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Black Documentation](https://black.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)
- [pylint Documentation](https://pylint.pycqa.org/)
- [bandit Documentation](https://bandit.readthedocs.io/)

## Support

For issues with the CI setup:
1. Check this documentation
2. Review the specific tool documentation
3. Check GitHub Actions logs for detailed error messages
4. Run `./scripts/run-ci-locally.sh` to reproduce issues locally 