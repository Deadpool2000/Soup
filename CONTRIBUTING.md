# Contributing to Soup

Thank you for your interest in contributing to Soup! We welcome bug reports, feature requests, and pull requests from the community.

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
```

### 2. Set Up Development Environment

Install the project in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

This installs:
- `pytest` for testing
- `ruff` for linting
- `pytest-cov` for coverage
- `httpx` for HTTP testing

### 3. Verify Setup

Run the test suite to confirm everything works:

```bash
pytest tests/ -v --tb=short
```

Run the linter:

```bash
ruff check soup_cli/ tests/
```

## Code Style

We use **ruff** for all code style and linting. Before committing, run:

```bash
# Check for issues
ruff check soup_cli/ tests/

# Auto-fix issues
ruff check --fix soup_cli/ tests/
```

### Style Guidelines

- **Line length:** 100 characters (enforced by ruff)
- **Imports:** Sorted and organized (ruff I rule)
- **Naming:** No single-letter variable names (ruff E741) — use `entry`, `part`, `length` instead of `l`, `p`, etc.
- **Lazy imports:** Heavy dependencies (torch, transformers, peft, trl, etc.) should be imported inside functions, not at module level, to keep the CLI responsive
- **Config validation:** Always use Pydantic v2 with `BaseModel` and `Field`
- **Output:** Use `rich.console.Console` for all output — never bare `print()`
- **Type hints:** Always include type hints for function parameters and return values

Example:

```python
# ❌ WRONG
from torch import cuda
import transformers

def train():
    print("Starting training")
    model = transformers.AutoModel.from_pretrained("llama-7b")

# ✅ CORRECT
def train():
    from torch import cuda
    import transformers
    
    console = Console()
    console.print("Starting training")
    model = transformers.AutoModel.from_pretrained("llama-7b")
```

## Project Structure

Key directories:

```
soup_cli/
  cli.py              - Main entry point, command routing
  commands/           - Command implementations (train, chat, eval, etc.)
  config/             - Config schema (schema.py) and loader (loader.py)
  data/               - Data loading and format conversion
  trainer/            - Training wrappers (SFT, DPO, GRPO, PPO, reward_model)
  monitoring/         - Callbacks and live dashboard
  experiment/         - SQLite experiment tracking
  utils/              - GPU detection, batch size estimation, error handling
  ui/                 - Web UI (FastAPI + HTML/JS)

templates/            - YAML config templates (chat, code, medical, vision, rlhf, reasoning)
tests/                - Test suite (40+ files, 600+ tests)
examples/             - Real-world config examples and datasets
```

## Running Tests

### All Tests

```bash
pytest tests/ -v --tb=short
```

### Single Test File

```bash
pytest tests/test_config.py -v
```

### Single Test

```bash
pytest tests/test_data.py::test_detect_alpaca_format -v
```

### With Coverage

```bash
pytest tests/ --cov=soup_cli --cov-report=html
```

### Test Categories

- `test_config.py` — Config loading and validation
- `test_data.py` — Data format detection and conversion
- `test_trainer_*.py` — Individual trainer tests
- `test_smoke_train.py` — Full pipeline tests (GPU required)
- `test_cli.py` — Command-line interface tests
- `test_errors.py` — Error message handling
- `test_*_command.py` — Specific command tests (chat, push, eval, etc.)

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write code following the style guidelines above
- Add tests for new functionality
- Update docstrings and comments
- Keep commits focused and logical

### 3. Run Tests & Lint

Before pushing, ensure everything passes:

```bash
# Lint first
ruff check --fix soup_cli/ tests/

# Then run tests
pytest tests/ -v --tb=short
```

If you've added new test files, increase the test count in `plan.md`.

### 4. Commit

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: descriptive message"
```

### 5. Push & Open a PR

```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub with:
- Clear title describing the change
- Description of what and why
- Reference any related issues (e.g., "Closes #123")
- Test results

## Submitting a Pull Request

### PR Template

Please use the following structure:

```markdown
## What's this PR about?

Brief description of the change.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement

## Testing

Describe how you tested this (e.g., `pytest tests/test_X.py -v`).

## Checklist

- [ ] Linting passes: `ruff check soup_cli/ tests/`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] New tests added for new functionality
- [ ] Docstrings and comments added
- [ ] No breaking changes (or documented)

## Related Issues

Closes #123 (if applicable)
```

## Architecture & Design Decisions

### Lazy Imports for Speed

Heavy ML imports (torch, transformers, trl) are imported inside command handlers so the CLI stays fast. Users can run `soup version` or `soup --help` instantly without waiting for PyTorch to load.

### Pydantic for Config Validation

All YAML configs are validated using Pydantic v2 models. These models are the single source of truth for valid fields and defaults. See `config/schema.py`.

### Trainers as Wrappers

`trainer/sft.py`, `trainer/dpo.py`, `trainer/grpo.py`, `trainer/ppo.py` wrap HuggingFace TRL trainers with:
- Auto quantization (BitsAndBytes, torchao QAT)
- Auto LoRA setup (PEFT)
- Auto batch size estimation
- Progress bar integration

### Experiment Tracking is SQLite

No external dependencies required. All runs, metrics, and eval results go to `~/.soup/experiments.db`.

### Data Format Normalization

Multiple formats (Alpaca, ShareGPT, ChatML, LLaVA, ShareGPT4V) are normalized to a unified `{"messages": [...]}` structure in `data/formats.py`.

## Adding a New Feature

### 1. New Training Task Type

If adding a new training algorithm (e.g., DPO, GRPO):

1. Create `trainer/your_trainer.py` with a class inheriting from `BaseTrainer`
2. Add Pydantic config class to `config/schema.py`
3. Add template to `templates/your.yaml` and `config/schema.py`
4. Update `commands/train.py` to route to your trainer
5. Add 30+ tests in `tests/test_your_trainer.py`
6. Update `CLAUDE.md` and `README.md`

### 2. New Data Format

1. Add detection and conversion logic to `data/formats.py`
2. Add tests in `tests/test_formats.py`
3. Update `data/loader.py` if needed
4. Document in `CLAUDE.md`

### 3. New Command

1. Create `commands/your_command.py` with a handler function
2. Register in `soup_cli/cli.py` with `@app.command()`
3. Add tests in `tests/test_your_command.py`
4. Update help text and README

## CI/CD

GitHub Actions runs on every push:
- **ruff** linting (must pass)
- **pytest** on Python 3.9, 3.11, 3.12 (must pass)

See `.github/workflows/ci.yml`.

## Releases

The project follows semantic versioning: `MAJOR.MINOR.PATCH`

### Version Bump Process

1. Update version in `pyproject.toml` and `soup_cli/__init__.py`
2. Run full test suite and linting
3. Update `CLAUDE.md`, `README.md`, `plan.md`
4. Commit with message: `Release v0.X.0`
5. Tag: `git tag v0.X.0 && git push --tags`
6. GitHub Actions auto-publishes to PyPI

See `CLAUDE.md` for the complete release checklist.

## Community

- **Issues:** Report bugs and request features on [GitHub Issues](https://github.com/MakazhanAlpamys/Soup/issues)
- **Discussions:** Ask questions on [GitHub Discussions](https://github.com/MakazhanAlpamys/Soup/discussions)
- **Code of Conduct:** Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Security:** Report security issues to [SECURITY.md](SECURITY.md)

## Questions?

- Check the [README](README.md) for quick start and features
- Check [CLAUDE.md](CLAUDE.md) for architecture details
- Open a GitHub Discussion for questions
- Join the community on Reddit ([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/))

Thank you for contributing! 🍲
