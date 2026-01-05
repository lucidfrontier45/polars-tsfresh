# Suggested Commands

## Development Commands
```bash
# Install dependencies
uv sync --no-dev  # runtime only
uv sync           # with dev dependencies

# Linting and Formatting
poe ruff_check     # check linting issues
poe ruff_fix       # auto-fix linting issues
poe ruff_format    # format code
poe pyrefly_check  # type checking
poe check          # run all checks (ruff + pyrefly)
poe format         # fix + format code

# Testing
poe test           # run tests with coverage

# Full validation
poe format && poe check && poe test
```

## Git Commands
```bash
git status
git add .
git commit -m "message"
git push
```

## File Operations
```bash
ls -la
find . -name "*.py"
grep -r "pattern" src/
```