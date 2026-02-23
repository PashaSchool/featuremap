# faultline

[![PyPI version](https://img.shields.io/pypi/v/faultline?color=blue)](https://pypi.org/project/faultline/)
[![Python](https://img.shields.io/pypi/pyversions/faultline)](https://pypi.org/project/faultline/)
[![License](https://img.shields.io/github/license/pashaSchool/faultline)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/pashaSchool/faultline?style=social)](https://github.com/pashaSchool/faultline)

> Turn your git history into a feature risk map — no Jira required.

**faultline** analyzes your git commit history to automatically detect features and modules in your codebase, then shows which ones are accumulating the most bug fixes — your technical debt hotspots.

No integrations. No configuration. Just point it at any git repo.

---

## Why faultline?

Engineering managers need to know *where* the technical debt is before they can act on it. Most tools require you to tag tickets in Jira or Linear. faultline reads the truth directly from your git history.

```
✗ payments    — health: 23   38 bug fixes / 112 commits (33.9%)
!  auth        — health: 54   12 bug fixes / 48 commits  (25.0%)
✓  dashboard   — health: 91    2 bug fixes / 67 commits   (3.0%)
```

---

## Installation

### Homebrew (macOS / Linux) — coming soon

```bash
brew install pashaSchool/tap/faultline
```

### pip

```bash
pip install faultline
```

### From source

```bash
git clone https://github.com/pashaSchool/faultline
cd faultline
pip install -e .
```

---

## Quick Start

```bash
# Analyze the current directory
faultline analyze .

# Analyze a specific repo
faultline analyze ./path/to/repo

# Focus on a source subdirectory (recommended for frontend/monorepo projects)
faultline analyze . --src src/

# Last 90 days, show top 5 risk zones
faultline analyze . --days 90 --top 5

# Save report to a custom path
faultline analyze . --output ./reports/feature-map.json

# Just print, don't save
faultline analyze . --no-save
```

---

## AI-Powered Feature Detection

By default faultline groups files by directory structure (fast, no API needed). With `--llm` enabled, Claude or a local Ollama model reads the full file tree and returns a **semantic feature map** — grouping files by business domain, not folder names.

```
Without --llm:  "components", "views", "hooks"   ← technical layers
With --llm:     "user-auth", "payments", "dashboard"  ← business features
```

### Anthropic Claude (cloud)

```bash
# Pass your API key directly
faultline analyze . --llm --api-key sk-ant-...

# Or use an environment variable
export ANTHROPIC_API_KEY=sk-ant-...
faultline analyze . --llm

# With source folder filter
faultline analyze . --llm --src src/
```

Get your API key at [console.anthropic.com](https://console.anthropic.com) → API Keys.
Uses **Claude Haiku** for cost-efficient analysis (~$0.001 per repo).

### Ollama (local, free, private)

Run analysis entirely on your machine — no API key, no data leaves your computer.
Recommended for private repositories.

```bash
# 1. Install Ollama
brew install ollama        # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model (one time, ~4.7 GB)
ollama pull qwen2.5-coder:7b

# 3. Start the server
ollama serve

# 4. Install the ollama package
pip install 'faultline[ollama]'

# 5. Run
faultline analyze . --llm --provider ollama --src src/
```

Use a lighter model if RAM is limited:

```bash
ollama pull llama3.2:3b
faultline analyze . --llm --provider ollama --model llama3.2:3b
```

The API key is validated **before** the analysis starts — no waiting to discover a bad key.
Falls back silently to heuristic mode if the LLM is unavailable.

---

## Focusing on a Subdirectory

Use `--src` to restrict analysis to a specific folder. Everything outside that path is ignored — including tooling, config files, and test infrastructure.

```bash
# Frontend app with sources in src/
faultline analyze . --src src/

# Next.js app
faultline analyze . --src app/

# Monorepo — analyze one package
faultline analyze . --src packages/api/src/
```

What `--src` automatically excludes regardless of location:
- Tooling: `.github/`, `.husky/`, `.storybook/`, `.circleci/`
- Build output: `dist/`, `build/`, `.next/`, `coverage/`
- Dependencies: `node_modules/`, `vendor/`, `.venv/`
- Binary and generated files: `.map`, `.lock`, `.woff`, `.ttf`, and others

---

## Output

### Terminal

```
╭─────────────────────────────────────────────╮
│             FeatureMap Analysis              │
│                                              │
│ Repository:          /path/to/repo           │
│ Analyzed:            last 365 days           │
│ Total commits:       847                     │
│ Features found:      12                      │
│ Bug fix commits:     143                     │
│ Average health score: 61.3/100               │
╰─────────────────────────────────────────────╯

╭─────────────────── Features by Risk ────────────────────╮
│   Feature       Health   Commits   Bug Fixes   Bug %    │
│ ✗ payments        23       112        38       33.9%    │
│ ! auth            54        48        12       25.0%    │
│ ✓ dashboard       91        67         2        3.0%    │
╰─────────────────────────────────────────────────────────╯

Top 3 risk zones:
  1. payments — 38 bug fixes out of 112 commits (33.9%)
  2. auth     — 12 bug fixes out of 48 commits  (25.0%)
```

### JSON

Results are saved to `.faultline/feature-map.json` by default:

```json
{
  "repo_path": "/path/to/repo",
  "analyzed_at": "2026-02-22T10:00:00Z",
  "total_commits": 847,
  "date_range_days": 365,
  "features": [
    {
      "name": "payments",
      "description": "Handles Stripe payment processing and subscription billing.",
      "health_score": 23.0,
      "bug_fix_ratio": 0.339,
      "bug_fixes": 38,
      "total_commits": 112,
      "authors": ["alice", "bob"],
      "paths": ["src/payments/stripe.py", "src/payments/webhooks.py"]
    }
  ]
}
```

---

## CLI Reference

### `faultline analyze [REPO_PATH]`

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | — | Subdirectory to focus on, e.g. `src/` or `app/` |
| `--days` | `365` | Days of git history to analyze |
| `--max-commits` | `5000` | Maximum commits to read |
| `--top` | `3` | Number of top risk zones to highlight |
| `--output` | `.faultline/feature-map.json` | Output file path |
| `--no-save` | — | Skip saving JSON output |
| `--llm` | `false` | Use AI for semantic feature detection |
| `--provider` | `anthropic` | LLM provider: `anthropic` or `ollama` |
| `--model` | — | Model override (default: `claude-haiku-4-5` / `qwen2.5-coder:7b`) |
| `--api-key` | env | Anthropic API key (`ANTHROPIC_API_KEY` env var) |
| `--ollama-url` | `http://localhost:11434` | Custom Ollama server URL |

---

## How It Works

1. **Reads git history** — up to `--max-commits` commits within the requested date range
2. **Collects tracked files** — respects `--src` filter, skips build output and tooling directories
3. **Maps files to features** — two modes:
   - **Heuristic** (default): groups by directory structure (`src/payments/` → `payments`)
   - **LLM** (`--llm`): sends the file tree to Claude or Ollama, gets back a semantic `{feature: [files]}` mapping based on business domain
4. **Scans commit history** — for each feature, counts total commits and bug fix commits per file
5. **Calculates health scores** — `100 - (bug_fix_ratio × 200)`, weighted by commit activity

---

## Health Score

| Score | Status | Meaning |
|-------|--------|---------|
| 75–100 | ✓ Green | Healthy — low bug fix ratio |
| 50–74 | ! Yellow | Watch — moderate technical debt |
| 0–49 | ✗ Red | Critical — high bug fix ratio |

---

## License

[Apache 2.0](LICENSE)
