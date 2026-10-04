# Development tasks. Everything assumes the nix devshell is active
# (direnv allow, or `nix develop -c just <task>`).

# Show available tasks.
default:
    @just --list

# Install git hooks (idempotent). The devshell shellHook does this too.
setup:
    lefthook install

# This is the only recipe that fetches Python packages; `eval-test` (inside `check`) then runs offline.
# Install JavaScript dependencies and the eval harness's Python venv, both strictly from their lockfiles.
install:
    pnpm install --frozen-lockfile
    uv sync --project eval --locked

# Rebuild dist/ on every change; serve it with `just serve` in another terminal.
dev:
    pnpm run dev

# Build the static site into dist/.
build:
    pnpm run build

# Serve dist/ on http://localhost:8000 (localhost counts as a secure context, so WebGPU and the microphone work).
serve: build
    python3 -m http.server 8000 --directory dist

# Run the unit tests.
test:
    pnpm run test

# Type-check without emitting.
typecheck:
    pnpm run typecheck

# Lint TypeScript with oxlint.
lint:
    pnpm run lint

# Format every file treefmt owns (nix, shell, TS/JS/CSS/HTML/JSON/YAML/Markdown), plus the justfile itself.
fmt:
    nix fmt
    just --fmt --unstable

# Verify formatting without rewriting anything, the way CI does.
fmt-check:
    nix flake check
    just --fmt --check --unstable

# Verify every justfile recipe has a description comment on the line above it.
justfile-lint:
    bash scripts/justfile-comment-lint.sh

# Lint the GitHub Actions workflows.
actions-lint:
    actionlint

# Full-history secret scan (the pre-commit hook only sees staged changes).
secrets:
    gitleaks git --redact

# Pin every GitHub Action to a 40-char SHA (--min-age 1 refuses releases younger than a day).
pin:
    pinact run --min-age 1 .github/workflows/*.yml

# Offline: verify every `uses:` is a 40-char SHA without calling the GitHub API.
pin-check:
    pinact run -fix=false -no-api .github/workflows/*.yml

# eval-run is left out on purpose: it needs a GPU and a real Chrome window.
# Every gate a change has to clear. CI runs exactly this recipe.
check: fmt-check justfile-lint actions-lint pin-check secrets typecheck lint test eval-lint eval-test build

# Lint and format-check the evaluation harness (Python) with ruff, without rewriting anything.
eval-lint:
    ruff check eval
    ruff format --check eval

# Run the evaluation harness unit tests (no network, no browser). Uses the venv `just install` synced; never re-locks.
eval-test:
    uv run --project eval --frozen --offline pytest eval/tests

# Download the pinned datasets and write `per` clips per dataset (16 kHz mono wav) plus eval/data/manifest.json.
eval-prepare per="100":
    uv run --project eval --locked python -m gemma4_eval.prepare --per-dataset "{{ per }}"

# Build the evaluation page into dist-eval/ (never deployed; dist/ stays untouched).
eval-build:
    pnpm exec tsdown -c tsdown.eval.config.ts

# Every model in `models` runs in turn under one run_id. `run=<run_id>` resumes that existing run (YYYYMMDDTHHMMSSZ) with its recorded limit/offset.
# Arguments are positional (`just eval-run e2b,e4b 10`) or name=value (`just eval-run models=e2b,e4b limit=10`).
# Transcribe the manifest in a real Chrome window (WebGPU); writes eval/results/raw/<run_id>/.
eval-run models="e2b" limit="" run="":
    uv run --project eval --locked python -m gemma4_eval.run_browser --models "{{ models }}" --limit "{{ limit }}" --run-id "{{ run }}"

# Aggregate every model of a run (the latest one when `run` is empty) into eval/results/<run_id>.{md,json} and latest.json.
eval-score run="":
    uv run --project eval --locked python -m gemma4_eval.score --run-id "{{ run }}"

# Arguments are positional only here (`just eval e2b,e4b 100`): name=value would land in the wrong step.
# Whole evaluation: prepare -> build -> run -> score.
eval models="e2b" per="100": (eval-prepare per) eval-build (eval-run models) eval-score
