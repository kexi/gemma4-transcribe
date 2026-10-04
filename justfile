# Development tasks. Everything assumes the nix devshell is active
# (direnv allow, or `nix develop -c just <task>`).

# Show available tasks.
default:
    @just --list

# Install git hooks (idempotent). The devshell shellHook does this too.
setup:
    lefthook install

# Install JavaScript dependencies from the lockfile (pnpm is the only supported package manager).
install:
    pnpm install --frozen-lockfile

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

# Every gate a change has to clear. CI runs exactly this recipe.
check: fmt-check justfile-lint actions-lint pin-check secrets typecheck lint test build
