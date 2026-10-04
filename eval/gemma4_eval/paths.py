"""リポジトリ内の既定パス。

`uv run --project eval` はカレントディレクトリを変えないため、cwd からの相対パスではなく
このファイルの位置から解決する（どこから実行しても同じ場所を指す）。
"""

from __future__ import annotations

from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_DIR.parent
DATA_DIR = EVAL_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
RESULTS_DIR = EVAL_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
CHROME_PROFILE_DIR = EVAL_DIR / ".chrome-profile"
DIST_EVAL_DIR = REPO_ROOT / "dist-eval"
