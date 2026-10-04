#!/bin/bash
# Every justfile recipe must carry a `#` comment on the line directly above it.
# `just --list` shows exactly that comment, so an undocumented recipe is one a
# reader cannot discover the purpose of without reading its body.
#
# Why not parse `just --dump --dump-format json`: that output drops the
# distinction between a recipe whose doc comment is absent and one whose
# comment sits above an intervening blank line, and it does not report the
# source line, so a failure could not name where to fix it.

set -uo pipefail

repo_root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
justfile="$repo_root/justfile"

if [ ! -f "$justfile" ]; then
    echo "justfile-comment-lint: $justfile がありません" >&2
    exit 1
fi

failed=0
previous_line=""
line_number=0

while IFS= read -r line || [ -n "$line" ]; do
    line_number=$((line_number + 1))

    # A recipe header starts at column 0 and ends in `:` (optionally followed
    # by dependencies). Assignments (`name := "x"`) are excluded by requiring
    # the character after the name not to be `=`, and settings (`set x := y`)
    # and comments never match the leading-name pattern below.
    is_recipe_header=false
    if [[ $line =~ ^[a-zA-Z_][a-zA-Z0-9_-]*([[:space:]]+[^:]*)?: ]] && [[ ! $line =~ := ]]; then
        is_recipe_header=true
    fi

    if [ "$is_recipe_header" = false ]; then
        previous_line="$line"
        continue
    fi

    recipe_name="${line%%[: ]*}"

    if [[ ! $previous_line =~ ^[[:space:]]*# ]]; then
        echo "justfile:$line_number: レシピ \"$recipe_name\" の直前行に説明コメントがありません" >&2
        failed=1
    fi

    previous_line="$line"
done <"$justfile"

if [ "$failed" -eq 0 ]; then
    echo "justfile-comment-lint: OK"
fi

exit "$failed"
