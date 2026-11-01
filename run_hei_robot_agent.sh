#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
cd "$ROOT"
exec uv run --locked --project "$ROOT" python -u -m sim_agent "$@"
