#!/usr/bin/env bash
# Guard: compileall + import smoke + CPU-safe unit tests.
# Usage: guard.sh [repo_root] ; exits non-zero on failure; prints PASS_COUNT to stdout.
set -u
ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY=/tmp/sglang-test/bin/python
cd "$ROOT" || exit 1

$PY -m compileall -q \
  python/sglang/srt/mem_cache/unified_cache/ \
  python/sglang/srt/mem_cache/storage/ \
  python/sglang/srt/mem_cache/pool_host/ \
  python/sglang/srt/mem_cache/_mamba_overflow_buffer.py \
  python/sglang/srt/server_args.py || { echo "GUARD-FAIL compileall" >&2; exit 1; }

PYTHONPATH=python timeout 300 $PY -c "
import sglang.srt.server_args
import sglang.srt.mem_cache.unified_cache.components.mamba_component
import sglang.srt.mem_cache.unified_cache.storage_attachment
import sglang.srt.mem_cache.storage.backend_factory
import sglang.srt.mem_cache.pool_host.mamba
print('imports OK')
" >/dev/null 2>&1 || { echo "GUARD-FAIL import" >&2; exit 1; }

# CPU-safe unit tests; guard-metric = passed count
out=$(PYTHONPATH=python timeout 900 $PY -m pytest \
  test/registered/unit/mem_cache/test_unified_mamba_views.py \
  test/registered/unit/mem_cache/test_hicache_file_lru_unit.py \
  -q --no-header -p no:cacheprovider 2>&1 | tail -2)
echo "$out" >&2
passed=$(echo "$out" | grep -oP '\d+(?= passed)' | head -1)
if [ -z "$passed" ]; then echo "GUARD-FAIL tests-no-result" >&2; exit 1; fi
echo "$passed"
