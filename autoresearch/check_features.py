#!/usr/bin/env python3
"""Unified-mamba-migration feature checklist. Prints PASS count as last line.
Usage: check_features.py [repo_root]
Each check prints NAME: PASS/FAIL; final line is just the integer count."""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
MC = ROOT / "python/sglang/srt/mem_cache"
UNI = MC / "unified_cache"

def has(path, *patterns):
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return False
    return all(re.search(p, text) for p in patterns)

checks = []
def check(name, ok):
    checks.append((name, bool(ok)))
    print(f"{name}: {'PASS' if ok else 'FAIL'}")

# 1. LRU-bounded file storage backend
check("lru_file_backend",
      (MC / "storage/lru_file.py").exists()
      and (MC / "storage/file/lru_file_evictor.py").exists()
      and has(MC / "storage/backend_factory.py", '"lru_file"', r"lru_file\.py|storage\.lru_file|lru_file\""))

# 2. Mamba companion-file semantics on unified path
# Companion-file semantics on the unified path = BACKUP_STORAGE phase building a
# MAMBA PoolTransfer (upstream base) + fork overflow fall-through (ported).
check("companion_write_read",
      has(UNI / "components/mamba_component.py",
          r"BACKUP_STORAGE", r"PoolName\.MAMBA", r"overflow_slot_ids"))
check("orphan_kv_guard",
      has(MC / "hybrid_cache/hybrid_cache_controller.py", r"pool_transfers"))
check("mamba_boundary_clamp",
      has(UNI / "components/mamba_component.py", r"mamba_boundary"))

# 3. Overflow allocator + flag
check("overflow_allocator",
      (MC / "_mamba_overflow_buffer.py").exists()
      and has(MC / "_mamba_overflow_buffer.py", r"overflow_alloc|OverflowAllocator")
      and has(MC / "pool_host/mamba.py", r"overflow_alloc", r"overflow_release"))
check("flag_mamba_overflow_size",
      has(ROOT / "python/sglang/srt/server_args.py", r"mamba[_-]overflow[_-]size"))

# 4. Prefetch gates + flags
check("flag_prefetch_threshold",
      has(ROOT / "python/sglang/srt/server_args.py", r"hicache[_-]prefetch[_-]threshold"))
check("flag_prefetch_capacity",
      has(ROOT / "python/sglang/srt/server_args.py", r"hicache[_-]prefetch[_-]capacity[_-]tokens"))
check("prefetch_capacity_wired",
      has(UNI / "storage_attachment.py", r"prefetch_capacity|capacity_tokens")
      or has(UNI / "components/mamba_component.py", r"prefetch_capacity|capacity_tokens")
      or has(MC / "hybrid_cache/hybrid_pool_assembler.py",
             r"def attach_hybrid_pool_to_unified_cache[\s\S]*?hicache_prefetch_capacity_tokens"))

# 5. Eviction-path mamba capture (v4 semantics) on unified path
# v4 leaf-demotion capture, unified: BACKUP_HOST commit stashes overflow ring
# slots in ComponentData.metadata instead of marking the node backuped, and
# BACKUP_STORAGE re-attaches overflow_slot_ids for ring release after archive.
check("evict_to_host_capture",
      has(UNI / "components/mamba_component.py",
          r"_mamba_overflow_indices", r"_mamba_overflow_slot_ids",
          r"BACKUP_HOST", r"BACKUP_STORAGE"))

# 6. Prompt anchor machinery
check("prompt_anchor",
      has(ROOT / "python/sglang/srt/managers/schedule_batch.py", r"mamba_prompt_anchor_seqlen"))

# 7. Fork class retired + backend switch
check("fork_class_deleted",
      not (MC / "hi_mamba_radix_cache.py").exists())
check("backend_switch",
      has(ROOT / "python/sglang/srt/server_args.py", r"radix_cache_backend|radix-cache-backend"))

# compileall on the migration-touched tree
targets = [str(UNI), str(MC / "storage"), str(MC / "pool_host"),
           str(MC / "_mamba_overflow_buffer.py"),
           str(ROOT / "python/sglang/srt/server_args.py")]
r = subprocess.run([sys.executable, "-m", "compileall", "-q", *targets])
check("compileall", r.returncode == 0)

print(sum(1 for _, ok in checks if ok))
