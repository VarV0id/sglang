# Unified-mamba migration — per-feature keep/adapt/drop notes

Migration target: fork HiMambaRadixCache feature set → upstream UnifiedRadixCache
(ComponentType.MAMBA). Base: deploy/synced-20260901 (upstream/main merged
2026-08-31). Work branch: autoresearch/unified-mamba-migration.
HiMambaRadixCache deleted; attach_hybrid_pool_to_mamba_cache removed.

| # | Feature | Verdict | Where it landed |
|---|---------|---------|-----------------|
| 1 | lru_file storage backend | **keep (already ported)** | `mem_cache/storage/lru_file.py` + `storage/file/lru_file_evictor.py`, registered `"lru_file"` in `backend_factory.py`. `--hicache-storage-backend-extra-config '{"max_size_gb": N}'` unchanged. No work needed. |
| 2 | Mamba companion files (.mamba_*.bin) | **adapt** | Upstream `mamba_component.build_hicache_transfers(BACKUP_STORAGE)` already writes the companion for host_value nodes, incl. `hit_policy=TRAILING_PAGES` — identical to fork's `mamba_archive_transfers`. Ported only the missing piece: overflow-backed nodes re-attach `overflow_slot_ids` from a metadata stash (see #5). Read-side gate: upstream `finalize_match_result_in_tree_core` derives `mamba_boundary_len` from `device_indices + host_hit_length`, so a missing companion is a natural miss (no explicit clamp needed — the clamp exists only in the fork's boundary bookkeeping). Orphan KV-only guard: lives in `hybrid_cache_controller._page_backup` (survived sync), still wired for the unified controller. |
| 3 | _MambaOverflowAllocator + --mamba-overflow-size | **keep (already ported)** | `_mamba_overflow_buffer.py`, `pool_host/mamba.py::MambaPoolHost(overflow_size=...)`, controller `_resolve_pool_transfers_allocation` fall-through (alloc → overflow_alloc → evict+retry) all present post-sync; `build_hybrid_mamba_stack` already passes `get_memory().mamba_overflow_size`. Flag `--mamba-overflow-size` (default 8, NS("memory")) present in server_args. |
| 4 | Prefetch gates | **adapt** | `--hicache-prefetch-threshold` (256, NS memory) present; upstream `prefetch_threshold` plumbing used as-is. `--hicache-prefetch-capacity-tokens` (0=auto) existed in server_args + controller + `build_hybrid_mamba_stack`, but the **unified** attach path dropped it: ported by adding `hicache_prefetch_capacity_tokens` through `attach_hybrid_pool_to_unified_cache → StackStrategy.build (all 7 strategies, default 0) → build_hybrid_mamba_stack → HybridCacheController`. Semantics identical (0=auto; unblocks ratio=1.0). Skip-logging / admission decoupling / last_hash alignment: upstream's rewritten prefetch path (`prefetch_from_storage` in unified_radix_cache.py) has its own gating; fork log lines not ported (upstream logs at its own sites). |
| 5 | Eviction-path mamba capture (v4) | **adapt** | Upstream already routes leaf demotion through BackupKV→`_execute_kv_backup`→controller→`commit_backup(BACKUP_HOST)`, which copies mamba D→H via `build_hicache_transfers(BACKUP_HOST)` — the v4 goal (mamba_host_value set before device free) is structurally covered, without v1's `_backup_mamba_before_tombstone` or v4's synchronous `torch.cuda.synchronize` hack. Ported the part upstream lacks: **overflow stash on BACKUP_HOST commit** — ring slots go to `ComponentData.metadata["_mamba_overflow_indices"/"_mamba_overflow_slot_ids"]` instead of `host_value` (ring rows recycle on archive ack, must never mark the node backuped), and BACKUP_STORAGE build re-attaches the ids for ring release. `PoolTransfer.overflow_slot_ids` already existed in the merged tree, as did the controller's release path. WIP branch wip/mamba-companion-evict-to-host-v4 retired (semantics ported). |
| 6 | Mamba prompt-anchor (strip_thinking_cache) | **keep (already adapted in sync)** | `req.kv.mamba_prompt_anchor_seqlen` in `schedule_batch.py`, pinned at end-of-prefill in `batch_result_processor.py`, `_cache_commit_len` anchor alignment — all survived the merge onto ReqKvInfo. Unified `cache_finished_req` consumes `prepare_for_caching_req` per component, and the mamba component's version reads the same `req.kv` fields. Upstream 70ac0c4b0e (mamba state corruption/slot leak on load_back abort) is present and complementary. |
| 7 | Misc hardening | **drop (superseded by upstream)** | (a) Sub-page prefix skip: unified `cache_finished_req` page-aligns via `RadixKey.page_aligned` and `insert` on a zero-length key is a no-op path in tree_core — the crash the fork guarded against doesn't exist there. (b) lock_ref decrement for evicted nodes: unified `dec_lock_ref` releases per-component locks unconditionally (`release_component_lock`), no evicted-skip branch. (c) `cache_len != page_aligned_len` defensive fallback: unified clamps with `min(cache_len, len(kv_indices))` upstream (the fork's exact fix), so no assert remains to defend. |

## Validation evidence (this loop)
- `autoresearch/check_features.py`: 14/14 (was 10 at baseline).
- Guard: compileall + import smoke (server_args, mamba_component,
  storage_attachment, backend_factory, pool_host.mamba) + CPU unit tests
  (test_unified_mamba_views, test_hicache_file_lru_unit,
  test_unified_mamba_overflow_backup): 43 passed, 8 skipped.
- New: `test/registered/unit/mem_cache/test_unified_mamba_overflow_backup.py`
  (6 tests) pins the ported overflow semantics.

## Not done here (per prompt constraints)
- No image build/push, no deploy, no branch pushes.
- GPU validation items 2-7 of the prompt (cold start smoke, companion coverage
  ~100%, cross-restart reuse, saturation soak, ratio=1.0 prefetch, strip-
  thinking soak) need the live cluster — run before/with the next image build.
