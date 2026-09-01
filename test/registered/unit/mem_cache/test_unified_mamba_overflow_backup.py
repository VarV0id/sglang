# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Ported fork semantics: mamba overflow ring on the UnifiedRadixCache path.

Covers the three behaviours added in the unified-mamba migration:
  1. BACKUP_HOST commit stashes overflow ring slots in ComponentData.metadata
     instead of marking the node mamba-backuped (ring rows recycle on archive
     ack and must never serve host->device restores).
  2. BACKUP_STORAGE build re-attaches overflow_slot_ids so the controller's
     archive-completion drain releases the ring slots.
  3. BACKUP_STORAGE commit clears the stash so a later backup is not treated
     as overflow-pending.

Runs CPU-only: no pool, no torch.cuda. The component under test only reads
node/component_data fields, so a minimal stub tree-core is enough.
"""
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
    ComponentType,
)


def _make_component():
    from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
        MambaComponent,
    )
    comp = MambaComponent.__new__(MambaComponent)
    comp._mamba_pool_host = None
    return comp


def _make_node(host_value=None, value=None):
    node = MagicMock()
    node.hash_value = ["h0"]
    cd = MagicMock()
    cd.host_value = host_value
    cd.value = value
    cd.metadata = {}
    node.component_data = {ComponentType.MAMBA: cd}
    return node, cd


class TestMambaOverflowBackupHostCommit(unittest.TestCase):
    def test_overflow_slots_stashed_not_marked_backuped(self):
        comp = _make_component()
        node, cd = _make_node()
        tr = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([9]),
            overflow_slot_ids=[9],
        )
        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_HOST, [tr], cache_actions=[]
        )
        self.assertIsNone(cd.host_value)  # must NOT be marked backuped
        self.assertEqual(cd.metadata["_mamba_overflow_slot_ids"], [9])
        self.assertTrue(torch.equal(cd.metadata["_mamba_overflow_indices"], torch.tensor([9])))

    def test_normal_commit_sets_host_value(self):
        comp = _make_component()
        node, cd = _make_node()
        tr = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([3]))
        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_HOST, [tr], cache_actions=[]
        )
        self.assertTrue(torch.equal(cd.host_value, torch.tensor([3])))
        self.assertEqual(cd.metadata, {})


class TestMambaOverflowBackupStorage(unittest.TestCase):
    def test_build_reattaches_overflow_slot_ids(self):
        comp = _make_component()
        node, cd = _make_node()
        cd.metadata["_mamba_overflow_indices"] = torch.tensor([7])
        cd.metadata["_mamba_overflow_slot_ids"] = [7]
        transfers = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_STORAGE)
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].overflow_slot_ids, [7])
        self.assertEqual(transfers[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)
        self.assertEqual(transfers[0].keys, ["h0"])

    def test_build_normal_host_value_path(self):
        comp = _make_component()
        node, cd = _make_node(host_value=torch.tensor([4]))
        transfers = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_STORAGE)
        self.assertEqual(len(transfers), 1)
        self.assertIsNone(transfers[0].overflow_slot_ids)

    def test_build_none_without_host_or_overflow(self):
        comp = _make_component()
        node, _ = _make_node()
        self.assertIsNone(comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_STORAGE))

    def test_commit_clears_stash(self):
        comp = _make_component()
        node, cd = _make_node()
        cd.metadata["_mamba_overflow_indices"] = torch.tensor([7])
        cd.metadata["_mamba_overflow_slot_ids"] = [7]
        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_STORAGE, [], cache_actions=[]
        )
        self.assertEqual(cd.metadata, {})


if __name__ == "__main__":
    unittest.main()
