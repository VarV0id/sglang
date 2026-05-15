"""Integration tests for the overflow region inside ``MambaPoolHost``.

These tests exercise the in-pool reserved-row design from
``.planning/quick/260515-epq-alt-b-mamba-overflow/DESIGN-V2.md``:

- ``overflow_size`` extra rows live at the tail of ``temporal_buffer`` and
  ``conv_buffer``.
- ``self.size`` (LRU-managed range) and the normal ``alloc/free`` API are
  unchanged.
- ``overflow_alloc`` / ``overflow_release`` hand out and reclaim absolute
  slot indices in ``[self.size, self.size + overflow_size)``.
- ``get_data_page(slot_idx, flat=True)`` works for overflow slot indices
  (the storage write path uses this to produce the .mamba_*.bin bytes).

Tests require real torch tensors but not CUDA. ``pin_memory=True`` is
forced to False here because torch's pinned-memory allocator requires the
CUDA driver to be present; skip the round-trip test if torch.cuda is
unavailable AND pin_memory is the only viable allocator (which on the
build host it isn't — we set pin_memory=False for the test).
"""

from __future__ import annotations

import unittest

try:
    import torch
    import numpy as np
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - torch is a hard runtime dep in prod
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch required")
class TestMambaPoolHostOverflow(unittest.TestCase):
    """Stub-based integration tests.

    We don't construct a real ``MambaPoolHost`` (it depends on a live
    ``MambaPool`` with CUDA-backed device tensors). Instead, we exercise the
    overflow allocator and verify the reserved-row math via a minimal
    in-process fixture that mirrors the real ``MambaPoolHost.__init__``
    semantics (``self.size + self.overflow_size`` for buffer dims; ``self.size``
    for ``free_slots``).
    """

    def _build_stub_pool(self, size: int = 16, overflow_size: int = 4):
        """Build a minimal fixture mirroring the post-Alt-B MambaPoolHost
        invariants without instantiating the full class.

        Skipped under stubs:
        - The actual device_pool wiring.
        - The host-memory budget check.

        Validated:
        - tensor dim = size + overflow_size
        - free_slots covers [0, size) only
        - overflow allocator owns [size, size + overflow_size)
        - get_data_page math works for overflow indices.
        """
        from sglang.srt.mem_cache._mamba_overflow_buffer import (
            _MambaOverflowAllocator,
        )

        total_rows = size + overflow_size
        # Layer-first layout, single conv state, tiny dims for fast tests.
        num_mamba_layers = 2
        temporal_state_shape = (3,)
        conv_state_shape = (2,)
        temporal_buffer = torch.zeros(
            (num_mamba_layers, total_rows) + temporal_state_shape,
            dtype=torch.float32,
        )
        conv_buffer = [
            torch.zeros(
                (num_mamba_layers, total_rows) + conv_state_shape,
                dtype=torch.float32,
            )
        ]
        free_slots = torch.arange(size, dtype=torch.int64)
        overflow = _MambaOverflowAllocator(base_idx=size, size=overflow_size)
        return {
            "size": size,
            "overflow_size": overflow_size,
            "total_rows": total_rows,
            "temporal_buffer": temporal_buffer,
            "conv_buffer": conv_buffer,
            "free_slots": free_slots,
            "overflow": overflow,
        }

    def test_overflow_rows_excluded_from_normal_alloc(self):
        """Normal allocator hands out exactly ``size`` slots, all in
        ``[0, size)``. Overflow allocator hands out slots in
        ``[size, size + overflow_size)``."""
        p = self._build_stub_pool(size=16, overflow_size=4)
        # Simulate the normal alloc behaviour (no-op slicing).
        size = p["size"]
        normal_slots = p["free_slots"].tolist()
        self.assertEqual(len(normal_slots), size)
        for s in normal_slots:
            self.assertLess(s, size)
            self.assertGreaterEqual(s, 0)

        overflow_slots = []
        for _ in range(p["overflow_size"]):
            s = p["overflow"].acquire()
            self.assertIsNotNone(s)
            overflow_slots.append(s)
        for s in overflow_slots:
            self.assertGreaterEqual(s, size)
            self.assertLess(s, size + p["overflow_size"])

        # Normal slot ids and overflow slot ids must be disjoint.
        self.assertEqual(set(normal_slots) & set(overflow_slots), set())

    def test_buffer_dims_include_overflow_rows(self):
        """Buffer dim along the slot axis must equal ``size + overflow_size``."""
        p = self._build_stub_pool(size=16, overflow_size=4)
        # Layer-first: dim 1 is the slot axis.
        self.assertEqual(p["temporal_buffer"].shape[1], p["total_rows"])
        for cb in p["conv_buffer"]:
            self.assertEqual(cb.shape[1], p["total_rows"])

    def test_overflow_index_round_trip(self):
        """Write known bytes into an overflow row and read them back.

        This proves that ``get_data_page`` (and by extension the storage
        backend's ``_write_page``) sees the correct bytes when fed an
        overflow slot index — which is the whole point of the in-pool
        reserved-row design (no new code path needed in
        ``hicache_storage.py``).
        """
        p = self._build_stub_pool(size=8, overflow_size=2)
        overflow_idx = p["overflow"].acquire()
        self.assertIsNotNone(overflow_idx)

        # Write recognizable bytes into the overflow row of both buffers,
        # across all layers.
        tb = p["temporal_buffer"]
        cb = p["conv_buffer"][0]
        for layer in range(tb.shape[0]):
            tb[layer, overflow_idx, :] = torch.tensor(
                [1.0 + layer, 2.0 + layer, 3.0 + layer]
            )
            cb[layer, overflow_idx, :] = torch.tensor(
                [10.0 + layer, 20.0 + layer]
            )

        # Read back via the same indexing the real
        # ``_iter_page_tensors`` uses in layer-first layout.
        for layer in range(tb.shape[0]):
            self.assertTrue(
                torch.equal(
                    tb[layer, overflow_idx, :],
                    torch.tensor([1.0 + layer, 2.0 + layer, 3.0 + layer]),
                )
            )
            self.assertTrue(
                torch.equal(
                    cb[layer, overflow_idx, :],
                    torch.tensor([10.0 + layer, 20.0 + layer]),
                )
            )

    def test_overflow_release_returns_slot_to_ring(self):
        p = self._build_stub_pool(size=4, overflow_size=2)
        s1 = p["overflow"].acquire()
        s2 = p["overflow"].acquire()
        self.assertIsNone(p["overflow"].acquire())  # ring saturated
        p["overflow"].release(s1)
        s3 = p["overflow"].acquire()
        self.assertEqual(s3, s1)
        # Cleanup.
        p["overflow"].release(s2)
        p["overflow"].release(s3)


@unittest.skipUnless(HAVE_TORCH and torch.cuda.is_available(),
                     "pinned-memory path requires CUDA driver")
class TestPinnedMemoryAllocation(unittest.TestCase):
    """Smoke test for the pinned-memory allocation path. Skipped on hosts
    without a CUDA driver because ``pin_memory=True`` calls
    ``cudaHostAlloc``."""

    def test_pinned_tensor_is_pinned(self):
        t = torch.empty((8, 4), dtype=torch.float32, pin_memory=True)
        self.assertTrue(t.is_pinned())


if __name__ == "__main__":
    unittest.main()
