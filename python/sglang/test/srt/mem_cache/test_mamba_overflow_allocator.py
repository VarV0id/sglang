"""Unit tests for ``_MambaOverflowAllocator``.

These tests do NOT require torch/CUDA — the allocator is a pure
integer-bookkeeping data structure that owns nothing but a list of refcounts.
"""

from __future__ import annotations

import threading
import unittest

from sglang.srt.mem_cache._mamba_overflow_buffer import _MambaOverflowAllocator


class TestMambaOverflowAllocator(unittest.TestCase):
    def test_acquire_release_basic(self):
        alloc = _MambaOverflowAllocator(base_idx=1000, size=8)

        slots = [alloc.acquire() for _ in range(8)]
        self.assertNotIn(None, slots)
        self.assertEqual(len(set(slots)), 8, "all 8 slots must be distinct")
        for s in slots:
            self.assertTrue(
                alloc.contains(s),
                f"slot {s} out of range [{alloc.base}, {alloc.base + alloc.size})",
            )

        # Ninth acquire fails — ring is saturated.
        self.assertIsNone(alloc.acquire())

        # Release one slot; the next acquire should succeed and return that
        # specific slot (round-robin scan starts from the slot after the one
        # just released).
        alloc.release(slots[3])
        reused = alloc.acquire()
        self.assertEqual(reused, slots[3])

    def test_retain_refcount(self):
        alloc = _MambaOverflowAllocator(base_idx=0, size=4)
        s = alloc.acquire()
        alloc.retain(s)  # refcount = 2
        alloc.release(s)  # refcount = 1, still in use
        # Acquire 3 more; ring has 3 free, all should succeed.
        more = [alloc.acquire() for _ in range(3)]
        self.assertNotIn(None, more)
        # Ring is now full (s + 3 more = 4 slots in use).
        self.assertIsNone(alloc.acquire())
        # Final release of s frees the last refcount.
        alloc.release(s)
        # Now one slot should be free again.
        self.assertIsNotNone(alloc.acquire())

    def test_double_release_does_not_crash(self):
        """Double-release should warn + count but not raise.

        Rationale: a double-release indicates a wiring bug but the scheduler
        thread must not crash on it — that would tank live requests.
        """
        alloc = _MambaOverflowAllocator(base_idx=0, size=2)
        s = alloc.acquire()
        alloc.release(s)
        alloc.release(s)  # second release on already-free slot
        stats = alloc.stats()
        self.assertEqual(stats["double_release"], 1)

    def test_thread_safety(self):
        """16 threads × 1000 acquire-release pairs each. Invariants:

        - no two threads ever see the same slot index simultaneously
        - all refcounts return to zero at the end
        - no exceptions raised
        """
        alloc = _MambaOverflowAllocator(base_idx=42, size=8)
        in_use_lock = threading.Lock()
        in_use: set[int] = set()
        collisions: list[int] = []
        errors: list[BaseException] = []

        def worker():
            try:
                for _ in range(1000):
                    s = alloc.acquire()
                    if s is None:
                        continue
                    with in_use_lock:
                        if s in in_use:
                            collisions.append(s)
                        in_use.add(s)
                    # tiny pause to maximise interleaving
                    with in_use_lock:
                        in_use.discard(s)
                    alloc.release(s)
            except BaseException as e:  # pragma: no cover - defensive
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(collisions, [], "no two threads should hold the same slot")

        # All refcounts back to zero — every slot acquirable again.
        for _ in range(8):
            s = alloc.acquire()
            self.assertIsNotNone(s)
        self.assertIsNone(alloc.acquire())

    def test_round_robin_distribution(self):
        """Across many acquire-release cycles, every slot should be used."""
        alloc = _MambaOverflowAllocator(base_idx=0, size=8)
        used: set[int] = set()
        for _ in range(100):
            s = alloc.acquire()
            self.assertIsNotNone(s)
            used.add(s)
            alloc.release(s)
        self.assertEqual(used, set(range(8)))

    def test_stats(self):
        alloc = _MambaOverflowAllocator(base_idx=0, size=4)
        acquired = [alloc.acquire() for _ in range(4)]
        self.assertIsNone(alloc.acquire())  # denial
        alloc.release(acquired[0])
        alloc.release(acquired[1])

        s = alloc.stats()
        self.assertEqual(s["acquires"], 4)
        self.assertEqual(s["releases"], 2)
        self.assertEqual(s["denials"], 1)
        self.assertEqual(s["in_use_peak"], 4)
        self.assertEqual(s["in_use_now"], 2)
        self.assertEqual(s["base"], 0)
        self.assertEqual(s["size"], 4)

    def test_contains_predicate(self):
        alloc = _MambaOverflowAllocator(base_idx=1000, size=8)
        self.assertTrue(alloc.contains(1000))
        self.assertTrue(alloc.contains(1007))
        self.assertFalse(alloc.contains(999))
        self.assertFalse(alloc.contains(1008))
        # Negative slot ids cannot fall inside any base >= 0 allocator.
        self.assertFalse(alloc.contains(-1))

    def test_size_zero_allocator(self):
        """A zero-size allocator is valid (overflow disabled).  acquire()
        always returns None; contains() is always False."""
        alloc = _MambaOverflowAllocator(base_idx=500, size=0)
        self.assertIsNone(alloc.acquire())
        self.assertFalse(alloc.contains(500))

    def test_acquire_out_of_range_release_raises(self):
        alloc = _MambaOverflowAllocator(base_idx=100, size=4)
        with self.assertRaises(ValueError):
            alloc.release(99)
        with self.assertRaises(ValueError):
            alloc.release(104)


if __name__ == "__main__":
    unittest.main()
