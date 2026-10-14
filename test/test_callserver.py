"""
Test cases for CallServer.

Covers allocation management, child-process calls (sync and async), shared
memory promotion and reuse, pre-allocated buffers, double-buffering, and the
standalone TCP server.

Run from the repository root with either:
    python -m pytest
    python -m unittest discover -s test
"""

from __future__ import annotations

import contextlib
import io
import operator
import os
import socket
import time
import unittest
from multiprocessing import Process
from typing import List
from unittest import mock

import numpy as np

from callserver import (
    REUSE,
    AllocationDescriptor,
    AllocationSet,
    AllocationSetManager,
    CallServer,
    run,
    run_console,
)


# Skip multiprocessing tests in environments where they may hang
SKIP_MULTIPROCESSING = os.environ.get("SKIP_MULTIPROCESSING_TESTS", "0") == "1"


# ============================================================================
# Test Cases - Unit tests that don't require multiprocessing
# ============================================================================


class TestAllocationDescriptor(unittest.TestCase):
    """Tests for AllocationDescriptor."""

    def test_creation(self):
        """Test creating an allocation descriptor."""
        desc = AllocationDescriptor("test_shm", (100, 100), np.float64)
        self.assertEqual(desc.name, "test_shm")
        self.assertEqual(desc.shape, (100, 100))
        self.assertEqual(desc.dtype, np.float64)

    def test_repr(self):
        """Test string representation."""
        desc = AllocationDescriptor("test_name", (50, 50), np.int32)
        repr_str = repr(desc)
        self.assertIn("test_name", repr_str)
        self.assertIn("50", repr_str)


class TestAllocationSet(unittest.TestCase):
    """Tests for AllocationSet functionality."""

    def test_alloc_and_map(self):
        """Test allocating and mapping shared memory."""
        alloc_set = AllocationSet(0, "test")

        try:
            # Allocate a buffer
            desc = alloc_set.alloc("buffer1", (100, 100), np.float64)
            self.assertEqual(desc.shape, (100, 100))
            self.assertEqual(desc.dtype, np.float64)

            # Map the buffer back
            array = alloc_set.from_sharedmemory(desc)
            self.assertEqual(array.shape, (100, 100))

            # Write and verify
            array[:] = 123.0
            array2 = alloc_set.from_sharedmemory(desc)
            self.assertEqual(array2[0, 0], 123.0)
        finally:
            alloc_set.close()

    def test_is_mappable(self):
        """Test is_mappable correctly identifies arrays to map."""
        alloc_set = AllocationSet(0, "test")

        try:
            self.assertFalse(alloc_set.is_mappable(None))
            self.assertTrue(alloc_set.is_mappable(np.ones((1000, 1000))))
            # Small arrays are sent via the pipe
            self.assertFalse(alloc_set.is_mappable(np.ones((2, 2))))
            # Object arrays cannot live in shared memory
            self.assertFalse(alloc_set.is_mappable(np.empty(1000, dtype=object)))
            self.assertFalse(alloc_set.is_mappable(42))
            self.assertFalse(alloc_set.is_mappable("string"))
        finally:
            alloc_set.close()

    def test_args_conversion(self):
        """Test nested arguments round-trip, with duplicates promoted once."""
        client = AllocationSet(0, "client")
        server = AllocationSet(0, "server")

        try:
            large = np.arange(1_000_000, dtype=np.float64).reshape(1000, 1000)
            args, kwargs = client.args_to_sharedmemory(
                large, 42, nested={"a": [large, (large * 2, 7)]}
            )
            self.assertEqual(len(client._linked_buffers), 2)

            args, kwargs = server.args_from_sharedmemory(*args, **kwargs)
            np.testing.assert_array_equal(args[0], large)
            self.assertEqual(args[1], 42)
            self.assertIs(kwargs["nested"]["a"][0], args[0])
            np.testing.assert_array_equal(kwargs["nested"]["a"][1][0], large * 2)
            self.assertEqual(kwargs["nested"]["a"][1][1], 7)
        finally:
            server.close()
            client.close()

    def test_nested_buffers_reused(self):
        """Test nested arrays reuse their buffers on later calls."""
        alloc_set = AllocationSet(0, "test")

        try:
            for i in range(3):
                payload = {"x": np.full((100, 100), i), "y": [np.full((100, 100), -i)]}
                (data,), _ = alloc_set.args_to_sharedmemory(payload)
                result = alloc_set.loads(data)
                self.assertEqual(result["x"][0, 0], i)
                self.assertEqual(result["y"][0][0, 0], -i)
            self.assertEqual(len(alloc_set._linked_buffers), 2)
        finally:
            alloc_set.close()

    def test_reuse_marker(self):
        """Test REUSE sends the existing buffer and None stays None."""
        alloc_set = AllocationSet(0, "test")

        try:
            with self.assertRaises(KeyError):
                alloc_set.args_to_sharedmemory(REUSE)

            first, _ = alloc_set.args_to_sharedmemory(np.ones((1000, 1000)))
            again, _ = alloc_set.args_to_sharedmemory(REUSE)
            self.assertEqual(len(alloc_set._linked_buffers), 1)
            self.assertIs(alloc_set.loads(again[0]), alloc_set.loads(first[0]))

            none_args, _ = alloc_set.args_to_sharedmemory(None)
            self.assertIsNone(alloc_set.loads(none_args[0]))
        finally:
            alloc_set.close()

    def test_retval_conversion(self):
        """Test nested return values round-trip."""
        server = AllocationSet(0, "server")
        client = AllocationSet(0, "client")

        try:
            large = np.ones((1000, 1000))
            data = server.retval_to_sharedmemory({"array": large, "pair": (large * 2, 42)})
            self.assertEqual(len(server._linked_buffers), 2)

            result = client.retval_from_sharedmemory(data)
            np.testing.assert_array_equal(result["array"], large)
            np.testing.assert_array_equal(result["pair"][0], large * 2)
            self.assertEqual(result["pair"][1], 42)
        finally:
            client.close()
            server.close()

    def test_repr(self):
        """Test AllocationSet string representation."""
        alloc_set = AllocationSet(0, "test_repr")
        try:
            self.assertIn("test_repr", repr(alloc_set))
            self.assertIn("ix=0", repr(alloc_set))
        finally:
            alloc_set.close()


class TestAllocationSetManager(unittest.TestCase):
    """Tests for AllocationSetManager functionality."""

    def tearDown(self):
        """Clean up any singleton managers."""
        for name in list(AllocationSetManager._managers.keys()):
            if "test" in name:
                manager = AllocationSetManager._managers.pop(name)
                manager.close()

    def test_singleton_pattern(self):
        """Test that managers with the same name are singletons."""
        manager1 = AllocationSetManager("singleton_test")
        manager2 = AllocationSetManager("singleton_test")
        self.assertIs(manager1, manager2)

    def test_none_name_not_singleton(self):
        """Test that None name creates independent instances."""
        manager1 = AllocationSetManager(None)
        manager2 = AllocationSetManager(None)
        self.assertIsNot(manager1, manager2)
        manager1.close()
        manager2.close()

    def test_add_allocation_set(self):
        """Test adding allocation sets."""
        manager = AllocationSetManager("add_test")

        set1 = manager.add_allocation_set("func1")
        self.assertIsInstance(set1, AllocationSet)
        self.assertEqual(set1.ix, 0)

        set2 = manager.add_allocation_set("func1")
        self.assertEqual(set2.ix, 1)

    def test_rotation_for_double_buffering(self):
        """Test that get_allocation_set rotates through sets."""
        manager = AllocationSetManager("rotation_test")

        # Add two sets
        manager.add_allocation_set("func")
        manager.add_allocation_set("func")

        # Get sets - should rotate
        set1 = manager.get_allocation_set("func")
        set2 = manager.get_allocation_set("func")
        set3 = manager.get_allocation_set("func")

        # Should rotate back to first set
        self.assertEqual(set1.ix, set3.ix)
        self.assertNotEqual(set1.ix, set2.ix)

    def test_get_active_id(self):
        """Test getting the active allocation set ID."""
        manager = AllocationSetManager("active_id_test")

        manager.add_allocation_set("func")
        manager.get_allocation_set("func")

        active_id = manager.get_active_id("func")
        self.assertEqual(active_id, 0)


class TestCallServerUnit(unittest.TestCase):
    """Unit tests for CallServer that don't require multiprocessing."""

    def test_repr_stopped(self):
        """Test string representation when stopped."""
        server = CallServer("test_repr")
        self.assertIn("test_repr", repr(server))
        self.assertIn("stopped", repr(server))
        self.assertIn("child", repr(server))

    def test_stop_without_start(self):
        """Test that stopping a never-started server doesn't raise."""
        server = CallServer("never_started")
        server.stop()  # Should not raise

    def test_allocation_set_management(self):
        """Test allocation set management methods."""
        server = CallServer("alloc_mgmt")
        try:
            # Add allocation sets
            alloc_set = server.add_allocation_set("test_func")
            self.assertIsInstance(alloc_set, AllocationSet)

            # Get allocation set
            alloc_set2 = server.get_allocation_set("test_func")
            self.assertIsInstance(alloc_set2, AllocationSet)
        finally:
            server.stop()


# ============================================================================
# Test Cases - Integration tests that require multiprocessing
# ============================================================================


class AllocationTracker:
    """Helper class to track shared memory allocations via logging."""

    def __init__(self):
        self.allocations: List[str] = []
        self._original_alloc = None

    def start(self, alloc_set: AllocationSet) -> None:
        """Start tracking allocations for the given AllocationSet."""
        self._original_alloc = alloc_set.alloc
        self.allocations = []

        def tracked_alloc(name, shape, dtype, init=None):
            self.allocations.append(name)
            return self._original_alloc(name, shape, dtype, init)

        alloc_set.alloc = tracked_alloc

    def stop(self, alloc_set: AllocationSet) -> None:
        """Stop tracking allocations."""
        if self._original_alloc is not None:
            alloc_set.alloc = self._original_alloc
            self._original_alloc = None

    @property
    def count(self) -> int:
        """Return the number of allocations tracked."""
        return len(self.allocations)


@unittest.skipIf(SKIP_MULTIPROCESSING, "Multiprocessing tests disabled")
class TestCallServerIntegration(unittest.TestCase):
    """Integration tests for CallServer with multiprocessing."""

    def test_simple_call(self):
        """Test a simple synchronous call."""
        with CallServer("test_simple") as server:
            result = server.call(np.add, np.ones((100, 100)), np.ones((100, 100)))
            self.assertEqual(result.shape, (100, 100))
            self.assertEqual(result[0, 0], 2.0)

    def test_async_call(self):
        """Test asynchronous call pattern."""
        with CallServer("test_async") as server:
            # Start async call
            server.call_async(np.multiply, np.full((100, 100), 3.0), np.full((100, 100), 4.0))

            # Get the result
            result = server.get_result()
            self.assertEqual(result[0, 0], 12.0)

    def test_context_manager(self):
        """Test context manager properly cleans up."""
        server = CallServer("test_context")
        with server:
            result = server.call(np.add, np.ones((10, 10)), np.ones((10, 10)))
            self.assertEqual(result[0, 0], 2.0)
        # Server should be stopped after exiting context
        self.assertFalse(server._started)

    def test_input_arrays_promoted_to_shared_memory(self):
        """Verify that input arrays are promoted to shared memory."""
        with CallServer("test_input_shm") as server:
            # Create large arrays (above MIN_SIZE threshold)
            a = np.ones((100, 100))  # 80000 bytes > 1024 MIN_SIZE
            b = np.ones((100, 100))

            # Get the allocation set for np.add
            alloc_set = server.get_allocation_set(np.add)

            # First call should allocate shared memory for inputs
            result = server.call(np.add, a, b)
            self.assertEqual(result[0, 0], 2.0)

            # Verify that descriptors were created for the input arguments
            # The args are named __arg0, __arg1, etc.
            self.assertIn("__arg0", alloc_set._descriptors)
            self.assertIn("__arg1", alloc_set._descriptors)

    def test_nested_arguments_round_trip(self):
        """Arrays nested in argument containers go through shared memory."""
        with CallServer("test_nested_args") as server:
            payload = {"a": [np.full((100, 100), 5.0)], "b": 1}
            alloc_set = server.get_allocation_set(operator.getitem)

            result = server.call(operator.getitem, payload, "a")

            self.assertIn("__arg0", alloc_set._descriptors)
            self.assertEqual(result[0][0, 0], 5.0)
            mapped = server._alloc_manager.get_allocation_set(operator.getitem, 0)
            self.assertTrue(any(result[0] is arr for arr in mapped._arrays.values()))
    def test_output_arrays_promoted_to_shared_memory(self):
        """Verify that output (return) arrays are promoted to shared memory."""
        with CallServer("test_output_shm") as server:
            a = np.ones((100, 100))
            b = np.ones((100, 100))

            # Perform the call
            result = server.call(np.add, a, b)

            # The result should be a numpy array backed by shared memory
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (100, 100))
            self.assertEqual(result[0, 0], 2.0)

            # The return value is allocated in shared memory by the server and
            # mapped (not copied) on the client side.
            alloc_set = server._alloc_manager.get_allocation_set(np.add, 0)
            self.assertEqual(len(alloc_set._mapped_buffers), 1)
            self.assertTrue(
                any(result is arr for arr in alloc_set._arrays.values()),
                "Result should be a view of a mapped shared memory buffer",
            )

    def test_shared_memory_allocated_once_and_reused(self):
        """Verify that shared memory is allocated only once and buffers reused."""
        with CallServer("test_reuse") as server:
            a = np.ones((100, 100))
            b = np.ones((100, 100))

            # Get the allocation set and track allocations
            alloc_set = server.get_allocation_set(np.add)
            tracker = AllocationTracker()
            tracker.start(alloc_set)

            # First call - should allocate buffers
            result1 = server.call(np.add, a, b)
            first_call_allocations = tracker.count
            self.assertEqual(result1[0, 0], 2.0)

            # Subsequent calls - should NOT allocate new buffers
            result2 = server.call(np.add, a, b)
            self.assertEqual(result2[0, 0], 2.0)
            result3 = server.call(np.add, a, b)
            self.assertEqual(result3[0, 0], 2.0)

            tracker.stop(alloc_set)

            # Verify no additional allocations were made after the first call
            self.assertEqual(tracker.count, first_call_allocations,
                            "Shared memory should be reused, not reallocated")

    def test_different_methods_use_different_allocation_sets(self):
        """Verify that different server-side methods use different buffers."""
        with CallServer("test_diff_methods") as server:
            a = np.ones((100, 100))
            b = np.ones((100, 100))

            # Call np.add
            result1 = server.call(np.add, a, b)
            self.assertEqual(result1[0, 0], 2.0)

            # Call np.multiply
            result2 = server.call(np.multiply, a, b)
            self.assertEqual(result2[0, 0], 1.0)

            # Verify that different allocation sets were used
            add_set = server._alloc_manager.get_allocation_set(np.add)
            mul_set = server._alloc_manager.get_allocation_set(np.multiply)

            # They should be different AllocationSet instances
            self.assertIsNot(add_set, mul_set)

            # Each should have its own descriptors
            self.assertIn("__arg0", add_set._descriptors)
            self.assertIn("__arg0", mul_set._descriptors)

            # The descriptors should point to different shared memory
            self.assertNotEqual(
                add_set._descriptors["__arg0"].name,
                mul_set._descriptors["__arg0"].name,
                "Different methods should use different shared memory buffers"
            )

    def test_preallocated_sets_no_extra_allocations(self):
        """Verify that pre-allocated sets result in no extra allocations."""
        with CallServer("test_prealloc") as server:
            # Pre-allocate buffers using CallServer.alloc
            a = CallServer.alloc(server, np.add, (100, 100), float)
            b = CallServer.alloc(server, np.add, (100, 100), float)

            # Initialize the pre-allocated buffers
            a[:] = 3.0
            b[:] = 4.0

            # Get the allocation set and track allocations
            alloc_set = server.get_allocation_set(np.add)
            tracker = AllocationTracker()
            tracker.start(alloc_set)

            # Call using pre-allocated buffers - should NOT allocate new buffers
            result = server.call(np.add, a, b)
            self.assertEqual(result[0, 0], 7.0)

            # Make another call with the same pre-allocated buffers
            a[:] = 5.0
            b[:] = 6.0
            result2 = server.call(np.add, a, b)
            self.assertEqual(result2[0, 0], 11.0)

            tracker.stop(alloc_set)

            # No new allocations should have been made for arguments
            # (only the return value might be allocated on first call)
            arg_allocations = [name for name in tracker.allocations
                               if name.startswith("__arg")]
            self.assertEqual(len(arg_allocations), 0,
                             "Pre-allocated buffers should not cause new allocations")

    def test_double_buffering_alternates(self):
        """Verify that buffers alternate when using double-buffering."""
        with CallServer("test_double_buffer_alt") as server:
            # Create two allocation sets for double-buffering
            server.add_allocation_set(np.add)
            server.add_allocation_set(np.add)

            a = np.ones((100, 100))
            b = np.ones((100, 100))

            # Track which allocation sets are used by checking active ID before each call
            used_sets: List[int] = []

            # Make several calls and track which set is active
            for i in range(6):
                # Make a call using the public API
                result = server.call(np.add, a, b)
                self.assertEqual(result[0, 0], 2.0)

                # Record which set was used (get_active_id returns the last used set)
                active_id = server._alloc_manager.get_active_id(np.add)
                used_sets.append(active_id)

            # With 2 sets, we should see rotation between them
            unique_sets = set(used_sets)
            self.assertEqual(len(unique_sets), 2,
                            "Double-buffering should use exactly 2 allocation sets")

            # Verify alternating pattern - adjacent calls should use different sets
            for i in range(1, len(used_sets)):
                self.assertNotEqual(used_sets[i], used_sets[i-1],
                                   f"Call {i} should use a different set than call {i-1}")


def _free_port() -> int:
    """Return a TCP port that is currently free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect(name: str, port: int, authkey: bytes, timeout: float = 10.0) -> CallServer:
    """Connect to a standalone server, retrying until it is listening."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return CallServer((name, port), authkey=authkey)
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


class TestRunConsole(unittest.TestCase):
    """Tests for the command-line entry point."""

    def test_port_is_required(self):
        """run_console must not fall back to a hardcoded port."""
        with mock.patch("sys.argv", ["callserver"]), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            run_console()


@unittest.skipIf(SKIP_MULTIPROCESSING, "Multiprocessing tests disabled")
class TestStandaloneServer(unittest.TestCase):
    """Integration tests for run() in standalone TCP mode."""

    def setUp(self):
        self.port = _free_port()
        self.authkey = b"test-authkey"
        self.process = Process(
            target=run, args=(("127.0.0.1", self.port, self.authkey),), daemon=True
        )
        self.process.start()

    def tearDown(self):
        self.process.terminate()
        self.process.join()

    def test_remote_call_and_reconnect(self):
        """A client can call the server, disconnect, and a new client can reconnect."""
        for _ in range(2):
            with _connect("test_remote", self.port, self.authkey) as server:
                self.assertIn("remote", repr(server))
                result = server.call(np.add, np.ones((100, 100)), np.ones((100, 100)))
                self.assertEqual(result[0, 0], 2.0)


def run_tests():
    """Run all tests."""
    unittest.main(module=__name__, exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
