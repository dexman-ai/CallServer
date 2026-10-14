"""
Shared memory allocation management for inter-process numpy array transfer.

This module provides classes for managing shared memory buffers used in RPC
communication. It handles the low-level details of creating, mapping, and
converting numpy arrays to/from shared memory segments.
"""

from __future__ import annotations

import io
import logging
import pickle
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class AllocationDescriptor:
    """
    Descriptor for a shared memory allocation.

    This lightweight class stores metadata about a shared memory buffer,
    allowing it to be reconstructed in another process. It is serializable
    and can be sent through pipes or sockets.

    Attributes:
        name: The system-level name of the shared memory segment.
        shape: The shape of the numpy array stored in the buffer.
        dtype: The data type of the numpy array.
    """

    __slots__ = ("name", "shape", "dtype")

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        """
        Initialize an AllocationDescriptor.

        Args:
            name: The system-level name of the shared memory segment.
            shape: The shape of the numpy array.
            dtype: The data type of the numpy array.
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"AllocationDescriptor(name={self.name!r}, shape={self.shape}, dtype={self.dtype})"


class Reuse:
    """
    Marker that sends the buffer already allocated at this position, without copying.

    Pass REUSE where an array was passed on an earlier call. The other side
    receives the existing shared memory buffer, with its contents unchanged.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "REUSE"


REUSE = Reuse()

_Promoted = Dict[int, Tuple[np.ndarray, AllocationDescriptor]]


class _SharedMemoryPickler(pickle.Pickler):
    """Pickler that replaces large arrays, at any depth, with AllocationDescriptors."""

    def __init__(
        self, file: io.BytesIO, argset: AllocationSet, prefix: str, promoted: _Promoted
    ) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._argset = argset
        self._prefix = prefix
        self._promoted = promoted
        self._slot = 0

    def _next_name(self) -> str:
        name = self._prefix if self._slot == 0 else f"{self._prefix}.{self._slot}"
        self._slot += 1
        return name

    def persistent_id(self, obj: Any) -> Optional[AllocationDescriptor]:
        if isinstance(obj, Reuse):
            name = self._next_name()
            if name not in self._argset._descriptors:
                raise KeyError(
                    f"REUSE at {name}: no buffer was allocated at this position on an earlier call."
                )
            return self._argset.get(name)
        if not self._argset.is_mappable(obj):
            return None
        # Consume a slot even for duplicates, so names depend only on position.
        name = self._next_name()
        hit = self._promoted.get(id(obj))
        if hit is not None:
            return hit[1]
        desc = self._argset.to_sharedmemory(name, obj)
        # Keep obj referenced so its id cannot be reused within this call.
        self._promoted[id(obj)] = (obj, desc)
        return desc


class _SharedMemoryUnpickler(pickle.Unpickler):
    """Unpickler that maps AllocationDescriptors back to shared memory arrays."""

    def __init__(self, file: io.BytesIO, argset: AllocationSet) -> None:
        super().__init__(file)
        self._argset = argset

    def persistent_load(self, pid: Any) -> np.ndarray:
        if not isinstance(pid, AllocationDescriptor):
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid!r}")
        return self._argset.from_sharedmemory(pid)


class AllocationSet:
    """
    Manages a set of shared memory allocations for inter-process communication.

    An AllocationSet represents a collection of shared memory buffers that can be
    used to transfer numpy arrays between processes. It handles the lifecycle of
    shared memory segments, including creation, mapping, and cleanup.

    The class supports two types of memory ownership:
        - Linked buffers: Created by this process and owned (will be unlinked on close)
        - Mapped buffers: Created by another process, only mapped for access

    Attributes:
        MIN_SIZE: Minimum array size (in bytes) to use shared memory instead of pipes.
        ix: Index identifier for this allocation set.

    Example:
        >>> alloc_set = AllocationSet(0, "my_function")
        >>> desc = alloc_set.alloc("buffer", (1024, 1024), np.float64)
        >>> array = alloc_set.from_sharedmemory(desc)
        >>> array[:] = 42.0
    """

    MIN_SIZE = 1024

    def __init__(self, ix: int, name: Optional[str] = None) -> None:
        """
        Initialize an AllocationSet.

        Args:
            ix: Index identifier for this allocation set.
            name: Optional name for logging and debugging purposes.
        """
        self.ix = ix
        self._name = name
        self._linked_buffers: List[shared_memory.SharedMemory] = []
        self._mapped_buffers: List[shared_memory.SharedMemory] = []
        self._arrays: Dict[str, np.ndarray] = {}
        self._descriptors: Dict[str, AllocationDescriptor] = {}

    def __repr__(self) -> str:
        """Return a string representation of the AllocationSet."""
        return (
            f"AllocationSet(ix={self.ix}, name={self._name!r}, "
            f"buffers={len(self._linked_buffers)}, mapped={len(self._mapped_buffers)})"
        )

    def alloc(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        init: Optional[np.ndarray] = None,
    ) -> AllocationDescriptor:
        """
        Allocate a new shared memory buffer.

        Creates a new shared memory segment and returns a descriptor that can be
        used to access the buffer from another process.

        Args:
            name: Logical name for the allocation (used for lookup).
            shape: Shape of the numpy array to allocate.
            dtype: Data type of the numpy array.
            init: Optional array to copy into the allocated buffer.

        Returns:
            An AllocationDescriptor containing the buffer metadata.
        """
        logger.info(f"alloc {self._name}({self.ix}): {name}")
        shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(shape) * np.dtype(dtype).itemsize)
        )
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        if init is not None:
            array[:] = init
        self._linked_buffers.append(shm)
        self._arrays[shm.name] = array
        desc = AllocationDescriptor(shm.name, shape, dtype)
        self._descriptors[name] = desc
        return desc

    def get(
        self, name: str, init: Optional[np.ndarray] = None
    ) -> AllocationDescriptor:
        """
        Get an existing allocation descriptor by logical name.

        Args:
            name: Logical name of the allocation.
            init: Optional array to copy into the buffer.

        Returns:
            The AllocationDescriptor for the named buffer.

        Raises:
            KeyError: If no allocation with the given name exists.
        """
        desc = self._descriptors[name]
        if init is not None:
            array = self._arrays[desc.name]
            array[:] = init
        return desc

    def is_mappable(self, arg: Any) -> bool:
        """
        Check if an object should be transferred via shared memory.

        Numpy arrays larger than MIN_SIZE bytes are mapped, unless their dtype
        holds Python objects, which cannot live in shared memory.

        Args:
            arg: The object to check.

        Returns:
            True if the object should use shared memory, False otherwise.
        """
        return (
            isinstance(arg, np.ndarray)
            and not arg.dtype.hasobject
            and arg.nbytes > AllocationSet.MIN_SIZE
        )

    def to_sharedmemory(self, name: str, array: np.ndarray) -> AllocationDescriptor:
        """
        Convert a numpy array to a shared memory allocation.

        If the array is already in shared memory (managed by this set), returns
        its existing descriptor. Otherwise, either reuses an existing allocation
        (if shape/dtype match) or creates a new one.

        Args:
            name: Logical name for the allocation.
            array: The numpy array to convert.

        Returns:
            An AllocationDescriptor for the shared memory buffer.

        Raises:
            AssertionError: If array is not a numpy array.
            AssertionError: If shape or dtype changed for an existing allocation.
        """
        assert isinstance(array, np.ndarray)
        for id, existing in self._arrays.items():
            if array is existing:
                return AllocationDescriptor(id, array.shape, array.dtype)
        if name in self._descriptors:
            assert (
                array.shape == self._descriptors[name].shape
                and array.dtype == self._descriptors[name].dtype
            ), f"Unexpected shape or dtype for {name}. Changing argument shape or dtype between calls is not supported."
            return self.get(name, array)

        return self.alloc(name, array.shape, array.dtype, array)

    def from_sharedmemory(self, descriptor: AllocationDescriptor) -> np.ndarray:
        """
        Map a shared memory buffer to a numpy array.

        If the buffer is already mapped in this set, returns the existing array.
        Otherwise, maps the shared memory segment and creates a new numpy array
        view of it.

        Args:
            descriptor: The AllocationDescriptor identifying the buffer.

        Returns:
            A numpy array backed by the shared memory buffer.

        Raises:
            AssertionError: If descriptor is not an AllocationDescriptor.
        """
        assert isinstance(descriptor, AllocationDescriptor)
        if descriptor.name in self._arrays:
            return self._arrays[descriptor.name]
        logger.info(f"from_sharedmemory {self._name}({self.ix}): {descriptor.name}")
        shm = shared_memory.SharedMemory(name=descriptor.name)
        self._mapped_buffers.append(shm)
        array = np.ndarray(descriptor.shape, dtype=descriptor.dtype, buffer=shm.buf)
        self._arrays[descriptor.name] = array
        return array
    
    def dumps(self, obj: Any, prefix: str, promoted: Optional[_Promoted] = None) -> bytes:
        """
        Pickle obj, moving large arrays at any depth to shared memory.

        Buffers are named prefix, prefix.1, prefix.2, ... in pickling order.
        Promoted arrays and REUSE markers each take one slot, so the same
        structure reuses the same buffers on later calls.

        Args:
            obj: The object to pickle.
            prefix: Name prefix for the buffers allocated for obj.
            promoted: Arrays already promoted in this call, keyed by id. An array
                that appears more than once is promoted only once.

        Returns:
            The pickled bytes, with large arrays replaced by AllocationDescriptors.
        """
        buf = io.BytesIO()
        _SharedMemoryPickler(buf, self, prefix, {} if promoted is None else promoted).dump(obj)
        return buf.getvalue()

    def loads(self, data: bytes) -> Any:
        """
        Unpickle data produced by dumps(), mapping descriptors to numpy arrays.

        Args:
            data: The pickled bytes.

        Returns:
            The unpickled object, with descriptors replaced by shared memory arrays.
        """
        return _SharedMemoryUnpickler(io.BytesIO(data), self).load()

    def args_to_sharedmemory(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[List[bytes], Dict[str, bytes]]:
        """
        Pickle function arguments, moving large arrays at any depth to shared memory.

        Each argument is pickled separately, so buffer names stay tied to the
        argument position or keyword even when other arguments change between
        calls. An array passed more than once in the same call is promoted once.

        Args:
            *args: Positional arguments to convert.
            **kwargs: Keyword arguments to convert.

        Returns:
            A tuple of (positional_args, keyword_args), each argument pickled to bytes.
        """
        promoted: _Promoted = {}
        positional_args = [
            self.dumps(arg, f"__arg{idx}", promoted) for idx, arg in enumerate(args)
        ]
        keyword_args = {k: self.dumps(v, f"__kw.{k}", promoted) for k, v in kwargs.items()}
        return positional_args, keyword_args

    def args_from_sharedmemory(
        self, *args: bytes, **kwargs: bytes
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Unpickle function arguments produced by args_to_sharedmemory().

        Args:
            *args: Pickled positional arguments.
            **kwargs: Pickled keyword arguments.

        Returns:
            A tuple of (positional_args, keyword_args) with descriptors replaced
            by their numpy array views.
        """
        positional_args = [self.loads(arg) for arg in args]
        keyword_args = {k: self.loads(v) for k, v in kwargs.items()}
        return positional_args, keyword_args

    def retval_to_sharedmemory(self, retval: Any) -> bytes:
        """
        Pickle a return value, moving large arrays at any depth to shared memory.

        Args:
            retval: The return value to convert.

        Returns:
            The pickled return value.
        """
        return self.dumps(retval, "__retval")

    def retval_from_sharedmemory(self, data: bytes) -> Any:
        """
        Unpickle a return value produced by retval_to_sharedmemory().

        Args:
            data: The pickled return value.

        Returns:
            The return value with descriptors replaced by numpy array views.
        """
        return self.loads(data)

    def close(self) -> None:
        """
        Release all shared memory resources.

        Closes and unlinks all linked buffers (owned by this process), and
        closes all mapped buffers (owned by other processes). After calling
        this method, the AllocationSet should not be used.
        """
        for buf in self._linked_buffers:
            buf.close()
            buf.unlink()
        for buf in self._mapped_buffers:
            buf.close()
        self._linked_buffers = []
        self._mapped_buffers = []
        self._arrays = {}
