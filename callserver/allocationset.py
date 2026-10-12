"""
Shared memory allocation management for inter-process numpy array transfer.

This module provides classes for managing shared memory buffers used in RPC
communication. It handles the low-level details of creating, mapping, and
converting numpy arrays to/from shared memory segments.
"""

from __future__ import annotations

import logging
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
        Check if an argument should be transferred via shared memory.

        Arguments are mapped to shared memory if they are numpy arrays larger
        than MIN_SIZE bytes. None values are also considered mappable if a
        corresponding allocation already exists.

        Args:
            arg: The argument to check.

        Returns:
            True if the argument should use shared memory, False otherwise.
        """
        return arg is None or (
            isinstance(arg, np.ndarray) and arg.nbytes > AllocationSet.MIN_SIZE
        )

    def to_sharedmemory(
        self, name: str, array: Optional[np.ndarray]
    ) -> Optional[AllocationDescriptor]:
        """
        Convert a numpy array to a shared memory allocation.

        If the array is already in shared memory (managed by this set), returns
        its existing descriptor. Otherwise, either reuses an existing allocation
        (if shape/dtype match) or creates a new one.

        Args:
            name: Logical name for the allocation.
            array: The numpy array to convert, or None.

        Returns:
            An AllocationDescriptor for the shared memory buffer, or None if
            array is None and no existing allocation exists.

        Raises:
            AssertionError: If array is not a numpy array (when not None).
            AssertionError: If shape or dtype changed for an existing allocation.
        """
        if array is None:
            return self.get(name) if name in self._descriptors else None
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
    
    def args_to_sharedmemory(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Convert function arguments to shared memory representations.

        Large numpy arrays in the arguments are converted to AllocationDescriptors,
        while other arguments are passed through unchanged.

        Args:
            *args: Positional arguments to convert.
            **kwargs: Keyword arguments to convert.

        Returns:
            A tuple of (positional_args, keyword_args) with arrays replaced by
            their AllocationDescriptors.
        """
        positional_args = [
            self.to_sharedmemory(f"__arg{idx}", arg) if self.is_mappable(arg) else arg
            for idx, arg in enumerate(args)
        ]
        keyword_args = {
            k: self.to_sharedmemory(k, v) if self.is_mappable(v) else v
            for k, v in kwargs.items()
        }
        return positional_args, keyword_args

    def args_from_sharedmemory(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Convert function arguments from shared memory representations.

        AllocationDescriptors in the arguments are mapped to numpy arrays,
        while other arguments are passed through unchanged.

        Args:
            *args: Positional arguments to convert.
            **kwargs: Keyword arguments to convert.

        Returns:
            A tuple of (positional_args, keyword_args) with descriptors replaced
            by their numpy array views.
        """
        positional_args = [
            self.from_sharedmemory(arg) if isinstance(arg, AllocationDescriptor) else arg
            for arg in args
        ]
        keyword_args = {
            k: self.from_sharedmemory(v) if isinstance(v, AllocationDescriptor) else v
            for k, v in kwargs.items()
        }
        return positional_args, keyword_args

    def retval_to_sharedmemory(self, retval: Any) -> Any:
        """
        Convert a function's return value to shared memory representation.

        Supports return values that are tuples, lists, dicts, or single values.
        Large numpy arrays are converted to AllocationDescriptors.

        Args:
            retval: The return value to convert.

        Returns:
            The return value with large arrays replaced by AllocationDescriptors.
        """
        if isinstance(retval, tuple):
            return tuple(
                self.to_sharedmemory(f"__retval{idx}", v) if self.is_mappable(v) else v
                for idx, v in enumerate(retval)
            )
        if isinstance(retval, list):
            return [
                self.to_sharedmemory(f"__retval{idx}", v) if self.is_mappable(v) else v
                for idx, v in enumerate(retval)
            ]
        if isinstance(retval, dict):
            return {
                k: self.to_sharedmemory(k, v) if self.is_mappable(v) else v
                for k, v in retval.items()
            }
        return (
            self.to_sharedmemory("__retval0", retval)
            if self.is_mappable(retval)
            else retval
        )

    def retval_from_sharedmemory(self, retval: Any) -> Any:
        """
        Convert a function's return value from shared memory representation.

        Supports return values that are tuples, lists, dicts, or single values.
        AllocationDescriptors are mapped to numpy arrays.

        Args:
            retval: The return value to convert.

        Returns:
            The return value with descriptors replaced by numpy array views.
        """
        if isinstance(retval, tuple):
            return tuple(
                self.from_sharedmemory(v) if isinstance(v, AllocationDescriptor) else v
                for v in retval
            )
        if isinstance(retval, list):
            return [
                self.from_sharedmemory(v) if isinstance(v, AllocationDescriptor) else v
                for v in retval
            ]
        if isinstance(retval, dict):
            return {
                k: self.from_sharedmemory(v) if isinstance(v, AllocationDescriptor) else v
                for k, v in retval.items()
            }
        return (
            self.from_sharedmemory(retval)
            if isinstance(retval, AllocationDescriptor)
            else retval
        )

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
