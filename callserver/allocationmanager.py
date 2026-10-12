"""
AllocationSetManager for managing multiple function allocation sets.

This module provides the AllocationSetManager class which coordinates multiple
AllocationSet instances for different functions in an RPC workflow.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

from .allocationset import AllocationSet


class AllocationSetManager:
    """
    Manages multiple AllocationSet instances for resource allocation in RPC workflows.

    This class implements a singleton pattern per name, ensuring that all references
    to the same named manager share state. It supports front- and back-buffering
    for function arguments, tracks active sets, and handles cleanup.

    The manager provides multiple allocation sets per function to support asynchronous
    call patterns where one set is being used for a call in progress while another
    is being prepared for the next call.

    Attributes:
        _managers: Class-level dict of all manager instances by name.

    Example:
        >>> manager = AllocationSetManager("my_server")
        >>> alloc_set = manager.add_allocation_set("my_function")
        >>> # Add a second set for async double-buffering
        >>> alloc_set2 = manager.add_allocation_set("my_function")
    """

    _managers: Dict[str, "AllocationSetManager"] = {}

    def __new__(cls, name: Optional[str] = None) -> "AllocationSetManager":
        """
        Return existing manager for the given name, or create a new one.

        Args:
            name: The unique name for this manager.

        Returns:
            An AllocationSetManager instance (existing or new).
        """
        if name is not None and name in cls._managers:
            return cls._managers[name]
        return super().__new__(cls)

    def __init__(self, name: Optional[str] = None) -> None:
        """
        Initialize the AllocationSetManager.

        If a manager with the given name already exists, this is a no-op.

        Args:
            name: The unique name for this manager.
        """
        if name is not None and name in AllocationSetManager._managers:
            return
        self._name = name
        self._arg_sets: Dict[str, List[AllocationSet]] = {}
        self._active_set_ids: Dict[str, int] = {}
        if self._name is not None:
            assert (
                self._name not in AllocationSetManager._managers
            ), f"An AllocationSetManager with name {self._name} already exists."
            AllocationSetManager._managers[self._name] = self

    def _make_name(self, fn: Union[Callable, str]) -> str:
        """
        Convert a function or string to a string name.

        Args:
            fn: A callable or string identifier.

        Returns:
            The function's __name__ attribute, or the string itself.
        """
        return fn.__name__ if callable(fn) else fn

    def add_allocation_set(self, fn: Union[Callable, str]) -> AllocationSet:
        """
        Add a new AllocationSet for the given function.

        Creates a new allocation set and adds it to the list of sets for the
        specified function. Multiple sets can be created for double-buffering.

        Args:
            fn: A callable or string identifier for the function.

        Returns:
            The newly created AllocationSet.
        """
        fn_name = self._make_name(fn)
        if fn_name not in self._arg_sets:
            self._arg_sets[fn_name] = []
            self._active_set_ids[fn_name] = -1

        arg_set = AllocationSet(
            len(self._arg_sets[fn_name]), f"{self._name}.{fn_name}"
        )
        self._arg_sets[fn_name].append(arg_set)
        return arg_set

    def get_allocation_set(
        self, fn: Union[Callable, str], idx: Optional[int] = None
    ) -> AllocationSet:
        """
        Get an AllocationSet for the given function.

        If idx is None, rotates to the next allocation set (for double-buffering).
        If idx is specified, returns or creates the set at that index.

        Args:
            fn: A callable or string identifier for the function.
            idx: Optional specific index to retrieve.

        Returns:
            The requested AllocationSet.
        """
        fn_name = self._make_name(fn)
        if fn_name not in self._arg_sets:
            self.add_allocation_set(fn_name)
        if idx is None:
            idx = self._active_set_ids[fn_name]
            idx = (1 + idx) % len(self._arg_sets[fn_name])
            self._active_set_ids[fn_name] = idx
        else:
            for i in range(len(self._arg_sets[fn_name]), idx + 1):
                self.add_allocation_set(fn_name)
        active_set = self._arg_sets[fn_name][idx]
        return active_set

    def get_active_id(self, fn: Union[Callable, str]) -> int:
        """
        Get the currently active allocation set index for a function.

        Args:
            fn: A callable or string identifier for the function.

        Returns:
            The index of the currently active AllocationSet.
        """
        fn_name = self._make_name(fn)
        return self._active_set_ids[fn_name]

    def close(self) -> None:
        """
        Release all resources held by all allocation sets.

        Closes all allocation sets for all functions managed by this manager.
        After calling this method, the manager should not be used.
        """
        for arg_set_group in self._arg_sets.values():
            for arg_set in arg_set_group:
                arg_set.close()
        self._arg_sets = {}

    def __del__(self) -> None:
        """Clean up resources when the manager is garbage collected."""
        self.close()


if __name__ == "__main__":
    manager = AllocationSetManager("test")
    manager.add_allocation_set("func1")
    manager2 = AllocationSetManager("test")
    assert manager is manager2
    assert manager2.get_allocation_set("func1") != None

    print("Test passed.")
    