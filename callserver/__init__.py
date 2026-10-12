"""
Shared-memory RPC module for high-performance inter-process communication.

This module provides an efficient Remote Procedure Call (RPC) mechanism that uses
shared memory for transferring large numpy arrays between processes on the same machine.
It is designed for use cases where low-latency, high-throughput data exchange is required,
such as real-time robot control, vision processing pipelines, or any compute-intensive
workload that benefits from zero-copy data sharing.

Key Features:
    - Automatic promotion of large numpy arrays to shared memory for zero-copy transfer
    - Support for both synchronous and asynchronous calls
    - Front/back buffer pattern for overlapping computation and communication
    - Connection to child processes or existing server processes via TCP

Limitations:
    - Only supports inter-process communication on the same machine (shared memory is local)
    - Cannot be used for distributed or remote execution across machines

Classes:
    CallServer: Main client interface for making RPC calls to a server process.
    AllocationSet: Manages a set of shared memory allocations for a single function.
    AllocationDescriptor: Descriptor for a shared memory buffer (name, shape, dtype).
    AllocationSetManager: Manages multiple AllocationSet instances for resource allocation.

Functions:
    run: Run the server loop, either on a child-process connection or as a
        standalone TCP server on a given (address, port, authkey) endpoint.
    run_console: Command-line entry point for running a standalone server.

Example:
    >>> import numpy as np
    >>> from callserver import CallServer
    >>> with CallServer('adder') as server:
    ...     result = server.call(np.add, np.ones((1024, 1024)), np.ones((1024, 1024)))
    >>> print(result[0, 0])
    2.0

For more details, see the callserver.md documentation file.
"""

from .allocationmanager import AllocationSetManager
from .allocationset import AllocationDescriptor, AllocationSet
from .callserver import DEFAULT_AUTHKEY, CallServer, run, run_console

__all__ = [
    "CallServer",
    "run",
    "run_console",
    "DEFAULT_AUTHKEY",
    "AllocationSet",
    "AllocationDescriptor",
    "AllocationSetManager",
]
