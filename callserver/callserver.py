"""
CallServer for shared-memory RPC with inter-process function execution.

This module provides the CallServer class for making RPC calls between processes
on the same machine, using shared memory for efficient numpy array transfer.
"""

from __future__ import annotations

import argparse
import threading
from multiprocessing import Pipe, Process
from multiprocessing.connection import Client, Connection, Listener
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from .allocationmanager import AllocationSetManager
from .allocationset import AllocationSet

DEFAULT_AUTHKEY = b"callserver"


class CallServer:
    """
    Client interface for shared-memory RPC with inter-process function execution.

    CallServer enables remote or local function execution with shared memory argument
    passing. It manages allocation sets for efficient inter-process communication,
    supports asynchronous and synchronous calls, and can operate as a client
    (connecting to an existing process) or spawn a child server process.

    Key Features:
        - Automatic promotion of large numpy arrays to shared memory
        - Support for both synchronous and asynchronous calls
        - Server-side state management via context objects
        - Connection to child processes or existing servers via TCP

    Limitations:
        - Only suitable for inter-process communication on the same machine
          (shared memory cannot be used across machines)
        - Only one async call can be in progress at a time

    Attributes:
        name: The unique name for this server instance.
        port: TCP port for remote connections (None for child process mode).
        context: Server-side state object or initializer.

    Example:
        >>> import numpy as np
        >>> from callserver import CallServer
        >>> server = CallServer('adder')
        >>> server.start()
        >>> result = server.call(np.add, np.ones((1024, 1024)), np.ones((1024, 1024)))
        >>> server.stop()
    """

    def __init__(
        self,
        name: Union[str, Tuple[str, Union[int, Tuple[str, int]]]],
        context: Optional[Any] = None,
        *args: Any,
        authkey: bytes = DEFAULT_AUTHKEY,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a CallServer client.

        Args:
            name: Either a string name (spawns a child process), or a tuple of
                (name, endpoint) where endpoint is a port number or (address, port)
                tuple for connecting to an existing server.
            context: Server-side context object, or a callable that will be
                invoked on the server to create the context.
            *args: Arguments passed to the context initializer.
            authkey: Authentication key used when connecting to a standalone
                server. Must match the key the server was started with.
            **kwargs: Keyword arguments passed to the context initializer.
        """
        self.name = name
        self.port: Optional[int] = None
        self.address: str = "127.0.0.1"
        if isinstance(name, tuple):
            self.name, endpoint = name
            self.address, self.port = (
                endpoint if isinstance(endpoint, tuple) else ("127.0.0.1", endpoint)
            )
        self.context = context
        self.context_args = args
        self.context_kwargs = kwargs
        self._alloc_manager = AllocationSetManager(self.name)
        self._call_in_progress = False
        self._started = False
        if self.port:
            self._pipe: Connection = Client(
                (self.address, self.port), authkey=authkey
            )
            self._server_process: Optional[Process] = None
        else:
            self._pipe, self._server_pipe = Pipe()
            self._server_process = Process(
                name=self.name, target=run, args=(self._server_pipe,), daemon=True
            )
        self._last_result: Any = None
        self._last_result_raw: Any = None
        self._call_unpack_fn: Optional[Callable] = None
        self._call_unpack_argset: Optional[AllocationSet] = None

    def __repr__(self) -> str:
        """Return a string representation of the CallServer."""
        mode = "remote" if self.port else "child"
        status = "started" if self._started else "stopped"
        return f"CallServer(name={self.name!r}, mode={mode}, status={status})"

    def __enter__(self) -> "CallServer":
        """Enter context manager, starting the server."""
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, stopping the server."""
        self.stop()

    def start(self) -> "CallServer":
        """
        Start the server process or connect to a remote server.

        For child process mode, spawns the server process. For remote mode,
        sends the initial configuration to the server.

        Returns:
            self, for method chaining.
        """
        if self._server_process:
            self._server_process.start()
        self._pipe.send(
            (self.name, self.context, self.context_args, self.context_kwargs)
        )
        self._started = True
        return self

    def add_allocation_set(self, fn: Union[Callable, str]) -> AllocationSet:
        """
        Add a new AllocationSet for the given function.

        Use this to pre-allocate additional buffer sets for double-buffering
        in async call patterns.

        Args:
            fn: A callable or string identifier for the function.

        Returns:
            The newly created AllocationSet.
        """
        return self._alloc_manager.add_allocation_set(fn)

    @staticmethod
    def alloc(
        callserver: Union["CallServer", str],
        fn: Union[Callable, str],
        shape: Tuple[int, ...],
        dtype: type = float,
        alloc_set_id: int = 0,
    ) -> np.ndarray:
        """
        Pre-allocate a shared memory buffer for a function.

        This static method can be used from both client and server code to
        pre-allocate buffers that will be used for argument passing.

        Args:
            callserver: A CallServer instance, or the server name string.
            fn: A callable or string identifier for the function.
            shape: Shape of the buffer to allocate.
            dtype: Data type of the buffer (default: float).
            alloc_set_id: Which allocation set to use (default: 0).

        Returns:
            A numpy array backed by shared memory.
        """
        alloc_mgr = (
            callserver._alloc_manager
            if isinstance(callserver, CallServer)
            else AllocationSetManager(callserver)
        )
        argset = alloc_mgr.get_allocation_set(fn, alloc_set_id)
        descriptor = argset.alloc("buffer", shape, dtype)
        return argset.from_sharedmemory(descriptor)

    def get_allocation_set(self, fn: Union[Callable, str]) -> AllocationSet:
        """
        Get the current AllocationSet for the given function.

        Args:
            fn: A callable or string identifier for the function.

        Returns:
            The current AllocationSet for the function.
        """
        return self._alloc_manager.get_allocation_set(fn)

    def _call_async(
        self,
        fn: Callable,
        argset: AllocationSet,
        unpack_fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Internal method to initiate an async call.

        Args:
            fn: The function to call on the server.
            argset: The AllocationSet containing the arguments.
            unpack_fn: Function to convert the result from shared memory.
            *args: Positional arguments (already converted to descriptors).
            **kwargs: Keyword arguments (already converted to descriptors).
        """
        self._call_in_progress = True
        self._call_unpack_fn = unpack_fn
        self._call_unpack_argset = argset
        self._pipe.send((fn, argset.ix, args, kwargs))

    def call_async(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        """
        Initiate an asynchronous function call.

        The call executes on the server in parallel. Use get_result() to
        retrieve the result when ready.

        Args:
            fn: The function to call on the server.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Raises:
            RuntimeError: If an async call is already in progress.
        """
        if self._call_in_progress:
            raise RuntimeError(
                "An async call is already in progress. Call get_result to "
                "complete the current call before starting a new one."
            )
        arg_set = self.get_allocation_set(fn)
        args, kwargs = arg_set.args_to_sharedmemory(*args, **kwargs)
        self._call_async(fn, arg_set, AllocationSet.retval_from_sharedmemory, *args, **kwargs)

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Make a synchronous function call.

        Equivalent to call_async() followed by get_result().

        Args:
            fn: The function to call on the server.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.
        """
        self.call_async(fn, *args, **kwargs)
        return self.get_result()

    def call_in_progress(self) -> bool:
        """
        Check if an async call is currently in progress.

        Returns:
            True if a call is in progress, False otherwise.
        """
        return self._call_in_progress

    def has_result(self) -> bool:
        """
        Check if a result is available without blocking.

        Returns:
            True if a result is ready to be retrieved, False otherwise.
        """
        return self._pipe.poll()

    def wait_for_result(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for a result to become available.

        If no async call is in progress, returns True immediately (a previous
        result may be available via get_result()).

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            True if a result is available (or no call is in progress),
            False if timeout expired while waiting.
        """
        if not self._call_in_progress:
            return True
        return self._pipe.poll(timeout)

    def get_result(self, timeout: Optional[float] = None) -> Any:
        """
        Retrieve the result of the current or last async call.

        If no call is in progress, returns the result of the last call.
        Otherwise, blocks until the result is available or timeout expires.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            The result of the function call, with numpy arrays converted
            from shared memory.

        Raises:
            TimeoutError: If timeout expires before result is available.
        """
        if not self._call_in_progress:
            return self._last_result
        if not self._pipe.poll(timeout):
            raise TimeoutError(
                f"Timed out waiting for result after {timeout} seconds"
            )
        self._last_result_raw = self._pipe.recv()
        if isinstance(self._last_result_raw, Exception):
            self._call_in_progress = False
            raise self._last_result_raw from self._last_result_raw
        self._last_result = self._call_unpack_fn(
            self._call_unpack_argset, self._last_result_raw
        )
        self._call_in_progress = False
        return self._last_result

    def stop(self) -> None:
        """
        Stop the server and clean up resources.

        Closes the connection and waits for the server process to terminate
        (if in child process mode).
        """
        if not self._started:
            return
        self._started = False
        self._pipe.close()
        if self._server_process is not None:
            self._server_process.join()
            self._server_process = None
            self._alloc_manager.close()

    def __del__(self) -> None:
        """Clean up resources when the client is garbage collected."""
        try:
            self.stop()
        except Exception:
            pass  # Ignore errors during cleanup

def run(
    endpoint_or_connection: Union[Connection, Tuple[str, int, bytes]],
) -> None:
    """
    Server loop for handling RPC calls.

    This function runs the main server loop, processing function calls from
    clients and returning results. It can operate in two modes:

    1. Child process mode: Receives a connection from a parent process
    2. Standalone mode: Listens for TCP connections on a specified endpoint.
       In this mode the function never returns; after a client disconnects,
       it waits for the next client.

    Args:
        endpoint_or_connection: Either a Connection object (for child process
            mode), or a tuple of (address, port, authkey) for standalone mode.

    Example:
        >>> from callserver import run
        >>> run(("127.0.0.1", 5000, b"callserver"))  # blocks forever
    """
    keep_alive = isinstance(endpoint_or_connection, tuple)
    pipe: Optional[Connection] = None if keep_alive else endpoint_or_connection
    config: Optional[Tuple] = None

    while keep_alive or pipe:
        if keep_alive:
            pipe = _wait_for_client_connection(*endpoint_or_connection)
        try:
            name, context_init, context_args, context_kwargs = pipe.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            if not keep_alive:
                return
            continue

        alloc_manager = AllocationSetManager(name)
        if not config or config != (name, context_init, context_args, context_kwargs):
            config = (name, context_init, context_args, context_kwargs)
            try:
                context = (
                    context_init(*context_args, **context_kwargs)
                    if callable(context_init)
                    else context_init
                )
            except Exception as e:
                try:
                    pipe.send(e)
                except (EOFError, BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                if not keep_alive:
                    return
                continue

        while pipe:
            try:
                cmd = pipe.recv()
            except (EOFError, BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                if not keep_alive:
                    return
                break

            fn, arg_set_idx, args, kwargs = cmd
            argset = alloc_manager.get_allocation_set(fn, arg_set_idx)
            args, kwargs = argset.args_from_sharedmemory(*args, **kwargs)
            result = (
                fn(context, *args, **kwargs) if context is not None else fn(*args, **kwargs)
            )
            result = argset.retval_to_sharedmemory(result)
            try:
                pipe.send(result)
            except (EOFError, BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                if not keep_alive:
                    return
                break


def _wait_for_client_connection(
    address: str,
    port: int,
    authkey: bytes,
) -> Connection:
    """
    Wait for a client connection on the specified endpoint.

    Starts a listener thread and waits for a client to connect.

    Args:
        address: The address to listen on.
        port: The port to listen on.
        authkey: Authentication key for the connection.

    Returns:
        The established Connection object.
    """
    listener_thread = threading.Thread(
        target=_wait_for_client_connection_thread,
        args=(address, port, authkey),
        daemon=True,
    )
    listener_thread.start()
    while listener_thread.is_alive():
        listener_thread.join(timeout=1)
    return listener_thread.connection


def _wait_for_client_connection_thread(
    address: str,
    port: int,
    authkey: bytes,
) -> None:
    """
    Thread target for listening for client connections.

    Creates a listener and waits for a client to connect. Stores the
    connection on the thread object for retrieval by the parent.

    Args:
        address: The address to listen on.
        port: The port to listen on.
        authkey: Authentication key for the connection.
    """
    with Listener((address, port), authkey=authkey) as listener:
        print(f"Listening for connections on {address}:{port}")
        threading.current_thread().connection = listener.accept()
        print("Connection accepted from", listener.last_accepted)


def run_console() -> None:
    """
    Command-line entry point for running a standalone RPC server.

    Parses command-line arguments for address, port, and authentication key,
    then starts the server loop. The server can be stopped with Ctrl+C.

    Command-line arguments:
        --port: Port to listen on (required)
        --address: Address to listen on (default: 127.0.0.1)
        --authkey: Authentication key (default: callserver)
    """
    parser = argparse.ArgumentParser(description="Start a standalone CallServer.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--address", type=str, default="127.0.0.1")
    parser.add_argument(
        "--authkey", type=str.encode, default=DEFAULT_AUTHKEY.decode()
    )
    args = parser.parse_args()
    try:
        run((args.address, args.port, args.authkey))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_console()
