# CallServer Usage Guide

`CallServer` runs Python functions in another process on the same machine. Numpy arrays above a size threshold in arguments and return values are transferred through shared memory, and all other data is sent through a pipe (child process) or socket (standalone server). CallServer can spawn a child server process or connect to an existing server process. It cannot be used for distributed execution across machines.

For runnable scripts, see the [examples](../examples) folder.

- [Key Features](#key-features)
- [Limitations](#limitations)
- [Usage](#usage)
- [Standalone Server](#standalone-server)
- [Notes](#notes)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)

## Key Features

- **Zero-serialization transfer**: Large numpy arrays are automatically promoted to shared memory instead of being pickled.
- **Zero-copy transfer**: Arrays can be pre-allocated in shared memory, requiring no copy and no serialization.
- **Transparent buffer management**: Arrays are promoted automatically wherever they appear in arguments and return values, including inside nested containers, dataclasses and other picklable objects. Shared memory is allocated on the first call and reused on subsequent calls.
- **Flexible call patterns**: Synchronous and asynchronous calls.
- **Double-buffering**: Front/back buffer pattern for overlapping server-side computation with client-side processing.
- **Connection modes**: Spawn a child process, or connect to a standalone server via TCP.
- **Server-side state**: Context objects can be passed to the server or created on the server.

## Limitations

- **Local machine only**: Shared memory cannot be used across machines.
- **Single async call**: Only one asynchronous call can be in progress at a time per `CallServer` instance.
- **Stable structure**: Within one argument or return value, buffers are assigned in pickling order. The arrays, their order, shapes and dtypes must stay the same between calls to the same function.
- **Picklable functions**: Server-side functions and context initializers are pickled by reference, so they must be defined at module level and importable by the server process.

## Usage

> On Windows and macOS, child processes re-import the main module, so client scripts need an `if __name__ == "__main__":` guard. It is omitted from the snippets below for brevity.

### 1. A simple example
```python
import numpy as np
from callserver import CallServer

server = CallServer('adder')
# start as child process
server.start()
# addition executes in the child process.
# the arguments and return value are exchanged through shared memory
result = server.call(np.add, np.ones((1024, 1024)), np.ones((1024, 1024)))
print(result[0, 0])  # 2.0
server.stop()
```

`CallServer` is also a context manager, which calls `start()` on entry and `stop()` on exit:
```python
with CallServer('adder') as server:
    result = server.call(np.add, a, b)
```

### 2. Server-Side State
Server-side state can be passed in via the CallServer constructor, or can be created by a server-side initializer.
If `context` is callable, it is called once inside the server process with the remaining constructor arguments, and its return value becomes the context. Otherwise, the object itself is pickled and sent to the server.
```python
import numpy as np
from callserver import CallServer

# The initializer can be a constructor or a factory function.
# It executes inside the server process.
def my_initializer(shape, value):
    # Create and return a state object
    return np.full(shape, value)

# arguments after the initializer are passed to it on the server
server = CallServer('my_server', my_initializer, (1024, 1024), 1000)
server.start()
```

Errors raised by the initializer are re-raised on the client by the next `call()` or `get_result()`.

### 3. Server-Side Functions
Server-side functions receive the context object as their first argument. If no context was given, functions are called with the call arguments only.
```python
# this executes inside the server process
def increment(state, value):
    return state + value

result = server.call(increment, 10)  # returns [[1010, 1010, ...
```

A more realistic example captures and processes camera frames on the server, and returns them through shared memory:
```python
import cv2
from callserver import CallServer

# called on the server side once, at startup
def create_context(device_id):
    return cv2.VideoCapture(device_id)

# executed on the server side on every call
def read_and_process_frame(cap):
    ret, frame = cap.read()
    # process frame as needed
    # ...
    return frame

server = CallServer('my_camera', create_context, 0)
server.start()

while True:
    frame = server.call(read_and_process_frame)  # this is the RPC call
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

server.stop()
```

### 4. Using Async Calls
Remote calls can execute asynchronously, allowing the client process to perform other work in parallel. Only one call can be in progress at a time: `call_async()` raises `RuntimeError` if the previous result has not been retrieved with `get_result()`.
```python
# start processing
server.call_async(read_and_process_frame)

# do local work while the server processes
do_local_computation()

# get the result (blocks until ready; raises TimeoutError if the timeout expires)
frame = server.get_result(timeout=1.0)
```

Use `has_result()` to poll without blocking, `wait_for_result(timeout)` to wait without retrieving, and `call_in_progress()` to check whether a call is outstanding.

### 5. Double-Buffering
Arrays returned by a call are views of shared memory, and that memory is reused by the next call to the same function. With a single allocation set, an async call therefore overwrites the previous result while the client may still be using it.

To avoid this, create two allocation sets for the function. `CallServer` rotates between them on every call, so the result of the previous call stays valid while the next call writes into the other set.
```python
# set up double-buffering
server.add_allocation_set(read_and_process_frame)
server.add_allocation_set(read_and_process_frame)

# prime the pipeline
server.call_async(read_and_process_frame)

while True:
    # get the previous result
    frame = server.get_result()
    # start next frame processing on the server
    server.call_async(read_and_process_frame)
    # use the previous frame buffer locally
    do_local_computation(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

See [examples/async_calls.py](../examples/async_calls.py).

### 6. Pre-Allocation of shared memory
Buffers allocated with `CallServer.alloc()` live in shared memory, so passing or returning them involves no copy at all.
```python
buffer = CallServer.alloc(server, increment, (1024, 1024), float)
result = server.call(increment, 10, buffer)

# On server side, the buffer arrives as a shared memory numpy array
def increment(context, value, buffer):
    buffer[:] = context # copy, no allocation
    buffer += value # in-place addition
    return buffer # buffer is already shared memory, so no allocation on return
```

Buffers can be allocated on the server side as well. The server name argument must match the name provided to the CallServer on the client.
```python
def my_initializer(shape, value):
    buffer = CallServer.alloc('my_server', increment, shape, float)
    buffer[:] = value
    return buffer
```

See [examples/preallocated_buffers.py](../examples/preallocated_buffers.py).

When pre-allocating buffers, follow these rules:
1. If the server has background threads pre-processing data directly in shared-memory buffers, you need to allocate an extra buffer (front- and back- buffers) and manage switching between them.
2. If the server-side function is only called synchronously, use a single allocation set.
3. If the server-side function is called async, use two allocation sets, with the same set of buffers in each (pass `alloc_set_id=0` and `alloc_set_id=1` to `CallServer.alloc()`). The CallServer will handle switching between the allocation sets between calls.
4. Mixing the back-buffer pattern and the double allocation-set pattern is not supported.

## Standalone Server

A server can also run as a separate, long-lived process. The port is required:

```bash
callserver --port 5000
# or
python -m callserver --port 5000 [--address 127.0.0.1] [--authkey callserver]
```

or from Python:

```python
from callserver import run, DEFAULT_AUTHKEY
run(("127.0.0.1", 5000, DEFAULT_AUTHKEY))  # blocks forever
```

Clients connect by passing a `(name, endpoint)` tuple, where `endpoint` is a port or an `(address, port)` tuple. The `authkey` must match the server's `--authkey`.

```python
with CallServer(('my_server', ('127.0.0.1', 5000)), authkey=b'callserver') as server:
    result = server.call(np.add, a, b)
```

- The server handles one client at a time. When a client disconnects, the server waits for the next one.
- If the next client uses the same name, context and context arguments, the existing server-side context is reused. Otherwise, it is re-created.
- Functions and context initializers sent to a standalone server must be importable by the server process, e.g. from installed packages, or from modules in the directory the server was started from with `python -m callserver`.

See [examples/standalone_client.py](../examples/standalone_client.py).

## Notes

- Always call `get_result()` after an asynchronous call before starting a new one.
- Returned arrays are views of shared memory that the next call to the same function reuses. Copy them (`result.copy()`) if you need to keep them beyond the next call, or use double-buffering.
- Arrays of 1024 bytes or less (`AllocationSet.MIN_SIZE`) are pickled and sent through the pipe or socket rather than promoted to shared memory. Arrays with an `object` dtype are always pickled.
- An array passed more than once in the same call is promoted once, and the server receives the same array object in each place.
- Pass `REUSE` where an array was passed on an earlier call to send the existing shared memory buffer unchanged, without copying. Raises `KeyError` if no buffer was allocated at that position yet.
- In child process mode, `stop()` waits for the server process to exit and releases all shared memory. In standalone mode, `stop()` only disconnects; the server keeps running.

## How It Works

```
┌─────────────────┐                  ┌─────────────────┐
│ Client process  │                  │ Server process  │
│                 │                  │                 │
│  CallServer     │ ── Pipe / TCP ─▶ │  run() loop     │
│                 │ ◀─ results ───── │                 │
│  Allocation-    │                  │  Allocation-    │
│  SetManager     │                  │  SetManager     │
│                 │  shared memory   │                 │
│  ┌───────────┐  │                  │  ┌───────────┐  │
│  │ Buffer A  │◀─┼──────────────────┼─▶│ Buffer A  │  │
│  └───────────┘  │                  │  └───────────┘  │
│  ┌───────────┐  │                  │  ┌───────────┐  │
│  │ Buffer B  │◀─┼──────────────────┼─▶│ Buffer B  │  │
│  └───────────┘  │                  │  └───────────┘  │
└─────────────────┘                  └─────────────────┘
```

1. On each call, the client pickles each argument separately. Large arrays found anywhere in an argument are replaced with `AllocationDescriptor`s (shared memory name, shape, dtype), copying the data into shared memory unless the array already lives there.
2. The function reference and the pickled arguments are sent through the pipe or socket.
3. The server unpickles the arguments, mapping descriptors back to numpy arrays, calls the function, and pickles the return value the same way.
4. The client unpickles the return value, mapping descriptors to numpy arrays.

Each process has one `AllocationSetManager` per server name, holding one or more `AllocationSet`s per function. Each `AllocationSet` tracks two kinds of shared memory buffers:

- **Linked buffers**: created by this process and owned by it (unlinked on close).
- **Mapped buffers**: created by the other process and only mapped for read/write access.

Buffers are created on the first call and reused on later calls. They are named by argument position or keyword (`__arg0`, `__arg1`, `__kw.name`, `__retval`), with `.1`, `.2`, ... appended for each further array within the same argument or return value.

## API Reference

### CallServer

The main client interface for making RPC calls.

**Constructor:**
```python
CallServer(name, context=None, *args, authkey=DEFAULT_AUTHKEY, **kwargs)
```
- `name`: Server identifier (string, spawns a child process), or `(name, endpoint)` tuple to connect to a standalone server, where `endpoint` is a port or an `(address, port)` tuple
- `context`: Server-side state object, or initializer callable that creates it on the server
- `*args`, `**kwargs`: Arguments passed to the context initializer
- `authkey`: Authentication key for connecting to a standalone server

**Methods:**
- `start()`: Start the server process or connect to a standalone server. Returns `self`.
- `call(fn, *args, **kwargs)`: Synchronous call. Returns the result.
- `call_async(fn, *args, **kwargs)`: Start an asynchronous call. Raises `RuntimeError` if a call is already in progress.
- `get_result(timeout=None)`: Retrieve the result of the async call, blocking until it is ready. Raises `TimeoutError` if the timeout expires. Returns the last result if no call is in progress.
- `wait_for_result(timeout=None)`: Wait for a result without retrieving it. Returns `True` if ready.
- `has_result()`: Check if a result is ready (non-blocking).
- `call_in_progress()`: Check if an async call is in progress.
- `add_allocation_set(fn)`: Add an allocation set for a function (call twice for double-buffering).
- `get_allocation_set(fn)`: Get the next allocation set for a function.
- `stop()`: Stop the server (child process mode) or disconnect (standalone mode), and clean up resources.

**Static Methods:**
- `CallServer.alloc(callserver, fn, shape, dtype=float, alloc_set_id=0)`: Pre-allocate a shared memory buffer for `fn`. `callserver` is a `CallServer` instance (client side) or the server name (server side). Returns a numpy array backed by shared memory.

### Functions

- `run(endpoint_or_connection)`: Run the server loop. Pass an `(address, port, authkey)` tuple to run a standalone TCP server.
- `run_console()`: Command-line entry point used by the `callserver` command and `python -m callserver`. Options: `--port` (required), `--address` (default `127.0.0.1`), `--authkey` (default `callserver`).

### Constants

- `DEFAULT_AUTHKEY`: Default authentication key (`b"callserver"`).
- `REUSE`: Marker argument that sends the buffer already allocated at its position, without copying. Instance of `Reuse`.

### AllocationSet

Manages shared memory buffers for a single function.

- `MIN_SIZE`: Arrays larger than this many bytes are promoted to shared memory (default 1024).
- `alloc(name, shape, dtype, init=None)`: Allocate a new buffer. Returns an `AllocationDescriptor`.
- `get(name, init=None)`: Get the descriptor of an existing buffer.
- `from_sharedmemory(descriptor)`: Map a descriptor to a numpy array.
- `close()`: Release all shared memory resources.

### AllocationSetManager

Manages the allocation sets of all functions for one server name. There is one instance per name in each process.

- `add_allocation_set(fn)`: Add a new set for a function.
- `get_allocation_set(fn, idx=None)`: Get the set at `idx`, creating it if needed. With `idx=None`, rotates to the next set.
- `get_active_id(fn)`: Index of the most recently used set.
- `close()`: Release all resources.

### AllocationDescriptor

Lightweight, picklable descriptor for a shared memory buffer (`name`, `shape`, `dtype`).
