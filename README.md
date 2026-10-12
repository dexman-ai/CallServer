# callserver

Fast inter-process function calls for Python, with numpy arrays passed through shared memory.

## Overview

`callserver` runs Python functions in another process on the same machine. Large numpy arrays in arguments and return values are moved to shared memory automatically, so they are not pickled or pushed through a pipe. Shared memory is allocated on the first call and reused on every call after that.

It is built for low-latency, high-throughput workloads that need a separate process, such as:

- Real-time robot control
- Vision processing pipelines
- Compute and data-intensive workloads that benefit from zero-copy data sharing

**Features**

- Automatic shared memory transfer of large numpy arrays, in both directions
- Zero-copy transfer with pre-allocated shared memory buffers
- Synchronous and asynchronous calls, with double-buffering to overlap work
- Persistent server-side state
- Runs the server as a child process, or connects to a standalone server over TCP

`callserver` works on a single machine only. It is not meant for distributed execution.

## Installation

```bash
pip install callserver
```

Requires Python 3.8+ and numpy.

## Usage
### Simple Example
Arrays are transferred via shared memory both ways. No serialization!
Shared memory is only allocated the first time the RPC is made. Subsequent calls reuse the same buffers.
```python
import numpy as np
from callserver import CallServer

if __name__ == "__main__":
    # Spawns a child process. np.add runs there; the arrays travel through shared memory.
    with CallServer("adder") as server:
        result = np.zeros((1024, 1024))
        ones = np.ones((1024, 1024))
        for _ in range(100):
            result = server.call(np.add, result, ones)
        print(result[0, 0])  # 100.0
```

### Server-Side State
A context passed to the server (instance or callable) holds state for subsequent calls:

```python
import numpy as np
from callserver import CallServer

class Accumulator:
    def __init__(self, shape):
        self.total = np.zeros(shape)

def accumulate(acc, values):
    acc.total += values
    return acc.total

if __name__ == "__main__":
    # Arguments after the initializer are passed to it on the server.
    with CallServer("accumulator", Accumulator, (512, 512)) as server:
        for _ in range(10):
            total = server.call(accumulate, np.ones((512, 512)))
        print(total[0, 0])  # 10.0
```

### Standalone Server
A server can also run as a separate, long-lived process that clients connect to over TCP. Start it from the command line (the port is required):

```bash
callserver --port 5000
```

Then connect by passing a `(name, (address, port))` tuple instead of a name:

```python
import numpy as np
from callserver import CallServer

with CallServer(("adder", ("127.0.0.1", 5000))) as server:
    result = server.call(np.add, np.ones((1024, 1024)), np.ones((1024, 1024)))
    print(result[0, 0])  # 2.0
```

The server handles one client at a time and keeps running after a client disconnects. Functions sent to it must be importable by the server process.

> Server-side functions must be defined at module level so they can be pickled. The `if __name__ == "__main__":` guard is required on Windows and macOS, because the child process re-imports the main module.

## Documentation

- [Usage guide](callserver/callserver.md): async calls, double-buffering, pre-allocated buffers, standalone servers, the API reference, and how it works.
- [Examples](examples): runnable scripts.

| Example | Shows |
| --- | --- |
| [simple_call.py](examples/simple_call.py) | Basic synchronous call |
| [server_state.py](examples/server_state.py) | Server-side state with a context initializer |
| [async_calls.py](examples/async_calls.py) | Async calls with double-buffering |
| [preallocated_buffers.py](examples/preallocated_buffers.py) | Zero-copy with pre-allocated shared memory |
| [standalone_client.py](examples/standalone_client.py) | Connecting to a standalone server |

## Development

```bash
pip install -e .[test]
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).
