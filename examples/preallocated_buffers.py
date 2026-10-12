"""
Pre-allocated shared memory buffers (zero-copy).

- The client pre-allocates its input buffer with CallServer.alloc() and writes
  into it directly, so passing it to the server copies nothing.
- The server pre-allocates its output buffer in the context initializer, and
  the server-side function writes into it in place, so returning it copies
  nothing either.

The name passed to CallServer.alloc() on the server side must match the name
given to the CallServer on the client side.

    python examples/preallocated_buffers.py
"""

import numpy as np

from callserver import CallServer

NAME = "scaler"
SHAPE = (1024, 1024)


def scale(ctx: dict, data: np.ndarray, factor: float) -> np.ndarray:
    """Server side: write the result into the pre-allocated output buffer."""
    np.multiply(data, factor, out=ctx["out"])
    return ctx["out"]


def create_context(shape) -> dict:
    """Runs once on the server: allocate the output buffer in shared memory."""
    return {"out": CallServer.alloc(NAME, scale, shape, np.float64)}


def main() -> None:
    with CallServer(NAME, create_context, SHAPE) as server:
        data = CallServer.alloc(server, scale, SHAPE, np.float64)
        for i in range(1, 4):
            data[:] = i  # write directly into shared memory
            result = server.call(scale, data, 10.0)
            print(f"{i} * 10 = {result[0, 0]}")


if __name__ == "__main__":
    main()
