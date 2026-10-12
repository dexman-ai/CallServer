"""
Asynchronous calls with double-buffering.

call_async() starts the call and returns immediately so the client can work in
parallel; get_result() blocks until the result is ready. Only one async call can
be in flight at a time.

Two allocation sets are created for the server-side function, so the result of
call N (in set A) stays valid while call N+1 writes into set B.

    python examples/async_calls.py
"""

import time

import numpy as np

from callserver import CallServer

SHAPE = (480, 640, 3)
FRAMES = 20


def produce_frame(frame_id: int) -> np.ndarray:
    """Server side: simulate a slow producer, e.g. a camera capture."""
    time.sleep(0.02)
    return np.full(SHAPE, frame_id % 256, dtype=np.uint8)


def consume_frame(frame: np.ndarray) -> None:
    """Client side: simulate slow processing of a frame."""
    time.sleep(0.02)


def main() -> None:
    with CallServer("producer") as server:
        # Front and back buffers.
        server.add_allocation_set(produce_frame)
        server.add_allocation_set(produce_frame)

        # Sequential baseline.
        start = time.perf_counter()
        for i in range(FRAMES):
            consume_frame(server.call(produce_frame, i))
        sequential = time.perf_counter() - start

        # Pipelined: the server produces frame i+1 while the client consumes frame i.
        start = time.perf_counter()
        server.call_async(produce_frame, 0)
        for i in range(FRAMES):
            frame = server.get_result()
            if i + 1 < FRAMES:
                server.call_async(produce_frame, i + 1)
            assert frame[0, 0, 0] == i % 256
            consume_frame(frame)
        pipelined = time.perf_counter() - start

        print(f"sequential: {sequential:.2f}s, pipelined: {pipelined:.2f}s")


if __name__ == "__main__":
    main()
