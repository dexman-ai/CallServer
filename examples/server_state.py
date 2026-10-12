"""
Server-side state.

The context initializer (here the Accumulator class) runs once inside the
server process. Every server-side function then receives the context object as
its first argument, so state persists between calls.

    python examples/server_state.py
"""

import numpy as np

from callserver import CallServer

SHAPE = (512, 512)


class Accumulator:
    def __init__(self, shape):
        self.total = np.zeros(shape)
        self.count = 0


# Server-side functions must be defined at module level so they can be pickled.
def accumulate(acc: Accumulator, values: np.ndarray) -> np.ndarray:
    acc.total += values
    acc.count += 1
    return acc.total


def mean(acc: Accumulator) -> np.ndarray:
    return acc.total / acc.count


def main() -> None:
    # Arguments after the context initializer are passed to it on the server.
    with CallServer("accumulator", Accumulator, SHAPE) as server:
        rng = np.random.default_rng(0)
        for _ in range(10):
            total = server.call(accumulate, rng.random(SHAPE))
        print(f"total[0, 0] = {total[0, 0]:.3f}")
        print(f"mean[0, 0]  = {server.call(mean)[0, 0]:.3f}")


if __name__ == "__main__":
    main()
