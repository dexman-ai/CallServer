"""
Simplest CallServer usage.

Runs np.add in a child process. Large numpy arguments and return values are
transferred through shared memory, which is allocated on the first call and
reused on every subsequent call.

    python examples/simple_call.py
"""

import numpy as np

from callserver import CallServer


def main() -> None:
    with CallServer("adder") as server:
        result = np.zeros((1024, 1024))
        ones = np.ones((1024, 1024))
        for _ in range(100):
            result = server.call(np.add, result, ones)
        print(f"result[0, 0] = {result[0, 0]}")  # 100.0


# The __main__ guard is required: on Windows and macOS the child process
# re-imports this module.
if __name__ == "__main__":
    main()
