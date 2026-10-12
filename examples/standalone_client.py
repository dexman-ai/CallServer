"""
Connect to an already-running standalone server.

1. Start the server from the examples directory, so that remote_functions.py is
   importable by the server:

       cd examples
       python -m callserver --port 5000

2. In another terminal, run this client:

       python examples/standalone_client.py --port 5000

The server keeps running after the client disconnects and accepts the next
client. Stop it with Ctrl+C.
"""

import argparse

import numpy as np

from callserver import CallServer
from remote_functions import normalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--address", default="127.0.0.1")
    args = parser.parse_args()

    # (name, (address, port)) connects to an existing server instead of
    # spawning a child process.
    with CallServer(("normalizer", (args.address, args.port))) as server:
        data = np.random.default_rng(0).normal(5.0, 2.0, (1024, 1024))
        result = server.call(normalize, data)
        print(f"mean={result.mean():.3f}, std={result.std():.3f}")


if __name__ == "__main__":
    main()
