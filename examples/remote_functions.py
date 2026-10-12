"""
Functions executed by the standalone server in standalone_client.py.

Functions are sent to the server by reference (module + name), so they must be
importable by the server process as well as the client.
"""

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()
