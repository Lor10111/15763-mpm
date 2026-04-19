"""
Stub tape classes — no warp dependency.
Tape / CondTape are only needed for warp autograd; they are unused
in the pure-PyTorch SVD path.
"""

from typing import Optional


class Tape:
    pass


class CondTape:
    def __init__(self, tape: Optional[Tape], cond: bool = True) -> None:
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
