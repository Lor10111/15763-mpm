"""
Pure-Python reader for Houdini classic BGEO v5 (binary, big-endian).

Only extracts particle positions — sufficient for visualizing CKMPM /
partio-style point-cloud dumps without installing Disney's partio.

Supports plain .bgeo and gzip-compressed .bgeo (auto-detected).
"""
from __future__ import annotations
import gzip, struct
from pathlib import Path
import numpy as np


def read_bgeo_positions(path) -> np.ndarray:
    """Read positions (N, 3) float32 from a Houdini BGEO v5 file."""
    path = str(path)
    with open(path, "rb") as f:
        data = f.read()

    # auto-detect gzip
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)

    p = 0
    magic = data[p:p+4]; p += 4
    if magic != b"Bgeo":
        raise ValueError(f"{path}: not a classic BGEO (magic={magic!r})")
    version_char = data[p:p+1]; p += 1                                 # 'V'
    (version,) = struct.unpack(">i", data[p:p+4]); p += 4
    if version != 5:
        raise ValueError(f"{path}: only BGEO v5 supported, got v{version}")

    (nPoints, nPrims, nPointGroups) = struct.unpack(">iii", data[p:p+12]); p += 12
    (nPrimGroups, nPointAttrib, nVertexAttrib, nPrimAttrib, nAttrib) = \
        struct.unpack(">iiiii", data[p:p+20]); p += 20

    # particleSize is # of 32-bit slots per point.
    # Layout: [0:3] = xyz, [3] = w (homogeneous), [4:] = subsequent attrs.
    particle_size = 4

    for _ in range(nPointAttrib):
        (name_len,) = struct.unpack(">H", data[p:p+2]); p += 2
        p += name_len                                                  # skip name
        (count, hou_type) = struct.unpack(">Hi", data[p:p+6]); p += 6
        if hou_type in (0, 1, 5):                                      # FLOAT, INT, VECTOR
            p += count * 4                                             # skip default values
            particle_size += count
        elif hou_type == 4:                                            # INDEXEDSTR
            (n_idx,) = struct.unpack(">i", data[p:p+4]); p += 4
            for _ in range(n_idx):
                (idx_name_len,) = struct.unpack(">H", data[p:p+2]); p += 2
                p += idx_name_len
            particle_size += count
        else:
            raise ValueError(f"{path}: unsupported attribute type {hou_type}")

    # Read raw point block: nPoints rows of particle_size 32-bit slots.
    raw_size = nPoints * particle_size * 4
    if p + raw_size > len(data):
        raise ValueError(
            f"{path}: declared {nPoints} points × {particle_size}×4 bytes "
            f"({raw_size}) exceeds remaining file size {len(data)-p}"
        )
    arr = np.frombuffer(data[p:p+raw_size], dtype=">f4").reshape(nPoints, particle_size)
    positions = np.ascontiguousarray(arr[:, :3]).astype(np.float32)
    return positions


if __name__ == "__main__":
    import sys
    pos = read_bgeo_positions(sys.argv[1])
    print(f"{sys.argv[1]}: {pos.shape[0]} points")
    print(f"  bbox min: {pos.min(axis=0)}")
    print(f"  bbox max: {pos.max(axis=0)}")
