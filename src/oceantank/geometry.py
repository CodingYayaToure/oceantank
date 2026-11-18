"""Tank geometries: same vertices/indices as the three.js geometries of the animation."""
from __future__ import annotations
import numpy as np


def plane_xz(width_x: float, depth_z: float, seg_x: int, seg_z: int):
    """PlaneGeometry(width_x, depth_z, seg_x, seg_z) then rotateX(-pi/2): horizontal plane, normal +Y."""
    gx1, gz1 = seg_x + 1, seg_z + 1
    sx, sz = width_x / seg_x, depth_z / seg_z
    ix = np.arange(gx1)
    iz = np.arange(gz1)
    x = ix * sx - width_x / 2
    z = iz * sz - depth_z / 2
    X, Z = np.meshgrid(x, z)                                  # (gz1, gx1)
    pos = np.stack([X, np.zeros_like(X), Z], axis=-1).reshape(-1, 3).astype(np.float32)
    iy, ixx = np.meshgrid(iz[:-1], ix[:-1], indexing="ij")
    a = ixx + gx1 * iy
    b = ixx + gx1 * (iy + 1)
    c = (ixx + 1) + gx1 * (iy + 1)
    d = (ixx + 1) + gx1 * iy
    idx = np.stack([np.stack([a, b, d], -1), np.stack([b, c, d], -1)], axis=2).reshape(-1)
    return pos, idx.astype(np.uint32)


def wall(ax: float, az: float, bx: float, bz: float, segs: int):
    """Vertical flank (unit height, scaled by the shader) between two points of the XZ plane."""
    length = float(np.hypot(bx - ax, bz - az))
    s = np.arange(segs + 1) / segs
    x = ax + (bx - ax) * s
    z = az + (bz - az) * s
    pos = np.zeros((2 * (segs + 1), 3), np.float32)
    pos[0::2, 0], pos[1::2, 0] = x, x
    pos[0::2, 1], pos[1::2, 1] = 0.0, 1.0
    pos[0::2, 2], pos[1::2, 2] = z, z
    au = np.repeat(s * length, 2).astype(np.float32)
    k = np.arange(segs) * 2
    idx = np.stack([k, k + 1, k + 2, k + 1, k + 3, k + 2], -1).reshape(-1)
    return pos, au, idx.astype(np.uint32)


def box(sx: float, sy: float, sz: float):
    """BoxGeometry centred at the origin, faces pointing outwards (CCW)."""
    h = np.array([sx, sy, sz]) / 2
    verts, idx = [], []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3        # (u, v, axis) is right-handed
        for sgn in (1, -1):
            quad = []
            for (su, sv) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis], p[u], p[v] = sgn * h[axis], su * h[u], sv * h[v]
                quad.append(p)
            if sgn < 0:
                quad = quad[::-1]
            n = len(verts)
            verts += quad
            idx += [n, n + 1, n + 2, n, n + 2, n + 3]
    return np.array(verts, np.float32), np.array(idx, np.uint32)
