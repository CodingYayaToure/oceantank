"""Orbit camera (same formulas as placeCamera() in the animation, three.js matrices)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    az: float = 0.0          # azimuth (rad)
    el: float = 0.62         # elevation (rad)
    zoom: float = 1.0
    fov: float = 32.0        # vertical field of view (deg)
    near: float = 0.1
    far: float = 300.0
    target: tuple = (0.0, 1.25, 0.0)

    def position(self, aspect: float) -> np.ndarray:
        hf = np.tan(self.fov * np.pi / 360.0) * aspect
        dist = (17.0 / hf + 3.6) * self.zoom
        ce = np.cos(self.el)
        return np.array([dist * ce * np.sin(self.az), 1.3 + dist * np.sin(self.el), dist * ce * np.cos(self.az)])

    def view(self, aspect: float) -> np.ndarray:
        eye = self.position(aspect)
        t = np.asarray(self.target, float)
        z = eye - t
        z /= np.linalg.norm(z)
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3] = x, y, z
        m[:3, 3] = -m[:3, :3] @ eye
        return m

    def projection(self, aspect: float) -> np.ndarray:
        f = 1.0 / np.tan(np.radians(self.fov) / 2)
        n, fa = self.near, self.far
        m = np.zeros((4, 4))
        m[0, 0], m[1, 1] = f / aspect, f
        m[2, 2], m[2, 3] = (fa + n) / (n - fa), 2 * fa * n / (n - fa)
        m[3, 2] = -1.0
        return m
