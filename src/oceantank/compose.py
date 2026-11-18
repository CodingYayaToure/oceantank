"""Compositing over the page background (CSS radial gradient) or over a plain colour."""
from __future__ import annotations
import numpy as np

ABYSS = (0x02, 0x0a, 0x1c)


def page_background(w: int, h: int) -> np.ndarray:
    """radial-gradient(ellipse 62% 46% at 50% 34%, #1b5cc4 0%, #0b2f73 34%, #041536 68%, #020a1c 100%)."""
    ys, xs = (np.arange(h) + .5)[:, None], (np.arange(w) + .5)[None, :]
    t = np.sqrt(((xs - .5 * w) / (.62 * w)) ** 2 + ((ys - .34 * h) / (.46 * h)) ** 2)
    stops = [0.0, .34, .68, 1.0]
    cols = np.array([(0x1b, 0x5c, 0xc4), (0x0b, 0x2f, 0x73), (0x04, 0x15, 0x36), ABYSS], float)
    return np.stack([np.interp(t, stops, cols[:, c]) for c in range(3)], -1)


def over(rgba_premult: np.ndarray, background="page") -> np.ndarray:
    """Premultiplied-alpha render output placed over a background -> RGB uint8."""
    h, w = rgba_premult.shape[:2]
    bg = page_background(w, h) if isinstance(background, str) and background == "page" else \
        np.broadcast_to(np.asarray(background, float), (h, w, 3))
    f = rgba_premult.astype(float)
    out = f[..., :3] + bg * (1 - f[..., 3:4] / 255.0)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
