"""Scene: ``build_glass()`` builds the glass tank, ``add_ocean()`` adds the water, ``render()`` produces the image.

Drawing order is the one three.js r128 uses for this scene: opaque objects first (surface, floor, edges), then
transparent objects sorted by ``renderOrder`` (ground -1, back flank 1, bubbles 2, side flanks 3, front flank 4,
particles 5, edge halos 6, glass 7); double-sided transparent materials are drawn in two
passes (back faces, then front faces), as in three.js r128.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import geometry as geo
from .camera import Camera
from .spec import Tank
from .timeline import Timeline, OceanState


@dataclass
class Layer:
    name: str
    program: str                      # shader name or 'basic'
    mode: str                         # 'triangles' | 'points' | 'lines'
    attrs: dict                       # attribute name -> float32 array (N, k)
    index: np.ndarray | None = None
    offset: tuple = (0.0, 0.0, 0.0)   # model translation
    transparent: bool = False
    blend: str = "none"               # 'none' | 'normal' | 'additive'
    depth_write: bool = True
    side: str = "front"               # 'front' | 'double'
    order: int = 0
    color: tuple = (1.0, 1.0, 1.0)    # basic material
    opacity: float = 1.0
    dynamic: str | None = None        # Timeline key of the buffers updated every frame


class Scene:
    def __init__(self, tank: Tank | None = None, surface_segments=(660, 200), hash_mode: str = "native"):
        self.tank = tank or Tank()
        self.hash_mode = hash_mode      # 'native': original shader; 'portable': hash without sin()
        self.surface_segments = surface_segments
        self.layers: list[Layer] = []
        self._gl = None

    # ------------------------------------------------------------------ construction (build)
    def build_glass(self):
        """Glass tank: 12 glowing edges (core + additive halo) and glass panes (opacity .045)."""
        L, W, GH = self.tank.length, self.tank.width, self.tank.glass_height
        x, z = L / 2, W / 2

        def beam(a, b):
            ln = float(np.linalg.norm(np.subtract(b, a)))
            along = [abs(b[i] - a[i]) > 0 for i in range(3)]
            for thick, kind in ((.045, "core"), (.2, "halo")):
                sz = [ln if along[i] else thick for i in range(3)]
                pos, idx = geo.box(*sz)
                mid = tuple((a[i] + b[i]) / 2 for i in range(3))
                if kind == "core":
                    self.layers.append(Layer("beam_core", "basic", "triangles", {"position": pos}, idx, mid,
                                             color=(0xd9 / 255, 0xef / 255, 1.0)))
                else:
                    self.layers.append(Layer("beam_halo", "basic", "triangles", {"position": pos}, idx, mid,
                                             transparent=True, blend="additive", depth_write=False, order=6,
                                             color=(0x6f / 255, 0xb8 / 255, 1.0), opacity=.16))
        for y in (0, GH):
            beam((-x, y, -z), (x, y, -z)); beam((-x, y, z), (x, y, z))
            beam((-x, y, -z), (-x, y, z)); beam((x, y, -z), (x, y, z))
        for cx, cz in ((-x, -z), (x, -z), (-x, z), (x, z)):
            beam((cx, 0, cz), (cx, GH, cz))
        pos, idx = geo.box(L, GH, W)
        self.layers.append(Layer("glass", "basic", "triangles", {"position": pos}, idx, (0, GH / 2, 0),
                                 transparent=True, blend="normal", depth_write=False, side="double", order=7,
                                 color=(0x9f / 255, 0xd2 / 255, 1.0), opacity=.045))
        self._gl = None
        return self

    def add_ocean(self):
        """Water: animated surface, 4 flanks seen through the glass, tank floor and reflections on the ground."""
        L, W = self.tank.length, self.tank.width
        sx, sz = self.surface_segments
        pos, idx = geo.plane_xz(L, W, sx, sz)
        self.layers.append(Layer("surface", "surface", "triangles", {"position": pos}, idx, side="double"))
        for (a, b, segs, order) in (((-L / 2, -W / 2), (L / 2, -W / 2), 660, 1),
                                    ((-L / 2, -W / 2), (-L / 2, W / 2), 160, 3),
                                    ((L / 2, -W / 2), (L / 2, W / 2), 160, 3),
                                    ((-L / 2, W / 2), (L / 2, W / 2), 660, 4)):
            p, au, i = geo.wall(a[0], a[1], b[0], b[1], segs)
            self.layers.append(Layer("wall", "wall", "triangles", {"position": p, "aU": au}, i,
                                     transparent=True, blend="normal", depth_write=False, side="double", order=order))
        p, i = geo.plane_xz(L, W, 1, 1)
        self.layers.append(Layer("floor", "floor", "triangles", {"position": p}, i, (0, .01, 0)))
        p, i = geo.plane_xz(90, 60, 1, 1)
        self.layers.append(Layer("ground", "ground", "triangles", {"position": p}, i, (0, -.04, 0),
                                 transparent=True, blend="normal", depth_write=False, order=-1))
        self._gl = None
        return self

    def add_particles(self, timeline: Timeline):
        """Spray, bubbles, steam and surface tracers (positions driven by ``timeline``)."""
        z3 = lambda n: np.zeros((n, 3), np.float32)
        add = lambda **kw: self.layers.append(Layer(mode="points", blend="additive", transparent=True,
                                                    depth_write=False, **kw))
        add(name="bubbles", program="bubbles", attrs={"position": z3(len(timeline.bubble_seeds)),
                                                      "aS": timeline.bubble_seeds}, order=2)
        add(name="spray", program="spray", attrs={"position": z3(len(timeline.spray_seeds)),
                                                  "aS": timeline.spray_seeds}, order=5)
        add(name="steam", program="steam", attrs={"position": z3(len(timeline.steam_seeds)),
                                                  "aS": timeline.steam_seeds}, order=5)
        add(name="heads", program="heads", attrs={"position": timeline.head_pos.copy(), "aK": timeline.head_k.reshape(-1, 1)},
            order=5, dynamic="heads")
        self.layers.append(Layer("segs", "segs", "lines", {"position": timeline.seg_pos.copy(), "aK": timeline.seg_k.reshape(-1, 1)},
                                 transparent=True, blend="additive", depth_write=False, order=5, dynamic="segs"))
        self._gl = None
        return self

    # ------------------------------------------------------------------ rendering
    def render(self, timeline: Timeline | OceanState, camera: Camera | None = None, size=(1280, 720),
               samples: int = 4) -> np.ndarray:
        """RGBA image (uint8, premultiplied alpha = raw WebGL canvas content), origin at the top left."""
        from .render_gl import GLRenderer
        if self._gl is None or self._gl.size != tuple(size) or self._gl.samples != samples:
            self._gl = GLRenderer(self, tuple(size), samples)
        return self._gl.render(timeline, camera or Camera())
