"""Ocean state over time: a port of apply()/advance()/stepTracers() from the HTML animation.

``Timeline`` provides the shader uniforms (uT, uC, uI, uRot, uR, uBurst, uDis) and the particle state.
The ``Mulberry32`` pseudo-random generator replays exactly the same sequence in Python and in
JavaScript (see tools/compare_with_html.py); a ``numpy.random.Generator`` can be used instead.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .spec import Tank

ZX0, ZX1 = -3.0, 5.5            # warm zone (same values as the animation)
PERIOD = 32.0
VMAX = 6.0
R0 = 1.85
NI = 1600


def ss(a: float, b: float, x: float) -> float:
    x = min(1.0, max(0.0, (x - a) / (b - a)))
    return x * x * (3 - 2 * x)


class Mulberry32:
    """Same generator as the one injected into the browser for the comparison."""
    def __init__(self, seed: int = 12345):
        self.a = seed & 0xFFFFFFFF

    @staticmethod
    def _imul(a: int, b: int) -> int:
        return (a * b) & 0xFFFFFFFF

    def random(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        a = self.a
        t = self._imul(a ^ (a >> 15), 1 | a)
        t = ((t + self._imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    def array(self, n: int) -> np.ndarray:
        return np.array([self.random() for _ in range(n)], dtype=np.float64)


class _NumpyRng:
    def __init__(self, seed=0):
        self.g = np.random.default_rng(seed)

    def random(self) -> float:
        return float(self.g.random())

    def array(self, n: int) -> np.ndarray:
        return self.g.random(n)


def make_rng(seed=0, kind: str = "numpy"):
    return Mulberry32(seed) if kind == "mulberry32" else _NumpyRng(seed)


@dataclass
class OceanState:
    """Water uniforms (GLSL uniform names)."""
    uT: float = 0.0
    uC: tuple = (-11.2, 0.0)
    uI: float = 0.0
    uRot: float = 0.0
    uR: float = R0
    uBurst: float = 0.0
    uDis: float = 0.0
    uZ: tuple = (ZX0, ZX1)


class Timeline:
    """Cyclone crossing the tank: reproduces state/apply/advance of the animation."""

    def __init__(self, tank: Tank | None = None, gain: float = 1.35, seed: int = 12345, rng: str = "mulberry32",
                 n_spray: int = 3200, n_bubbles: int = 900, n_steam: int = 1700, n_tracers: int = 2600,
                 itab: np.ndarray | None = None, start_ready: bool = True, zone: tuple = (ZX0, ZX1)):
        self.tank = tank or Tank()
        L, W = self.tank.length, self.tank.width
        self.L, self.W = L, W
        self.X0, self.X1 = -L / 2 + 3.8, L / 2 - 3.8
        self.gain = gain
        self.t = 0.0
        self.clock = 0.0
        self.rot = 0.0
        self.zone = (float(zone[0]), float(zone[1]))      # warm zone [z0, z1]: shader tint and vortex expansion
        self.state = OceanState(uZ=self.zone)
        self.rng = make_rng(seed, rng)
        self.itab = self._build_itab() if itab is None else np.asarray(itab, np.float32)
        # particle seeds, in the animation's order: spray, bubbles, steam, then tracers
        f32 = np.float32
        self.spray_seeds = self.rng.array(n_spray * 4).astype(f32).reshape(-1, 4)
        self.bubble_seeds = self.rng.array(n_bubbles * 4).astype(f32).reshape(-1, 4)
        self.steam_seeds = self.rng.array(n_steam * 4).astype(f32).reshape(-1, 4)
        self.n_tracers = n_tracers
        self.tx = np.zeros(n_tracers, f32)
        self.tz = np.zeros(n_tracers, f32)
        self.tl = np.zeros(n_tracers, f32)
        for i in range(n_tracers):
            self._respawn(i)
            self.tl[i] = f32(float(self.tl[i]) * self.rng.random())
        self.head_pos = np.zeros((n_tracers, 3), f32)
        self.seg_pos = np.zeros((n_tracers * 2, 3), f32)
        self.head_k = np.ones(n_tracers, f32)
        self.seg_k = np.tile(np.array([1, 0], f32), n_tracers)
        if start_ready:                      # the animation's "readable start state"
            self.t, self.rot = PERIOD * 0.3, 20.0
            for _ in range(90):
                self.advance(1 / 30)

    # ---- intensity model (SST -> intensity), identical to the animation ----
    @staticmethod
    def sst_at(x: float) -> float:
        warm = ss(ZX0 - 2.6, ZX0 + 1.6, x) * (1 - ss(ZX1 - 1.6, ZX1 + 2.6, x))
        cold = ss(7.6, 11.4, x)
        return 27.6 + 3.0 * warm - 3.2 * cold - 1.1 * ss(ZX0 + 1.5, ZX1 + .5, x) * warm

    def _build_itab(self) -> np.ndarray:
        if self.zone != (ZX0, ZX1):                       # custom warm zone: same law through the vectorised model
            from .model import Params, intensity_table
            return intensity_table(Params(x0=self.X0, x1=self.X1, z0=self.zone[0], z1=self.zone[1])).astype(np.float32)
        tab = np.zeros(NI + 1, np.float32)
        I, dx = .04, (self.X1 - self.X0) / NI
        for k in range(NI + 1):
            x = self.X0 + dx * k
            sst = self.sst_at(x)
            tg = max(0.0, (sst - 26.5) / 4) ** .9
            lam = 1.5 if tg > I else (1.4 if sst < 26.5 else 5.2)
            I += (tg - I) * min(1.0, dx / lam)
            tab[k] = I
        return tab

    def inorm(self, u: float) -> float:
        f = min(NI - .001, max(0.0, u * NI))
        k = int(math.floor(f))
        a = f - k
        return (float(self.itab[k]) * (1 - a) + float(self.itab[k + 1]) * a) * (1 - ss(.97, 1, u))

    def slope(self, u: float) -> float:
        d = .004
        return (self.inorm(min(1.0, u + d)) - self.inorm(max(0.0, u - d))) / ((self.X1 - self.X0) * 2 * d)

    # ---- particles ----
    def _respawn(self, i: int):
        f32 = np.float32
        self.tx[i] = f32((self.rng.random() - .5) * (self.L - .4))
        self.tz[i] = f32((self.rng.random() - .5) * (self.W - .4))
        self.tl[i] = f32(3 + self.rng.random() * 9)

    def step_tracers(self, dt: float, I: float, cx: float, cz: float, t: float):
        f32 = np.float32
        tx, tz = self.tx.astype(np.float64), self.tz.astype(np.float64)
        dx, dz = tx - cx, tz - cz
        r = np.hypot(dx, dz) + 1e-4
        prof = np.where(r < R0, r / R0, (R0 / r) ** .85 * np.exp(-(r - R0) / 11))
        V = VMAX * I * prof
        vx = V * dz / r - .22 * V * dx / r + .06 * np.sin(tz * 1.3 + t)
        vz = -V * dx / r - .22 * V * dz / r + .06 * np.cos(tx * 1.1 - t)
        self.tx = (tx + vx * dt).astype(f32)
        self.tz = (tz + vz * dt).astype(f32)
        self.tl = (self.tl.astype(np.float64) - dt).astype(f32)
        dead = ((self.tl < 0) | ((r < .45 * R0) & (I > .3))
                | (np.abs(self.tx) > self.L / 2 - .1) | (np.abs(self.tz) > self.W / 2 - .1))
        for i in np.nonzero(dead)[0]:         # ascending order: same random consumption as the JS loop
            self._respawn(int(i))
            vx[i] = vz[i] = 0.0
        self.head_pos[:, 0], self.head_pos[:, 2] = self.tx, self.tz
        tau = .11
        self.seg_pos[0::2, 0], self.seg_pos[0::2, 2] = self.tx, self.tz
        self.seg_pos[1::2, 0] = (self.tx - vx * tau).astype(f32)
        self.seg_pos[1::2, 2] = (self.tz - vz * tau).astype(f32)

    # ---- time ----
    def apply(self, dt: float):
        u = (self.t % PERIOD) / PERIOD
        In = self.inorm(u)
        I = In * 1.18 * self.gain
        cx = self.X0 + (self.X1 - self.X0) * u
        cz = .75 * math.sin(u * math.pi * 3)
        R = R0 * (1.1 - .28 * min(In / .95, 1.1) + .45 * ss(self.zone[1] - 1.5, self.zone[1] + 6, cx))
        self.rot += dt * 2.7 * (.45 + .75 * I)
        s = self.state
        s.uT, s.uC, s.uI, s.uRot, s.uR = self.clock, (cx, cz), I, self.rot, R
        s.uBurst = min(1.0, max(0.0, self.slope(u) / .26))
        s.uDis = ss(8.0, 10.6, cx)
        self.step_tracers(dt, I, cx, cz, self.clock)

    def run_to(self, t_target: float, dt: float = 1 / 30):
        """Advance (without rendering) from the start of the cycle to ``t_target`` seconds, in [0, 32]."""
        self.t = self.clock = self.rot = 0.0
        while self.t + dt <= t_target:
            self.advance(dt)
        self.advance(max(t_target - self.t, 0.0))

    def advance(self, dt: float):
        self.t += dt
        self.clock += dt
        if self.t >= PERIOD:
            self.t -= PERIOD
        self.apply(dt)
