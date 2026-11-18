"""Pure, vectorised version of the cyclone model used by the animation (no OpenGL).

The animation's cyclone is driven by a small phenomenological chain::

    SST(x)  ->  target intensity I*(x)  ->  relaxation along the track  ->  I(u)  ->  storm state (position, radius, ...)

This module exposes that chain as plain NumPy functions so that it can be plotted, swept over thousands of parameter sets
(every field of :class:`Params` may be a scalar or an array, broadcast together) and fed back to the renderer through
``Timeline(itab=...)``. With default parameters it reproduces ``Timeline`` exactly (checked by the test-suite).
All lengths are in tank units.
"""
from __future__ import annotations
from dataclasses import dataclass, fields, replace
import numpy as np

PERIOD = 32.0                 # seconds of animation per crossing
NI = 1600                     # resolution of the intensity table
R0 = 1.85                     # reference vortex radius
VMAX = 6.0                    # peak tracer speed for I = 1 (tank units / s)
COLD_X0, COLD_X1 = 7.6, 11.4  # cold-water ramp (fixed: also hard-coded in the shaders)
DIS_X0, DIS_X1 = 8.0, 10.6    # dissipation ramp (fixed, shaders)


def ss(a, b, x):
    """Cubic smoothstep, vectorised."""
    s = np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@dataclass(frozen=True)
class Params:
    """Parameters of the SST profile and of the intensity law. Each field may be a scalar or a NumPy array."""
    # sea-surface temperature profile (deg C)
    sst_base: float = 27.6      # open-water SST
    warm_amp: float = 3.0       # warm-zone anomaly
    cold_amp: float = 3.2       # cooling in the cold region at the end of the tank
    wake_cool: float = 1.1      # cold-wake proxy inside the warm zone
    z0: float = -3.0            # warm zone start (tank units)
    z1: float = 5.5             # warm zone end
    # intensity law
    threshold: float = 26.5     # genesis threshold (deg C, Gray 1968)
    norm: float = 4.0           # K that map to I* = 1
    exponent: float = 0.9
    lam_grow: float = 1.5       # relaxation length while intensifying
    lam_cold: float = 1.4       # relaxation length while decaying over cold water
    lam_warm: float = 5.2       # relaxation length while decaying over warm water
    i0: float = 0.04            # intensity at the left end of the track
    # animation scaling and track
    gain: float = 1.35
    x0: float = -11.2
    x1: float = 11.2

    def replace(self, **kw) -> "Params":
        return replace(self, **kw)

    @staticmethod
    def for_tank(tank) -> "Params":
        """Track ends used by the animation for a given ``Tank``."""
        return Params(x0=-tank.length / 2 + 3.8, x1=tank.length / 2 - 3.8)


def sst(x, p: Params = Params()):
    """Sea-surface temperature (deg C) at position ``x`` along the track (broadcasts with array parameters)."""
    warm = ss(p.z0 - 2.6, p.z0 + 1.6, x) * (1.0 - ss(p.z1 - 1.6, p.z1 + 2.6, x))
    cold = ss(COLD_X0, COLD_X1, x)
    wake = ss(p.z0 + 1.5, p.z1 + 0.5, x) * warm
    return p.sst_base + p.warm_amp * warm - p.cold_amp * cold - p.wake_cool * wake


def target_intensity(sst_c, p: Params = Params()):
    """I*(SST) = max(0, (SST - threshold) / norm) ** exponent."""
    return np.maximum(0.0, (np.asarray(sst_c, dtype=float) - p.threshold) / p.norm) ** p.exponent


def _broadcast_shape(p: Params):
    return np.broadcast(*[np.asarray(getattr(p, f.name), dtype=float) for f in fields(p)]).shape


def _relax(p: Params, ni: int):
    """Generator of the relaxation along the track: yields ``(k, I_k)`` for k = 0..ni (arrays over the ensemble).

    dI/dx = (I* - I) / lambda,  lambda in {lam_grow, lam_cold, lam_warm}, explicit Euler. The loop is over the track
    (sequential); the arithmetic is over the ensemble (vectorised).
    """
    shape = _broadcast_shape(p)
    cur = np.broadcast_to(np.asarray(p.i0, dtype=float), shape).copy()
    dx = (np.asarray(p.x1, dtype=float) - np.asarray(p.x0, dtype=float)) / ni
    for k in range(ni + 1):
        x = p.x0 + dx * k
        s = sst(x, p)
        tg = target_intensity(s, p)
        lam = np.where(tg > cur, p.lam_grow, np.where(s < p.threshold, p.lam_cold, p.lam_warm))
        cur = cur + (tg - cur) * np.minimum(1.0, dx / lam)
        yield k, cur


def intensity_table(p: Params = Params(), ni: int = NI) -> np.ndarray:
    """Intensity along the track (law of ``_relax``). Shape ``(..., ni + 1)``; a vectorised copy of ``Timeline._build_itab``."""
    tab = np.empty(_broadcast_shape(p) + (ni + 1,))
    for k, cur in _relax(p, ni):
        tab[..., k] = cur
    return tab


def intensity_stats(p: Params = Params(), ni: int = NI) -> dict:
    """Streaming summary of the same computation, in O(ensemble) memory (for very large parameter sweeps).

    Returns the peak intensity ``I_peak`` (with the animation gain), its track fraction ``u_peak`` and the final value.
    """
    shape = _broadcast_shape(p)
    peak = np.full(shape, -np.inf)
    kpk = np.zeros(shape, dtype=int)
    cur = None
    for k, cur in _relax(p, ni):
        better = cur > peak
        peak = np.where(better, cur, peak)
        kpk = np.where(better, k, kpk)
    return {"I_peak": peak * 1.18 * np.asarray(p.gain, dtype=float), "u_peak": kpk / ni, "I_end": cur * 1.18 * np.asarray(p.gain, dtype=float)}


def inorm(tab: np.ndarray, u, ni: int | None = None):
    """Intensity (before gain) at track fraction ``u`` in [0, 1]: linear interpolation of the table, end fade-out."""
    ni = (tab.shape[-1] - 1) if ni is None else ni
    u = np.asarray(u, dtype=float)
    f = np.minimum(ni - 0.001, np.maximum(0.0, u * ni))
    k = np.floor(f).astype(int)
    a = f - k
    v = tab[..., k] * (1.0 - a) + tab[..., k + 1] * a
    return v * (1.0 - ss(0.97, 1.0, u))


def cycle_signals(p: Params, tab: np.ndarray, t, period: float = PERIOD) -> dict:
    """Storm state at animation times ``t`` (seconds), exactly as ``Timeline.apply`` computes it (scalar parameters)."""
    t = np.asarray(t, dtype=float)
    u = (t % period) / period
    In = inorm(tab, u)
    I = In * 1.18 * p.gain
    cx = p.x0 + (p.x1 - p.x0) * u
    cz = 0.75 * np.sin(u * np.pi * 3)
    R = R0 * (1.1 - 0.28 * np.minimum(In / 0.95, 1.1) + 0.45 * ss(p.z1 - 1.5, p.z1 + 6.0, cx))
    d = 0.004
    slope = (inorm(tab, np.minimum(1.0, u + d)) - inorm(tab, np.maximum(0.0, u - d))) / ((p.x1 - p.x0) * 2 * d)
    return {"t": t, "u": u, "cx": cx, "cz": cz, "In": In, "I": I, "R": R,
            "burst": np.clip(slope / 0.26, 0.0, 1.0), "dis": ss(DIS_X0, DIS_X1, cx),
            "sst": sst(cx, p), "v_tracer": VMAX * I}


def wind_profile(r, I, r0: float = R0, vmax: float = VMAX):
    """Tangential speed of the animation's modified Rankine vortex (tank units / s)."""
    r = np.asarray(r, dtype=float) + 1e-4
    prof = np.where(r < r0, r / r0, (r0 / r) ** 0.85 * np.exp(-(r - r0) / 11.0))
    return vmax * I * prof


def cycle_events(p: Params, tab: np.ndarray, n: int = 3201) -> dict:
    """Key instants of one crossing (seconds): warm-zone entry/exit, peak intensity, rapid-intensification window."""
    t = np.linspace(0.0, PERIOD, n, endpoint=False)
    s = cycle_signals(p, tab, t)
    span = p.x1 - p.x0
    ev = {"t_enter": float((p.z0 - p.x0) / span * PERIOD), "t_exit": float((p.z1 - p.x0) / span * PERIOD),
          "t_peak": float(t[np.argmax(s["I"])]), "I_peak": float(s["I"].max())}
    ri = s["burst"] > 0.5
    ev["t_ri_start"] = float(t[ri][0]) if ri.any() else None
    ev["t_ri_end"] = float(t[ri][-1]) if ri.any() else None
    return ev
