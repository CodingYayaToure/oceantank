"""High-performance tools: parallel parameter sweeps, convergence studies, scaling benchmarks.

Design
------
* The relaxation along the track is *sequential* (each step depends on the previous one) but *independent between
  ensemble members*. We therefore loop over the track and vectorise over members (SIMD/NumPy), then spread chunks of
  members over processes (``multiprocessing``, ``spawn`` so it behaves the same on Linux, macOS, Windows and Colab).
* Chunks are deterministic functions of their parameters: the result is identical for any number of workers.
* ``scaling`` reports wall time, speedup, parallel efficiency and an Amdahl fit (serial fraction).
"""
from __future__ import annotations
import itertools
import multiprocessing as mp
import os
import time
import numpy as np
from .model import Params, intensity_stats, intensity_table, NI
from . import physics as ph


def grid(**axes) -> tuple[dict, tuple]:
    """Cartesian product of parameter axes -> (flat arrays, shape). ``grid(warm_amp=[1,2,3], lam_grow=[1,2])``."""
    names = list(axes)
    vals = [np.atleast_1d(np.asarray(axes[n], dtype=float)) for n in names]
    mesh = np.meshgrid(*vals, indexing="ij")
    return {n: m.ravel() for n, m in zip(names, mesh)}, tuple(len(v) for v in vals)


def _chunk_job(args):
    flat, ni, base = args
    p = base.replace(**flat)
    st = intensity_stats(p, ni)
    return {k: np.asarray(v) for k, v in st.items()}


def sweep(axes: dict, base: Params = Params(), ni: int = NI, workers: int = 1, chunk: int | None = None) -> dict:
    """Parameter sweep over the Cartesian product of ``axes`` (peak intensity, location of the peak, final intensity).

    Returns ``{"I_peak","u_peak","I_end"}`` reshaped to the grid shape plus ``"axes"`` and ``"seconds"``.
    """
    flat, shape = grid(**axes)
    n = int(np.prod(shape))
    if chunk is None:
        chunk = max(1, -(-n // (4 * workers))) if workers > 1 else n
    pieces = [(({k: v[i:i + chunk] for k, v in flat.items()}), ni, base) for i in range(0, n, chunk)]
    t0 = time.perf_counter()
    if workers > 1 and len(pieces) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(pieces))) as pool:
            out = pool.map(_chunk_job, pieces)
    else:
        out = [_chunk_job(a) for a in pieces]
    dt = time.perf_counter() - t0
    res = {k: np.concatenate([o[k] for o in out]).reshape(shape) for k in out[0]}
    res.update(axes={k: np.asarray(v) for k, v in axes.items()}, seconds=dt, size=n)
    return res


def convergence(base: Params = Params(), levels=(50, 100, 200, 400, 800, 1600), ref_factor: int = 16) -> dict:
    """Time-step convergence of the phenomenological law (explicit Euler): error of I_peak vs a fine reference run.

    Returns the levels, the errors, and the observed order ``p`` of ``err ~ h^p`` (least-squares slope in log-log).
    """
    levels = np.asarray(levels, dtype=int)
    ref = float(intensity_stats(base, int(levels[-1]) * ref_factor)["I_peak"])
    err = np.array([abs(float(intensity_stats(base, int(n))["I_peak"]) - ref) for n in levels])
    ok = err > 0
    slope = np.polyfit(np.log(1.0 / levels[ok]), np.log(err[ok]), 1)[0]
    return {"levels": levels, "h": 1.0 / levels, "error": err, "order": float(slope), "reference": ref}


def convergence_physics(q: ph.PhysParams = ph.PhysParams(), mp_: Params = Params(), levels=(50, 100, 200, 400),
                        ref_factor: int = 8) -> dict:
    """Convergence of the RK4 energy-balance integrator: max-norm error of the whole V(t) curve on the nodes common to
    every level (levels are nested), against a reference run ``ref_factor`` times finer."""
    levels = np.asarray(levels, dtype=int)
    n_ref = int(levels[-1]) * ref_factor
    ref = ph.run(q, mp_, n_ref)["v"]
    err = np.array([np.max(np.abs(ph.run(q, mp_, int(n))["v"] - ref[..., :: n_ref // int(n)])) for n in levels])
    ok = err > 1e-14
    slope = np.polyfit(np.log(1.0 / levels[ok]), np.log(err[ok]), 1)[0] if ok.sum() > 1 else float("nan")
    return {"levels": levels, "h": 1.0 / levels, "error": err, "order": float(slope), "reference": float(ref.max())}


def scaling(n_members: int = 20000, workers=(1, 2), ni: int = NI, repeats: int = 1, chunk: int = 10000) -> dict:
    """Strong scaling of ``sweep``: same problem, increasing number of processes. Includes the vectorisation gain.

    ``serial_loop`` is a pure-Python loop over members (one scalar run per member, extrapolated from a sample) — the
    baseline that vectorisation removes. The chunk size is the same for every worker count, so that the comparison does
    not mix parallelism with cache effects.
    """
    rng = np.random.default_rng(0)
    axes_sample = {"warm_amp": rng.uniform(0.0, 5.0, 40)}
    t0 = time.perf_counter()
    for w in axes_sample["warm_amp"]:
        intensity_stats(Params(warm_amp=float(w)), ni)
    t_scalar = (time.perf_counter() - t0) / 40 * n_members          # extrapolated
    axes = {"warm_amp": np.linspace(0.0, 5.0, n_members)}
    times = []
    for w in workers:
        best = min(sweep(axes, ni=ni, workers=w, chunk=chunk)["seconds"] for _ in range(repeats))
        times.append(best)
    times = np.array(times)
    w = np.asarray(workers, dtype=float)
    speed = times[0] / times
    eff = speed / (w / w[0])
    f = None
    if len(w) > 1 and w[-1] > 1:                                      # Amdahl: S = 1 / (f + (1 - f)/n)
        s_, n_ = speed[-1], w[-1] / w[0]
        f = float(np.clip((1.0 / s_ - 1.0 / n_) / (1.0 - 1.0 / n_), 0.0, 1.0))
    return {"workers": np.asarray(workers), "seconds": times, "speedup": speed, "efficiency": eff, "amdahl_serial_fraction": f,
            "n_members": n_members, "scalar_loop_seconds": t_scalar, "vectorisation_gain": t_scalar / times[0],
            "cpu_count": os.cpu_count()}


def render_scaling(workers=(1, 2), n_frames: int = 8, size=(480, 270)) -> dict:
    """Strong scaling of parallel frame rendering (software OpenGL). Also checks pixel identity serial vs parallel."""
    from .api import Viewer
    v = Viewer(size=size)
    times = np.linspace(2.0, 30.0, n_frames)
    res, ref = [], None
    for w in workers:
        t0 = time.perf_counter()
        frames = v.render_frames(times, workers=w)
        res.append(time.perf_counter() - t0)
        arrs = [np.asarray(f) for f in frames]
        if ref is None:
            ref = arrs
        else:
            identical = all(np.array_equal(a, b) for a, b in zip(ref, arrs))
    sec = np.array(res); w_ = np.asarray(workers, dtype=float)
    return {"workers": np.asarray(workers), "seconds": sec, "speedup": sec[0] / sec, "efficiency": sec[0] / sec / (w_ / w_[0]),
            "frames": n_frames, "identical": True if len(workers) == 1 else bool(identical)}


def _chunk_phys(args):
    flat, n_steps, base_p, base_q = args
    kp = {k: v for k, v in flat.items() if k in model_keys()[0]}
    kq = {k: v for k, v in flat.items() if k in model_keys()[1]}
    r = ph.run(base_q.replace(**kq), base_p.replace(**kp), n_steps)
    v = r["v"]
    j = v.argmax(axis=-1)
    return {"v_peak": v.max(axis=-1), "u_peak": j / n_steps, "v_end": v[..., -1], "dT_max": r["dT"].max(axis=-1)}


def model_keys():
    from .dashboard import P_KEYS, Q_KEYS   # same registry of field names as the dashboard
    return P_KEYS, Q_KEYS


def sweep_physics(axes: dict, base_p: Params = Params(), base_q: ph.PhysParams = ph.PhysParams(), n_steps: int = 400,
                  workers: int = 1, chunk: int | None = None) -> dict:
    """Parameter sweep of the energy-balance model (RK4) over the Cartesian product of ``axes``.

    Axis names may be fields of :class:`Params` (environment) or :class:`PhysParams` (physics), e.g.
    ``{"warm_amp": ..., "speed_ms": ...}``. Memory per chunk is ``chunk * n_steps * 4 * 8`` bytes, hence the small default.
    """
    flat, shape = grid(**axes)
    n = int(np.prod(shape))
    if chunk is None:
        chunk = max(1, min(2000, -(-n // (4 * workers)) if workers > 1 else n))
    pieces = [({k: v[i:i + chunk] for k, v in flat.items()}, n_steps, base_p, base_q) for i in range(0, n, chunk)]
    t0 = time.perf_counter()
    if workers > 1 and len(pieces) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(pieces))) as pool:
            out = pool.map(_chunk_phys, pieces)
    else:
        out = [_chunk_phys(a) for a in pieces]
    res = {k: np.concatenate([o[k] for o in out]).reshape(shape) for k in out[0]}
    res.update(axes={k: np.asarray(v) for k, v in axes.items()}, seconds=time.perf_counter() - t0, size=n)
    return res
