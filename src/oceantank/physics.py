"""Energy-balance intensity model (reduced, Emanuel-type) driven by the animation's SST track.

This is the *physical* counterpart of the animation's phenomenological SST law (``oceantank.model``). It integrates

    d(V^2/2)/dt = eta/h * [ eps * Ck * V * (k0* - k) - Cd * V^3 ],   eps = (Ts - To) / To

i.e. dV/dt = a (Vp^2 - V^2) with a = eta * Cd / h and the potential intensity Vp^2 = (Ck/Cd) * eps * (k0* - k)
(Emanuel, 1986, in reduced form), plus a relaxation equation for the sea-surface cooling dT under the storm
(cold wake). Units are SI. The SST seen by the storm is the animation's SST profile sampled along the track, with the
track mapped to ``length_km`` travelled at ``speed_ms``.

Honest status: a reduced model (no vertical structure, no convection, no shear, no dry air). ``eta``, ``ck_cd`` and the
viability threshold are *calibrated, not measured*. Every field of :class:`PhysParams` (except the booleans) may be an
array: the ensemble is then integrated in one vectorised loop.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from .model import Params, sst as sst_profile, NI

KELVIN = 273.15
CP = 1005.0      # J/kg/K
LV = 2.5e6       # J/kg
EPS = 0.622


def saturation_vapor_pressure(t_kelvin):
    """Saturation vapour pressure over water (hPa), Bolton (1980)."""
    t_c = np.asarray(t_kelvin, dtype=float) - KELVIN
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def saturation_humidity(t_kelvin, p_hpa=1010.0):
    e = saturation_vapor_pressure(t_kelvin)
    return EPS * e / (np.asarray(p_hpa, dtype=float) - (1.0 - EPS) * e)


def enthalpy_disequilibrium(sst_k, p_hpa=1010.0, rh_air=0.8, dt_air=1.0):
    """k0* - k (J/kg): enthalpy of saturated sea air minus that of the surface air."""
    sst_k = np.asarray(sst_k, dtype=float)
    t_air = sst_k - dt_air
    return CP * (sst_k - t_air) + LV * (saturation_humidity(sst_k, p_hpa) - rh_air * saturation_humidity(t_air, p_hpa))


def potential_intensity(sst_k, t_out_k=200.0, ck_cd=0.5, rh_air=0.8, p_hpa=1010.0, dt_air=1.0):
    """Potential maximum wind Vp (m/s); 0 when the energy balance is negative."""
    sst_k = np.asarray(sst_k, dtype=float)
    vp2 = ck_cd * (sst_k - t_out_k) / t_out_k * enthalpy_disequilibrium(sst_k, p_hpa, rh_air, dt_air)
    return np.sqrt(np.maximum(vp2, 0.0))


def viability(sst_k, threshold_k=26.5 + KELVIN, width_k=0.5):
    """Empirical 0..1 factor (logistic around the Gray threshold) standing for what the model does not resolve."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(sst_k, dtype=float) - threshold_k) / width_k))


@dataclass(frozen=True)
class PhysParams:
    cd: float = 1.5e-3              # drag coefficient
    ck_cd: float = 0.5              # Ck/Cd (calibrated)
    h_bl: float = 2000.0            # effective boundary-layer depth (m)
    eta: float = 0.25               # spin-up efficiency (calibrated)
    t_out_k: float = 200.0          # outflow temperature (K)
    rh_air: float = 0.8
    dt_air: float = 1.0
    viab_threshold_c: float = 27.0  # effective viability threshold (deg C), close to Gray's 26.5
    viab_width_k: float = 0.4
    tau_cool_s: float = 12 * 3600.0 # relaxation time of the sea-surface cooling
    cool_k: float = 1.5             # equilibrium cooling (K) for V = v_ref and U = u_ref
    v_ref: float = 60.0
    u_ref: float = 5.0
    cool_max_k: float = 6.0
    length_km: float = 2000.0       # length of the track
    speed_ms: float = 6.0           # translation speed (m/s)
    v0: float = 12.0                # initial wind (m/s)
    use_viability: bool = True
    cooling: bool = True            # sea-surface cooling feedback on/off

    @property
    def a(self):
        return self.eta * self.cd / self.h_bl

    @property
    def duration_s(self):
        return np.asarray(self.length_km) * 1000.0 / np.asarray(self.speed_ms)

    def replace(self, **kw) -> "PhysParams":
        return replace(self, **kw)


def vp_effective(sst_k, q: PhysParams):
    vp = potential_intensity(sst_k, q.t_out_k, q.ck_cd, q.rh_air, dt_air=q.dt_air)
    if q.use_viability:
        vp = vp * viability(sst_k, np.asarray(q.viab_threshold_c) + KELVIN, q.viab_width_k)
    return vp


def _shape(q: PhysParams, mp: Params):
    from .model import _broadcast_shape
    parts = [np.asarray(getattr(q, f), dtype=float) for f in
             ("cd", "ck_cd", "h_bl", "eta", "t_out_k", "rh_air", "dt_air", "viab_threshold_c", "viab_width_k", "tau_cool_s",
              "cool_k", "v_ref", "u_ref", "cool_max_k", "length_km", "speed_ms", "v0")]
    return np.broadcast(np.empty(_broadcast_shape(mp)), *parts).shape


def run(q: PhysParams = PhysParams(), mp: Params = Params(), n_steps: int = NI) -> dict:
    """Integrate (V, dT) along the track with RK4 on ``n_steps`` steps. Outputs have shape ``(..., n_steps + 1)``."""
    shape = _shape(q, mp)
    dur = np.broadcast_to(q.duration_s, shape)
    h = dur / n_steps
    span = np.asarray(mp.x1, dtype=float) - np.asarray(mp.x0, dtype=float)
    speed = np.asarray(q.speed_ms, dtype=float)

    def sst_bg_k(t):
        return sst_profile(mp.x0 + span * t / dur, mp) + KELVIN

    def f(t, y):
        v, dT = y
        vp = vp_effective(sst_bg_k(t) - dT, q)
        dv = q.a * (vp * vp - v * v)
        if q.cooling:
            eq = np.minimum(q.cool_max_k, q.cool_k * (v / q.v_ref) ** 2 * (q.u_ref / np.maximum(speed, 2.0)))
            ddT = (eq - dT) / q.tau_cool_s
        else:
            ddT = np.zeros_like(v)
        return np.stack([dv, ddT])

    y = np.stack([np.broadcast_to(np.asarray(q.v0, dtype=float), shape).copy(), np.zeros(shape)])
    V = np.empty(shape + (n_steps + 1,)); DT = np.empty_like(V); BG = np.empty_like(V); VP = np.empty_like(V)
    t = np.zeros(shape)
    for i in range(n_steps + 1):
        bg = sst_bg_k(t)
        V[..., i], DT[..., i], BG[..., i] = y[0], y[1], bg
        VP[..., i] = vp_effective(bg - y[1], q)
        if i == n_steps:
            break
        k1 = f(t, y); k2 = f(t + h / 2, y + h / 2 * k1); k3 = f(t + h / 2, y + h / 2 * k2); k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    tt = np.linspace(0.0, 1.0, n_steps + 1) * dur[..., None]
    return {"t": tt, "u": np.linspace(0.0, 1.0, n_steps + 1), "v": V, "dT": DT, "sst_bg_k": BG, "sst_seen_k": BG - DT, "vp": VP}


def intensity_table(q: PhysParams = PhysParams(), mp: Params = Params(), v_norm: float = 70.0, ni: int = NI) -> np.ndarray:
    """Physical wind history normalised to the animation's intensity scale (``V / v_norm``): usable as ``Timeline(itab=...)``.

    The normalisation ``v_norm`` (m/s) is a *visual* choice that maps the physical wind to the animation's I scale
    (the built-in law peaks near 0.95).
    """
    return run(q, mp, ni)["v"] / v_norm


# ---- verification helpers (used by the test-suite and the convergence study) ------------------------------------------
def analytic_const_vp(t, v0, vp, a):
    """Exact solution of dV/dt = a (vp^2 - V^2) for constant vp."""
    t = np.asarray(t, dtype=float)
    if vp == 0.0:
        return v0 / (1.0 + a * v0 * t)
    r = v0 / vp
    if r < 1.0:
        return vp * np.tanh(np.arctanh(r) + a * vp * t)
    if r > 1.0:
        return vp / np.tanh(np.arctanh(1.0 / r) + a * vp * t)
    return np.full_like(t, vp)


def integrate_const_vp(v0, vp, a, t_end, n, method="rk4"):
    """Fixed-step integration of dV/dt = a (vp^2 - V^2); ``method`` in {"euler", "rk4"}. Returns V(t_end)."""
    h = t_end / n
    f = lambda v: a * (vp * vp - v * v)
    v = float(v0)
    for _ in range(n):
        if method == "euler":
            v += h * f(v)
        else:
            k1 = f(v); k2 = f(v + h / 2 * k1); k3 = f(v + h / 2 * k2); k4 = f(v + h * k3)
            v += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return v


RI_MS_PER_DAY = 15.4   # NHC rapid-intensification threshold: +30 kt in 24 h (= 15.4 m/s per day)


def events(r: dict, q: PhysParams = PhysParams(), mp: Params = Params(), v_genesis: float = 17.5) -> dict:
    """Key instants (days) of one run (scalar parameters): warm-zone entry/exit, genesis, peak, rapid intensification.

    * entry/exit: the storm crosses the warm zone ``[z0, z1]`` of the track (same definition as the animation);
    * genesis: first time V >= ``v_genesis`` (17.5 m/s, tropical-storm threshold of the Saffir-Simpson scale);
    * RI: after genesis, window where dV/dt >= 15.4 m/s per day (NHC definition, +30 kt / 24 h).
    """
    day = np.asarray(r["t"]) / 86400.0
    v = np.asarray(r["v"])
    dur = float(day[-1])
    span = float(mp.x1) - float(mp.x0)
    t_in, t_out = (float(mp.z0) - float(mp.x0)) / span * dur, (float(mp.z1) - float(mp.x0)) / span * dur
    dvdt = np.gradient(v, day)
    gen = v >= v_genesis
    t_gen = float(day[gen][0]) if gen.any() else None
    after = day >= (t_gen if t_gen is not None else np.inf)
    ri = (dvdt >= RI_MS_PER_DAY) & after
    first = lambda m: float(day[m][0]) if m.any() else None
    last = lambda m: float(day[m][-1]) if m.any() else None
    return {"t_enter": t_in, "t_exit": t_out, "t_genesis": t_gen, "t_peak": float(day[np.argmax(v)]),
            "v_peak": float(v.max()), "t_ri_start": first(ri), "t_ri_end": last(ri), "dvdt": dvdt,
            "ri_max": float(dvdt[after].max()) if after.any() else 0.0}
