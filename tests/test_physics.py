"""Verification (does the code solve the equations?) and validation against literature bounds for the energy-balance model."""
import numpy as np
from oceantank import Timeline, physics as ph
from oceantank.model import Params


def test_saturation_vapour_pressure_tables():
    assert abs(ph.saturation_vapor_pressure(293.15) - 23.37) / 23.37 < 0.01      # 20 deg C
    assert abs(ph.saturation_vapor_pressure(303.15) - 42.43) / 42.43 < 0.01      # 30 deg C


def test_clausius_clapeyron_rate():
    e1, e2 = ph.saturation_vapor_pressure(297.65), ph.saturation_vapor_pressure(298.65)
    assert 0.058 < np.log(e2 / e1) < 0.065                                        # about 6 % per K at 25 deg C (L / Rv T^2 = 6.1 %)


def test_potential_intensity_bounds():
    assert 60.0 < float(ph.potential_intensity(303.15)) < 110.0                  # Vp(30 C), literature range
    assert float(ph.potential_intensity(303.15)) > float(ph.potential_intensity(299.15))
    assert float(ph.potential_intensity(150.0)) == 0.0                           # negative energy balance


def _order(method, ns):
    v0, vp, a, T = 12.0, 60.0, 1.875e-7, 2.0 * 86400
    exact = float(ph.analytic_const_vp(T, v0, vp, a))
    err = [abs(ph.integrate_const_vp(v0, vp, a, T, n, method) - exact) for n in ns]
    return -np.polyfit(np.log(ns), np.log(err), 1)[0], err


def test_rk4_is_fourth_order_and_euler_first_order():
    p4, e4 = _order("rk4", [4, 8, 16, 32])
    p1, e1 = _order("euler", [100, 200, 400, 800])
    assert 3.6 < p4 < 4.5 and e4[-1] < 1e-4                                 # measured ~4.3 (pre-asymptotic)
    assert 0.9 < p1 < 1.1


def test_run_matches_analytic_solution_for_constant_sst():
    mp = Params(sst_base=29.0, warm_amp=0.0, cold_amp=0.0, wake_cool=0.0)
    q = ph.PhysParams(cooling=False, use_viability=True)
    r = ph.run(q, mp, n_steps=800)
    vp = float(ph.vp_effective(29.0 + ph.KELVIN, q))
    exact = ph.analytic_const_vp(r["t"], q.v0, vp, q.a)
    assert np.max(np.abs(r["v"] - exact)) / vp < 1e-6


def test_trends_and_cooling_feedback():
    peak = lambda **kw: ph.run(ph.PhysParams(**kw.pop("q", {})), Params(**kw)).get("v").max()
    assert peak(warm_amp=1.0) < peak(warm_amp=3.0) < peak(warm_amp=5.0)
    assert peak(q=dict(cooling=True)) < peak(q=dict(cooling=False))              # the cold wake weakens the storm


def test_ensemble_equals_scalar():
    amp = np.array([1.0, 2.5, 4.0])
    ens = ph.run(ph.PhysParams(), Params(warm_amp=amp), n_steps=200)["v"]
    assert ens.shape == (3, 201)
    for i in range(3):
        one = ph.run(ph.PhysParams(), Params(warm_amp=float(amp[i])), n_steps=200)["v"]
        assert np.allclose(ens[i], one, rtol=0, atol=1e-9)


def test_physical_table_drives_the_animation():
    tab = ph.intensity_table()
    assert tab.shape == (1601,) and 0.1 < tab.max() < 1.5
    tl = Timeline(start_ready=False, itab=tab)
    tl.run_to(20.0)
    assert tl.state.uI > 0.5


def test_events_are_consistent():
    r = ph.run()
    ev = ph.events(r)
    assert 0.0 < ev["t_enter"] < ev["t_peak"] < ev["t_exit"] < ph.PhysParams().duration_s / 86400
    assert ev["t_genesis"] is not None and ev["t_genesis"] < ev["t_peak"]
    assert abs(ev["v_peak"] - r["v"].max()) < 1e-12
    cold = ph.events(ph.run(ph.PhysParams(), Params(sst_base=24.0, warm_amp=0.0)), mp=Params(sst_base=24.0, warm_amp=0.0))
    assert cold["v_peak"] < 30 and cold["t_ri_start"] is None
