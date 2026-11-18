import numpy as np
import pytest
from oceantank import hpc
from oceantank.model import Params, intensity_stats


def test_grid_shape_and_order():
    flat, shape = hpc.grid(a=[1, 2, 3], b=[10, 20])
    assert shape == (3, 2) and flat["a"].tolist() == [1, 1, 2, 2, 3, 3] and flat["b"].tolist() == [10, 20] * 3


def test_sweep_matches_scalar_runs():
    ax = {"warm_amp": [0.0, 2.0, 4.0], "lam_grow": [1.0, 2.0]}
    r = hpc.sweep(ax, ni=200)
    assert r["I_peak"].shape == (3, 2)
    for i, w in enumerate(ax["warm_amp"]):
        for j, l in enumerate(ax["lam_grow"]):
            assert abs(r["I_peak"][i, j] - float(intensity_stats(Params(warm_amp=w, lam_grow=l), 200)["I_peak"])) < 1e-12


def test_parallel_sweep_is_identical_to_serial():
    ax = {"warm_amp": np.linspace(0, 5, 64), "cold_amp": np.linspace(0, 4, 4)}
    a, b = hpc.sweep(ax, ni=150, workers=1), hpc.sweep(ax, ni=150, workers=2)
    for k in ("I_peak", "u_peak", "I_end"):
        assert np.array_equal(a[k], b[k])


def test_euler_convergence_is_first_order():
    c = hpc.convergence(levels=(100, 200, 400, 800))
    assert 0.7 < c["order"] < 1.3
    assert c["error"][-1] < c["error"][0]


def test_rk4_convergence_in_physics():
    # The coupled right-hand side is non-smooth (sqrt(max(.,0)), min(), logistic), so the observed order on the whole wind curve
    # is ~2.6, below the formal order 4 that test_physics measures on the smooth constant-SST problem.
    c = hpc.convergence_physics(levels=(50, 100, 200, 400))
    assert c["order"] > 2.0 and c["error"][-1] < c["error"][0] / 50


def test_scaling_report_is_consistent():
    s = hpc.scaling(n_members=2000, workers=(1,))
    assert s["speedup"][0] == 1.0 and s["vectorisation_gain"] > 5


def test_parallel_render_is_pixel_identical():
    from oceantank import Viewer
    v = Viewer(size=(160, 90))
    t = [3.0, 11.0, 19.0, 27.0]
    ser, par = v.render_frames(t, workers=1), v.render_frames(t, workers=2)
    for a, b in zip(ser, par):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_physics_sweep_matches_scalar_and_parallel_is_identical():
    from oceantank import physics as ph
    ax = {"warm_amp": [1.0, 3.0, 5.0], "speed_ms": [3.0, 6.0, 12.0]}
    a = hpc.sweep_physics(ax, n_steps=150)
    for i, w in enumerate(ax["warm_amp"]):
        for j, s in enumerate(ax["speed_ms"]):
            one = ph.run(ph.PhysParams(speed_ms=s), Params(warm_amp=w), 150)["v"].max()
            assert abs(a["v_peak"][i, j] - one) < 1e-9
    b = hpc.sweep_physics(ax, n_steps=150, workers=2, chunk=3)
    assert np.array_equal(a["v_peak"], b["v_peak"])
    # slow storms get a colder wake but also more time to spin up: the net effect is not monotonic (see docs/MODELS.md)
    assert a["dT_max"][2, 0] > a["dT_max"][2, 2]
