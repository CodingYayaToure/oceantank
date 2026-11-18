"""The vectorised model must reproduce Timeline exactly (default parameters) and behave sensibly on ensembles."""
import numpy as np
import pytest
from oceantank import Timeline
from oceantank.model import Params, intensity_table, cycle_signals, sst, cycle_events, wind_profile, PERIOD


def test_table_matches_timeline_exactly():
    tl = Timeline(start_ready=False)
    tab = intensity_table(Params()).astype(np.float32)
    assert tab.shape == tl.itab.shape
    assert np.array_equal(tab, tl.itab)


def test_sst_matches_timeline_sst_at():
    x = np.linspace(-11.2, 11.2, 57)
    assert np.allclose(sst(x), [Timeline.sst_at(v) for v in x], atol=1e-12)


@pytest.mark.parametrize("t", [0.5, 6.0, 12.6, 14.0, 20.0, 27.5, 31.0])
def test_cycle_signals_match_timeline_state(t):
    tl = Timeline(start_ready=False)
    tl.run_to(t)
    tl.advance(0)
    s = cycle_signals(Params(), tl.itab.astype(float), t)
    st = tl.state
    assert abs(float(s["cx"]) - st.uC[0]) < 1e-9 and abs(float(s["cz"]) - st.uC[1]) < 1e-9
    assert abs(float(s["I"]) - st.uI) < 1e-6
    assert abs(float(s["R"]) - st.uR) < 1e-6
    assert abs(float(s["burst"]) - st.uBurst) < 1e-6
    assert abs(float(s["dis"]) - st.uDis) < 1e-9


def test_ensemble_equals_scalar_loop():
    rng = np.random.default_rng(1)
    amp, lg = rng.uniform(0, 5, 6), rng.uniform(0.8, 3, 6)
    ens = intensity_table(Params(warm_amp=amp, lam_grow=lg), ni=400)
    assert ens.shape == (6, 401)
    for i in range(6):
        one = intensity_table(Params(warm_amp=float(amp[i]), lam_grow=float(lg[i])), ni=400)
        assert np.allclose(ens[i], one, rtol=0, atol=1e-13)      # not bitwise: array pow may differ by 1 ulp


def test_physical_trends():
    base = Params()
    peak = lambda **kw: intensity_table(base.replace(**kw)).max()
    assert peak(warm_amp=1.0) < peak(warm_amp=3.0) < peak(warm_amp=5.0)          # warmer water, stronger storm
    assert peak(sst_base=24.0, warm_amp=0.0) < 0.05                                # no warm zone, no intensification
    assert peak(lam_grow=3.0) < peak(lam_grow=1.0)                                 # slower growth, lower peak


def test_custom_zone_drives_timeline():
    tl = Timeline(start_ready=False, zone=(-6.0, 2.0))
    assert tl.state.uZ == (-6.0, 2.0)
    tab = intensity_table(Params(z0=-6.0, z1=2.0)).astype(np.float32)
    assert np.array_equal(tl.itab, tab)
    ev = cycle_events(Params(z0=-6.0, z1=2.0), tab.astype(float))
    assert ev["t_enter"] < ev["t_peak"] < PERIOD


def test_wind_profile_peaks_at_core_radius():
    r = np.linspace(0.01, 8, 800)
    v = wind_profile(r, 1.0)
    assert abs(r[np.argmax(v)] - 1.85) < 0.02 and abs(v.max() - 6.0) < 0.05


def test_streaming_stats_equal_table():
    from oceantank.model import intensity_stats
    rng = np.random.default_rng(3)
    p = Params(warm_amp=rng.uniform(0, 5, 8), gain=rng.uniform(1, 2, 8))
    tab, st = intensity_table(p, ni=300), intensity_stats(p, ni=300)
    assert np.allclose(st["I_peak"], tab.max(axis=-1) * 1.18 * p.gain, atol=1e-13)
    assert np.array_equal(st["u_peak"], tab.argmax(axis=-1) / 300)
