import numpy as np
import pytest

pytest.importorskip("matplotlib")
from oceantank import dashboard as D


def test_compute_is_consistent_between_models():
    st = D.compute({})
    assert st["v_phys"].shape == st["v_repo"].shape == st["days"].shape == (1601,)
    assert 40 < st["ev"]["v_peak"] < 90 and 0 < st["ev"]["t_enter"] < st["ev"]["t_exit"] < st["duration_days"]
    assert st["sens"]["repo"].shape == st["sens"]["phys"].shape == (41,)


def test_cursor_quantities_and_scenarios():
    st = D.compute({})
    c0, c1 = D.at(st, 0.0), D.at(st, 1.0)
    assert c0["i"] == 0 and c1["i"] == 1600 and c0["cx"] < c1["cx"]
    cold = D.compute(D.PRESETS["Cold basin (no warm zone)"])
    assert cold["ev"]["v_peak"] < st["ev"]["v_peak"] / 2
    assert D.compute({"cooling": False})["ev"]["v_peak"] > st["ev"]["v_peak"]


def test_panels_render_png_and_move_cursor():
    st = D.compute({})
    pn = D.Panels(figsize=(8, 4.5), dpi=60)
    pn.draw(st, 0.3)
    a = pn.png()
    pn.move(st, 0.7)
    assert a[:8] == b"\x89PNG\r\n\x1a\n" and pn.png() != a


def test_dashboard_widgets_drive_the_viewer():
    pytest.importorskip("ipywidgets")
    d = D.Dashboard(size=(160, 90))
    first = d.img.value
    d.set(drive="physics", warm_amp=5.0)
    assert d.img.value != first and d.flat["warm_amp"] == 5.0
    tm = d.timeline_model()
    assert tm["itab"].shape == (1601,) and tm["itab"].dtype == np.float32
