import os, subprocess, sys, json
from pathlib import Path
import numpy as np
import pytest
from oceantank import Tank, Camera, Timeline, Scene, Mulberry32, over
from oceantank import geometry as geo, shaders

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- geometry
def test_plane_matches_threejs_layout():
    pos, idx = geo.plane_xz(30, 8, 660, 200)
    assert pos.shape == (661 * 201, 3) and idx.shape == (660 * 200 * 6,)
    assert pos[:, 0].min() == -15 and pos[:, 0].max() == 15 and pos[:, 2].min() == -4 and pos[:, 2].max() == 4
    a, b, c = (pos[idx[:3]][k] for k in range(3))
    assert np.cross(b - a, c - a)[1] > 0                      # front face points towards +Y


def test_box_faces_point_outward():
    v, i = geo.box(2, 4, 6)
    assert len(i) == 36
    for t in i.reshape(-1, 3):
        a, b, c = v[t]
        n = np.cross(b - a, c - a)
        centre = (a + b + c) / 3
        assert np.dot(n, centre) > 0


def test_wall_attributes():
    pos, au, idx = geo.wall(-15, -4, 15, -4, 660)
    assert pos.shape == (2 * 661, 3) and au.max() == pytest.approx(30, abs=1e-4) and len(idx) == 660 * 6
    assert set(np.unique(pos[:, 1])) == {0.0, 1.0}


# ---------------------------------------------------------------- camera
def test_camera_targets_center_and_distance():
    cam = Camera()
    asp = 16 / 9
    v = cam.view(asp)
    tgt = v @ np.array([0, 1.25, 0, 1])
    assert abs(tgt[0]) < 1e-12 and abs(tgt[1]) < 1e-12 and tgt[2] < 0
    p = cam.projection(asp) @ (v @ np.array([0, 1.25, 0, 1]))
    assert abs(p[0] / p[3]) < 1e-9 and abs(p[1] / p[3]) < 1e-9       # the target projects to the centre of the screen


# ---------------------------------------------------------------- timeline
def test_mulberry32_matches_javascript():
    r = Mulberry32(12345)
    ref = [0.9797282677609473, 0.3067522644996643, 0.484205421525985, 0.817934412509203, 0.5094283693470061]
    assert [r.random() for _ in range(5)] == pytest.approx(ref, abs=1e-15)
    for _ in range(20000):                  # Node: 5 draws, then 20000, then the next one
        r.random()
    assert r.random() == pytest.approx(0.5074967634864151, abs=1e-15)


def test_timeline_deterministic_and_start_state():
    a, b = Timeline(seed=7), Timeline(seed=7)
    assert a.t == pytest.approx(12.6) and a.clock == pytest.approx(3.0)
    for _ in range(10):
        a.advance(1 / 30); b.advance(1 / 30)
    assert np.array_equal(a.head_pos, b.head_pos) and a.state == b.state
    assert 0.0 <= a.state.uBurst <= 1.0 and a.state.uR > 1.0


def test_cyclone_crosses_the_tank_over_one_cycle():
    tl = Timeline(start_ready=False)
    tl.run_to(0.5); x0 = tl.state.uC[0]
    tl.run_to(16.0); xm = tl.state.uC[0]
    tl.run_to(31.0); x1 = tl.state.uC[0]
    assert x0 < -10 and abs(xm) < 1.0 and x1 > 10         # from -11 to +11: it moves along the whole length


def test_intensity_table_shape():
    tl = Timeline(start_ready=False)
    assert len(tl.itab) == 1601 and tl.itab.max() > 0.5 and tl.itab.min() >= 0.0


# ---------------------------------------------------------------- shaders
def test_all_shaders_present_and_hash_swap():
    for n in shaders.NAMES:
        assert shaders.raw(n, "vert") and shaders.raw(n, "frag")
    native = shaders.parametrize(shaders.raw("surface", "frag"), Tank(), "native")
    port = shaders.parametrize(shaders.raw("surface", "frag"), Tank(), "portable")
    assert shaders.HASH_NATIVE_1 in native and shaders.HASH_NATIVE_1 not in port
    assert shaders.raw("surface", "frag") == native                        # default tank: original shader unchanged


def test_tank_parametrization_changes_constants():
    t = Tank(length=20, width=6, depth=2.0)
    s = shaders.parametrize(shaders.raw("ground", "frag"), t)
    assert "vec2(10.,3.)" in s and "step(3.,p.y)" in s
    assert "const float D=2.; const float HW=3.;" in shaders.parametrize(shaders.raw("surface", "vert"), t)


# ---------------------------------------------------------------- rendu GL
SIZE = (480, 270)


def test_render_layers_and_determinism():
    tl = Timeline()
    glass = Scene().build_glass()
    g = glass.render(tl, Camera(), SIZE)
    assert g.shape == (SIZE[1], SIZE[0], 4)
    assert g[0, 0, 3] == 0                                                # outside the tank: transparent
    assert (g[..., 3] > 30).mean() < 0.05                                 # only the edges are clearly visible
    assert 0.15 < (g[..., 3] > 0).mean() < 0.6                            # the glass panes tint the silhouette (alpha ~ .045)
    full = Scene().build_glass().add_ocean()
    a = full.render(tl, Camera(), SIZE)
    b = full.render(tl, Camera(), SIZE)
    assert np.array_equal(a, b)                                           # deterministic rendering
    assert (a[..., 3] == 255).mean() > 0.15                               # the water is opaque
    assert a[..., :3].mean() > g[..., :3].mean()


def test_native_vs_portable_hash_differ_but_same_layout():
    tl = Timeline()
    n = Scene().build_glass().add_ocean().render(tl, Camera(), SIZE)
    p = Scene(hash_mode="portable").build_glass().add_ocean().render(tl, Camera(), SIZE)
    assert (n[..., 3] > 0).tolist() == (p[..., 3] > 0).tolist() or abs(((n[..., 3] > 0) != (p[..., 3] > 0)).mean()) < 0.01
    assert not np.array_equal(n, p)


def test_compose_on_page_background():
    tl = Timeline()
    img = Scene().build_glass().render(tl, Camera(), SIZE)
    rgb = over(img)
    assert rgb.dtype == np.uint8 and rgb.shape == (SIZE[1], SIZE[0], 3)
    assert rgb[int(.34 * SIZE[1]), SIZE[0] // 2].sum() > rgb[2, 2].sum()   # gradient centre is lighter than the corners


# ---------------------------------------------------------------- comparison with the HTML animation (optional)
HTML, THREE = os.environ.get("OCEANTANK_HTML"), os.environ.get("OCEANTANK_THREE")


@pytest.mark.skipif(not (HTML and THREE), reason="set OCEANTANK_HTML and OCEANTANK_THREE (+ playwright) to compare with the browser")
def test_matches_html_canvas(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "tools/compare_with_html.py"), HTML, THREE, str(tmp_path), "640", "360",
                        "0", "--portable", "--particles=none"], capture_output=True, text=True, timeout=600)
    m = json.loads(r.stdout.strip().splitlines()[-1])
    assert m["mae"] < 1.0 and m["psnr_db"] > 35


def test_viewer_notebook_api(tmp_path):
    from oceantank import Viewer
    v = Viewer(size=(320, 180))
    a, b = v.frame(at=3), v.frame(at=20)
    assert a.size == (320, 180) and not np.array_equal(np.asarray(a), np.asarray(b))      # the cyclone has moved
    g = v.gif(str(tmp_path / "c.gif"), duration=2, fps=4, progress=False)
    from PIL import Image
    assert Image.open(g).n_frames == 8
    glass = Viewer(size=(320, 180), ocean=False).frame()
    assert glass.size == (320, 180)
