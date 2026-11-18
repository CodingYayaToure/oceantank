"""Generate examples/oceantank_dashboard.ipynb (requires nbformat)."""
import nbformat as nbf

C, M = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell
nb = nbf.v4.new_notebook()
nb.cells = [
M("# oceantank dashboard: a cyclone crossing a warm zone\n"
  "The 3-D animation and the scientific plots move together. Two storm models are driven by the **same SST track** and compared:\n\n"
  "* **repo law**: the animation's phenomenological SST → intensity relaxation (`oceantank.model`);\n"
  "* **energy balance**: a reduced Emanuel-type model with sea-surface cooling, RK4 (`oceantank.physics`).\n\n"
  "Sections: **1** interactive dashboard · **2** verification (convergence) · **3** parallel parameter sweeps · "
  "**4** strong scaling · **5** parallel rendering.\n\n"
  "No GPU needed. Run the cells from top to bottom; the install cell takes about 1 minute on Colab."),
M("## 0. Install (once per session)"),
C('''import importlib.util, os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")      # Mesa software rendering: works without a display or GPU

if IN_COLAB:                                             # OpenGL/EGL libraries of the Colab machine
    subprocess.run("apt-get -qq update && apt-get -qq install -y libegl1 libegl-mesa0 libgl1 libgles2 libgl1-mesa-dri",
                   shell=True, check=False)

if importlib.util.find_spec("oceantank") is None or importlib.util.find_spec("oceantank.dashboard") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                           "git+https://github.com/CodingYayaToure/oceantank.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "moderngl", "ipywidgets", "matplotlib"])
print("oceantank ready, CPUs:", os.cpu_count())'''),
M("## 1. Interactive dashboard\n"
  "* **Scenario** loads a preset (cold basin, hot anomaly, fast / slow mover, long or short warm zone, no cooling).\n"
  "* **Animation driven by** chooses which model drives the 3-D storm (intensity, warm-zone tint, gain).\n"
  "* The **time slider / play** moves the 3-D frame and the cursor of the six panels together.\n"
  "* Moving a parameter recomputes both models (vectorised, tens of ms) and redraws everything; the status line gives "
  "zone entry / exit, genesis, peak and the maximum intensification rate (RI = +15.4 m/s per day, NHC definition).\n\n"
  "Reading the panels: **A** SST seen by the storm (the cold wake lowers it) · **B** wind of both models and the potential "
  "intensity Vp · **C** dV/dt with the RI threshold · **D** radial wind profile · **E** ensemble: peak wind vs warm-zone anomaly · "
  "**F** wind vs SST (phase plane)."),
C('''from oceantank.dashboard import Dashboard

dash = Dashboard(size=(480, 270))
dash.show();'''),
M("## 2. Verification: does the code solve its equations?\n"
  "Observed order of accuracy: error against a much finer run. Euler (repo law) must give order 1; RK4 on the smooth constant-SST problem "
  "gives 4 (see `tests/test_physics.py`); on the coupled problem, whose right-hand side is not smooth (`max`, `min`, logistic), "
  "the observed order is lower."),
C('''import numpy as np, matplotlib.pyplot as plt
from oceantank import hpc

ce, cr = hpc.convergence(), hpc.convergence_physics()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for a, c, name, col, k in ((ax[0], ce, "Euler (repo law), peak intensity", "#2a78d6", 1), (ax[1], cr, "RK4 (energy balance), whole V(t) curve", "#eb6834", 4)):
    a.loglog(c["h"], c["error"], "o-", color=col, lw=2)
    ref = c["error"][-1] * (c["h"] / c["h"][-1]) ** k
    a.loglog(c["h"], ref, "--", color="#52514e", lw=1, label=f"formal order {k}")
    a.set_title(f"{name}\\nobserved order = {c['order']:.2f}", fontsize=10, loc="left")
    a.set_xlabel("step h (fraction of the track)"); a.set_ylabel("error"); a.grid(True, color="#e6e5e1"); a.legend(frameon=False)
plt.show()'''),
M("## 3. Parallel parameter sweeps\n"
  "Each ensemble member is an independent run: the track loop is sequential, the arithmetic is vectorised over members (NumPy) and chunks of "
  "members are spread over processes (`multiprocessing`, `spawn`). The result is identical for any number of workers."),
C('''import time
workers = os.cpu_count() or 1
amp = np.linspace(0.0, 6.0, 80)

r1 = hpc.sweep({"warm_amp": amp, "lam_grow": np.linspace(0.6, 3.5, 80)}, workers=workers)          # repo law: 6,400 members
r2 = hpc.sweep_physics({"warm_amp": amp, "speed_ms": np.linspace(2.0, 14.0, 80)}, workers=workers)   # energy balance: 6,400 members
print(f"repo law: {r1['size']} members in {r1['seconds']:.2f} s; energy balance: {r2['size']} members in {r2['seconds']:.2f} s ({workers} workers)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
m1 = ax[0].pcolormesh(r1["axes"]["lam_grow"], amp, r1["I_peak"], cmap="Blues", shading="auto")
ax[0].set(title="Repo law: peak intensity I", xlabel="growth length λ", ylabel="warm-zone anomaly (K)"); fig.colorbar(m1, ax=ax[0])
m2 = ax[1].pcolormesh(r2["axes"]["speed_ms"], amp, r2["v_peak"], cmap="Oranges", shading="auto")
ax[1].set(title="Energy balance: peak wind (m/s)", xlabel="translation speed (m/s)", ylabel="warm-zone anomaly (K)"); fig.colorbar(m2, ax=ax[1])
plt.show()'''),
M("Reading the maps: the repo law has no ceiling (its peak keeps growing with the anomaly), whereas the energy balance saturates near the potential intensity Vp. "
  "On the right map, a slower storm cools the water more (colder wake) **but** also has more time to spin up. The net effect on peak wind "
  "is not monotonic: it depends on the warm-zone anomaly. This is a property of the reduced model, to be validated on observations before any "
  "operational claim."),
M("## 4. Strong scaling\n"
  "Same problem, increasing number of processes. `vectorisation_gain` compares with a Python loop over members (extrapolated from a 40-member sample)."),
C('''s = hpc.scaling(n_members=200000, workers=tuple(sorted({1, workers})))
print(f"CPUs: {s['cpu_count']} | vectorisation gain vs Python loop: x{s['vectorisation_gain']:.0f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
ax[0].plot(s["workers"], s["speedup"], "o-", color="#2a78d6", lw=2, label="measured")
ax[0].plot(s["workers"], s["workers"] / s["workers"][0], "--", color="#52514e", label="ideal")
ax[0].set(title="Speed-up", xlabel="processes", ylabel="T(1) / T(n)"); ax[0].legend(frameon=False); ax[0].grid(True, color="#e6e5e1")
ax[1].plot(s["workers"], s["efficiency"], "o-", color="#eb6834", lw=2); ax[1].set_ylim(0, 1.1)
ax[1].set(title="Parallel efficiency", xlabel="processes", ylabel="speed-up / n"); ax[1].grid(True, color="#e6e5e1")
plt.show()
print({k: s[k] for k in ("workers", "seconds", "speedup", "efficiency", "amdahl_serial_fraction")})'''),
M("## 5. Parallel rendering of the animation\n"
  "Every frame is a pure function of its time (seeded timeline), so frames can be rendered by independent processes, each with its own "
  "single-threaded software-GL context, and the result is **pixel-identical** to the serial one (tested). On a 2-vCPU machine the gain is small "
  "because Mesa llvmpipe already uses both cores for one frame; the benefit grows with the number of cores."),
C('''from oceantank import Viewer
from IPython.display import Image as IPImage

viewer = Viewer(size=(480, 270))
viewer.set_model(**dash.timeline_model())            # the scenario currently selected in the dashboard
t0 = time.perf_counter()
path = viewer.gif("cyclone_parallel.gif", fps=4, workers=workers, progress=False)
print(f"{workers} worker(s): {time.perf_counter() - t0:.1f} s")
IPImage(filename=path)'''),
M("## Limitations\n"
  "* The animation's law is **phenomenological** (tank units); the energy balance is **reduced** (no vertical structure, convection, shear or dry air); "
  "`η`, `Ck/Cd` and the viability threshold are calibrated, not measured.\n"
  "* The comparison in panels B/E uses a *visual* scale V = v_norm · I to put both models in m/s.\n"
  "* Neither model is a weather forecast. Real validation needs observations (best-track data, SST products)."),
]
nbf.write(nb, "examples/oceantank_dashboard.ipynb")
print("written", len(nb.cells), "cells")
