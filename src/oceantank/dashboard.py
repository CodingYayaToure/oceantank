"""Interactive dashboard for Jupyter / Colab: the OpenGL animation and the scientific plots move together.

    from oceantank.dashboard import Dashboard
    Dashboard().show()

Two storm models are driven by the same SST track and compared side by side:

* **repo law**: the phenomenological SST -> intensity relaxation of the animation (``oceantank.model``);
* **physical model**: a reduced Emanuel-type energy balance with sea-surface cooling (``oceantank.physics``).

Either one can drive the 3-D animation. Moving a parameter recomputes both models (vectorised, a few ms), re-renders the
frame and redraws the six panels; moving the time slider (or pressing play) moves the cursor in every panel and the
camera frame in lock-step.
"""
from __future__ import annotations
import io
import os
from dataclasses import fields
import numpy as np
from . import model, physics as ph
from .model import Params, PERIOD, R0, ss
from .physics import PhysParams

# palette (validated reference palette: categorical slots 1 and 2, neutral inks)
C_REPO, C_PHYS = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE, ZONE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#f2e6d0"
SAFFIR = ((17.5, "TS"), (33.0, "C1"), (43.0, "C2"), (50.0, "C3"), (58.0, "C4"), (70.0, "C5"))

P_KEYS = {f.name for f in fields(Params)}
Q_KEYS = {f.name for f in fields(PhysParams)}

# key, label, min, max, step, default, group
SLIDERS = [
    ("sst_base", "Background SST (°C)", 24.0, 31.0, 0.1, 27.6, "Ocean"),
    ("warm_amp", "Warm-zone anomaly (K)", 0.0, 6.0, 0.1, 3.0, "Ocean"),
    ("cold_amp", "Cold-water anomaly (K)", 0.0, 6.0, 0.1, 3.2, "Ocean"),
    ("wake_cool", "Cold wake in zone (K)", 0.0, 3.0, 0.1, 1.1, "Ocean"),
    ("z0", "Warm zone start (x)", -8.0, 4.0, 0.1, -3.0, "Ocean"),
    ("z1", "Warm zone end (x)", 0.0, 9.0, 0.1, 5.5, "Ocean"),
    ("lam_grow", "Growth length λ (repo law)", 0.5, 4.0, 0.1, 1.5, "Repo law"),
    ("gain", "Visual gain", 0.8, 2.0, 0.05, 1.35, "Repo law"),
    ("eta", "Spin-up efficiency η", 0.05, 0.6, 0.01, 0.25, "Physics"),
    ("ck_cd", "Ck / Cd", 0.2, 1.2, 0.05, 0.5, "Physics"),
    ("speed_ms", "Translation speed (m/s)", 2.0, 15.0, 0.5, 6.0, "Physics"),
    ("cool_k", "Cooling at V = 60 m/s (K)", 0.0, 4.0, 0.1, 1.5, "Physics"),
    ("v_norm", "Visual scale V → I (m/s)", 40.0, 100.0, 1.0, 70.0, "Physics"),
]
DEFAULTS = {k: d for k, _, _, _, _, d, _ in SLIDERS}
DEFAULTS.update(cooling=True, use_viability=True, drive="Repo law")

PRESETS = {
    "Default crossing": {},
    "Cold basin (no warm zone)": {"warm_amp": 0.0, "sst_base": 25.5},
    "Hot anomaly (+5 K)": {"warm_amp": 5.0},
    "Fast mover (12 m/s): weak wake": {"speed_ms": 12.0},
    "Slow mover (3 m/s): strong cold wake": {"speed_ms": 3.0},
    "Long warm zone": {"z0": -6.0, "z1": 7.0},
    "Late, short warm zone": {"z0": 1.0, "z1": 4.0},
    "No cooling feedback": {"cooling": False},
}


def split(flat: dict) -> tuple[Params, PhysParams, float]:
    p = Params(**{k: float(v) for k, v in flat.items() if k in P_KEYS})
    q = PhysParams(**{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in flat.items() if k in Q_KEYS})
    return p, q, float(flat.get("v_norm", 70.0))


def radius(In, cx, p: Params):
    """Radius of maximum wind of the animation (tank units), as in ``model.cycle_signals``."""
    return R0 * (1.1 - 0.28 * np.minimum(np.asarray(In) / 0.95, 1.1) + 0.45 * ss(p.z1 - 1.5, p.z1 + 6.0, cx))


def compute(flat: dict) -> dict:
    """Both models along the track for the parameters ``flat`` (a few ms, vectorised)."""
    p, q, v_norm = split({**DEFAULTS, **flat})
    tab = model.intensity_table(p)
    run = ph.run(q, p)
    days = run["t"] / 86400.0
    u = run["u"]
    v_repo = model.inorm(tab, u) * v_norm
    v_phys = run["v"]
    ev = ph.events(run, q, p)
    x = p.x0 + (p.x1 - p.x0) * u
    return dict(p=p, q=q, v_norm=v_norm, tab=tab, run=run, days=days, u=u, x=x, v_repo=v_repo, v_phys=v_phys, ev=ev,
                dvdt_repo=np.gradient(v_repo, days), dvdt_phys=ev["dvdt"], sst_track=model.sst(x, p),
                duration_days=float(days[-1]), sens=sensitivity(p, q, v_norm))


def sensitivity(p: Params, q: PhysParams, v_norm: float, n: int = 41, lo: float = 0.0, hi: float = 6.0) -> dict:
    """Peak wind as a function of the warm-zone anomaly: one vectorised ensemble per model (n members each)."""
    amp = np.linspace(lo, hi, n)
    pe = p.replace(warm_amp=amp)
    st = model.intensity_stats(pe)
    return {"warm_amp": amp, "repo": st["I_peak"] / (1.18 * p.gain) * v_norm, "phys": ph.run(q, pe, 400)["v"].max(axis=-1)}


def at(state: dict, u: float) -> dict:
    """Instantaneous quantities at track fraction ``u`` (the cursor)."""
    i = int(round(np.clip(u, 0.0, 1.0) * (len(state["u"]) - 1)))
    p = state["p"]
    In_repo = float(model.inorm(state["tab"], u))
    In_phys = state["v_phys"][i] / state["v_norm"]
    return dict(i=i, day=float(state["days"][i]), cx=float(state["x"][i]), sst=float(state["sst_track"][i]),
                sst_seen=float(state["run"]["sst_seen_k"][i] - 273.15), v_repo=float(state["v_repo"][i]),
                v_phys=float(state["v_phys"][i]), vp=float(state["run"]["vp"][i]), dT=float(state["run"]["dT"][i]),
                R_repo=float(radius(In_repo, state["x"][i], p)), R_phys=float(radius(In_phys, state["x"][i], p)))


class Panels:
    """The six panels. ``draw`` builds the static part once per parameter change, ``move`` only the time cursor."""

    def __init__(self, figsize=(13.0, 7.2), dpi=100):
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor=SURFACE, layout="constrained")
        gs = self.fig.add_gridspec(2, 3)
        self.ax = {k: self.fig.add_subplot(gs[i // 3, i % 3]) for i, k in enumerate("ABCDEF")}
        self.cur = []

    @staticmethod
    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=INK2, length=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    def draw(self, st: dict, u: float):
        for a in self.ax.values():
            a.clear()
        self.cur = []                    # cleared with the axes
        d, ev, A = st["days"], st["ev"], self.ax
        zone = (ev["t_enter"], ev["t_exit"])
        for k in "ABC":
            A[k].axvspan(*zone, color=ZONE, lw=0, zorder=0)
        # A: SST
        self._style(A["A"], "A  Sea-surface temperature under the track", "time (days)", "°C")
        A["A"].plot(d, st["sst_track"], color=INK2, lw=2, label="background SST")
        A["A"].plot(d, st["run"]["sst_seen_k"] - 273.15, color=C_PHYS, lw=2, label="SST seen by the storm (cold wake)")
        A["A"].axhline(st["q"].viab_threshold_c, color=INK2, lw=1, ls=(0, (4, 3)))
        A["A"].text(d[-1], st["q"].viab_threshold_c, " viability\n threshold", fontsize=7, color=INK2, va="center")
        A["A"].legend(fontsize=7.5, frameon=False, loc="lower left")
        A["A"].text(np.mean(zone), A["A"].get_ylim()[1], "warm zone", ha="center", va="top", fontsize=8, color=INK2)
        # B: wind
        self._style(A["B"], "B  Maximum wind: repo law vs energy balance", "time (days)", "m/s")
        A["B"].plot(d, st["run"]["vp"], color=INK2, lw=1.4, ls=(0, (4, 3)), label="potential intensity Vp")
        A["B"].plot(d, st["v_repo"], color=C_REPO, lw=2.2, label="repo law")
        A["B"].plot(d, st["v_phys"], color=C_PHYS, lw=2.2, label="energy balance")
        for v, name in SAFFIR:
            A["B"].axhline(v, color=GRID, lw=0.8, zorder=0)
            A["B"].text(d[-1], v, name, fontsize=6.5, color=INK2, va="bottom", ha="right")
        for key, c, lbl in (("v_repo", C_REPO, "repo"), ("v_phys", C_PHYS, "phys")):
            j = int(np.argmax(st[key]))
            A["B"].plot(d[j], st[key][j], "o", color=c, ms=6, mec=SURFACE, mew=1.5)
        if ev["t_ri_start"] is not None:
            A["B"].axvspan(ev["t_ri_start"], ev["t_ri_end"], ymin=0, ymax=0.03, color=C_PHYS, lw=0)
            A["B"].text(ev["t_ri_start"], 3.0, " RI window (energy balance)", fontsize=7, color=INK2, va="bottom")
        A["B"].legend(fontsize=7.5, frameon=False, loc="upper left")
        # C: intensification rate
        self._style(A["C"], "C  Intensification rate dV/dt", "time (days)", "m/s per day")
        A["C"].plot(d, st["dvdt_repo"], color=C_REPO, lw=2, label="repo law")
        A["C"].plot(d, st["dvdt_phys"], color=C_PHYS, lw=2, label="energy balance")
        A["C"].axhline(ph.RI_MS_PER_DAY, color=INK2, lw=1, ls=(0, (4, 3)))
        A["C"].text(d[-1], ph.RI_MS_PER_DAY, " RI\n 15.4", fontsize=7, color=INK2, va="bottom")
        A["C"].axhline(0.0, color=GRID, lw=1)
        top = 1.1 * float(max(np.max(st["dvdt_repo"]), np.max(st["dvdt_phys"]), 20.0))
        A["C"].set_ylim(-60.0, top)      # the repo law's end-of-track fade-out has a very steep tail: clipped on purpose
        if ev["t_ri_start"] is not None:
            A["C"].axvspan(ev["t_ri_start"], ev["t_ri_end"], color=C_PHYS, alpha=0.10, lw=0)
        A["C"].legend(fontsize=7.5, frameon=False, loc="upper right")
        # D: radial profile (filled by move)
        self._style(A["D"], "D  Wind profile at the cursor", "radius (tank units)", "tangential speed (m/s)")
        # E: sensitivity
        s = st["sens"]
        self._style(A["E"], "E  Peak wind vs warm-zone anomaly (ensemble)", "warm-zone anomaly (K)", "peak wind (m/s)")
        A["E"].plot(s["warm_amp"], s["repo"], color=C_REPO, lw=2, label="repo law")
        A["E"].plot(s["warm_amp"], s["phys"], color=C_PHYS, lw=2, label="energy balance")
        A["E"].axvline(st["p"].warm_amp, color=INK2, lw=1, ls=(0, (4, 3)))
        A["E"].legend(fontsize=7.5, frameon=False, loc="lower right")
        # F: phase plane
        self._style(A["F"], "F  Phase plane: wind vs SST seen", "SST seen by the storm (°C)", "m/s")
        sst_axis = np.linspace(22.0, 32.0, 120)
        A["F"].plot(sst_axis, ph.vp_effective(sst_axis + ph.KELVIN, st["q"]), color=INK2, lw=1.4, ls=(0, (4, 3)), label="Vp(SST)")
        A["F"].plot(st["run"]["sst_seen_k"] - 273.15, st["v_phys"], color=C_PHYS, lw=2, label="energy balance")
        A["F"].plot(st["sst_track"], st["v_repo"], color=C_REPO, lw=2, label="repo law")
        A["F"].legend(fontsize=7.5, frameon=False, loc="upper left")
        self.move(st, u)

    def move(self, st: dict, u: float) -> dict:
        for art in self.cur:
            art.remove()
        self.cur = []
        c, A = at(st, u), self.ax
        for k in "ABC":
            self.cur.append(A[k].axvline(c["day"], color=INK, lw=1.2))
        self.cur.append(A["A"].plot(c["day"], c["sst"], "o", color=INK2, ms=7, mec=SURFACE, mew=1.5)[0])
        self.cur.append(A["A"].plot(c["day"], c["sst_seen"], "o", color=C_PHYS, ms=7, mec=SURFACE, mew=1.5)[0])
        self.cur.append(A["B"].plot(c["day"], c["v_repo"], "o", color=C_REPO, ms=7, mec=SURFACE, mew=1.5)[0])
        self.cur.append(A["B"].plot(c["day"], c["v_phys"], "o", color=C_PHYS, ms=7, mec=SURFACE, mew=1.5)[0])
        self.cur.append(A["C"].plot(c["day"], st["dvdt_repo"][c["i"]], "o", color=C_REPO, ms=7, mec=SURFACE, mew=1.5)[0])
        self.cur.append(A["C"].plot(c["day"], st["dvdt_phys"][c["i"]], "o", color=C_PHYS, ms=7, mec=SURFACE, mew=1.5)[0])
        r = np.linspace(0.0, 10.0, 200)
        shape = lambda R: model.wind_profile(r, 1.0, r0=R, vmax=1.0)
        self.cur.append(A["D"].plot(r, c["v_repo"] * shape(c["R_repo"]), color=C_REPO, lw=2)[0])
        self.cur.append(A["D"].plot(r, c["v_phys"] * shape(c["R_phys"]), color=C_PHYS, lw=2)[0])
        A["D"].set_ylim(0, max(75.0, 1.05 * max(c["v_repo"], c["v_phys"], float(np.max(st["v_phys"])))))
        self.cur.append(A["E"].plot(st["p"].warm_amp, np.interp(st["p"].warm_amp, st["sens"]["warm_amp"], st["sens"]["phys"]), "o",
                                    color=C_PHYS, ms=7, mec=SURFACE, mew=1.5)[0])
        self.cur.append(A["F"].plot(c["sst_seen"], c["v_phys"], "o", color=C_PHYS, ms=7, mec=SURFACE, mew=1.5)[0])
        self.cur.append(A["F"].plot(c["sst"], c["v_repo"], "o", color=C_REPO, ms=7, mec=SURFACE, mew=1.5)[0])
        title = (f"day {c['day']:.2f} of {st['duration_days']:.2f}  ·  x = {c['cx']:+.1f}  ·  SST seen {c['sst_seen']:.1f} °C  ·  "
                 f"repo {c['v_repo']:.0f} m/s  ·  physics {c['v_phys']:.0f} m/s (Vp {c['vp']:.0f})")
        self.fig.suptitle(title, fontsize=10.5, color=INK, x=0.01, ha="left")
        return c

    def png(self) -> bytes:
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", facecolor=SURFACE)
        return buf.getvalue()


def static_report(path: str, flat: dict | None = None, t: float = 16.0, figsize=(13.0, 7.2)) -> str:
    """Write the six panels for the parameters ``flat`` at animation time ``t`` (s) to a PNG (no GL, no widgets)."""
    st = compute(flat or {})
    pn = Panels(figsize)
    pn.draw(st, t / PERIOD)
    pn.fig.savefig(path, facecolor=SURFACE)
    return path


class Dashboard:
    """Widgets + animation frame + plots. ``Dashboard().show()`` in a notebook cell."""

    def __init__(self, size=(480, 270), viewer=None, render: bool = True):
        import ipywidgets as w
        from .api import Viewer
        self.w = w
        self.flat = dict(DEFAULTS)
        self.render_frames_enabled = render
        self.viewer = viewer or (Viewer(size=size) if render else None)
        self.panels = Panels()
        self.state = compute(self.flat)
        self.t = 16.0
        self._busy = False
        self._build()
        self.refresh()

    # ---- UI ---------------------------------------------------------------------------------------------------------
    def _build(self):
        w = self.w
        self.controls = {}
        groups = {}
        for key, label, lo, hi, step, default, group in SLIDERS:
            s = w.FloatSlider(value=default, min=lo, max=hi, step=step, description=label, continuous_update=False,
                              style={"description_width": "170px"}, layout=w.Layout(width="360px"), readout_format=".2f")
            s.observe(lambda ch, k=key: self._on_param(k, ch["new"]), names="value")
            self.controls[key] = s
            groups.setdefault(group, []).append(s)
        self.cooling = w.Checkbox(value=True, description="Sea-surface cooling feedback", indent=False)
        self.viab = w.Checkbox(value=True, description="Viability factor (Gray threshold)", indent=False)
        self.cooling.observe(lambda ch: self._on_param("cooling", ch["new"]), names="value")
        self.viab.observe(lambda ch: self._on_param("use_viability", ch["new"]), names="value")
        groups["Physics"] += [self.cooling, self.viab]
        self.drive = w.ToggleButtons(options=["Repo law", "Energy balance"], value="Repo law", description="Animation driven by",
                                     style={"description_width": "initial", "button_width": "120px"})
        self.drive.observe(lambda ch: self._on_param("drive", ch["new"]), names="value")
        self.preset = w.Dropdown(options=list(PRESETS), description="Scenario", style={"description_width": "60px"},
                                 layout=w.Layout(width="360px"))
        self.preset.observe(lambda ch: self.apply_preset(ch["new"]), names="value")
        self.time = w.FloatSlider(value=self.t, min=0.0, max=PERIOD, step=0.25, description="Animation time (s)", readout_format=".2f",
                                  continuous_update=False, style={"description_width": "130px"}, layout=w.Layout(width="620px"))
        self.time.observe(lambda ch: self.set_time(ch["new"]), names="value")
        self.play = w.Play(value=self.t, min=0.0, max=PERIOD, step=0.5, interval=450, description="Play")
        w.jslink((self.play, "value"), (self.time, "value"))
        self.az = w.FloatSlider(value=0.0, min=-1.2, max=1.2, step=0.05, description="Camera azimuth", continuous_update=False,
                                style={"description_width": "110px"}, layout=w.Layout(width="360px"))
        self.az.observe(lambda ch: self.refresh(), names="value")
        self.live = w.Checkbox(value=self.render_frames_enabled, description="Render 3-D frame", indent=False,
                               disabled=self.viewer is None)
        self.live.observe(lambda ch: self.refresh(), names="value")
        self.gif_btn = w.Button(description="Export GIF (parallel)", icon="film", layout=w.Layout(width="360px"))
        self.gif_btn.on_click(lambda _: self.export_gif())
        self.msg = w.HTML()
        self.img = w.Image(format="png", layout=w.Layout(width="480px"))
        self.fig_img = w.Image(format="png", layout=w.Layout(width="100%", max_width="1100px"))
        tabs = w.Tab(children=[w.VBox(groups["Ocean"]), w.VBox(groups["Repo law"]), w.VBox(groups["Physics"])])
        for i, t in enumerate(("Ocean", "Repo law", "Physics")):
            tabs.set_title(i, t)
        left = w.VBox([self.preset, self.drive, tabs, self.az, self.live, self.gif_btn, self.msg], layout=w.Layout(width="390px"))
        top = w.HBox([left, self.img], layout=w.Layout(align_items="flex-start"))
        self.ui = w.VBox([top, w.HBox([self.play, self.time]), self.fig_img])

    def show(self):
        from IPython.display import display
        display(self.ui)
        return self

    # ---- state ------------------------------------------------------------------------------------------------------
    def _on_param(self, key, value):
        self.flat[key] = value
        self.refresh(recompute=True)

    def set(self, **kw):
        """Set parameters programmatically (updates the widgets and redraws once)."""
        self.flat.update(kw)
        for k, v in kw.items():
            wd = self.controls.get(k) or {"cooling": self.cooling, "use_viability": self.viab}.get(k)
            if k == "drive":
                wd = self.drive
                v = "Energy balance" if v in ("Energy balance", "physics") else "Repo law"
                self.flat["drive"] = v
            if wd is not None and wd.value != v:
                self._busy = True
                wd.value = v
                self._busy = False
        self.refresh(recompute=True)

    def apply_preset(self, name):
        self.set(**{**DEFAULTS, **PRESETS[name]})

    def set_time(self, t):
        self.t = float(t)
        self.refresh()

    def timeline_model(self) -> dict:
        st = self.state
        tab = st["run"]["v"] / st["v_norm"] if self.flat["drive"] == "Energy balance" else st["tab"]
        return dict(itab=np.asarray(tab, np.float32), zone=(st["p"].z0, st["p"].z1), gain=st["p"].gain)

    def refresh(self, recompute: bool = False):
        if self._busy:
            return
        if recompute:
            self.state = compute(self.flat)
            self.panels.draw(self.state, self.t / PERIOD)
        else:
            self.panels.move(self.state, self.t / PERIOD)
        self.fig_img.value = self.panels.png()
        if self.viewer is not None and self.live.value:
            self.viewer.set_model(**self.timeline_model())
            frame = self.viewer.frame(at=self.t, az=float(self.az.value))
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            self.img.value = buf.getvalue()
        ev = self.state["ev"]
        fmt = lambda x: "–" if x is None else f"{x:.2f}"
        self.msg.value = (f"<small>zone entry {fmt(ev['t_enter'])} d · exit {fmt(ev['t_exit'])} d · genesis {fmt(ev['t_genesis'])} d · "
                          f"peak {ev['v_peak']:.0f} m/s at {fmt(ev['t_peak'])} d · max dV/dt {ev['ri_max']:.0f} m/s/day"
                          f"{' (RI)' if ev['ri_max'] >= ph.RI_MS_PER_DAY else ''}</small>")

    # ---- HPC: parallel export -----------------------------------------------------------------------------------------
    def export_gif(self, path: str = "cyclone_dashboard.gif", fps: float = 8.0, workers: int | None = None) -> str:
        """Render the current scenario as a GIF, frames in parallel (``workers`` defaults to the CPU count)."""
        if self.viewer is None:
            raise RuntimeError("no viewer: create the Dashboard with render=True")
        self.viewer.set_model(**self.timeline_model())
        self.msg.value = "<small>rendering GIF…</small>"
        self.viewer.gif(path, fps=fps, az=float(self.az.value), workers=workers or os.cpu_count() or 1, progress=False)
        self.refresh()
        return path
