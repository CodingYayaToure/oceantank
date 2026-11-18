"""Figures of the documentation: docs/dashboard.png and docs/hpc_benchmarks.png (needs matplotlib and docs/benchmarks.json)."""
import json
import numpy as np
from matplotlib.figure import Figure
from oceantank.dashboard import static_report, C_REPO, C_PHYS, INK2, GRID, SURFACE

static_report("docs/dashboard.png", {}, t=16.0)

b = json.load(open("docs/benchmarks.json"))
fig = Figure(figsize=(13.0, 3.6), dpi=100, facecolor=SURFACE, layout="constrained")
ax = fig.subplots(1, 4)
for a, key, name, col, k in ((ax[0], "conv_euler", "Euler, repo law", C_REPO, 1), (ax[1], "conv_rk4", "RK4, energy balance", C_PHYS, 4)):
    c = b[key]; h, e = np.array(c["h"]), np.array(c["error"])
    a.loglog(h, e, "o-", color=col, lw=2)
    a.loglog(h, e[-1] * (h / h[-1]) ** k, "--", color=INK2, lw=1, label=f"formal order {k}")
    a.set_title(f"{name}: observed order {c['order']:.2f}", fontsize=9.5, loc="left")
    a.set_xlabel("step h (fraction of the track)", fontsize=8.5); a.set_ylabel("error", fontsize=8.5); a.legend(frameon=False, fontsize=8)
s, r = b["sweep"], b["render"]
for a, d, title, col in ((ax[2], s, f"Parameter sweep ({s['n_members']:,} members)", C_REPO), (ax[3], r, f"Parallel rendering ({r['frames']} frames)", C_PHYS)):
    w = np.array(d["workers"])
    a.plot(w, d["speedup"], "o-", color=col, lw=2, label="measured")
    a.plot(w, w / w[0], "--", color=INK2, lw=1, label="ideal")
    a.set_title(f"{title}\nefficiency {d['efficiency'][-1]*100:.0f} % on {w[-1]} processes", fontsize=9.5, loc="left")
    a.set_xlabel("processes", fontsize=8.5); a.set_ylabel("speed-up", fontsize=8.5); a.set_xticks(w); a.legend(frameon=False, fontsize=8)
for a in ax:
    a.set_facecolor(SURFACE); a.grid(True, color=GRID); a.tick_params(labelsize=8)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.savefig("docs/hpc_benchmarks.png", facecolor=SURFACE)
print("figures written")
