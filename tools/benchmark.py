"""Run the HPC benchmarks and write docs/benchmarks.json (run as a script: multiprocessing uses ``spawn``)."""
import json
import sys
import numpy as np
from oceantank import hpc


def ser(o):
    if isinstance(o, dict):
        return {k: ser(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main(out="docs/benchmarks.json", members=200_000, workers=(1, 2), frames=8):
    res = dict(sweep=hpc.scaling(n_members=members, workers=workers, repeats=2),
               render=hpc.render_scaling(workers=workers, n_frames=frames),
               conv_euler=hpc.convergence(), conv_rk4=hpc.convergence_physics())
    json.dump(ser(res), open(out, "w"), indent=1)
    print(json.dumps(ser({k: res[k] for k in ("sweep", "render")}), indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
