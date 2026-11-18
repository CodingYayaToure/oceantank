"""Simple API for notebooks (Jupyter, Colab) and scripts: ``Viewer(...).frame(at=14)`` returns a PIL image."""
from __future__ import annotations
import os
from PIL import Image
from . import Tank, Camera, Timeline, Scene, over

_WORKER = None


def _worker_init(cfg):
    """Pool initialiser: one single-threaded software-GL context per process (avoids llvmpipe oversubscription)."""
    global _WORKER
    os.environ.setdefault("LP_NUM_THREADS", "1")
    _WORKER = Viewer(**cfg)


def _worker_frame(job):
    at, az, el, zoom = job
    return _WORKER.frame(at=at, az=az, el=el, zoom=zoom).tobytes()


class Viewer:
    """Glass tank (+ water, + particles) ready to be rendered at any instant of the 32 s cycle.

    >>> v = Viewer(size=(960, 540))
    >>> v.frame(at=14, az=0.4)            # PIL image (displays by itself in a notebook)
    >>> v.gif("cycle.gif")                # one full cycle
    """

    def __init__(self, size=(960, 540), ocean: bool = True, particles: bool = True,
                 hash_mode: str = "native", seed: int = 12345, tank: Tank | None = None, timeline: dict | None = None):
        self.size, self.seed, self.tank = tuple(size), seed, tank or Tank()
        self._cfg = dict(size=self.size, ocean=ocean, particles=particles, hash_mode=hash_mode, seed=seed, tank=self.tank, timeline=timeline)
        self.timeline_kw = dict(timeline or {})          # extra Timeline arguments: gain, itab, zone
        self._tl = self._new_timeline()
        self.scene = Scene(self.tank, hash_mode=hash_mode).build_glass()
        if ocean:
            self.scene.add_ocean()
            if particles:
                self.scene.add_particles(self._tl)

    def _new_timeline(self) -> Timeline:
        return Timeline(self.tank, seed=self.seed, rng="mulberry32", start_ready=False, **self.timeline_kw)

    def set_model(self, **timeline_kw) -> "Viewer":
        """Change the storm model driving the animation (``gain``, ``itab``, ``zone``) without rebuilding the GL scene."""
        self.timeline_kw = dict(timeline_kw)
        self._cfg["timeline"] = dict(timeline_kw)
        return self

    def frame(self, at: float = 14.0, az: float = 0.0, el: float = 0.62, zoom: float = 1.0) -> Image.Image:
        """Image at time ``at`` (s; 0 = formation ... 32 = end of the cycle)."""
        tl = self._new_timeline()
        tl.run_to(at)
        tl.advance(0)
        rgba = self.scene.render(tl, Camera(az=az, el=el, zoom=zoom), self.size)
        return Image.fromarray(over(rgba))

    def render_frames(self, times, az: float = 0.0, el: float = 0.62, zoom: float = 1.0, workers: int = 1) -> list:
        """Frames at the given instants. ``workers > 1`` renders in parallel (one process + GL context per worker).

        Every frame is a pure function of its instant (a fresh, seeded Timeline per frame), so the result is
        pixel-identical whatever ``workers`` is and in whatever order the frames are computed.
        """
        times = [float(t) for t in times]
        if workers <= 1 or len(times) < 2:
            return [self.frame(at=t, az=az, el=el, zoom=zoom) for t in times]
        import multiprocessing as mp
        ctx = mp.get_context("spawn")      # fork would duplicate an initialised GL context
        jobs = [(t, az, el, zoom) for t in times]
        with ctx.Pool(min(workers, len(times)), initializer=_worker_init, initargs=(self._cfg,)) as pool:
            raw = pool.map(_worker_frame, jobs, chunksize=max(1, len(jobs) // (4 * workers)))
        return [Image.frombytes("RGB", self.size, b) for b in raw]

    def gif(self, path: str = "cycle.gif", start: float = 0.0, duration: float = 32.0, fps: float = 8.0,
            speed: float = 1.0, az: float = 0.0, el: float = 0.62, progress: bool = True, workers: int | None = None) -> str:
        """GIF animation of ``duration`` simulated seconds (32 = one full cycle).

        ``workers`` (default ``None``) renders the frames with :meth:`render_frames` (parallel when > 1); the default
        path advances one timeline incrementally.
        """
        n, step = max(1, round(duration * fps / speed)), speed / fps
        if workers is not None:
            imgs = self.render_frames([start + (k + 1) * step for k in range(n)], az=az, el=el, workers=workers)
            frames = [im.convert("P", palette=Image.ADAPTIVE, colors=255) for im in imgs]
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
            return path
        tl = self._new_timeline()
        tl.run_to(start)
        cam, frames = Camera(az=az, el=el), []
        for k in range(n):
            tl.advance(step)
            img = Image.fromarray(over(self.scene.render(tl, cam, self.size)))
            frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=255))
            if progress and k % 25 == 0:
                print(f"frame {k + 1}/{n} (cyclone at x = {tl.state.uC[0]:.1f})")
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
        return path
