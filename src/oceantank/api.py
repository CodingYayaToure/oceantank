"""Simple API for notebooks (Jupyter, Colab) and scripts: ``Viewer(...).frame(at=14)`` returns a PIL image."""
from __future__ import annotations
from PIL import Image
from . import Tank, Camera, Timeline, Scene, over


class Viewer:
    """Glass tank (+ water, + particles) ready to be rendered at any instant of the 32 s cycle.

    >>> v = Viewer(size=(960, 540))
    >>> v.frame(at=14, az=0.4)            # PIL image (displays by itself in a notebook)
    >>> v.gif("cycle.gif")                # one full cycle
    """

    def __init__(self, size=(960, 540), ocean: bool = True, particles: bool = True,
                 hash_mode: str = "native", seed: int = 12345, tank: Tank | None = None):
        self.size, self.seed, self.tank = tuple(size), seed, tank or Tank()
        self._tl = self._new_timeline()
        self.scene = Scene(self.tank, hash_mode=hash_mode).build_glass()
        if ocean:
            self.scene.add_ocean()
            if particles:
                self.scene.add_particles(self._tl)

    def _new_timeline(self) -> Timeline:
        return Timeline(self.tank, seed=self.seed, rng="mulberry32", start_ready=False)

    def frame(self, at: float = 14.0, az: float = 0.0, el: float = 0.62, zoom: float = 1.0) -> Image.Image:
        """Image at time ``at`` (s; 0 = formation ... 32 = end of the cycle)."""
        tl = self._new_timeline()
        tl.run_to(at)
        tl.advance(0)
        rgba = self.scene.render(tl, Camera(az=az, el=el, zoom=zoom), self.size)
        return Image.fromarray(over(rgba))

    def gif(self, path: str = "cycle.gif", start: float = 0.0, duration: float = 32.0, fps: float = 8.0,
            speed: float = 1.0, az: float = 0.0, el: float = 0.62, progress: bool = True) -> str:
        """GIF animation of ``duration`` simulated seconds (32 = one full cycle)."""
        tl = self._new_timeline()
        tl.run_to(start)
        n, step = max(1, round(duration * fps / speed)), speed / fps
        cam, frames = Camera(az=az, el=el), []
        for k in range(n):
            tl.advance(step)
            img = Image.fromarray(over(self.scene.render(tl, cam, self.size)))
            frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=255))
            if progress and k % 25 == 0:
                print(f"frame {k + 1}/{n} (cyclone at x = {tl.state.uC[0]:.1f})")
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
        return path
