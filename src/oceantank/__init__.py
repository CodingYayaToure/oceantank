"""oceantank: a rectangular glass tank filled with ocean water, rendered like the HTML/three.js animation."""
from .spec import Tank
from .camera import Camera
from .timeline import Timeline, OceanState, Mulberry32
from .scene import Scene
from .compose import over, page_background
from .api import Viewer

__all__ = ["Viewer", "Tank", "Camera", "Timeline", "OceanState", "Mulberry32", "Scene", "over", "page_background"]
__version__ = "0.1.2"
