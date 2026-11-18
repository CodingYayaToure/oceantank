"""Tank dimensions (same names and default values as the HTML animation)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Tank:
    """Rectangular glass tank standing on the ground, filled with water up to ``depth``.

    x: length (``length``), z: width (``width``), y: height. The origin is at the centre of the tank floor.
    """
    length: float = 30.0        # L
    width: float = 8.0          # W
    depth: float = 3.2          # D: resting height of the water
    glass_height: float = 4.3   # GH: height of the glass panes (above the water)

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2
