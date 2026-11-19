"""Shared drawing helpers for the terrain environments."""

from __future__ import annotations

import numpy as np

# Elevation colour ramp: deep water, shallows, sand, grass, forest, rock, snow.
_RAMP_STOPS = [0.00, 0.28, 0.33, 0.38, 0.60, 0.80, 1.00]
_RAMP_COLORS = np.array(
    [
        [24, 54, 110],
        [64, 124, 180],
        [216, 201, 146],
        [112, 165, 84],
        [48, 104, 60],
        [120, 112, 104],
        [244, 244, 248],
    ],
    dtype=np.float64,
)
_OUTLINE_COLOR = np.array([255, 255, 255], dtype=np.uint8)


def terrain_image(elevation: np.ndarray, cell_size: int, relief: float = 4.0) -> np.ndarray:
    """Colour an elevation map with hill shading, scaled up by ``cell_size``.

    ``relief`` sets the shading strength per unit of slope between samples.
    """
    colors = np.stack(
        [np.interp(elevation, _RAMP_STOPS, _RAMP_COLORS[:, i]) for i in range(3)], axis=-1
    )
    # Light from the top left, so slopes read as relief.
    if elevation.shape[0] > 1 and elevation.shape[1] > 1:
        d_row, d_col = np.gradient(elevation)
        colors = colors * np.clip(1.0 - relief * (d_row + d_col), 0.6, 1.4)[..., None]
    frame = np.clip(colors, 0, 255).astype(np.uint8)
    return frame.repeat(cell_size, axis=0).repeat(cell_size, axis=1)


def draw_marker(
    frame: np.ndarray,
    cell: tuple[int, int],
    color: tuple[int, int, int],
    cell_size: int,
    radius: float = 0.45,
) -> None:
    """Draw a white-outlined disc in one cell of a rendered frame, in place.

    ``radius`` is a fraction of the cell size.
    """
    top, left = cell[0] * cell_size, cell[1] * cell_size
    center = (cell_size - 1) / 2.0
    rows, cols = np.ogrid[:cell_size, :cell_size]
    distance = np.hypot(rows - center, cols - center)
    radius = max(cell_size * radius, 1.0)
    patch = frame[top : top + cell_size, left : left + cell_size]
    patch[distance <= radius] = _OUTLINE_COLOR
    patch[distance <= radius - max(cell_size * 0.04, 1.0)] = color


class MatplotlibViewer:
    """A reusable window for ``render_mode="human"``."""

    def __init__(self) -> None:
        self._figure = None
        self._image = None

    def show(self, frame: np.ndarray, fps: float) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None:
            plt.ion()
            self._figure, axes = plt.subplots()
            axes.set_axis_off()
            self._image = axes.imshow(frame)
        else:
            self._image.set_data(frame)
        plt.pause(1.0 / fps)

    def close(self) -> None:
        if self._figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._figure)
            self._figure = None
            self._image = None
