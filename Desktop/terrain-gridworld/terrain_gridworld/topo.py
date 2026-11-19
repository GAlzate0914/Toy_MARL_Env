"""Line-drawn contour-map rendering with a pixel-art RC car for the summit environment."""

from __future__ import annotations

import numpy as np

_PAPER_COLOR = "#ffffff"
_LINE_COLOR = "#111111"
_GRID_COLOR = "#000000"
_SUMMIT_COLOR = "#f2b705"

# A top-down RC car facing up. Each character is one sprite pixel.
_CAR_ROWS = [
    "..................",
    "......KKKKKK......",
    ".....KYRRRRYK.....",
    ".....KRRRRRRK.....",
    "...KKKRRwwRRKKK...",
    "...KKKRRwwRRKKK...",
    "...KKKRWWWWRKKK...",
    ".....KRWWWWRK.....",
    ".....KRRwwRRK.....",
    ".....KRRwwRRK.....",
    ".....KRRwwRRK.....",
    ".....KRDDDDRK.....",
    "...KKKRDDDDRKKK...",
    "...KKKRRwwRRKKK...",
    "...KKKRRRRRRKKK...",
    ".....KDRRRRDK.....",
    "......KKKKKKG.....",
    "............G.....",
]
_CAR_PALETTE = {
    "K": (32, 32, 36),     # tyres and outline
    "R": (214, 48, 49),    # body
    "D": (150, 30, 32),    # body shadow
    "W": (150, 210, 240),  # windscreen
    "w": (245, 245, 245),  # racing stripe
    "Y": (255, 220, 80),   # headlights
    "G": (70, 70, 74),     # antenna
}
CAR_BODY_COLOR = _CAR_PALETTE["R"]

# np.rot90 turns counter-clockwise, so these are quarter turns from facing up.
_QUARTER_TURNS = {"up": 0, "left": 1, "down": 2, "right": 3}


def _build_sprite() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = len(_CAR_ROWS)
    colors = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    for r, row in enumerate(_CAR_ROWS):
        for c, char in enumerate(row):
            if char != ".":
                colors[r, c] = _CAR_PALETTE[char]
                mask[r, c] = True
    return colors, mask, np.array([list(row) for row in _CAR_ROWS])


_CAR_COLORS, _CAR_MASK, _CAR_CHARS = _build_sprite()


def topo_background(
    surface: np.ndarray,
    cell_size: int,
    goal: tuple[int, int],
    start: tuple[int, int] | None = None,
    interval: float = 0.1,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Draw the terrain as a contour map and return it as an RGB array.

    Thin contour lines, one every ``interval`` of elevation, with a faint
    cell grid, a ring at the start and a flag on the peak. Without ``base``
    the lines are black on white; with ``base``, an RGB image of the same
    size, softer dark lines are drawn over it. The output is
    ``surface.shape`` pixels.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    height, width = surface.shape
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_PAPER_COLOR)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_axis_off()
    if base is not None:
        axes.imshow(base, extent=(0, width, height, 0), interpolation="nearest")
    grid_color, grid_alpha = (("#ffffff", 0.35) if base is not None else (_GRID_COLOR, 0.08))

    for i in range(1, width // cell_size):
        axes.axvline(i * cell_size, color=grid_color, linewidth=0.6, alpha=grid_alpha)
    for i in range(1, height // cell_size):
        axes.axhline(i * cell_size, color=grid_color, linewidth=0.6, alpha=grid_alpha)

    x = np.arange(width) + 0.5
    y = np.arange(height) + 0.5
    lines = np.arange(interval, 1.0, interval)
    if len(lines):
        if base is None:
            axes.contour(x, y, surface, levels=lines, colors=_LINE_COLOR, linewidths=1.0)
        else:
            widths = [1.1 if round(level / interval) % 4 == 0 else 0.6 for level in lines]
            axes.contour(x, y, surface, levels=lines, colors="#1d2418", linewidths=widths,
                         alpha=0.55)

    if start is not None:
        sy, sx = (start[0] + 0.5) * cell_size, (start[1] + 0.5) * cell_size
        axes.plot(sx, sy, "o", markersize=cell_size * 0.5, markerfacecolor="none",
                  markeredgecolor="#ffffff" if base is not None else _LINE_COLOR,
                  markeredgewidth=1.5, alpha=0.8 if base is not None else 0.25)

    # Plant the flag on the highest point, which lies inside the goal cell.
    top, left = goal[0] * cell_size, goal[1] * cell_size
    patch = surface[top : top + cell_size, left : left + cell_size]
    peak_row, peak_col = np.unravel_index(np.argmax(patch), patch.shape)
    gy, gx = top + peak_row + 0.5, left + peak_col + 0.5
    pole = cell_size * 0.42
    axes.plot([gx, gx], [gy + pole * 0.3, gy - pole], color=_LINE_COLOR, linewidth=2.0,
              solid_capstyle="round")
    axes.fill(
        [gx, gx + pole * 0.85, gx], [gy - pole, gy - pole * 0.75, gy - pole * 0.5],
        facecolor=_SUMMIT_COLOR, edgecolor=_LINE_COLOR, linewidth=1.0,
    )

    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def draw_car(
    frame: np.ndarray,
    cell: tuple[int, int],
    heading: str,
    cell_size: int,
    body: tuple[int, int, int] | None = None,
) -> bool:
    """Paste the car sprite into a cell, facing ``heading``, in place.

    ``body`` repaints the car in another colour; the default is red. The
    sprite is scaled up by a whole number so its pixels stay crisp. Returns
    False, drawing nothing, if the cell is too small for the sprite.
    """
    scale = int(cell_size * 0.9) // _CAR_MASK.shape[0]
    if scale < 1:
        return False
    sprite = _CAR_COLORS
    if body is not None:
        sprite = _CAR_COLORS.copy()
        sprite[_CAR_CHARS == "R"] = body
        sprite[_CAR_CHARS == "D"] = tuple(int(v * 0.7) for v in body)
    turns = _QUARTER_TURNS[heading]
    colors = np.rot90(sprite, turns).repeat(scale, 0).repeat(scale, 1)
    mask = np.rot90(_CAR_MASK, turns).repeat(scale, 0).repeat(scale, 1)
    size = mask.shape[0]
    top = cell[0] * cell_size + (cell_size - size) // 2
    left = cell[1] * cell_size + (cell_size - size) // 2
    patch = frame[top : top + size, left : left + size]
    patch[mask] = colors[mask]
    return True


def draw_battery(
    frame: np.ndarray, cell: tuple[int, int], fraction: float | None, cell_size: int
) -> None:
    """Draw a small battery gauge along the top of a cell, in place.

    Green above half charge, amber above a fifth, red below. ``fraction`` of
    None (unlimited charge) draws nothing.
    """
    if fraction is None or not np.isfinite(fraction) or cell_size < 12:
        return
    fraction = min(max(float(fraction), 0.0), 1.0)
    top, left = cell[0] * cell_size, cell[1] * cell_size
    height = max(cell_size // 9, 3)
    width = int(cell_size * 0.7)
    x0 = left + (cell_size - width) // 2
    y0 = top + 2
    frame[y0 : y0 + height, x0 : x0 + width] = (30, 30, 34)
    color = (70, 200, 90) if fraction > 0.5 else (240, 180, 40) if fraction > 0.2 else (225, 60, 50)
    filled = int(round((width - 2) * fraction))
    if filled > 0:
        frame[y0 + 1 : y0 + height - 1, x0 + 1 : x0 + 1 + filled] = color


def draw_record(frame: np.ndarray, cell: tuple[int, int], cell_size: int) -> None:
    """Outline a cell in gold: the highest point recorded so far, in place."""
    top, left = cell[0] * cell_size, cell[1] * cell_size
    width = max(cell_size // 16, 2)
    gold = (255, 205, 40)
    patch = frame[top : top + cell_size, left : left + cell_size]
    patch[:width] = gold
    patch[-width:] = gold
    patch[:, :width] = gold
    patch[:, -width:] = gold
