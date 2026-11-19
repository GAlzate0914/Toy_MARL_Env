from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TABLE_SIZE = 256
_TABLE_MASK = _TABLE_SIZE - 1
# Unit-gradient 2-D Perlin noise is bounded by +/- sqrt(2)/2.
_PERLIN_BOUND = np.sqrt(2.0) / 2.0


@dataclass(frozen=True)
class TerrainConfig:
    scale: float = 16.0
    octaves: int = 4
    persistence: float = 0.5
    lacunarity: float = 2.0
    exponent: float = 1.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.octaves < 1:
            raise ValueError(f"octaves must be at least 1, got {self.octaves}")
        if not 0 < self.persistence <= 1:
            raise ValueError(f"persistence must be in (0, 1], got {self.persistence}")
        if self.lacunarity < 1:
            raise ValueError(f"lacunarity must be at least 1, got {self.lacunarity}")
        if self.exponent <= 0:
            raise ValueError(f"exponent must be positive, got {self.exponent}")


class PerlinNoise2D:
    """A single layer of 2-D gradient noise, vectorized with NumPy."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._permutation = rng.permutation(_TABLE_SIZE)
        angles = rng.uniform(0.0, 2.0 * np.pi, _TABLE_SIZE)
        self._gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        # Perlin noise is exactly zero on lattice points. A random sub-cell
        # offset stops a regular cell grid from landing on them, which would
        # otherwise flatten the whole layer when scale divides the coordinates.
        self._offset = rng.uniform(0.0, _TABLE_SIZE, 2)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = x + self._offset[0]
        y = y + self._offset[1]
        x_floor = np.floor(x).astype(np.int64)
        y_floor = np.floor(y).astype(np.int64)
        x_frac = x - x_floor
        y_frac = y - y_floor

        def corner(dx: int, dy: int) -> np.ndarray:
            hashed = self._permutation[(x_floor + dx) & _TABLE_MASK]
            hashed = self._permutation[(hashed + y_floor + dy) & _TABLE_MASK]
            gradient = self._gradients[hashed]
            return gradient[..., 0] * (x_frac - dx) + gradient[..., 1] * (y_frac - dy)

        u = _fade(x_frac)
        v = _fade(y_frac)
        bottom = _lerp(corner(0, 0), corner(1, 0), u)
        top = _lerp(corner(0, 1), corner(1, 1), u)
        return _lerp(bottom, top, v)


def _fade(t: np.ndarray) -> np.ndarray:
    """Quintic easing, which keeps the noise smooth across lattice boundaries."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def generate_terrain(
    height: int,
    width: int,
    seed: int | None = None,
    config: TerrainConfig | None = None,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Generate an elevation map.

    Args:
        height: Number of rows.
        width: Number of columns.
        seed: Terrain seed. The same seed and config reproduce the same world.
        config: Generator settings. Defaults to ``TerrainConfig()``.
        origin: World coordinates ``(row, col)`` of the map's first cell. Use
            with ``normalize=False`` to generate adjoining chunks.

    Returns:
        A ``float32`` array of shape ``(height, width)`` with values in [0, 1].
    """
    if height < 1 or width < 1:
        raise ValueError(f"map size must be positive, got {height}x{width}")
    config = config or TerrainConfig()

    rows = (np.arange(height) + origin[0]) / config.scale
    cols = (np.arange(width) + origin[1]) / config.scale
    y, x = np.meshgrid(rows, cols, indexing="ij")

    # Each octave gets an independent generator, as in Minecraft. Spawning from
    # a SeedSequence means raising `octaves` adds detail without changing the
    # layers that already exist.
    octave_seeds = np.random.SeedSequence(seed).spawn(config.octaves)
    total = np.zeros_like(x)
    amplitude = 1.0
    frequency = 1.0
    amplitude_sum = 0.0
    for octave_seed in octave_seeds:
        noise = PerlinNoise2D(np.random.default_rng(octave_seed))
        total += amplitude * noise(x * frequency, y * frequency)
        amplitude_sum += amplitude
        amplitude *= config.persistence
        frequency *= config.lacunarity

    if config.normalize:
        low, high = total.min(), total.max()
        if high > low:
            elevation = (total - low) / (high - low)
        else:
            elevation = np.full_like(total, 0.5)
    else:
        elevation = 0.5 + 0.5 * total / (amplitude_sum * _PERLIN_BOUND)

    elevation = np.clip(elevation, 0.0, 1.0) ** config.exponent
    return elevation.astype(np.float32)


def generate_hills(
    height: int,
    width: int,
    seed: int | None = None,
    hills: int = 3,
    radius: float = 2.0,
    resolution: int = 1,
    warp: float = 0.6,
) -> np.ndarray:
    """Generate a flat plain with a few smooth hills.

    The tallest hill has height 1, the others are clearly lower, and everywhere
    outside the hills is exactly 0. Hills are centred on cells. Their outlines
    are bent by a smooth random warp, so contours flow instead of forming
    perfect circles, and neighbouring hills can run into each other.

    Args:
        height: Number of rows of cells.
        width: Number of columns of cells.
        seed: Seed for hill positions and heights.
        hills: Number of hills.
        radius: Hill radius, in cells.
        resolution: Samples per cell along each side. The result has shape
            ``(height * resolution, width * resolution)``.
        warp: How far hill outlines are bent, in cells. 0 gives round hills.

    Returns:
        A ``float32`` array with values in [0, 1].
    """
    if height < 1 or width < 1:
        raise ValueError(f"map size must be positive, got {height}x{width}")
    if hills < 1:
        raise ValueError(f"hills must be at least 1, got {hills}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if warp < 0:
        raise ValueError(f"warp must be non-negative, got {warp}")

    rng = np.random.default_rng(seed)
    cells = height * width
    count = min(hills, cells)
    centers = None
    for _ in range(1000):
        picks = rng.choice(cells, size=count, replace=False)
        candidate = np.stack(np.unravel_index(picks, (height, width)), axis=-1) + 0.5
        gaps = np.linalg.norm(candidate[:, None] - candidate[None], axis=-1)
        if count == 1 or gaps[np.triu_indices(count, 1)].min() >= 1.6 * radius:
            centers = candidate
            break
    if centers is None:  # the map is too small to keep the hills apart
        centers = candidate

    # The first hill is the summit; the rest step down so the summit is clear.
    heights = np.concatenate(([1.0], np.sort(rng.uniform(0.4, 0.75, count - 1))[::-1]))

    rows = (np.arange(height * resolution) + 0.5) / resolution
    cols = (np.arange(width * resolution) + 0.5) / resolution
    y, x = np.meshgrid(rows, cols, indexing="ij")
    if warp > 0:
        # Two smooth noise fields nudge each point, bending the hill outlines.
        warp_x, warp_y = PerlinNoise2D(rng), PerlinNoise2D(rng)
        y, x = (
            y + warp * 2 * warp_y(x / 3.0, y / 3.0),
            x + warp * 2 * warp_x(x / 3.0, y / 3.0),
        )

    # Hills are added, so where two meet their contours merge smoothly.
    surface = np.zeros_like(x)
    for (row, col), peak in zip(centers, heights):
        distance = np.hypot(y - row, x - col) / radius
        surface += np.where(distance < 1.0, peak * np.cos(distance * np.pi / 2) ** 2, 0.0)
    surface /= surface.max()
    return surface.astype(np.float32)
