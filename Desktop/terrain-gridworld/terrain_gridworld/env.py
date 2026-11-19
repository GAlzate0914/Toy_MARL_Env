"""A Gymnasium grid world of procedurally generated terrain."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from terrain_gridworld.rendering import MatplotlibViewer, terrain_image
from terrain_gridworld.terrain import TerrainConfig, generate_terrain


class TerrainGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        terrain: TerrainConfig | dict[str, Any] | None = None,
        terrain_seed: int | None = None,
        render_mode: str | None = None,
        cell_size: int = 12,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError(f"map size must be positive, got {width}x{height}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        if cell_size < 1:
            raise ValueError(f"cell_size must be positive, got {cell_size}")

        self.width = width
        self.height = height
        self.terrain_config = (
            TerrainConfig(**terrain) if isinstance(terrain, dict) else terrain or TerrainConfig()
        )
        self.terrain_seed = terrain_seed
        self.render_mode = render_mode
        self.cell_size = cell_size

        self.observation_space = spaces.Box(0.0, 1.0, shape=(height, width), dtype=np.float32)
        self.action_space = spaces.Discrete(1)

        self.elevation: np.ndarray | None = None
        self._active_terrain_seed: int | None = None
        self._viewer = MatplotlibViewer()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        terrain_seed = options.get("terrain_seed", self.terrain_seed)
        if terrain_seed is None:
            terrain_seed = int(self.np_random.integers(2**31))
        self._active_terrain_seed = terrain_seed
        self.elevation = generate_terrain(
            self.height, self.width, seed=terrain_seed, config=self.terrain_config
        )

        if self.render_mode == "human":
            self.render()
        return self.elevation.copy(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.elevation is None:
            raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")

        if self.render_mode == "human":
            self.render()
        return self.elevation.copy(), 0.0, False, False, self._info()

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; nothing was drawn.")
            return None
        if self.elevation is None:
            raise RuntimeError("call reset() before render()")
        frame = terrain_image(self.elevation, self.cell_size)
        if self.render_mode == "rgb_array":
            return frame
        self._viewer.show(frame, self.metadata["render_fps"])
        return None

    def close(self) -> None:
        self._viewer.close()

    def _info(self) -> dict[str, Any]:
        return {"terrain_seed": self._active_terrain_seed}
