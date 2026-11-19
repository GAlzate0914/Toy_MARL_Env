"""A toy-text environment: drive an RC car to the highest point of the terrain."""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import colorize

from terrain_gridworld.rendering import MatplotlibViewer, draw_marker, terrain_image
from terrain_gridworld.terrain import TerrainConfig, generate_hills, generate_terrain
from terrain_gridworld.topo import draw_battery, draw_car, draw_record, topo_background

# Same action numbering as FrozenLake and the other toy-text environments.
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
# Fifth action: park the car, switching it off for the rest of the episode.
PARK = 4
_MOVES = {LEFT: (0, -1), DOWN: (1, 0), RIGHT: (0, 1), UP: (-1, 0)}
_ACTION_NAMES = {LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up", PARK: "Park"}
_HEADINGS = {LEFT: "left", DOWN: "down", RIGHT: "right", UP: "up"}
_OPPOSITE = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}

# Fallback agent marker for cells too small for the car sprite.
_AGENT_COLOR = (220, 40, 40)
# A car with an empty battery is drawn grey.
_DEAD_COLOR = (120, 120, 124)

# Perlin settings for terrain="noise": landforms a few cells across.
SUMMIT_TERRAIN = TerrainConfig(scale=3.0, octaves=5)
# Gentle rolling ground between the hills: smooth enough for clean contours.
GROUND_TERRAIN = TerrainConfig(scale=3.0, octaves=3, persistence=0.45)


def steer(
    heading: str,
    last_move: int | None,
    cell: tuple[int, int],
    next_cell: tuple[int, int],
    action: int,
) -> tuple[str, int | None]:
    """Work out which way a car faces after a move.

    A move that undoes the previous one is drawn as the car reversing, so it
    keeps facing the same way. Bumping into the edge turns the car to face the
    intended direction. Returns the new heading and the move to remember.
    """
    delta = (next_cell[0] - cell[0], next_cell[1] - cell[1])
    for move, offset in _MOVES.items():
        if offset == delta:
            if last_move is not None and move == _OPPOSITE[last_move]:
                return heading, None  # reversing; the next move drives forward
            return _HEADINGS[move], move
    return _HEADINGS[action], None


def battery_text(battery: float, capacity: float) -> str:
    if capacity == float("inf"):
        return "unlimited"
    return f"{battery:.1f}/{capacity:g} ({100 * battery / capacity:.0f}%)"


class TerrainSummitEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        terrain: str | TerrainConfig | dict[str, Any] = "hills",
        hills: int = 3,
        hill_radius: float = 3.5,
        ground: float = 0.1,
        terrain_seed: int | None = 0,
        start: str | tuple[int, int] = "corner",
        is_slippery: bool = False,
        referee: bool = False,
        battery: float | None = 100.0,
        move_cost: float = 0.5,
        climb_cost: float = 30.0,
        render_mode: str | None = None,
        map_style: str = "terrain",
        resolution: int = 48,
    ) -> None:
        if width < 1 or height < 1 or width * height < 2:
            raise ValueError(f"map needs at least 2 cells, got {width}x{height}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        if resolution < 1:
            raise ValueError(f"resolution must be positive, got {resolution}")
        if not 0 <= ground < 1:
            raise ValueError(f"ground must be in [0, 1), got {ground}")
        if map_style not in ("terrain", "lines"):
            raise ValueError(f"unsupported map_style: {map_style!r}")
        if battery is not None and battery <= 0:
            raise ValueError(f"battery must be positive or None, got {battery}")
        if move_cost < 0 or climb_cost < 0:
            raise ValueError("move_cost and climb_cost must not be negative")

        self.nrow, self.ncol = height, width
        if terrain == "hills":
            self.terrain_config = None
        elif terrain == "noise":
            self.terrain_config = SUMMIT_TERRAIN
        elif isinstance(terrain, dict):
            self.terrain_config = TerrainConfig(**terrain)
        elif isinstance(terrain, TerrainConfig):
            self.terrain_config = terrain
        else:
            raise ValueError(f"unsupported terrain: {terrain!r}")
        if terrain_seed is None:
            terrain_seed = int(np.random.default_rng().integers(2**31))
        self.terrain_seed = terrain_seed
        self.is_slippery = is_slippery
        self.referee = referee
        self.capacity = float("inf") if battery is None else float(battery)
        self.move_cost = float(move_cost)
        self.climb_cost = float(climb_cost)
        self.render_mode = render_mode
        self.resolution = resolution
        self.map_style = map_style
        # Elevation between drawn contour lines.
        self.contour_interval = 0.05 if map_style == "terrain" else 0.1

        if self.terrain_config is None:
            hill_surface = generate_hills(
                height, width, terrain_seed, hills, hill_radius, resolution
            )
            self.surface = hill_surface
            if ground > 0:
                rolling = generate_terrain(
                    height * resolution,
                    width * resolution,
                    seed=terrain_seed,
                    config=dataclasses.replace(GROUND_TERRAIN, scale=GROUND_TERRAIN.scale * resolution),
                )
                # Fading the ground out towards the hill tops keeps the tallest
                # hill's peak at exactly 1, so it is always the summit.
                self.surface = hill_surface + ground * rolling * (1.0 - hill_surface)
        else:
            # The fine surface is sampled `resolution` times per cell, so the
            # scale is converted from cells to samples.
            fine_config = dataclasses.replace(
                self.terrain_config, scale=self.terrain_config.scale * resolution
            )
            self.surface = generate_terrain(
                height * resolution, width * resolution, seed=terrain_seed, config=fine_config
            )
        self.elevation = self.surface.reshape(height, resolution, width, resolution).max(axis=(1, 3))
        self.goal = tuple(int(i) for i in np.unravel_index(np.argmax(self.elevation), (height, width)))
        if float(self.elevation.max()) == float(self.elevation.min()):
            raise ValueError("terrain is flat, so it has no single highest point")

        if start == "corner":
            corners = np.array([(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)])
            distance = np.abs(corners - np.array(self.goal)).sum(axis=1)
            self.start: tuple[int, int] | None = tuple(int(i) for i in corners[np.argmax(distance)])
        elif start == "lowest":
            lowest = np.argwhere(self.elevation == self.elevation.min())
            distance = np.abs(lowest - np.array(self.goal)).sum(axis=1)
            self.start = tuple(int(i) for i in lowest[np.argmax(distance)])
        elif start == "random":
            self.start = None
        else:
            row, col = start
            if not (0 <= row < height and 0 <= col < width):
                raise ValueError(f"start {start} is outside the {height}x{width} map")
            if (row, col) == self.goal:
                raise ValueError(f"start {start} is the summit")
            self.start = (int(row), int(col))

        self.observation_space = spaces.Discrete(height * width)
        self.action_space = spaces.Discrete(5)
        self.P = self._build_transitions()

        self.s: int | None = None
        self.lastaction: int | None = None
        self.heading = "right"
        self._last_move: int | None = None
        self.battery = self.capacity
        self.parked = False
        self.recorded: tuple[int, int] | None = None
        self.summit_recorded = False
        self._background: np.ndarray | None = None
        self._viewer = MatplotlibViewer()

    def to_state(self, row: int, col: int) -> int:
        return row * self.ncol + col

    def to_cell(self, state: int) -> tuple[int, int]:
        return divmod(int(state), self.ncol)

    def battery_fraction(self, battery: float) -> float | None:
        """Charge left as a fraction of capacity, or None if unlimited."""
        return None if self.capacity == float("inf") else battery / self.capacity

    def move_charge(self, state: int, next_state: int) -> float:
        """Battery charge used by driving from one cell to another."""
        climb = float(self.elevation[self.to_cell(next_state)] - self.elevation[self.to_cell(state)])
        return self.move_cost + self.climb_cost * max(climb, 0.0)

    def _record(self, state: int) -> bool:
        """Update the record of the highest cell visited; True if it is newly the summit."""
        cell = self.to_cell(state)
        if self.recorded is None or self.elevation[cell] > self.elevation[self.recorded]:
            self.recorded = cell
        if cell == self.goal and not self.summit_recorded:
            self.summit_recorded = True
            return True
        return False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        if self.start is None:
            goal_state = self.to_state(*self.goal)
            state = int(self.np_random.integers(self.observation_space.n - 1))
            self.s = state + (state >= goal_state)
        else:
            self.s = self.to_state(*self.start)
        self.lastaction = None
        self.heading = "right"
        self._last_move = None
        self.battery = self.capacity
        self.parked = False
        self.recorded = None
        self.summit_recorded = False
        self._record(self.s)

        if self.render_mode == "human":
            self.render()
        return self.s, self._info(1.0)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self.s is None:
            raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")

        action = int(action)
        self.lastaction = action
        prob, stalled, reward = 1.0, False, 0.0
        if self.parked or self.battery <= 0:
            pass  # a parked or flat car does not move
        elif action == PARK:
            self.parked = True
        else:
            transitions = self.P[self.s][action]
            index = self.np_random.choice(len(transitions), p=[t[0] for t in transitions])
            prob, next_state = transitions[index][0], transitions[index][1]
            charge = self.move_charge(self.s, next_state)
            if charge > self.battery:
                self.battery = 0.0
                stalled = True
            else:
                self.battery -= charge
                self._update_heading(self.s, next_state, action)
                self.s = next_state
                reward = float(self._record(next_state))

        terminated = (
            self.parked
            or self.battery <= 0
            or (self.referee and self.to_cell(self.s) == self.goal)
        )
        info = self._info(prob)
        info["stalled"] = stalled
        if self.render_mode == "human":
            self.render()
        return self.s, reward, bool(terminated), False, info

    def render(self) -> str | np.ndarray | None:
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; nothing was drawn.")
            return None
        if self.s is None:
            raise RuntimeError("call reset() before render()")
        if self.render_mode == "ansi":
            return self._render_text()
        frame = self._render_frame()
        if self.render_mode == "rgb_array":
            return frame
        self._viewer.show(frame, self.metadata["render_fps"])
        return None

    def close(self) -> None:
        self._viewer.close()

    def _build_transitions(self) -> dict[int, dict[int, list[tuple[float, int, float, bool]]]]:
        goal_state = self.to_state(*self.goal)
        transitions = {}
        for state in range(self.nrow * self.ncol):
            transitions[state] = {}
            for action in range(4):
                if self.referee and state == goal_state:
                    transitions[state][action] = [(1.0, state, 0.0, True)]
                    continue
                if self.is_slippery:
                    outcomes = [((action - 1) % 4, 1 / 3), (action, 1 / 3), ((action + 1) % 4, 1 / 3)]
                else:
                    outcomes = [(action, 1.0)]
                transitions[state][action] = []
                for move, prob in outcomes:
                    next_state = self._move(state, move)
                    reached = next_state == goal_state
                    transitions[state][action].append(
                        (prob, next_state, float(reached), bool(self.referee and reached))
                    )
            transitions[state][PARK] = [(1.0, state, 0.0, True)]
        return transitions

    def _move(self, state: int, action: int) -> int:
        row, col = self.to_cell(state)
        d_row, d_col = _MOVES[action]
        row = min(max(row + d_row, 0), self.nrow - 1)
        col = min(max(col + d_col, 0), self.ncol - 1)
        return self.to_state(row, col)

    def _update_heading(self, state: int, next_state: int, action: int) -> None:
        self.heading, self._last_move = steer(
            self.heading, self._last_move, self.to_cell(state), self.to_cell(next_state), action
        )

    def _info(self, prob: float) -> dict[str, Any]:
        return {
            "prob": prob,
            "elevation": float(self.elevation[self.to_cell(self.s)]),
            "battery": self.battery,
            "parked": self.parked,
            "recorded": self.recorded,
            "recorded_elevation": float(self.elevation[self.recorded]),
            "summit_recorded": self.summit_recorded,
        }

    def _render_text(self) -> str:
        """One character per cell: elevation from 0 to 9, S start, G summit."""
        levels = np.minimum((self.elevation * 10).astype(int), 9)
        grid = [[str(level) for level in row] for row in levels]
        if self.start is not None:
            grid[self.start[0]][self.start[1]] = "S"
        grid[self.goal[0]][self.goal[1]] = "G"
        row, col = self.to_cell(self.s)
        grid[row][col] = colorize(grid[row][col], "red", highlight=True)

        lines = []
        if self.lastaction is not None:
            lines.append(f"  ({_ACTION_NAMES[self.lastaction]})")
        lines.extend("".join(line) for line in grid)
        lines.append(f"Battery: {battery_text(self.battery, self.capacity)}")
        return "\n".join(lines) + "\n"

    def _map_background(self) -> np.ndarray:
        """The rendered map without cars. It never changes, so it is drawn once."""
        if self._background is None:
            base = None
            if self.map_style == "terrain":
                # Low ground is grassland (no water); hills rise through
                # forest and rock to snow.
                shade = np.interp(self.surface, [0.0, 0.3, 1.0], [0.4, 0.55, 1.0])
                base = terrain_image(shade, 1, relief=0.35 * self.resolution)
            self._background = topo_background(
                self.surface, self.resolution, self.goal, self.start, base=base,
                interval=self.contour_interval,
            )
        return self._background

    def _render_frame(self) -> np.ndarray:
        frame = self._map_background().copy()
        if self.recorded is not None:
            draw_record(frame, self.recorded, self.resolution)
        cell = self.to_cell(self.s)
        body = _DEAD_COLOR if self.battery <= 0 else None
        if not draw_car(frame, cell, self.heading, self.resolution, body=body):
            draw_marker(frame, cell, _AGENT_COLOR, self.resolution, radius=0.3)
        draw_battery(frame, cell, self.battery_fraction(self.battery), self.resolution)
        return frame
