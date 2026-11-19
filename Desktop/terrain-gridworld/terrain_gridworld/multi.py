"""Several RC cars searching the same terrain for its summit."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import colorize

from terrain_gridworld.rendering import draw_marker
from terrain_gridworld.summit import _DEAD_COLOR, PARK, TerrainSummitEnv, battery_text, steer
from terrain_gridworld.topo import draw_battery, draw_car, draw_record

CAR_NAMES = ("Red", "Yellow", "Blue", "Green")
CAR_COLORS = {
    "Red": (214, 48, 49),
    "Yellow": (245, 197, 24),
    "Blue": (40, 112, 224),
    "Green": (52, 168, 83),
}
_TEXT_COLORS = {"Red": "red", "Yellow": "yellow", "Blue": "blue", "Green": "green"}


class MultiCarSummitEnv(TerrainSummitEnv):
    def __init__(self, n_cars: int = 3, start: str = "corners", **kwargs: Any) -> None:
        if not 1 <= n_cars <= len(CAR_NAMES):
            raise ValueError(f"n_cars must be 1 to {len(CAR_NAMES)}, got {n_cars}")
        if start not in ("corners", "random"):
            raise ValueError(f"start must be 'corners' or 'random', got {start!r}")
        super().__init__(start="random", **kwargs)

        self.n_cars = n_cars
        self.car_names = CAR_NAMES[:n_cars]
        self.start_mode = start
        n_cells = self.nrow * self.ncol
        if n_cars > n_cells - 1:
            raise ValueError(f"a {self.nrow}x{self.ncol} map has no room for {n_cars} cars")

        self.starts: list[tuple[int, int]] | None = None
        if start == "corners":
            corners = sorted(
                {(0, 0), (0, self.ncol - 1), (self.nrow - 1, 0), (self.nrow - 1, self.ncol - 1)}
                - {self.goal},
                key=lambda c: (-(abs(c[0] - self.goal[0]) + abs(c[1] - self.goal[1])), c),
            )
            if len(corners) < n_cars:
                raise ValueError(f"not enough free corners for {n_cars} cars")
            self.starts = corners[:n_cars]

        self.observation_space = spaces.MultiDiscrete([n_cells] * n_cars)
        self.action_space = spaces.MultiDiscrete([5] * n_cars)
        self.positions = np.zeros(n_cars, dtype=np.int64)
        self.batteries = np.full(n_cars, self.capacity)
        self.parked_cars = np.zeros(n_cars, dtype=bool)
        self.headings = ["right"] * n_cars
        self._last_moves: list[int | None] = [None] * n_cars
        self._on_top: int | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        if self.starts is not None:
            cells = [self.to_state(*cell) for cell in self.starts]
        else:
            goal_state = self.to_state(*self.goal)
            choices = [s for s in range(self.nrow * self.ncol) if s != goal_state]
            cells = self.np_random.choice(choices, size=self.n_cars, replace=False)
        self.positions = np.array(cells, dtype=np.int64)
        self.s = int(self.positions[0])
        self.lastaction = None
        self.headings = ["right"] * self.n_cars
        self._last_moves = [None] * self.n_cars
        self._on_top = None
        self.batteries = np.full(self.n_cars, self.capacity)
        self.parked_cars = np.zeros(self.n_cars, dtype=bool)
        self.recorded = None
        self.summit_recorded = False
        for state in self.positions:
            self._record(int(state))

        if self.render_mode == "human":
            self.render()
        return self.positions.copy(), self._team_info(None, [])

    def step(
        self, action: np.ndarray | list[int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.s is None:
            raise RuntimeError("call reset() before step()")
        actions = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(actions):
            raise ValueError(f"invalid action: {action!r}")

        winner = None
        stalled = []
        for i, car_action in enumerate(actions):
            state = int(self.positions[i])
            if self.parked_cars[i] or self.batteries[i] <= 0:
                continue  # parked or flat: stays put
            if int(car_action) == PARK:
                self.parked_cars[i] = True
                continue
            transitions = self.P[state][int(car_action)]
            index = self.np_random.choice(len(transitions), p=[t[0] for t in transitions])
            next_state = transitions[index][1]
            charge = self.move_charge(state, next_state)
            if charge > self.batteries[i]:
                self.batteries[i] = 0.0
                stalled.append(i)
                continue
            self.batteries[i] -= charge
            self.headings[i], self._last_moves[i] = steer(
                self.headings[i], self._last_moves[i],
                self.to_cell(state), self.to_cell(next_state), int(car_action),
            )
            self.positions[i] = next_state
            if self._record(next_state):
                winner = i
        self.s = int(self.positions[0])
        self.lastaction = actions.tolist()

        all_stopped = bool(np.all(self.parked_cars | (self.batteries <= 0)))
        on_summit = any(self.to_cell(s) == self.goal for s in self.positions)
        terminated = all_stopped or (self.referee and on_summit)
        reward = float(winner is not None)
        # Draw the car that recorded the summit on top of any car sharing its cell.
        if winner is not None:
            self._on_top = winner
        if self.render_mode == "human":
            self.render()
        return self.positions.copy(), reward, terminated, False, self._team_info(winner, stalled)

    def _team_info(self, winner: int | None, stalled: list[int]) -> dict[str, Any]:
        rows, cols = np.divmod(self.positions, self.ncol)
        return {
            "elevations": self.elevation[rows, cols].astype(np.float64),
            "batteries": self.batteries.copy(),
            "parked": self.parked_cars.copy(),
            "recorded": self.recorded,
            "recorded_elevation": float(self.elevation[self.recorded]),
            "summit_recorded": self.summit_recorded,
            "winner": winner,
            "stalled": stalled,
        }

    def _render_text(self) -> str:
        """One character per cell: elevation 0 to 9, G summit, a letter per car."""
        levels = np.minimum((self.elevation * 10).astype(int), 9)
        grid = [[str(level) for level in row] for row in levels]
        grid[self.goal[0]][self.goal[1]] = "G"
        for name, state in zip(self.car_names, self.positions):
            row, col = self.to_cell(state)
            grid[row][col] = colorize(name[0], _TEXT_COLORS[name], bold=True)
        lines = ["".join(line) for line in grid]
        for name, battery, parked in zip(self.car_names, self.batteries, self.parked_cars):
            status = " (parked)" if parked else " (flat)" if battery <= 0 else ""
            lines.append(f"{name}: {battery_text(battery, self.capacity)}{status}")
        return "\n".join(lines) + "\n"

    def _render_frame(self) -> np.ndarray:
        frame = self._map_background().copy()
        if self.recorded is not None:
            draw_record(frame, self.recorded, self.resolution)
        order = list(range(self.n_cars))
        if self._on_top is not None:
            order.remove(self._on_top)
            order.append(self._on_top)
        for i in order:
            name, state, heading = self.car_names[i], self.positions[i], self.headings[i]
            cell = self.to_cell(state)
            body = _DEAD_COLOR if self.batteries[i] <= 0 else CAR_COLORS[name]
            if not draw_car(frame, cell, heading, self.resolution, body=body):
                draw_marker(frame, cell, body, self.resolution, radius=0.3)
            draw_battery(frame, cell, self.battery_fraction(self.batteries[i]), self.resolution)
        return frame
