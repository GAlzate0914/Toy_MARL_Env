from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
PARK = 4
_MOVES = {LEFT: (0, -1), DOWN: (1, 0), RIGHT: (0, 1), UP: (-1, 0)}

Cell = tuple[int, int]

# A teammate's hill: don't start climbing within this many cells of a car that
# is already climbing.
_CLIMBER_RANGE = 4

# Taller hills are wider, so a known peak covers more ground the higher it is:
# about 1.5 cells for a small bump and 6 for a big hill.
_PEAK_RANGE_BASE = 1.0
_PEAK_RANGE_PER_HEIGHT = 7.0

# Searching cars try to stay at least this far apart.
_SPACING = 4

# Without a referee, drivers assume the ground never rises faster than this
# per cell (they raise it if they ever measure a steeper step).
DEFAULT_MAX_SLOPE = 0.65

# Battery planning: a typical climbing step gains about this much height, and
# a climb is only started with this much charge to spare.
_RISE_PER_STEP = 0.15
_CLIMB_MARGIN = 1.25


@dataclass
class SharedMap:
    elevations: dict[Cell, float] = field(default_factory=dict)
    visited: set[Cell] = field(default_factory=set)
    ruled_out: dict[Cell, set[Cell]] = field(default_factory=dict)
    peaks: dict[Cell, str] = field(default_factory=dict)
    positions: dict[int, Cell] = field(default_factory=dict)
    climbing: dict[int, bool] = field(default_factory=dict)
    targets: dict[int, Cell | None] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    messages: list[tuple[int, str, str]] = field(default_factory=list)
    step: int = 0
    steepest: float = 0.0
    finished: bool = False
    batteries: dict[int, float] = field(default_factory=dict)
    active: dict[int, bool] = field(default_factory=dict)
    flags: dict[Cell, str] = field(default_factory=dict)

    def clear(self) -> None:
        names = dict(self.names)
        self.__init__()
        self.names = names

    def say(self, car_id: int, text: str) -> None:
        self.messages.append((self.step, self.names.get(car_id, "Car"), text))


class ContourClimber:
    def __init__(
        self,
        nrow: int,
        ncol: int,
        contour_interval: float = 0.05,
        seed: int | None = None,
        *,
        shared: SharedMap | None = None,
        car_id: int = 0,
        name: str = "Car",
        referee: bool = True,
        max_slope: float = DEFAULT_MAX_SLOPE,
        capacity: float = float("inf"),
        move_cost: float = 0.0,
        climb_cost: float = 0.0,
    ) -> None:
        if contour_interval <= 0:
            raise ValueError(f"contour_interval must be positive, got {contour_interval}")
        if max_slope <= 0:
            raise ValueError(f"max_slope must be positive, got {max_slope}")
        self.referee = referee
        self.max_slope = max_slope
        self.capacity = capacity
        self.move_cost = move_cost
        self.climb_cost = climb_cost
        self.nrow = nrow
        self.ncol = ncol
        self.contour_interval = contour_interval
        self._seed = seed
        self._owns_map = shared is None
        self.shared = shared if shared is not None else SharedMap()
        self.car_id = car_id
        self.name = name
        self.shared.names[car_id] = name

    @classmethod
    def for_env(cls, env: Any, seed: int | None = None) -> ContourClimber:
        """Build a driver sized for a ``TerrainSummitEnv`` (wrapped or not)."""
        env = env.unwrapped
        return cls(
            env.nrow, env.ncol, env.contour_interval, seed, referee=env.referee,
            **battery_settings(env),
        )

    @property
    def elevations(self) -> dict[Cell, float]:
        return self.shared.elevations

    @property
    def visited(self) -> set[Cell]:
        return self.shared.visited

    @property
    def ruled_out(self) -> dict[Cell, set[Cell]]:
        return self.shared.ruled_out

    @property
    def peaks(self) -> list[Cell]:
        """False peaks this driver has found."""
        return [cell for cell, name in self.shared.peaks.items() if name == self.name]

    def reset(self, state: int, info: dict[str, Any]) -> None:
        if self._owns_map:
            self.shared.clear()
        self._rng = np.random.default_rng(self._seed)
        self.cell: Cell = self._to_cell(state)
        self.elevations[self.cell] = float(info["elevation"])
        self.visited.add(self.cell)
        self.mode = "searching"
        self._return_to: Cell | None = None
        self._last_move: int | None = None
        self._announced: set[str] = set()
        self.battery = float(info.get("battery", self.capacity))
        self._publish()

    def act(self) -> int:
        """Choose the next action."""
        if self._owns_map:
            self.shared.step += 1
        if self.mode in ("parked", "flat"):
            return PARK
        if not self.referee:
            self._open = self.open_cells()
            if not self._open:
                return self._finish_action()
        if self.mode == "backtracking":
            return self._toward(self._return_to)
        if self.mode == "climbing":
            reason = self._reason_to_leave_hill() or self._too_low_to_keep_climbing()
            if reason:
                self.shared.say(self.car_id, reason)
                self.mode = "searching"
            else:
                action = self._climb_action()
                if action is not None:
                    return action
                self._end_climb()
                self.mode = "searching"
            self._publish()
        return self._search_action()

    def observe(self, state: int, info: dict[str, Any]) -> None:
        """Update memory with the result of the last action."""
        previous, new = self.cell, self._to_cell(state)
        here, there = self.elevations[previous], float(info["elevation"])
        self.battery = float(info.get("battery", self.battery))
        if self.battery <= 0 and self.mode != "flat":
            self.mode = "flat"
            self.cell = new
            self.shared.say(self.car_id, f"Out of battery at {new}.")
            self._publish()
            return
        if self.mode in ("parked", "flat"):
            return
        self.elevations[new] = there
        move = self._move_between(previous, new)
        first_visit = new not in self.visited
        self.visited.add(new)
        self.cell = new

        if self.mode == "backtracking":
            # Back where it came from, or (after a slip) somewhere even higher.
            if new == self._return_to or there > self.elevations[self._return_to]:
                self.mode = "climbing"
                self._return_to = None
        elif self.mode == "climbing":
            if new != previous and there <= here:
                # Not going up: remember that, and reverse.
                self.ruled_out.setdefault(previous, set()).add(new)
                self.mode = "backtracking"
                self._return_to = previous
        elif self.mode == "searching" and first_visit and self._contour(there) > self._contour(here):
            reason = self._hill_already_covered(new)
            if reason is None and not self.referee and not self._could_beat_best_near(new):
                reason = "nothing on it could beat the best so far"
            if reason is None and not self._can_afford_climb(there):
                reason = f"with {self._percent()} charge I can't afford the climb; {self._flag_hill(new)}"
            if reason:
                # Say why a hill is skipped once, not at every contour line.
                if reason not in self._announced:
                    self._announced.add(reason)
                    self.shared.say(self.car_id, f"Crossed a contour at {new}, but {reason}. Moving on.")
            else:
                self.mode = "climbing"
                taken = self._take_over_flag(new)
                if taken:
                    self.shared.say(
                        self.car_id,
                        f"Crossed a contour at {new}. Taking the hill {taken} flagged "
                        f"({self._percent()} charge).",
                    )
                elif len(self.shared.names) > 1:
                    self.shared.say(self.car_id, f"Crossed a contour at {new}. Climbing this hill.")

        if move is not None:
            self._last_move = move
            self.shared.steepest = max(self.shared.steepest, abs(there - here))
        self._publish()

    # Working out where the summit is, without a referee

    def best_known(self) -> tuple[Cell, float]:
        """The highest cell anyone has visited, and its elevation."""
        cell = max(self.visited, key=lambda c: (self.elevations[c], c))
        return cell, self.elevations[cell]

    def slope_limit(self) -> float:
        """The steepest rise per cell the driver currently allows for."""
        return max(self.max_slope, 1.2 * self.shared.steepest)

    def open_cells(self) -> set[Cell]:
        """Unvisited cells that could still be higher than the best known cell."""
        visited = list(self.visited)
        heights = np.array([self.elevations[c] for c in visited])
        best = heights.max()
        rows, cols = np.divmod(np.arange(self.nrow * self.ncol), self.ncol)
        v_rows = np.array([c[0] for c in visited])
        v_cols = np.array([c[1] for c in visited])
        distance = np.abs(rows[:, None] - v_rows) + np.abs(cols[:, None] - v_cols)
        bound = (heights + self.slope_limit() * distance).min(axis=1)
        return {
            (int(r), int(c))
            for r, c, b in zip(rows, cols, bound)
            if b > best + 1e-9 and (int(r), int(c)) not in self.visited
        }

    def _could_beat_best_near(self, cell: Cell, radius: int = 2) -> bool:
        return any(_distance(cell, other) <= radius for other in self.open_cells())

    def _finish_action(self) -> int:
        """Nothing unexplored could be higher: the highest point is on record, so park."""
        best, height = self.best_known()
        if not self.shared.finished:
            self.shared.finished = True
            self.shared.say(
                self.car_id,
                f"Nothing unexplored can beat {best} at {height:.2f}. "
                "That's the highest point. Everyone park.",
            )
        self.mode = "parked"
        self.shared.say(self.car_id, f"Parking at {self.cell} with {self._percent()} charge.")
        self._publish()
        return PARK

    # Battery planning
    def _percent(self) -> str:
        if self.capacity == float("inf"):
            return "unlimited"
        return f"{100 * self.battery / self.capacity:.0f}%"

    def climb_estimate(self, height: float) -> float:
        """Charge needed to climb from ``height`` to above the best cell found."""
        _, best = self.best_known()
        rise = max(best - height, 0.0) + self.contour_interval
        steps = rise / _RISE_PER_STEP + 2
        return self.climb_cost * rise + self.move_cost * steps

    def _can_afford_climb(self, height: float) -> bool:
        return self.battery >= _CLIMB_MARGIN * self.climb_estimate(height)

    def _flag_hill(self, cell: Cell) -> str:
        """Flag an unaffordable hill for a teammate (once), and say what was done."""
        if len(self.shared.names) <= 1:
            return "nobody else can take it"
        for flag in self.shared.flags:
            if _distance(cell, flag) <= _CLIMBER_RANGE:
                return f"it's already flagged at {flag}"
        self.shared.flags[cell] = self.name
        return f"flagging {cell} for a teammate"

    def _take_over_flag(self, cell: Cell) -> str | None:
        """Clear a teammate's flag on the hill this car is starting to climb."""
        for flag, owner in list(self.shared.flags.items()):
            if _distance(cell, flag) <= _CLIMBER_RANGE and owner != self.name:
                del self.shared.flags[flag]
                return owner
        return None

    def _too_low_to_keep_climbing(self) -> str | None:
        steep_step = self.move_cost + self.climb_cost * 0.3
        if self.battery < steep_step:
            return f"Down to {self._percent()} charge, giving up this climb; {self._flag_hill(self.cell)}."
        return None

    def _end_climb(self) -> None:
        """Report why a climb ended: a peak, or nothing left worth checking."""
        here = self.elevations[self.cell]
        neighbours = [n for _, n in self._neighbours(self.cell)]
        is_peak = all(
            n in self.ruled_out.get(self.cell, set())
            or (n in self.elevations and self.elevations[n] <= here)
            for n in neighbours
        )
        if is_peak:
            self.shared.peaks[self.cell] = self.name
            if self.referee:
                self.shared.say(self.car_id, f"Topped a false peak at {self.cell}. Searching again.")
            else:
                best, height = self.best_known()
                note = "the highest so far" if best == self.cell else f"lower than {best}"
                self.shared.say(
                    self.car_id,
                    f"Topped a peak at {self.cell}, height {here:.2f}, {note}. Searching on.",
                )
        else:
            self.shared.say(
                self.car_id, "Nothing higher on this hill could beat the best so far. Searching on."
            )

    # Coordination
    def _publish(self) -> None:
        self.shared.positions[self.car_id] = self.cell
        self.shared.climbing[self.car_id] = self.mode in ("climbing", "backtracking")
        self.shared.batteries[self.car_id] = self.battery
        self.shared.active[self.car_id] = self.mode not in ("parked", "flat")
        if self.mode != "searching":
            self.shared.targets[self.car_id] = None

    def _teammates(self) -> list[int]:
        """Teammates still driving."""
        return [
            i for i in self.shared.positions
            if i != self.car_id and self.shared.active.get(i, True)
        ]

    def _hill_already_covered(self, cell: Cell) -> str | None:
        for other in self._teammates():
            if self.shared.climbing.get(other) and _distance(cell, self.shared.positions[other]) <= _CLIMBER_RANGE:
                return f"{self.shared.names[other]} is already climbing it"
        for peak, finder in self.shared.peaks.items():
            if _distance(cell, peak) <= self._peak_range(peak):
                who = "I" if finder == self.name else finder
                return f"{who} already topped it at {peak}"
        return None

    def _peak_range(self, peak: Cell) -> float:
        return _PEAK_RANGE_BASE + _PEAK_RANGE_PER_HEIGHT * self.elevations[peak]

    def _reason_to_leave_hill(self) -> str | None:
        for peak, finder in self.shared.peaks.items():
            # On the upper slopes of a hill that has already been topped.
            if _distance(self.cell, peak) <= max(2.0, 0.7 * self._peak_range(peak)):
                who = "I" if finder == self.name else finder
                return f"{who} already topped this hill at {peak}. Leaving it."
        for other in self._teammates():
            # Two cars on one hill: the one with more charge keeps it.
            if not (self.shared.climbing.get(other)
                    and _distance(self.cell, self.shared.positions[other]) <= 2):
                continue
            mine, theirs = self.battery, self.shared.batteries.get(other, 0.0)
            if theirs > mine or (theirs == mine and other < self.car_id):
                return (f"{self.shared.names[other]} is on this hill too, with more charge. "
                        "Leaving it to them.")
        return None

    # Climbing
    def _climb_action(self) -> int | None:
        here = self.elevations[self.cell]
        excluded = self.ruled_out.get(self.cell, set())
        known_higher, unknown = [], []
        for action, neighbour in self._neighbours(self.cell):
            if neighbour in excluded:
                continue
            if neighbour in self.elevations:
                if self.elevations[neighbour] > here:
                    known_higher.append((self.elevations[neighbour], action))
            elif self.referee or neighbour in self._open:
                unknown.append(action)
        if known_higher:
            return max(known_higher)[1]
        if unknown:
            return self._prefer_straight(unknown)
        return None

    # Searching
    def _search_action(self) -> int:
        others = [self.shared.positions[i] for i in self._teammates()]
        fresh = [
            (a, n) for a, n in self._neighbours(self.cell)
            if n not in self.visited and (self.referee or n in self._open)
        ]
        if fresh:
            self.shared.targets[self.car_id] = None
            if not others:
                return self._prefer_straight([a for a, _ in fresh])

            def score(option: tuple[int, Cell]) -> tuple[int, bool, float]:
                action, cell = option
                room = min(_distance(cell, other) for other in others)
                return (min(room, _SPACING), action == self._last_move, self._rng.random())

            return max(fresh, key=score)[0]

        target = self._choose_target(others)
        self.shared.targets[self.car_id] = target
        if target is None:
            return int(self._rng.integers(4))
        return self._toward(target)

    def _choose_target(self, others: list[Cell]) -> Cell | None:
        """The nearest unexplored cell, steering clear of teammates' areas."""
        if self.referee:
            unvisited = [
                (r, c) for r in range(self.nrow) for c in range(self.ncol)
                if (r, c) not in self.visited
            ]
        else:
            unvisited = sorted(self._open)
        if not unvisited:
            return None
        claims = [t for i, t in self.shared.targets.items()
                  if i != self.car_id and t is not None and self.shared.active.get(i, True)]
        busy = claims + others
        # Hills teammates flagged, if this car could afford to climb them.
        flags = [f for f, owner in self.shared.flags.items()
                 if owner != self.name and self._can_afford_climb(self.elevations.get(f, 0.0))]

        def cost(cell: Cell) -> tuple[float, Cell]:
            crowding = 0
            if busy:
                crowding = max(0, _SPACING - min(_distance(cell, b) for b in busy))
            pull = 8 if flags and min(_distance(cell, f) for f in flags) <= 2 else 0
            return (_distance(cell, self.cell) + 5 * crowding - pull, cell)

        return min(unvisited, key=cost)

    # Helpers
    def _prefer_straight(self, actions: list[int]) -> int:
        if self._last_move in actions:
            return self._last_move
        return int(self._rng.choice(actions))

    def _toward(self, target: Cell) -> int:
        d_row, d_col = target[0] - self.cell[0], target[1] - self.cell[1]
        if abs(d_row) >= abs(d_col) and d_row != 0:
            return DOWN if d_row > 0 else UP
        if d_col != 0:
            return RIGHT if d_col > 0 else LEFT
        return int(self._rng.integers(4))

    def _neighbours(self, cell: Cell) -> list[tuple[int, Cell]]:
        result = []
        for action, (d_row, d_col) in _MOVES.items():
            row, col = cell[0] + d_row, cell[1] + d_col
            if 0 <= row < self.nrow and 0 <= col < self.ncol:
                result.append((action, (row, col)))
        return result

    def _contour(self, elevation: float) -> int:
        return math.floor(elevation / self.contour_interval + 1e-9)

    def _to_cell(self, state: int) -> Cell:
        return divmod(int(state), self.ncol)

    @staticmethod
    def _move_between(a: Cell, b: Cell) -> int | None:
        for action, offset in _MOVES.items():
            if (b[0] - a[0], b[1] - a[1]) == offset:
                return action
        return None


class CarTeam:
    def __init__(
        self,
        nrow: int,
        ncol: int,
        names: tuple[str, ...] = ("Red", "Yellow", "Blue"),
        contour_interval: float = 0.05,
        seed: int | None = None,
        communicate: bool = True,
        referee: bool = True,
        capacity: float = float("inf"),
        move_cost: float = 0.0,
        climb_cost: float = 0.0,
    ) -> None:
        self.communicate = communicate
        self.referee = referee
        self.shared = SharedMap()
        self.cars = [
            ContourClimber(
                nrow, ncol, contour_interval,
                seed=None if seed is None else seed + i,
                shared=self.shared if communicate else None,
                car_id=i, name=name, referee=referee,
                capacity=capacity, move_cost=move_cost, climb_cost=climb_cost,
            )
            for i, name in enumerate(names)
        ]

    @classmethod
    def for_env(cls, env: Any, seed: int | None = None, communicate: bool = True) -> CarTeam:
        """Build a team sized for a ``MultiCarSummitEnv`` (wrapped or not)."""
        env = env.unwrapped
        return cls(
            env.nrow, env.ncol, env.car_names, env.contour_interval, seed, communicate,
            referee=env.referee, **battery_settings(env),
        )

    @property
    def messages(self) -> list[tuple[int, str, str]]:
        """Radio messages, as ``(step, name, text)``."""
        if self.communicate:
            return self.shared.messages
        return sorted(m for car in self.cars for m in car.shared.messages)

    @property
    def modes(self) -> list[str]:
        return [car.mode for car in self.cars]

    def reset(self, states: np.ndarray, info: dict[str, Any]) -> None:
        self.shared.clear()
        for car, info_i in zip(self.cars, _per_car(states, info)):
            car.reset(*info_i)

    def act(self) -> np.ndarray:
        self.shared.step += 1
        for car in self.cars:
            car.shared.step = self.shared.step
        return np.array([car.act() for car in self.cars], dtype=np.int64)

    def observe(self, states: np.ndarray, info: dict[str, Any]) -> None:
        for car, info_i in zip(self.cars, _per_car(states, info)):
            car.observe(*info_i)


def battery_settings(env: Any) -> dict[str, float]:
    """The battery rules of an environment, for building drivers."""
    return {"capacity": env.capacity, "move_cost": env.move_cost, "climb_cost": env.climb_cost}


def _per_car(states: np.ndarray, info: dict[str, Any]) -> list[tuple[int, dict[str, float]]]:
    batteries = info.get("batteries", [float("inf")] * len(states))
    return [
        (int(state), {"elevation": float(elevation), "battery": float(battery)})
        for state, elevation, battery in zip(states, info["elevations"], batteries)
    ]


def _distance(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
