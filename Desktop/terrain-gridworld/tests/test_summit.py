import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import terrain_gridworld  # noqa: F401  (registers the environments)
from terrain_gridworld import TerrainSummitEnv
from terrain_gridworld.summit import DOWN, LEFT, PARK, RIGHT, UP
from terrain_gridworld.topo import CAR_BODY_COLOR, draw_car


@pytest.mark.parametrize("render_mode", ["rgb_array", "ansi"])
@pytest.mark.parametrize("start", ["lowest", "random"])
def test_passes_gymnasium_env_checker(render_mode, start):
    env = gym.make("TerrainSummit-v0", render_mode=render_mode, start=start)
    check_env(env.unwrapped, skip_render_check=False)
    env.close()


def test_spaces_and_toy_text_style_registration():
    env = gym.make("TerrainSummit-v0", width=20, height=10)
    assert env.observation_space == gym.spaces.Discrete(200)
    assert env.action_space == gym.spaces.Discrete(5)
    assert env.spec.max_episode_steps == 1000
    env.close()
    assert TerrainSummitEnv(referee=True, resolution=8).action_space == gym.spaces.Discrete(5)


def test_goal_is_highest_cell_and_start_is_farthest_corner():
    for seed in range(10):
        env = TerrainSummitEnv(terrain_seed=seed, resolution=8)
        assert env.elevation[env.goal] == env.elevation.max()
        assert env.start[0] in (0, 15) and env.start[1] in (0, 15)
        distance = abs(env.start[0] - env.goal[0]) + abs(env.start[1] - env.goal[1])
        assert distance >= 15
        state, info = env.reset(seed=0)
        assert env.to_cell(state) == env.start
        assert info["elevation"] == env.elevation[env.start]


def test_lowest_start_option():
    env = TerrainSummitEnv(start="lowest", resolution=8)
    assert env.elevation[env.start] == env.elevation.min()


def test_map_is_fixed_and_reproducible():
    assert np.array_equal(TerrainSummitEnv().elevation, TerrainSummitEnv().elevation)
    assert not np.array_equal(
        TerrainSummitEnv(terrain_seed=1).elevation, TerrainSummitEnv(terrain_seed=2).elevation
    )
    env = TerrainSummitEnv()
    before = env.elevation.copy()
    env.reset(seed=1)
    env.reset(seed=2)
    assert np.array_equal(env.elevation, before)


def test_random_seed_map_is_recorded():
    env = TerrainSummitEnv(terrain_seed=None)
    assert np.array_equal(env.elevation, TerrainSummitEnv(terrain_seed=env.terrain_seed).elevation)


def test_defaults_to_16_by_16_with_a_fine_surface():
    env = TerrainSummitEnv()
    assert env.elevation.shape == (16, 16)
    assert env.surface.shape == (768, 768)
    assert env.elevation[env.goal] == env.surface.max()


def test_cell_elevation_is_the_highest_point_in_its_patch():
    env = TerrainSummitEnv(resolution=16)
    assert env.elevation[2, 5] == env.surface[32:48, 80:96].max()


def test_random_start_never_starts_on_the_summit():
    env = TerrainSummitEnv(width=2, height=1, start="random")
    for seed in range(50):
        state, _ = env.reset(seed=seed)
        assert env.to_cell(state) != env.goal


def test_custom_start():
    env = TerrainSummitEnv(start=(3, 4))
    state, _ = env.reset(seed=0)
    assert env.to_cell(state) == (3, 4)


@pytest.mark.parametrize("start", [(-1, 0), (0, 16), "middle"])
def test_invalid_start_is_rejected(start):
    with pytest.raises((ValueError, TypeError)):
        TerrainSummitEnv(start=start)


def test_start_on_summit_is_rejected():
    goal = TerrainSummitEnv().goal
    with pytest.raises(ValueError):
        TerrainSummitEnv(start=goal)


@pytest.mark.parametrize(
    ("action", "expected"),
    [(LEFT, (5, 4)), (DOWN, (6, 5)), (RIGHT, (5, 6)), (UP, (4, 5))],
)
def test_non_slippery_moves(action, expected):
    env = TerrainSummitEnv(is_slippery=False, start=(5, 5), terrain_seed=3)
    env.reset(seed=0)
    state, reward, terminated, truncated, info = env.step(action)
    assert env.to_cell(state) == expected
    assert (reward, terminated, truncated, info["prob"]) == (0.0, False, False, 1.0)


def test_edge_bump_stays_put():
    env = TerrainSummitEnv(is_slippery=False, start=(0, 0), terrain_seed=3)
    env.reset(seed=0)
    state, *_ = env.step(UP)
    assert env.to_cell(state) == (0, 0)


def test_slippery_moves_match_frozenlake():
    env = TerrainSummitEnv(is_slippery=True, start=(5, 5), terrain_seed=3)
    outcomes = {env.to_cell(s) for _, s, _, _ in env.P[env.to_state(5, 5)][RIGHT]}
    assert outcomes == {(4, 5), (5, 6), (6, 5)}  # up, right and down

    counts = {}
    for seed in range(3000):
        env.reset(seed=seed)
        state, *_ = env.step(RIGHT)
        counts[env.to_cell(state)] = counts.get(env.to_cell(state), 0) + 1
    assert set(counts) == outcomes
    assert all(abs(n / 3000 - 1 / 3) < 0.04 for n in counts.values())


def beside_the_summit():
    row, col = TerrainSummitEnv(resolution=8).goal
    return ((row, col - 1), RIGHT) if col > 0 else ((row, col + 1), LEFT)


def test_with_a_referee_reaching_the_summit_pays_one_and_ends():
    start, action = beside_the_summit()
    env = TerrainSummitEnv(start=start, referee=True, resolution=8)
    env.reset(seed=0)
    state, reward, terminated, truncated, _ = env.step(action)
    assert env.to_cell(state) == env.goal
    assert (reward, terminated, truncated) == (1.0, True, False)


def test_reaching_the_summit_records_it_and_carries_on():
    start, action = beside_the_summit()
    env = TerrainSummitEnv(start=start, resolution=8)
    _, info = env.reset(seed=0)
    assert not info["summit_recorded"] and info["recorded"] == start
    state, reward, terminated, _, info = env.step(action)
    assert env.to_cell(state) == env.goal
    assert (reward, terminated) == (1.0, False)
    assert info["summit_recorded"] and info["recorded"] == env.goal
    assert info["recorded_elevation"] == env.elevation.max()
    # Driving away keeps the record and pays nothing more.
    back = LEFT if action == RIGHT else RIGHT
    state, reward, terminated, _, info = env.step(back)
    assert env.to_cell(state) == start
    assert (reward, terminated, info["recorded"]) == (0.0, False, env.goal)
    _, reward, _, _, _ = env.step(action)
    assert reward == 0.0  # only the first visit pays


def test_record_tracks_the_highest_cell_visited():
    env = TerrainSummitEnv(start=(8, 8), resolution=8, battery=None)
    env.reset(seed=0)
    seen = [(8, 8)]
    for action in [RIGHT, RIGHT, DOWN, LEFT, UP, UP]:
        state, _, _, _, info = env.step(action)
        seen.append(env.to_cell(state))
        assert info["recorded"] == max(seen, key=lambda c: env.elevation[c])


def test_parking_ends_the_episode():
    env = TerrainSummitEnv(resolution=8)
    state, _ = env.reset(seed=0)
    after, reward, terminated, _, info = env.step(PARK)
    assert (after, reward, terminated, info["parked"]) == (state, 0.0, True, True)
    assert info["battery"] == 150.0  # parking is free


# Battery


def test_moves_drain_the_battery_by_distance_and_climb():
    env = TerrainSummitEnv(start=(5, 5), terrain_seed=3, resolution=8, move_cost=0.5, climb_cost=30)
    env.reset(seed=0)
    for action in [RIGHT, DOWN, LEFT, UP, UP]:
        before_cell, before = env.to_cell(env.s), env.battery
        state, _, _, _, info = env.step(action)
        climb = max(float(env.elevation[env.to_cell(state)] - env.elevation[before_cell]), 0.0)
        assert before - info["battery"] == pytest.approx(0.5 + 30 * climb)


def test_downhill_and_flat_cost_only_the_move():
    env = TerrainSummitEnv(resolution=8)
    env.reset(seed=0)
    env.elevation = np.zeros_like(env.elevation)
    _, _, _, _, info = env.step(RIGHT)
    assert info["battery"] == pytest.approx(150 - 0.5)


def test_edge_bump_costs_a_move():
    env = TerrainSummitEnv(start=(0, 0), terrain_seed=3, resolution=8)
    env.reset(seed=0)
    _, _, _, _, info = env.step(UP)
    assert info["battery"] == pytest.approx(150 - 0.5)


def test_a_move_the_battery_cannot_pay_for_stalls_and_ends_the_episode():
    env = TerrainSummitEnv(start=(5, 5), terrain_seed=3, resolution=8, battery=1.0, climb_cost=1000)
    state, _ = env.reset(seed=0)
    neighbours = {RIGHT: (5, 6), DOWN: (6, 5), LEFT: (5, 4), UP: (4, 5)}
    uphill = max(neighbours, key=lambda a: env.elevation[neighbours[a]])
    assert env.elevation[neighbours[uphill]] > env.elevation[5, 5]
    after, _, terminated, _, info = env.step(uphill)
    assert after == state and terminated and info["stalled"]
    assert info["battery"] == 0.0


def test_running_flat_ends_the_episode():
    env = TerrainSummitEnv(resolution=8, battery=1.0)
    env.reset(seed=0)
    env.elevation = np.zeros_like(env.elevation)
    _, _, terminated, _, info = env.step(RIGHT)
    assert not terminated and info["battery"] == pytest.approx(0.5)
    _, _, terminated, _, info = env.step(RIGHT)
    assert terminated and info["battery"] == 0.0


def test_unlimited_battery_never_runs_out():
    env = TerrainSummitEnv(resolution=8, battery=None, render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(300):
        _, _, terminated, _, info = env.step(UP)
        assert not terminated
    assert info["battery"] == float("inf")
    env.render()


@pytest.mark.parametrize("kwargs", [{"battery": 0}, {"battery": -5}, {"move_cost": -1}, {"climb_cost": -1}])
def test_invalid_battery_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TerrainSummitEnv(resolution=8, **kwargs)


def test_flat_car_is_drawn_grey_with_an_empty_gauge():
    env = TerrainSummitEnv(resolution=32, battery=0.5, render_mode="rgb_array")
    env.reset(seed=0)
    env.elevation = np.zeros_like(env.elevation)
    env.step(RIGHT)
    frame = env.render()
    row, col = env.to_cell(env.s)
    cell = frame[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
    from terrain_gridworld.topo import CAR_BODY_COLOR

    assert not (cell == CAR_BODY_COLOR).all(axis=-1).any()
    assert (cell == (120, 120, 124)).all(axis=-1).sum() > 20


@pytest.mark.parametrize("slippery", [False, True])
def test_transition_table(slippery):
    env = TerrainSummitEnv(is_slippery=slippery, resolution=8)
    goal_state = env.to_state(*env.goal)
    for state, actions in env.P.items():
        assert set(actions) == {0, 1, 2, 3, PARK}
        assert actions[PARK] == [(1.0, state, 0.0, True)]
        for action in range(4):
            assert sum(p for p, *_ in actions[action]) == pytest.approx(1.0)
            for _, next_state, reward, done in actions[action]:
                assert not done
                assert reward == float(next_state == goal_state)


@pytest.mark.parametrize("slippery", [False, True])
def test_transition_table_with_a_referee(slippery):
    env = TerrainSummitEnv(is_slippery=slippery, referee=True, resolution=8)
    goal_state = env.to_state(*env.goal)
    for state, actions in env.P.items():
        for action, transitions in actions.items():
            assert sum(p for p, *_ in transitions) == pytest.approx(1.0)
            for _, next_state, reward, terminated in transitions:
                if action == PARK:
                    assert (next_state, reward, terminated) == (state, 0.0, True)
                elif state == goal_state:
                    assert (next_state, reward, terminated) == (goal_state, 0.0, True)
                else:
                    assert terminated == (next_state == goal_state)
                    assert reward == float(terminated)


@pytest.mark.parametrize("slippery", [False, True])
def test_value_iteration_solves_it(slippery):
    from examples.summit_demo import run, value_iteration

    env = gym.make("TerrainSummit-v0", is_slippery=slippery, referee=True, battery=None)
    policy, _ = value_iteration(env.unwrapped)
    success, _ = run(env, policy, episodes=20, seed=0)
    assert success == 1.0


def test_ansi_render_marks_start_goal_and_agent():
    env = TerrainSummitEnv(width=16, height=8, render_mode="ansi", is_slippery=False)
    env.reset(seed=0)
    text = env.render()
    rows = text.strip("\n").split("\n")
    assert len(rows) == 9  # 8 map rows and a battery line
    assert rows[-1] == "Battery: 150.0/150 (100%)"
    assert "G" in text and "\x1b[" in text
    env.step(RIGHT)
    assert env.render().startswith("  (Right)")


def test_rgb_render_shape_and_car():
    env = TerrainSummitEnv(width=10, height=5, resolution=32, render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (160, 320, 3) and frame.dtype == np.uint8
    row, col = env.start
    cell = frame[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
    assert (cell == CAR_BODY_COLOR).all(axis=-1).sum() > 20
    assert not (frame == CAR_BODY_COLOR).all(axis=-1)[:, : col * 32].any()


def test_background_is_drawn_once():
    env = TerrainSummitEnv(render_mode="rgb_array", is_slippery=False)
    env.reset(seed=0)
    env.render()
    background = env._background
    env.step(DOWN)
    env.render()
    assert env._background is background


def test_car_faces_the_way_it_moved():
    env = TerrainSummitEnv(is_slippery=False, start=(5, 5), terrain_seed=3)
    env.reset(seed=0)
    assert env.heading == "right"
    for action, heading in [(UP, "up"), (LEFT, "left"), (DOWN, "down"), (RIGHT, "right")]:
        env.step(action)
        assert env.heading == heading


def test_slipping_car_faces_the_direction_it_slid():
    env = TerrainSummitEnv(is_slippery=True, start=(5, 5), terrain_seed=3)
    for seed in range(30):
        env.reset(seed=seed)
        state, *_ = env.step(RIGHT)
        moved = {(4, 5): "up", (6, 5): "down", (5, 6): "right"}[env.to_cell(state)]
        assert env.heading == moved


def test_car_turns_with_heading():
    frames = {}
    for heading in ("up", "right"):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        assert draw_car(frame, (0, 0), heading, 64)
        frames[heading] = (frame == CAR_BODY_COLOR).all(axis=-1)
    rows, cols = np.nonzero(frames["up"])
    assert np.ptp(rows) > np.ptp(cols)  # longer than wide facing up
    rows, cols = np.nonzero(frames["right"])
    assert np.ptp(cols) > np.ptp(rows)


def test_tiny_cells_fall_back_to_a_dot():
    env = TerrainSummitEnv(resolution=8, render_mode="rgb_array")
    env.reset(seed=0)
    assert env.render().shape == (128, 128, 3)


def test_human_render_runs_headless():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    env = TerrainSummitEnv(render_mode="human")
    env.reset(seed=0)
    env.step(RIGHT)
    env.close()


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        TerrainSummitEnv().step(RIGHT)


@pytest.mark.parametrize("kwargs", [{"width": 1, "height": 1}, {"render_mode": "video"}, {"resolution": 0}])
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TerrainSummitEnv(**kwargs)



def test_ground_zero_gives_three_hills_on_a_flat_plain():
    env = TerrainSummitEnv(resolution=16, ground=0)
    surface = env.surface
    assert (surface == 0).mean() > 0.5
    interior = surface[1:-1, 1:-1]
    peaks = (
        (interior > 0)
        & (interior >= surface[:-2, 1:-1]) & (interior >= surface[2:, 1:-1])
        & (interior >= surface[1:-1, :-2]) & (interior >= surface[1:-1, 2:])
    )
    assert len(np.unique(interior[peaks])) == 3
    assert env.elevation.max() > 0.99
    assert np.sort(np.unique(interior[peaks]))[-2] < 0.85


@pytest.mark.parametrize("seed", range(20))
def test_rolling_ground_never_outgrows_the_summit(seed):
    flat = TerrainSummitEnv(terrain_seed=seed, resolution=8, ground=0)
    rolling = TerrainSummitEnv(terrain_seed=seed, resolution=8, ground=0.9)
    assert rolling.goal == flat.goal
    assert rolling.surface.max() == pytest.approx(flat.surface.max())
    assert (rolling.surface >= flat.surface).all()
    assert (rolling.surface == 0).mean() < 0.05  # the plain is no longer flat


@pytest.mark.parametrize("map_style", ["terrain", "lines"])
def test_map_styles_render(map_style):
    env = TerrainSummitEnv(render_mode="rgb_array", map_style=map_style, resolution=32)
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (512, 512, 3)
    white = (frame == 255).all(axis=-1).mean()
    assert white > 0.5 if map_style == "lines" else white < 0.05


@pytest.mark.parametrize("kwargs", [{"ground": 1.0}, {"ground": -0.1}, {"map_style": "sketch"}])
def test_invalid_ground_and_style_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TerrainSummitEnv(resolution=8, **kwargs)


def test_hill_count_and_radius_are_configurable():
    env = TerrainSummitEnv(hills=1, hill_radius=2.5, resolution=8, ground=0)
    assert env.elevation.max() > 0.99
    assert (env.elevation > 0).sum() > 9


def test_noise_terrain_is_still_available():
    env = TerrainSummitEnv(terrain="noise", render_mode="rgb_array", resolution=16)
    assert (env.surface == 0).mean() < 0.05
    env.reset(seed=0)
    assert env.render().shape == (256, 256, 3)
    TerrainSummitEnv(terrain={"scale": 2.0}, resolution=8)


def test_unknown_terrain_is_rejected():
    with pytest.raises(ValueError):
        TerrainSummitEnv(terrain="mountains")


def test_defaults_are_toy_text_style():
    env = TerrainSummitEnv()
    assert not env.is_slippery
    assert all(len(env.P[s][a]) == 1 for s in env.P for a in env.P[s])


def test_terrain_style_has_no_water():
    from terrain_gridworld.rendering import _RAMP_COLORS

    env = TerrainSummitEnv(render_mode="rgb_array", resolution=16)
    env.reset(seed=0)
    frame = env.render().astype(int)
    water = (frame[..., 2] > frame[..., 1] + 20) & (frame[..., 2] > frame[..., 0] + 40)
    assert water.mean() < 0.001
    assert _RAMP_COLORS is not None


def test_reversing_keeps_the_car_facing_forward():
    env = TerrainSummitEnv(start=(5, 5), terrain_seed=3)
    env.reset(seed=0)
    env.step(RIGHT)
    assert env.heading == "right"
    env.step(LEFT)  # backs up
    assert env.heading == "right"
    env.step(LEFT)  # now drives forward to the left
    assert env.heading == "left"


# ContourClimber


def run_climber(terrain_seed, **env_kwargs):
    from terrain_gridworld import ContourClimber

    env_kwargs.setdefault("battery", None)  # unlimited unless a test sets one
    env = gym.make("TerrainSummit-v0", terrain_seed=terrain_seed, resolution=8, **env_kwargs)
    agent = ContourClimber.for_env(env, seed=0)
    state, info = env.reset(seed=0)
    agent.reset(state, info)
    history = [(agent.mode, agent.cell, info["elevation"])]
    while True:
        state, _, terminated, truncated, info = env.step(agent.act())
        agent.observe(state, info)
        history.append((agent.mode, agent.cell, info["elevation"]))
        if terminated or truncated:
            agent.last_info = info
            return agent, terminated, history


def test_climber_backtracks_after_going_down():
    _, _, history = run_climber(0)
    for (mode, cell, height), (next_mode, next_cell, next_height) in zip(history, history[1:]):
        if next_mode == "backtracking":
            assert mode == "climbing" and next_height <= height
    backtracks = [i for i, (mode, *_) in enumerate(history) if mode == "backtracking"]
    assert backtracks
    for i in backtracks:
        # The very next move returns to the cell the probe started from.
        assert history[i + 1][1] == history[i - 1][1]


def test_climber_starts_climbing_on_crossing_a_contour_line():
    _, _, history = run_climber(0)
    interval = TerrainSummitEnv(resolution=8).contour_interval
    for (mode, _, height), (next_mode, _, next_height) in zip(history, history[1:]):
        if mode == "searching" and next_mode == "climbing":
            assert int(next_height / interval) > int(height / interval)


def test_climber_only_ever_climbs_up():
    _, _, history = run_climber(1)
    for (mode, _, height), (next_mode, _, next_height) in zip(history, history[1:]):
        if mode == "climbing" and next_mode == "climbing":
            assert next_height > height


def test_climber_copes_with_slipping():
    agent, terminated, _ = run_climber(0, is_slippery=True, max_episode_steps=3000)
    assert terminated and agent.mode == "parked"
    assert agent.last_info["summit_recorded"]


@pytest.mark.parametrize("terrain_seed", range(30))
def test_climber_with_a_referee_reaches_the_summit(terrain_seed):
    agent, terminated, _ = run_climber(terrain_seed, referee=True)
    assert terminated
    assert agent.cell == TerrainSummitEnv(terrain_seed=terrain_seed, resolution=8).goal


@pytest.mark.parametrize("terrain_seed", range(30))
def test_climber_records_the_summit_and_parks(terrain_seed):
    agent, terminated, _ = run_climber(terrain_seed)
    assert terminated and agent.mode == "parked"
    goal = TerrainSummitEnv(terrain_seed=terrain_seed, resolution=8).goal
    assert agent.last_info["summit_recorded"]
    assert agent.last_info["recorded"] == goal
    assert agent.open_cells() == set()
    assert agent.best_known()[0] == goal


def test_climber_runs_flat_on_a_small_battery():
    agent, terminated, _ = run_climber(0, battery=20)
    assert terminated and agent.mode == "flat"
    assert agent.last_info["battery"] <= 0


def test_climber_skips_climbs_it_cannot_afford():
    from terrain_gridworld import ContourClimber

    agent = ContourClimber(16, 16, referee=False, capacity=100, move_cost=0.5, climb_cost=30)
    agent.reset(0, {"elevation": 0.0, "battery": 100.0})
    agent.elevations[(15, 15)] = 0.9  # a high peak is already known
    agent.visited.add((15, 15))
    assert agent.climb_estimate(0.1) == pytest.approx(30 * 0.85 + 0.5 * (0.85 / 0.15 + 2))
    agent.battery = 20.0
    agent.observe(1, {"elevation": 0.1, "battery": 20.0})  # crosses two contour lines
    assert agent.mode == "searching"
    assert "can't afford the climb" in agent.shared.messages[-1][2]


def test_climber_gives_up_a_climb_when_nearly_flat():
    from terrain_gridworld import ContourClimber

    agent = ContourClimber(16, 16, referee=False, capacity=100, move_cost=0.5, climb_cost=30)
    agent.reset(0, {"elevation": 0.0, "battery": 100.0})
    agent.observe(1, {"elevation": 0.1, "battery": 90.0})
    assert agent.mode == "climbing"
    agent.observe(2, {"elevation": 0.2, "battery": 5.0})
    agent.act()
    assert agent.mode == "searching"
    assert "giving up this climb" in agent.shared.messages[-1][2]


def test_climber_does_not_need_to_visit_every_cell():
    visited = [len(run_climber(seed)[0].visited) for seed in range(10)]
    assert np.mean(visited) < 0.8 * 256


def test_open_cells_follow_the_slope_bound():
    from terrain_gridworld import ContourClimber

    agent = ContourClimber(1, 6, referee=False, max_slope=0.5)
    agent.reset(0, {"elevation": 0.0})  # cell (0, 0) at height 0
    agent.observe(1, {"elevation": 0.0})
    agent.observe(2, {"elevation": 0.9})  # a steep step: the limit rises to 1.2 * 0.9
    assert agent.slope_limit() == pytest.approx(1.08)
    # Nothing unvisited can beat 0.9 within reach of (0, 1) at 0.0... except
    # cells next to the 0.9 cell, which could be higher.
    assert agent.open_cells() == {(0, 3), (0, 4), (0, 5)}


def test_climber_rejects_bad_slope():
    from terrain_gridworld import ContourClimber

    with pytest.raises(ValueError):
        ContourClimber(16, 16, max_slope=0)


def test_climber_rejects_bad_interval():
    from terrain_gridworld import ContourClimber

    with pytest.raises(ValueError):
        ContourClimber(16, 16, contour_interval=0)
