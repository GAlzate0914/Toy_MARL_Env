import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import terrain_gridworld  # noqa: F401  (registers the environments)
from terrain_gridworld import CarTeam, ContourClimber, MultiCarSummitEnv, SharedMap
from terrain_gridworld.multi import CAR_COLORS
from terrain_gridworld.summit import DOWN, LEFT, PARK, RIGHT, UP

# Environment


@pytest.mark.parametrize("render_mode", ["rgb_array", "ansi"])
@pytest.mark.parametrize("start", ["corners", "random"])
def test_passes_gymnasium_env_checker(render_mode, start):
    env = gym.make("TerrainSummitTeam-v0", render_mode=render_mode, start=start, resolution=16)
    check_env(env.unwrapped, skip_render_check=False)
    env.close()


def test_spaces_and_registration():
    env = gym.make("TerrainSummitTeam-v0", resolution=8)
    assert env.observation_space == gym.spaces.MultiDiscrete([256] * 3)
    assert env.action_space == gym.spaces.MultiDiscrete([5] * 3)
    assert env.spec.max_episode_steps == 500
    assert env.unwrapped.car_names == ("Red", "Yellow", "Blue")


def test_cars_start_in_different_corners_farthest_first():
    env = MultiCarSummitEnv(resolution=8)
    states, info = env.reset(seed=0)
    cells = [env.to_cell(s) for s in states]
    assert len(set(cells)) == 3
    assert all(r in (0, 15) and c in (0, 15) for r, c in cells)
    distances = [abs(r - env.goal[0]) + abs(c - env.goal[1]) for r, c in cells]
    assert distances == sorted(distances, reverse=True)
    assert np.allclose(info["elevations"], [env.elevation[c] for c in cells])
    assert info["winner"] is None


def test_random_starts_are_separate_and_off_the_summit():
    env = MultiCarSummitEnv(width=3, height=1, start="random", n_cars=2, resolution=8)
    for seed in range(30):
        states, _ = env.reset(seed=seed)
        cells = [env.to_cell(s) for s in states]
        assert len(set(cells)) == 2 and env.goal not in cells


def test_every_car_moves_each_step():
    env = MultiCarSummitEnv(resolution=8, start="random")
    states, _ = env.reset(seed=1)
    before = [env.to_cell(s) for s in states]
    actions = [RIGHT, DOWN, LEFT]
    after_states, *_ = env.step(actions)
    for (r, c), action, after in zip(before, actions, after_states):
        d_row, d_col = {LEFT: (0, -1), DOWN: (1, 0), RIGHT: (0, 1), UP: (-1, 0)}[action]
        expected = (min(max(r + d_row, 0), 15), min(max(c + d_col, 0), 15))
        assert env.to_cell(after) == expected


def put_car_beside_the_summit(env, car):
    row, col = env.goal
    beside, action = ((row, col - 1), RIGHT) if col > 0 else ((row, col + 1), LEFT)
    env.positions[car] = env.to_state(*beside)
    return action


def test_with_a_referee_the_first_car_on_the_summit_ends_it():
    env = MultiCarSummitEnv(resolution=8, referee=True)
    env.reset(seed=0)
    action = put_car_beside_the_summit(env, 1)
    _, reward, terminated, truncated, info = env.step([UP, action, UP])
    assert (reward, terminated, truncated, info["winner"]) == (1.0, True, False, 1)


def test_reaching_the_summit_records_it_and_the_team_carries_on():
    env = MultiCarSummitEnv(resolution=8)
    _, info = env.reset(seed=0)
    assert not info["summit_recorded"]
    action = put_car_beside_the_summit(env, 1)
    _, reward, terminated, _, info = env.step([UP, action, UP])
    assert (reward, terminated, info["winner"]) == (1.0, False, 1)
    assert info["summit_recorded"] and info["recorded"] == env.goal
    _, reward, terminated, _, info = env.step([UP, UP, UP])
    assert (reward, terminated, info["winner"], info["recorded"]) == (0.0, False, None, env.goal)


def test_episode_ends_when_every_car_has_parked():
    env = MultiCarSummitEnv(resolution=8)
    env.reset(seed=0)
    _, _, terminated, _, info = env.step([PARK, PARK, UP])
    assert not terminated and list(info["parked"]) == [True, True, False]
    states, _, _, _, _ = env.step([UP, UP, UP])  # parked cars stay put
    _, _, terminated, _, info = env.step([UP, UP, PARK])
    assert terminated and all(info["parked"])


def test_each_car_has_its_own_battery():
    env = MultiCarSummitEnv(resolution=8, start="random")
    env.reset(seed=1)
    env.elevation = np.zeros_like(env.elevation)
    _, _, _, _, info = env.step([UP, PARK, UP])
    assert list(info["batteries"]) == [149.5, 150.0, 149.5]


def test_flat_cars_stop_and_the_episode_ends_when_all_are_flat():
    env = MultiCarSummitEnv(resolution=8, battery=1.0)
    states, _ = env.reset(seed=0)
    env.elevation = np.zeros_like(env.elevation)
    env.step([RIGHT, PARK, RIGHT])
    after, _, terminated, _, info = env.step([RIGHT, UP, LEFT])
    assert terminated
    assert list(info["batteries"]) == [0.0, 1.0, 0.0]
    stuck, _, _, _, _ = env.step([DOWN, DOWN, DOWN])
    assert list(stuck) == list(after)


def test_stalling_on_a_climb_is_reported():
    env = MultiCarSummitEnv(resolution=8, battery=1.0, climb_cost=1000)
    env.reset(seed=0)
    action = put_car_beside_the_summit(env, 0)
    _, _, _, _, info = env.step([action, PARK, PARK])
    assert info["stalled"] == [0] and info["batteries"][0] == 0.0


def test_cars_are_drawn_in_their_colours():
    env = MultiCarSummitEnv(render_mode="rgb_array", resolution=32)
    states, _ = env.reset(seed=0)
    frame = env.render()
    for name, state in zip(env.car_names, states):
        row, col = env.to_cell(state)
        cell = frame[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
        assert (cell == CAR_COLORS[name]).all(axis=-1).sum() > 20


def test_text_render_shows_a_letter_per_car():
    env = MultiCarSummitEnv(render_mode="ansi", resolution=8)
    env.reset(seed=0)
    text = env.render()
    for letter in "RYB":
        assert letter in text
    assert "G" in text


@pytest.mark.parametrize("kwargs", [{"n_cars": 0}, {"n_cars": 5}, {"start": "middle"}])
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MultiCarSummitEnv(resolution=8, **kwargs)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        MultiCarSummitEnv(resolution=8).step([0, 0, 0])


# Team


def run_team(terrain_seed, communicate=True, **env_kwargs):
    env_kwargs.setdefault("battery", None)  # unlimited unless a test sets one
    env = gym.make("TerrainSummitTeam-v0", terrain_seed=terrain_seed, resolution=8, **env_kwargs)
    team = CarTeam.for_env(env, seed=0, communicate=communicate)
    states, info = env.reset(seed=0)
    team.reset(states, info)
    paths = [[car.cell] for car in team.cars]
    steps = 0
    while True:
        states, _, terminated, truncated, info = env.step(team.act())
        team.observe(states, info)
        steps += 1
        for path, car in zip(paths, team.cars):
            path.append(car.cell)
        if terminated or truncated:
            return team, terminated, steps, paths, info


@pytest.mark.parametrize("terrain_seed", range(25))
def test_team_records_the_summit_then_parks(terrain_seed):
    team, terminated, _, paths, info = run_team(terrain_seed)
    assert terminated
    goal = MultiCarSummitEnv(terrain_seed=terrain_seed, resolution=8).goal
    assert info["summit_recorded"] and info["recorded"] == goal
    assert any(goal in path for path in paths)
    assert team.modes == ["parked"] * 3 and all(info["parked"])
    assert any(text.startswith("Nothing unexplored can beat") for _, _, text in team.messages)


@pytest.mark.parametrize("terrain_seed", range(10))
def test_team_with_a_referee_reaches_the_summit(terrain_seed):
    _, terminated, _, paths, info = run_team(terrain_seed, referee=True)
    assert terminated
    goal = MultiCarSummitEnv(terrain_seed=terrain_seed, resolution=8).goal
    assert paths[info["winner"]][-1] == goal


def test_default_battery_team_usually_finishes_and_always_records():
    results = [run_team(s, battery=150.0) for s in range(30)]
    assert all(info["summit_recorded"] for *_, info in results)
    finished = [team.modes == ["parked"] * 3 for team, *_ in results]
    assert np.mean(finished) > 0.6


def test_talking_beats_silence_on_a_battery():
    talking = np.mean([run_team(s, battery=100.0)[4]["summit_recorded"] for s in range(40)])
    silent = np.mean([run_team(s, communicate=False, battery=100.0)[4]["summit_recorded"] for s in range(40)])
    assert talking > silent


def test_a_car_hands_an_unaffordable_hill_to_a_teammate():
    team, _, _, _, info = run_team(18, battery=100.0)
    texts = [(name, text) for _, name, text in team.messages]
    flagged = [i for i, (_, t) in enumerate(texts) if "flagging" in t]
    taken = [i for i, (_, t) in enumerate(texts) if "Taking the hill" in t]
    assert flagged and taken and taken[0] > flagged[0]
    assert texts[flagged[0]][0] != texts[taken[0]][0]
    assert info["summit_recorded"]


def test_the_car_with_more_charge_keeps_a_shared_hill():
    shared = SharedMap()
    kw = dict(shared=shared, seed=0, referee=False, capacity=100, move_cost=0.5, climb_cost=30)
    red = ContourClimber(16, 16, car_id=0, name="Red", **kw)
    blue = ContourClimber(16, 16, car_id=1, name="Blue", **kw)
    red.reset(5 * 16 + 5, {"elevation": 0.0, "battery": 40.0})
    blue.reset(5 * 16 + 7, {"elevation": 0.0, "battery": 90.0})
    red.mode, blue.mode = "climbing", "climbing"
    red._publish(), blue._publish()
    assert "Blue is on this hill too" in (red._reason_to_leave_hill() or "")
    assert blue._reason_to_leave_hill() is None


def test_cars_share_one_map():
    team, *_ = run_team(0)
    assert all(car.shared is team.shared for car in team.cars)
    for path in run_team(0)[3]:
        assert set(path) <= team.shared.visited


def overlap(paths):
    owners = {}
    for i, path in enumerate(paths):
        for cell in path:
            owners.setdefault(cell, set()).add(i)
    return sum(len(o) > 1 for o in owners.values()) / len(owners)


def test_talking_cars_overlap_far_less_than_silent_ones():
    talking = np.mean([overlap(run_team(s)[3]) for s in range(30)])
    silent = np.mean([overlap(run_team(s, communicate=False)[3]) for s in range(30)])
    assert talking < 0.1
    assert silent > 3 * talking


def test_talking_cars_overlap_little_with_a_referee():
    talking = np.mean([overlap(run_team(s, referee=True)[3]) for s in range(30)])
    assert talking < 0.05


def test_talking_cars_never_reclimb_a_teammates_peak():
    for seed in range(30):
        team, *_ = run_team(seed)
        found = [text for _, _, text in team.messages if text.startswith("Topped a false peak")]
        assert len(found) == len(set(found))


def test_talking_team_is_faster_on_average():
    talking = np.mean([run_team(s)[2] for s in range(40)])
    silent = np.mean([run_team(s, communicate=False)[2] for s in range(40)])
    assert talking < silent


def test_a_car_skips_a_hill_a_teammate_is_climbing():
    shared = SharedMap()
    red = ContourClimber(16, 16, shared=shared, car_id=0, name="Red", seed=0)
    blue = ContourClimber(16, 16, shared=shared, car_id=1, name="Blue", seed=0)
    red.reset(5 * 16 + 5, {"elevation": 0.0})
    blue.reset(5 * 16 + 9, {"elevation": 0.0})
    red.observe(5 * 16 + 6, {"elevation": 0.2})  # Red crosses contours and climbs
    assert red.mode == "climbing"
    blue.observe(5 * 16 + 8, {"elevation": 0.2})  # Blue crosses nearby
    assert blue.mode == "searching"
    assert "Red is already climbing it" in shared.messages[-1][2]


def test_a_car_skips_a_hill_a_teammate_has_topped():
    shared = SharedMap()
    red = ContourClimber(16, 16, shared=shared, car_id=0, name="Red", seed=0)
    blue = ContourClimber(16, 16, shared=shared, car_id=1, name="Blue", seed=0)
    red.reset(0, {"elevation": 0.0})
    blue.reset(10 * 16 + 10, {"elevation": 0.0})
    shared.elevations[(7, 10)] = 0.7
    shared.peaks[(7, 10)] = "Red"
    blue.observe(9 * 16 + 10, {"elevation": 0.2})
    assert blue.mode == "searching"
    assert "Red already topped it" in shared.messages[-1][2]


def test_searching_cars_claim_different_targets():
    shared = SharedMap()
    cars = [ContourClimber(16, 16, shared=shared, car_id=i, name=n, seed=0)
            for i, n in enumerate(["Red", "Yellow"])]
    cars[0].reset(0, {"elevation": 0.0})
    cars[1].reset(1, {"elevation": 0.0})
    shared.visited.update((r, c) for r in range(16) for c in range(16) if (r, c) not in {(15, 15), (15, 0)})
    for car in cars:
        car.act()
    assert shared.targets[0] != shared.targets[1]


def test_silent_team_keeps_separate_maps():
    team = CarTeam(16, 16, seed=0, communicate=False)
    assert len({id(car.shared) for car in team.cars}) == 3


def test_single_climber_still_works_alone():
    env = gym.make("TerrainSummit-v0", resolution=8, battery=None)
    agent = ContourClimber.for_env(env, seed=0)
    state, info = env.reset(seed=0)
    agent.reset(state, info)
    while True:
        state, _, terminated, truncated, info = env.step(agent.act())
        agent.observe(state, info)
        if terminated or truncated:
            break
    assert terminated
    assert agent.shared.messages == [] or all(m[1] == "Car" for m in agent.shared.messages)
