import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import terrain_gridworld  # noqa: F401  (registers the environment)
from terrain_gridworld import TerrainConfig, TerrainGridWorldEnv, generate_terrain


# Terrain


def test_shape_type_and_range():
    terrain = generate_terrain(32, 48, seed=1)
    assert terrain.shape == (32, 48)
    assert terrain.dtype == np.float32
    assert terrain.min() == 0.0 and terrain.max() == 1.0


def test_same_seed_matches_and_different_seed_differs():
    assert np.array_equal(generate_terrain(32, 32, seed=7), generate_terrain(32, 32, seed=7))
    assert not np.array_equal(generate_terrain(32, 32, seed=7), generate_terrain(32, 32, seed=8))


@pytest.mark.parametrize("seed", range(20))
def test_terrain_is_smooth(seed):
    terrain = generate_terrain(32, 32, seed=seed).astype(np.float64)
    neighbour_correlation = np.corrcoef(terrain[:, :-1].ravel(), terrain[:, 1:].ravel())[0, 1]
    assert neighbour_correlation > 0.85  # uncorrelated noise would be near 0
    assert np.median(np.abs(np.diff(terrain, axis=1))) < 0.1  # white noise is near 0.29


def test_scale_that_divides_coordinates_does_not_flatten_terrain():
    config = TerrainConfig(scale=1.0, octaves=1, normalize=False)
    assert generate_terrain(32, 32, seed=3, config=config).std() > 0.01


def test_chunks_join_seamlessly_without_normalization():
    config = TerrainConfig(normalize=False)
    world = generate_terrain(32, 64, seed=5, config=config)
    left = generate_terrain(32, 32, seed=5, config=config)
    right = generate_terrain(32, 32, seed=5, config=config, origin=(0, 32))
    assert np.array_equal(np.hstack([left, right]), world)


def test_unnormalized_terrain_stays_in_range():
    config = TerrainConfig(normalize=False, persistence=1.0, octaves=8)
    for seed in range(20):
        terrain = generate_terrain(32, 32, seed=seed, config=config)
        assert terrain.min() >= 0.0 and terrain.max() <= 1.0


def test_negative_origin_is_supported():
    config = TerrainConfig(normalize=False)
    world = generate_terrain(16, 16, seed=2, config=config, origin=(-8, -8))
    corner = generate_terrain(8, 8, seed=2, config=config)
    assert np.array_equal(world[8:, 8:], corner)


def test_more_octaves_add_detail_to_the_same_landscape():
    coarse = generate_terrain(32, 32, seed=4, config=TerrainConfig(octaves=2, normalize=False))
    fine = generate_terrain(32, 32, seed=4, config=TerrainConfig(octaves=5, normalize=False))
    assert np.corrcoef(coarse.ravel(), fine.ravel())[0, 1] > 0.8


@pytest.mark.parametrize(
    "bad",
    [
        {"scale": 0},
        {"octaves": 0},
        {"persistence": 0},
        {"persistence": 1.5},
        {"lacunarity": 0.5},
        {"exponent": 0},
    ],
)
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        TerrainConfig(**bad)


def test_one_cell_map_does_not_divide_by_zero():
    assert generate_terrain(1, 1, seed=0).item() == 0.5


# Environment


def test_passes_gymnasium_env_checker():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    env = gym.make("TerrainGridWorld-v0", render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=False)
    env.close()


def test_registered_and_configurable_through_make():
    env = gym.make("TerrainGridWorld-v0", width=20, height=10, terrain={"scale": 8.0})
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10, 20)
    assert env.unwrapped.terrain_config.scale == 8.0
    env.close()


def test_defaults_to_32_by_32():
    obs, _ = TerrainGridWorldEnv().reset(seed=0)
    assert obs.shape == (32, 32)


def test_seeded_reset_is_reproducible():
    first, _ = TerrainGridWorldEnv().reset(seed=11)
    second, _ = TerrainGridWorldEnv().reset(seed=11)
    assert np.array_equal(first, second)


def test_each_reset_makes_a_new_map_by_default():
    env = TerrainGridWorldEnv()
    first, _ = env.reset(seed=0)
    second, _ = env.reset()
    assert not np.array_equal(first, second)


def test_fixed_terrain_seed_keeps_the_map():
    env = TerrainGridWorldEnv(terrain_seed=42)
    first, _ = env.reset(seed=0)
    second, _ = env.reset(seed=1)
    assert np.array_equal(first, second)


def test_info_seed_replays_a_map():
    env = TerrainGridWorldEnv()
    obs, info = env.reset(seed=3)
    replay, _ = env.reset(options={"terrain_seed": info["terrain_seed"]})
    assert np.array_equal(obs, replay)


def test_step_is_a_no_op():
    env = TerrainGridWorldEnv()
    before, _ = env.reset(seed=0)
    after, reward, terminated, truncated, _ = env.step(0)
    assert np.array_equal(before, after)
    assert (reward, terminated, truncated) == (0.0, False, False)


def test_invalid_action_is_rejected():
    env = TerrainGridWorldEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(1)


def test_observation_cannot_mutate_the_world():
    env = TerrainGridWorldEnv()
    obs, _ = env.reset(seed=0)
    before = env.elevation.copy()
    obs[:] = 0
    assert np.array_equal(env.elevation, before)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        TerrainGridWorldEnv().step(0)


def test_rgb_render_shape():
    env = TerrainGridWorldEnv(width=20, height=10, cell_size=8, render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (80, 160, 3)
    assert frame.dtype == np.uint8


def test_render_works_on_a_single_row_grid():
    env = TerrainGridWorldEnv(width=8, height=1, render_mode="rgb_array")
    env.reset(seed=0)
    assert env.render().shape == (12, 96, 3)


def test_human_render_runs_headless():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    env = TerrainGridWorldEnv(render_mode="human")
    env.reset(seed=0)
    env.step(0)
    env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": 0},
        {"render_mode": "ansi"},
        {"cell_size": 0},
    ],
)
def test_invalid_env_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TerrainGridWorldEnv(**kwargs)
