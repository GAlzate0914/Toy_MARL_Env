"""Procedural-terrain grid worlds for Gymnasium."""

from gymnasium.envs.registration import register

from terrain_gridworld.agents import CarTeam, ContourClimber, SharedMap
from terrain_gridworld.env import TerrainGridWorldEnv
from terrain_gridworld.multi import MultiCarSummitEnv
from terrain_gridworld.summit import TerrainSummitEnv
from terrain_gridworld.terrain import PerlinNoise2D, TerrainConfig, generate_terrain

__all__ = [
    "CarTeam",
    "ContourClimber",
    "MultiCarSummitEnv",
    "SharedMap",
    "PerlinNoise2D",
    "TerrainConfig",
    "TerrainGridWorldEnv",
    "TerrainSummitEnv",
    "generate_terrain",
]

register(id="TerrainGridWorld-v0", entry_point="terrain_gridworld.env:TerrainGridWorldEnv")
register(
    id="TerrainSummit-v0",
    entry_point="terrain_gridworld.summit:TerrainSummitEnv",
    max_episode_steps=1000,
)
register(
    id="TerrainSummitTeam-v0",
    entry_point="terrain_gridworld.multi:MultiCarSummitEnv",
    max_episode_steps=500,
)
