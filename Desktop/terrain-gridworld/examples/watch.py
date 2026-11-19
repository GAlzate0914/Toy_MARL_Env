
"python examples/watch.py team random  # a new random map each time"

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import matplotlib.pyplot as plt

import terrain_gridworld
from terrain_gridworld import CarTeam, ContourClimber


def watch_team(terrain_seed):
    env = gym.make("TerrainSummitTeam-v0", terrain_seed=terrain_seed, render_mode="human")
    team = CarTeam.for_env(env, seed=0)
    states, info = env.reset(seed=0)
    team.reset(states, info)
    shown = 0
    step = 0
    while True:
        states, reward, terminated, truncated, info = env.step(team.act())
        team.observe(states, info)
        step += 1
        for msg_step, name, text in team.messages[shown:]:
            print(f"[{msg_step:>3}] {name}: {text}")
        shown = len(team.messages)
        if reward > 0:
            print(f"[{step:>3}] *** Summit recorded by {team.cars[info['winner']].name} ***")
        if terminated or truncated:
            break
    charge = ", ".join(f"{car.name} {car._percent()} ({car.mode})" for car in team.cars)
    print(_verdict(info), f"after {step} steps. Charge left: {charge}")
    return env


def _verdict(info):
    return "Summit recorded" if info["summit_recorded"] else "Summit NOT recorded"


def watch_solo(terrain_seed):
    env = gym.make("TerrainSummit-v0", terrain_seed=terrain_seed, render_mode="human")
    agent = ContourClimber.for_env(env, seed=0)
    state, info = env.reset(seed=0)
    agent.reset(state, info)
    mode = agent.mode
    shown = 0
    step = 0
    while True:
        state, reward, terminated, truncated, info = env.step(agent.act())
        agent.observe(state, info)
        step += 1
        if agent.mode != mode:
            print(f"[{step:>3}] {mode} -> {agent.mode}")
            mode = agent.mode
        for msg_step, _, text in agent.shared.messages[shown:]:
            print(f"[{msg_step:>3}] {text}")
        shown = len(agent.shared.messages)
        if reward > 0:
            print(f"[{step:>3}] *** Summit recorded ***")
        if terminated or truncated:
            break
    print(_verdict(info), f"after {step} steps. Battery left: {agent._percent()} ({agent.mode})")
    return env


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "team"
    seed_arg = sys.argv[2] if len(sys.argv) > 2 else "0"
    seed = None if seed_arg == "random" else int(seed_arg)
    env = watch_solo(seed) if which == "solo" else watch_team(seed)
    # Keep the window open on the final frame until it is closed.
    plt.ioff()
    plt.show()
    env.close()
