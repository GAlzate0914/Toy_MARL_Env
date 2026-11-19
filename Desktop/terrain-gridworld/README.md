# Terrain Grid World

Configurable Gymnasium grid worlds where every cell has a
continuous elevation: an RC car searching for the summit of a few hills, and a
terrain-only world of Minecraft-style fractal Perlin noise.

## Install

```bash
pip install -e ".[dev]"
pytest
python examples/climber_demo.py climber.gif
python examples/watch.py          # watch the three cars live in a window
```

## Find the summit

`TerrainSummit-v0` is a toy-text environment in the style of Gymnasium's
FrozenLake and CliffWalking. An RC car drives around a 16×16 grid of grassland
with three smooth hills, looking for the top of the tallest one, on a limited
battery.

The observation is the car's cell, `row * width + col`. `info["elevation"]` is
an altimeter reading for that cell, so a driver can tell whether it is
climbing or descending, and `info["battery"]` is its remaining charge.

```python
import gymnasium as gym
import terrain_gridworld
from terrain_gridworld import ContourClimber

env = gym.make("TerrainSummit-v0", render_mode="rgb_array")
agent = ContourClimber.for_env(env, seed=0)
state, info = env.reset(seed=0)
agent.reset(state, info)
while True:
    state, reward, terminated, truncated, info = env.step(agent.act())
    agent.observe(state, info)
    if terminated or truncated:
        break
print(info["summit_recorded"], info["recorded"], info["battery"])
```

Actions are `0` left, `1` down, `2` right, `3` up, and `4` park.

**Recording the summit.** The environment keeps a record of the highest cell
the car has visited (`info["recorded"]`, outlined in gold on the map). The
first time that is the true summit, the step pays a reward of 1 and
`info["summit_recorded"]` becomes True. The car does not have to stay there or
come back: the summit only has to be recorded. Every other step pays 0.

**Battery.** Each car starts with 150 units of charge. Every move costs 0.5,
plus 30 per unit of elevation gained, so the strain of a steep climb is what
drains it: climbing from the plain to the top of the tallest hill costs about
30 on its own. Driving downhill or on the flat costs only the 0.5. A move the
battery cannot pay for does not happen and leaves it empty. Parking is free.
A car with an empty battery is drawn grey, and each car has a charge gauge.

**Ending.** The episode ends when the car parks (action 4) or runs flat. With
`referee=True` it also ends as soon as the car reaches the summit, like
FrozenLake.

### The ContourClimber driver

`ContourClimber` is a scripted driver that knows only its own cell, the
altimeter and its battery, and remembers what it has seen. Its current
behaviour is in `agent.mode`:

- **Searching:** drives across new ground, preferring to keep going straight.
  Driving uphill across a contour line onto new ground starts a climb.
- **Climbing:** tries neighbouring cells that might be higher, heading for the
  highest it knows about.
- **Backtracking:** if a move did not go up, it reverses to where it came from
  and tries another direction. The car is drawn reversing, still facing forward.
- **Parked** or **flat:** finished, or out of charge.

If it runs out of ways up, it has topped a peak (recorded in `agent.peaks`), and
it goes back to searching.

**Knowing when it is done.** The car keeps exploring until no unexplored cell
could still be higher than the best cell it has found. It assumes the ground
never rises faster than 0.65 per cell (the steepest step on these maps is
about 0.6), and raises that limit if it ever measures a steeper step. So an
unexplored cell can be at most `measured height + 0.65 × distance` above any
cell already measured. If even that bound is no higher than the best cell
found, the cell is ruled out without a visit. When nothing is left open, the
car parks: the highest point is already on record (`agent.open_cells()` and
`agent.best_known()` show this). Flat ground gets ruled out in lanes about 3
cells apart, while slopes near tall peaks get checked closely.

**Saving charge.** Before starting a climb, the car estimates the charge
needed to get higher than the best cell found so far (`agent.climb_estimate`).
If it cannot afford that with a 25% margin, it does not climb. It also gives up
a climb partway when its charge drops below what one steep step would cost.
And it never climbs a hill that the slope bound says cannot beat the best cell
found, which is where most of the saving comes from.

Without batteries (`battery=None`) the car always records the summit and
finishes checking the map, in about 210 steps. With the default 150 units, one
car records the summit on about 70% of maps but runs flat before it can finish
checking.

`python examples/climber_demo.py climber.gif` records it as an animation, with
its trail, what it is doing and its charge at each step. Cells ruled out
without a visit are greyed.

| Option | Default | Effect |
| --- | --- | --- |
| `width`, `height` | 16 | Grid size |
| `hills` | 3 | Number of hills |
| `hill_radius` | 3.5 | Hill size, in cells |
| `ground` | 0.1 | Height of the rolling grassland between hills; 0 makes it flat |
| `terrain` | `"hills"` | `"noise"` for rugged Perlin terrain, or Perlin settings as below (`scale` in cells) |
| `terrain_seed` | 0 | Which map; None for a random one |
| `start` | `"corner"` | Corner farthest from the summit, `"lowest"`, `"random"` or `(row, col)` |
| `is_slippery` | False | FrozenLake-style slipping: 1/3 intended, 1/3 each sideways |
| `battery` | 150 | Starting charge per car; None for unlimited |
| `move_cost` | 0.5 | Charge used by every move |
| `climb_cost` | 30 | Extra charge per unit of elevation gained |
| `referee` | False | True: reaching the summit also ends the episode |
| `render_mode` | None | `"ansi"`, `"rgb_array"` or `"human"` |
| `map_style` | `"terrain"` | `"terrain"` for a shaded map, `"lines"` for black contours on white |
| `resolution` | 48 | Terrain samples and pixels per cell |

Episodes are cut off after 1,000 steps. The tallest hill is always the summit:
the rolling ground fades out towards the hill tops, so it can never raise a
smaller hill above it. `env.surface` holds the fine terrain, `env.elevation`
the 16×16 cell values (each the highest point in its cell), and `env.P` the
movement model in FrozenLake's format (it leaves out the battery, and its
reward for entering the summit applies only the first time).

The `"terrain"` style shades grassland, forest, rock and snow, with no water,
and draws a contour line every 0.05 of elevation. Both styles show a faint
grid, a flag on the peak and a ring at the start. The map is drawn once, so
later frames are fast. The text render shows each cell's elevation from 0 to 9,
with `S` the start, `G` the summit, the car highlighted, and a battery line.

`python examples/summit_demo.py` plans the shortest route with value
iteration (with a referee and unlimited charge), for comparison.

## Three cars that talk to each other

`TerrainSummitTeam-v0` puts red, yellow and blue RC cars on the same map, each
starting in a different corner with its own battery. Every step moves all
three. The environment records the highest cell any car has visited, and the
team gets a reward of 1 the first time that is the summit. The episode ends
when every car has parked or run flat. Episodes are cut off after 500 steps.

```python
import gymnasium as gym
import terrain_gridworld
from terrain_gridworld import CarTeam

env = gym.make("TerrainSummitTeam-v0", render_mode="rgb_array")
team = CarTeam.for_env(env, seed=0)
states, info = env.reset(seed=0)        # one cell per car
team.reset(states, info)
while True:
    states, reward, terminated, truncated, info = env.step(team.act())  # one action per car
    team.observe(states, info)
    if terminated or truncated:
        break
print(info["summit_recorded"], info["batteries"])
for step, name, text in team.messages:
    print(step, name, text)
```

Info holds `elevations`, `batteries` and `parked` per car, plus `recorded`,
`recorded_elevation`, `summit_recorded`, `winner` (the car that recorded the
summit on this step) and `stalled` (cars whose move failed for lack of charge).

The cars are `ContourClimber` drivers sharing one map and radio channel:

- **One shared map:** every cell any car has visited, its elevation, and every
  direction already found not to go uphill.
- **Spreading out:** a searching car prefers new cells at least 4 cells from
  its teammates, and when it runs out of new neighbours it heads for an
  unexplored area no teammate has claimed.
- **One car per hill:** a car that crosses a contour line does not climb if a
  teammate is already climbing that hill, or if anyone has already topped it.
  Taller peaks cover a wider area. If two cars end up on one hill, the one with
  more charge keeps it.
- **Handing off hills:** a car that cannot afford a climb, or runs low partway
  up, flags the hill on the radio. Teammates with enough charge steer towards
  flagged hills, and the first to cross onto one takes it over.
- **Ruling out together:** a cell ruled out by one car's measurements is ruled
  out for everyone, so the team finishes when nothing anywhere is left open,
  and then every car parks.
- **Radio messages:** each of these decisions is logged in `team.messages`.

On 200 maps:

| | Summit recorded | Finished checking the map | Average steps |
| --- | --- | --- | --- |
| Three cars that talk, 150 charge | 100% | 72% | 73 |
| Three silent cars, 150 charge | 90% | 0% | 113 |
| One car, 150 charge | 71% | 0% | 103 |
| Three cars that talk, 100 charge | 95% | 4% | 75 |
| Three silent cars, 100 charge | 79% | 0% | 83 |
| One car, 100 charge | 46% | 0% | 73 |

`CarTeam(..., communicate=False)` gives the silent cars, which ignore each
other: each has to convince itself, so they never finish checking, and they
climb the same hills repeatedly. With unlimited charge, the talking team
always records the summit and finishes checking in about 70 steps.

`python examples/team_demo.py team.gif` records the team as an animation, with
each car's trail, charge and mode, the radio messages, and ruled-out cells
greyed. Add `--silent` for cars that don't communicate, or `--battery 100` for
a tighter battery. `python examples/team_demo.py handoff.gif 18 --battery 100`
shows Yellow, low on charge, flagging a hill that Red then takes over.

`MultiCarSummitEnv` accepts every `TerrainSummitEnv` option, plus `n_cars` (1
to 4; the fourth car is green) and `start` (`"corners"` or `"random"`).

## Terrain only

```python
import gymnasium as gym
import terrain_gridworld

env = gym.make("TerrainGridWorld-v0", render_mode="rgb_array")
elevation, info = env.reset(seed=0)  # float32 array of shape (32, 32), values in [0, 1]
frame = env.render()
```

The world has no agent yet. `step(0)` is a placeholder no-op that returns the
same terrain with a reward of 0, and episodes never end on their own.

## Configure

```python
env = gym.make(
    "TerrainGridWorld-v0",
    width=64,
    height=48,
    terrain={"scale": 24.0, "octaves": 5, "persistence": 0.45, "exponent": 1.4},
    terrain_seed=42,   # same map every episode; omit for a new map each reset
)
```

| Terrain setting | Default | Effect |
| --- | --- | --- |
| `scale` | 16.0 | Size of the largest landforms, in cells |
| `octaves` | 4 | Noise layers; more gives finer detail |
| `persistence` | 0.5 | Amplitude kept per octave; higher is rougher |
| `lacunarity` | 2.0 | Frequency growth per octave |
| `exponent` | 1.0 | Above 1 flattens lowlands and sharpens peaks |
| `normalize` | True | Stretch each map to span exactly [0, 1] |

To replay a map, pass the seed from `info["terrain_seed"]` back through
`env.reset(options={"terrain_seed": seed})`.

## Terrain without the environment

```python
from terrain_gridworld import TerrainConfig, generate_terrain

elevation = generate_terrain(32, 32, seed=1)

# Adjoining chunks line up exactly when normalize is off.
config = TerrainConfig(normalize=False)
left = generate_terrain(32, 32, seed=1, config=config)
right = generate_terrain(32, 32, seed=1, config=config, origin=(0, 32))
```

Noise repeats every 256 lattice cells per octave, so very large worlds will
eventually tile. At the default scale that is about 4,000 cells across.
