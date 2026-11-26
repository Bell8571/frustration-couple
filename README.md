# Shared register

Two gradient descenders share one vector of 64 coordinates. The left copy is driven toward target `a` and the right copy toward target `b`, each by descending squared error. A hidden mask marks the 32 coordinates whose targets differ. The release rule sees only the coupling residual. A coordinate whose residual stays hot is uncoupled, and after that the two copies of that coordinate move independently.

## Dynamics

Compatible coordinates share one target drawn uniformly from `[-1, 1]`. Each incompatible coordinate gets a second target offset by a gap whose absolute size is drawn from `[0.5, 2]`. The mask is true on those coordinates and is not an input to the release rule. Both states start at zero.

While a coordinate is coupled, each side is pulled toward the midpoint with strength `J`:

```
pull = J * (midpoint - y)
force = -(y - target) + pull
```

The residual is `max(|pull_left|, |pull_right|)`: the absolute agreement correction, taken before `dt` scales the step. `J = 0` skips the pull, so the residual is identically zero and each side follows only its own target.

The reported loss on a set of coordinates is the mean of `0.5 * ((y_left - a)^2 + (y_right - b)^2)`. Disagreement is the mean of `|y_left - y_right|`. An average over an empty set is reported as 0. That is why an empty mask shows zero mask loss even though the quantity was not measured on any coordinate.

### Gradient flow

Release, hybrid, and identical runs use `momentum = 0`. The step is forward Euler on the first-order flow

```
y <- y + dt * force
```

with `dt = 0.05`.

### Heavy-ball momentum

The memory run is a heavy-ball momentum control. Velocity and position use semi-implicit (symplectic) Euler

```
v <- momentum * v + dt * force
y <- y + dt * v
```

with `momentum = 0.9` and with velocity starting at zero. From rest, one step displaces the state by `dt^2 * force`. `_advance` takes this branch whenever `momentum != 0`. In the gradient-flow branch the same `force` is a rate; in this branch it is the acceleration.

## Modes

`evaluate_grid` records four modes on one draw of the targets (seed 42), for `J` from 0 to 2 in steps of 0.1. Calibration and the measured run are 400 steps each.

| Mode | Agreement `J` | Release | Momentum | Role |
| --- | --- | --- | --- | --- |
| release | the sweep | on | 0 | Main curve. Hot coordinates are dropped. |
| hybrid | the sweep | off | 0 | Both sides stay coupled for the whole run. |
| identical | the sweep | on | 0 | The two targets are the same vector, so the mask is empty. |
| memory | 0 | on | 0.9 | Heavy-ball control with the agreement pull turned off. |

### Nulls

- **`J = 0`.** The targets may still disagree. The coupling correction is identically zero, the residual is zero, and the release rule drops nothing. On the seed-42 release curve this is the `J=0.00` line: nothing released, precision 1, recall 0.
- **Identical targets.** `make_problem(..., identical=True)` copies one target vector and leaves the mask empty. Residuals stay at zero and nothing is released.
- **Momentum without agreement.** The memory mode runs heavy-ball momentum at `J = 0` with release enabled. The coupling residual stays zero, so nothing is released.

Hybrid is the matched control for release: the same targets and the same `J`, with the release rule disabled. Masked coordinates stay coupled. Their disagreement falls as `J` grows, and their loss stays positive because one shared value has two different targets.

## Why recall stays near one half

`RESIDUAL_PERCENTILE = 75` and the 50-step persistence cap limit recall to about 0.5–0.6 by construction. Ranking average precision can still be 1.0.

Calibration averages the residual over 400 steps while every coordinate is coupled. The threshold is the 75th percentile of that 64-vector. The 32 agreeing coordinates sit near residual zero and the 32 masked coordinates sit above them, so the 75th percentile lands inside the masked group. Over the next 400 steps a coordinate is released only when its instantaneous residual stays above the threshold, by more than `RESIDUAL_FLOOR` (`1e-12`), for 50 consecutive steps. The cooler masked coordinates do not qualify.

Average precision scores the ranking of that calibration residual, highest first. It is 1.0 when every masked coordinate outranks every agreeing one, which can happen while the percentile gate releases only the hotter part of the mask. In the seed 42–47 sweep from `python register.py` on the tested NumPy, every release run with `J > 0` has precision 1 and average precision 1, and recall lies between 0.50 and 0.59.

## Precision, recall, and average precision

A hit is a released coordinate that lies on the hidden mask. `false_releases` is `n_released - hits`.

- **Precision** is `hits / n_released` when anything was released. Precision is 1 when the released set is empty (no false releases yet).
- **Recall** is `hits / n_mask` when the mask is non-empty, and 0 when the mask is empty.
- **Average precision** ranks the calibration-time mean residual against the hidden mask. It is 1 when every disagreeing coordinate outranks every agreeing one. It is 0 when the mask is empty, because there is nothing to rank. Equal scores keep their original index order (`argsort` with `mergesort`). At `J = 0` every residual is zero, so the seed-42 value is that tie-break among zeros.

## Install

Tested on Python 3.12.3 with NumPy 1.26.4.

```bash
python -m pip install -r requirements-dev.txt
```

`requirements.txt` declares NumPy (`numpy>=1.26,<3`). `requirements-dev.txt` adds pytest (`pytest>=8,<10`) for the test run.

## Run

```bash
python test_register.py
pytest
python register.py
```

`python test_register.py` runs the five checks and prints `all register checks passed`. `pytest` collects those same tests. `python register.py` writes `results/register_scan.csv` and `results/register_summary.txt`, and prints the summary. The `results/` directory is gitignored.

## `results/register_summary.txt`

Text report for the seed-42 grid and a small multi-seed check.

The header restates the setup (`n=64`, `n_diff=32`, `dt`, calibration length, run length, persistence, percentile) and two conventions: precision is 1 when the released set is empty, and average precision ranks the calibration residual against the hidden mask.

- **release curve, seed 42.** One line per `J`. Fields are `released`, `false`, `precision`, `recall`, `AP`, `residD`, `residAgree`, `lossD`, and `disagreeD`. `residD` and `residAgree` are the mean calibration residuals on the mask and off it.
- **hybrid, identical, and memory.** One line per run of that mode. Fields are `released`, `false`, `coupled`, `lossD`, `disagreeD`, `lossAgree`, and `disagreeAgree`. `coupled` is how many coordinates are still shared at the end (64 when the rule releases nothing). Memory contributes one line, `J=0.00`.
- **extra release seeds.** Seeds 43–47, each run at the first positive `J` on the seed-42 release curve that released at least one coordinate, and at `J = 1` and `J = 2`. Fields are `released`, `false`, `recall`, and `AP`.

`results/register_scan.csv` is the same runs as a table. Columns: `seed`, `mode`, `align_j`, `n_released`, `n_mask`, `false_releases`, `precision`, `recall`, `average_precision`, `threshold`, `residual_d`, `residual_agree`, `disagree_d`, `disagree_agree`, `loss_d`, `loss_agree`, `still_coupled`.
