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

The recall ceiling is `RESIDUAL_PERCENTILE = 75`. On seeds 42–47 the 50-step persistence cap leaves recall unchanged.

Calibration averages the residual over 400 steps with every coordinate coupled. The threshold is a percentile of those 64 means. Agreeing means are exactly 0 and masked means are positive, so the 75th percentile falls inside the masked group. The measured run compares the instantaneous residual with that threshold. Near equilibrium the instantaneous value sits above the calibration mean, so a few masked coordinates whose calibration mean is under the percentile still stay hot for every run step. Coordinates whose instantaneous residual stays under the threshold stay cold for the whole run.

On these seeds, at every `J` from 0.1 to 2.0, that hot set is fixed for all 400 run steps. A coordinate above the threshold stays above it on every step, and a coordinate below it never crosses. Recall is therefore the same at every persistence from 1 to 400. `PERSISTENCE = 50` is inside that range and does not change the released set. The ceiling is the percentile alone.

Sweeping the percentile on the same 6 seeds and the same 20 values of `J` (120 runs; persistence, `dt`, and the step counts left at their defaults) moves mean recall as follows. Every run has 0 false releases and average precision 1. The denominator 3840 is 120 runs times 32 masked coordinates.

| Percentile | Mean recall |
| --- | --- |
| 25 | 1 (3840/3840) |
| 50 | 1 (3840/3840) |
| 60 | 0.825 (3167/3840) |
| 75 | 0.539 (2069/3840) |
| 90 | 0.259 (993/3840) |

At the default 75th percentile, recall on an individual run lies between 0.50 and 0.59.

## Precision, recall, and average precision

A hit is a released coordinate that lies on the hidden mask. `false_releases` is `n_released - hits`.

- **Precision** is `hits / n_released` when anything was released. Precision is 1 when the released set is empty (no false releases yet).
- **Recall** is `hits / n_mask` when the mask is non-empty, and 0 when the mask is empty.
- **Average precision** ranks the calibration-time mean residual against the hidden mask. It is 1 when every disagreeing coordinate outranks every agreeing one. It is 0 when the mask is empty, because there is nothing to rank. Equal scores keep their original index order (`argsort` with `mergesort`). At `J = 0` every residual is zero, so the seed-42 value is that tie-break among zeros.

For `J > 0`, zero false releases and average precision 1 follow from the residual. Agreeing coordinates start at the same value and share a target, so the two copies stay equal. Their midpoint pull is 0, and their coupling residual is exactly 0 on every step. A release requires a residual strictly above the threshold by `RESIDUAL_FLOOR`, and the threshold is a percentile of nonnegative values, so an agreeing coordinate stays unreleased. Masked calibration residuals are positive while the coupling term is on, and each of them outranks every agreeing coordinate.

The `J = 0` line and the memory null skip the coupling term, so the residual is 0 on the mask as well and the rule releases nothing.

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

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## How to cite

Anyone using this code or its results is asked to cite this work. If you use this software or its results, please cite it as below.

```bibtex
@software{Bell2026frustration-couple,
  author       = {Bell, Jason},
  organization = {Bell Corporate Labs LLC},
  title        = {frustration-couple},
  year         = {2026},
  date         = {2026-11-26},
  version      = {0.1.0},
  license      = {MIT},
  url          = {https://github.com/Bell8571/frustration-couple}
}
```

The same metadata is in [CITATION.cff](CITATION.cff). Jason Bell is affiliated with Bell Corporate Labs LLC.
