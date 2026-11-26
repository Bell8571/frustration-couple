"""Two updaters, one shared register, a blind agreement pull.

L descends ||y - a||^2. R descends ||y - b||^2. On a hidden mask the targets
differ. The release rule never reads that mask. It watches the absolute
agreement correction and drops a coordinate whose residual stays high.

J = 0 is the null: the two sides may disagree, but the coupling correction is
identically zero, so nothing is released. Disabling release is the hybrid:
they keep negotiating and pay internal loss on the mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_COORDINATES = 64
N_DIFFER = 32
DT = 0.05
CALIBRATION_STEPS = 400
RUN_STEPS = 400
PERSISTENCE = 50
RESIDUAL_PERCENTILE = 75.0
RESIDUAL_FLOOR = 1e-12
MOMENTUM = 0.9


def make_problem(
    n: int,
    n_diff: int,
    rng: np.random.Generator,
    *,
    identical: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Targets a, b and the hidden disagreement mask.

    Compatible coordinates share a target. Incompatible coordinates get a gap
    drawn from [0.5, 2]. The mask is not passed to the release rule.
    """
    targets = rng.uniform(-1.0, 1.0, size=n)
    partner = targets.copy()
    mask = np.zeros(n, dtype=bool)
    if identical or n_diff <= 0:
        return targets, partner, mask
    chosen = rng.choice(n, size=n_diff, replace=False)
    gaps = rng.uniform(0.5, 2.0, size=n_diff) * rng.choice(np.array([-1.0, 1.0]), size=n_diff)
    partner[chosen] = targets[chosen] + gaps
    mask[chosen] = True
    return targets, partner, mask


def _advance(
    y_left: np.ndarray,
    y_right: np.ndarray,
    v_left: np.ndarray,
    v_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    coupled: np.ndarray,
    align_j: float,
    momentum: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One Euler step. Residual is the absolute agreement rate, before dt."""
    pull_left = np.zeros_like(y_left)
    pull_right = np.zeros_like(y_right)
    if align_j != 0.0 and np.any(coupled):
        middle = 0.5 * (y_left + y_right)
        pull_left[coupled] = align_j * (middle[coupled] - y_left[coupled])
        pull_right[coupled] = align_j * (middle[coupled] - y_right[coupled])
    residual = np.maximum(np.abs(pull_left), np.abs(pull_right))
    force_left = -(y_left - target_left) + pull_left
    force_right = -(y_right - target_right) + pull_right
    if momentum == 0.0:
        y_left = y_left + dt * force_left
        y_right = y_right + dt * force_right
    else:
        v_left = momentum * v_left + dt * force_left
        v_right = momentum * v_right + dt * force_right
        y_left = y_left + dt * v_left
        y_right = y_right + dt * v_right
    return y_left, y_right, v_left, v_right, residual


def _losses(
    y_left: np.ndarray,
    y_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    disagree = np.abs(y_left - y_right)
    loss = 0.5 * ((y_left - target_left) ** 2 + (y_right - target_right) ** 2)
    rest = ~mask

    def _mean(values: np.ndarray, chosen: np.ndarray) -> float:
        if not np.any(chosen):
            return 0.0
        return float(np.mean(values[chosen]))

    return {
        "disagree_d": _mean(disagree, mask),
        "disagree_agree": _mean(disagree, rest),
        "loss_d": _mean(loss, mask),
        "loss_agree": _mean(loss, rest),
    }


def run_case(
    target_left: np.ndarray,
    target_right: np.ndarray,
    mask: np.ndarray,
    *,
    align_j: float,
    release: bool,
    momentum: float = 0.0,
    calibration_steps: int = CALIBRATION_STEPS,
    run_steps: int = RUN_STEPS,
    persistence: int = PERSISTENCE,
    percentile: float = RESIDUAL_PERCENTILE,
    dt: float = DT,
) -> dict[str, float | int]:
    """Calibrate a residual threshold, then optionally release hot coordinates."""
    n = target_left.shape[0]
    y_left = np.zeros(n, dtype=np.float64)
    y_right = np.zeros(n, dtype=np.float64)
    v_left = np.zeros(n, dtype=np.float64)
    v_right = np.zeros(n, dtype=np.float64)
    coupled = np.ones(n, dtype=bool)
    residual_sum = np.zeros(n, dtype=np.float64)

    for _ in range(calibration_steps):
        y_left, y_right, v_left, v_right, residual = _advance(
            y_left, y_right, v_left, v_right,
            target_left, target_right, coupled, align_j, momentum, dt,
        )
        residual_sum += residual

    mean_residual = residual_sum / calibration_steps
    threshold = float(np.percentile(mean_residual, percentile))
    streak = np.zeros(n, dtype=np.int32)
    released = np.zeros(n, dtype=bool)

    for _ in range(run_steps):
        y_left, y_right, v_left, v_right, residual = _advance(
            y_left, y_right, v_left, v_right,
            target_left, target_right, coupled, align_j, momentum, dt,
        )
        if not release:
            continue
        hot = residual > threshold + RESIDUAL_FLOOR
        streak = np.where(hot, streak + 1, 0)
        newly = (streak >= persistence) & coupled
        released[newly] = True
        coupled[newly] = False
        streak[newly] = 0

    hits = int(np.count_nonzero(released & mask))
    n_released = int(np.count_nonzero(released))
    n_mask = int(np.count_nonzero(mask))
    false_releases = n_released - hits
    precision = 1.0 if n_released == 0 else hits / n_released
    recall = 0.0 if n_mask == 0 else hits / n_mask
    ranking = _average_precision(mean_residual, mask)
    measured = _losses(y_left, y_right, target_left, target_right, mask)
    rest = ~mask
    return {
        "align_j": float(align_j),
        "n_released": n_released,
        "n_mask": n_mask,
        "false_releases": false_releases,
        "precision": float(precision),
        "recall": float(recall),
        "average_precision": ranking,
        "threshold": threshold,
        "residual_d": _mean_where(mean_residual, mask),
        "residual_agree": _mean_where(mean_residual, rest),
        "still_coupled": int(np.count_nonzero(coupled)),
        **measured,
    }


def _mean_where(values: np.ndarray, chosen: np.ndarray) -> float:
    if not np.any(chosen):
        return 0.0
    return float(np.mean(values[chosen]))


def _average_precision(score: np.ndarray, mask: np.ndarray) -> float:
    """Ranking of the calibration residual against the hidden mask.

    1 means every disagreeing coordinate outranks every agreeing one.
    0 when the mask is empty, because there is nothing to rank.
    """
    n_mask = int(np.count_nonzero(mask))
    if n_mask == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    ranked = mask[order]
    hits = 0
    total = 0.0
    for rank, is_hit in enumerate(ranked, start=1):
        if is_hit:
            hits += 1
            total += hits / rank
    return float(total / n_mask)


def evaluate_grid(
    align_values: tuple[float, ...],
    *,
    seed: int,
    n: int = N_COORDINATES,
    n_diff: int = N_DIFFER,
) -> list[dict]:
    """Release sweep, hybrid control, identical-target control, and J=0 memory."""
    rng = np.random.default_rng(seed)
    left, right, mask = make_problem(n, n_diff, rng)
    same_left, same_right, same_mask = make_problem(n, n_diff, np.random.default_rng(seed), identical=True)
    rows: list[dict] = []
    for align_j in align_values:
        for mode, targets, this_mask, release, momentum in (
            ("release", (left, right), mask, True, 0.0),
            ("hybrid", (left, right), mask, False, 0.0),
            ("identical", (same_left, same_right), same_mask, True, 0.0),
        ):
            row = run_case(
                targets[0], targets[1], this_mask,
                align_j=align_j, release=release, momentum=momentum,
            )
            row["mode"] = mode
            row["seed"] = seed
            rows.append(row)
    memory = run_case(
        left, right, mask, align_j=0.0, release=True, momentum=MOMENTUM,
    )
    memory["mode"] = "memory"
    memory["seed"] = seed
    rows.append(memory)
    return rows


def _format_row(row: dict) -> str:
    return (
        f"{row['seed']},{row['mode']},{row['align_j']:.2f},{row['n_released']},"
        f"{row['n_mask']},{row['false_releases']},{row['precision']:.6f},"
        f"{row['recall']:.6f},{row['average_precision']:.6f},{row['threshold']:.8e},"
        f"{row['residual_d']:.8e},{row['residual_agree']:.8e},"
        f"{row['disagree_d']:.8f},{row['disagree_agree']:.8e},"
        f"{row['loss_d']:.8f},{row['loss_agree']:.8e},{row['still_coupled']}"
    )


def write_report(rows: list[dict], path: Path) -> str:
    header = (
        "seed,mode,align_j,n_released,n_mask,false_releases,precision,recall,"
        "average_precision,threshold,residual_d,residual_agree,disagree_d,"
        "disagree_agree,loss_d,loss_agree,still_coupled"
    )
    body = "\n".join(_format_row(row) for row in rows)
    path.write_text(header + "\n" + body + "\n", encoding="utf-8")
    return header


def summarize(rows: list[dict]) -> str:
    """Text curve for the seed-42 release mode and the four controls."""
    lines = [
        "Shared register. Blind agreement. Release from the coupling residual only.",
        f"n={N_COORDINATES} n_diff={N_DIFFER} dt={DT} calibration={CALIBRATION_STEPS} "
        f"run={RUN_STEPS} persistence={PERSISTENCE} percentile={RESIDUAL_PERCENTILE}",
        "precision is 1 when the released set is empty (no false releases yet).",
        "average_precision ranks the calibration residual against the hidden mask.",
    ]
    release_rows = [row for row in rows if row["mode"] == "release" and row["seed"] == 42]
    lines.append("release curve, seed 42:")
    for row in release_rows:
        lines.append(
            f"  J={row['align_j']:.2f} released={row['n_released']} "
            f"false={row['false_releases']} precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} AP={row['average_precision']:.3f} "
            f"residD={row['residual_d']:.4f} residAgree={row['residual_agree']:.4e} "
            f"lossD={row['loss_d']:.4f} disagreeD={row['disagree_d']:.4f}"
        )
    for mode in ("hybrid", "identical", "memory"):
        chosen = [row for row in rows if row["mode"] == mode and row["seed"] == 42]
        lines.append(f"{mode}:")
        for row in chosen:
            lines.append(
                f"  J={row['align_j']:.2f} released={row['n_released']} "
                f"false={row['false_releases']} coupled={row['still_coupled']} "
                f"lossD={row['loss_d']:.4f} disagreeD={row['disagree_d']:.4f} "
                f"lossAgree={row['loss_agree']:.4e} disagreeAgree={row['disagree_agree']:.4e}"
            )
    extra = [row for row in rows if row["mode"] == "release" and row["seed"] != 42]
    if extra:
        lines.append("extra release seeds:")
        for row in extra:
            lines.append(
                f"  seed={row['seed']} J={row['align_j']:.2f} "
                f"released={row['n_released']} false={row['false_releases']} "
                f"recall={row['recall']:.3f} AP={row['average_precision']:.3f}"
            )
    return "\n".join(lines)


def main() -> None:
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    align_values = tuple(round(0.1 * k, 2) for k in range(0, 21))
    rows = evaluate_grid(align_values, seed=42)
    release_rows = [row for row in rows if row["mode"] == "release"]
    positive = [row for row in release_rows if row["align_j"] > 0.0 and row["n_released"] > 0]
    onset = positive[0]["align_j"] if positive else align_values[-1]
    probe = tuple(sorted({onset, 1.0, 2.0}))
    for seed in range(43, 48):
        rng = np.random.default_rng(seed)
        left, right, mask = make_problem(N_COORDINATES, N_DIFFER, rng)
        for align_j in probe:
            row = run_case(left, right, mask, align_j=align_j, release=True)
            row["mode"] = "release"
            row["seed"] = seed
            rows.append(row)
    write_report(rows, out / "register_scan.csv")
    text = summarize(rows)
    (out / "register_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
