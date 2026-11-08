"""Nulls for the shared register. The release rule may see only residuals."""

from __future__ import annotations

import numpy as np

import register


def test_j0_releases_nothing() -> None:
    rng = np.random.default_rng(42)
    left, right, mask = register.make_problem(64, 32, rng)
    assert int(np.count_nonzero(mask)) == 32
    row = register.run_case(left, right, mask, align_j=0.0, release=True)
    assert row["n_released"] == 0
    assert row["false_releases"] == 0
    assert row["residual_d"] == 0.0
    assert row["residual_agree"] == 0.0


def test_identical_targets_release_nothing() -> None:
    rng = np.random.default_rng(7)
    left, right, mask = register.make_problem(64, 32, rng, identical=True)
    assert not np.any(mask)
    assert np.array_equal(left, right)
    row = register.run_case(left, right, mask, align_j=1.0, release=True)
    assert row["n_released"] == 0
    assert row["residual_d"] == 0.0


def test_hybrid_stays_coupled_with_high_loss() -> None:
    left = np.array([0.0, 0.0, 1.0, -0.5], dtype=np.float64)
    right = np.array([0.0, 1.0, 1.0, 0.5], dtype=np.float64)
    mask = np.array([False, True, False, True])
    align_j = 2.0
    row = register.run_case(
        left, right, mask,
        align_j=align_j, release=False,
        calibration_steps=800, run_steps=800,
    )
    gap = np.abs(left - right)
    assert row["still_coupled"] == 4
    assert row["n_released"] == 0
    assert row["disagree_agree"] < 1e-8
    assert row["loss_agree"] < 1e-8
    expected_disagree = float(np.mean(gap[mask]) / (1.0 + align_j))
    expected_loss = float(np.mean((gap[mask] * align_j / (2.0 * (1.0 + align_j))) ** 2))
    assert abs(row["disagree_d"] - expected_disagree) < 1e-6
    assert abs(row["loss_d"] - expected_loss) < 1e-6
    assert row["loss_d"] > 0.05
    assert row["disagree_d"] < 0.5 * float(np.mean(gap[mask]))


def test_momentum_at_j0_releases_nothing() -> None:
    rng = np.random.default_rng(42)
    left, right, mask = register.make_problem(64, 32, rng)
    row = register.run_case(
        left, right, mask, align_j=0.0, release=True, momentum=register.MOMENTUM,
    )
    assert row["n_released"] == 0
    assert row["residual_d"] == 0.0


def test_release_hits_only_the_mask() -> None:
    rng = np.random.default_rng(42)
    left, right, mask = register.make_problem(64, 32, rng)
    row = register.run_case(left, right, mask, align_j=1.0, release=True)
    assert row["false_releases"] == 0
    assert row["n_released"] > 0
    assert row["precision"] == 1.0
    assert row["average_precision"] == 1.0


def main() -> None:
    test_j0_releases_nothing()
    test_identical_targets_release_nothing()
    test_hybrid_stays_coupled_with_high_loss()
    test_momentum_at_j0_releases_nothing()
    test_release_hits_only_the_mask()
    print("all register checks passed")


if __name__ == "__main__":
    main()
