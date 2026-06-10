"""Tests for dataset split logic. Skipped when the ml extra (torch) is absent."""

import pytest

pytest.importorskip("torch", reason="requires the `ml` extra")

from solarscan.data.infrared_modules import Sample, class_counts, stratified_split  # noqa: E402


def _make_samples() -> list[Sample]:
    # 3 classes with 100, 50, 10 samples — imbalanced like the real data.
    samples = []
    for label, n in [(0, 100), (1, 50), (2, 10)]:
        samples.extend(Sample(path=f"img_{label}_{i}.jpg", label=label) for i in range(n))
    return samples


def test_split_is_disjoint_and_covers_all():
    samples = _make_samples()
    train, val, test = stratified_split(samples, val_ratio=0.2, test_ratio=0.2, seed=0)
    paths = lambda xs: {s.path for s in xs}  # noqa: E731
    assert paths(train) & paths(val) == set()
    assert paths(train) & paths(test) == set()
    assert paths(val) & paths(test) == set()
    assert len(train) + len(val) + len(test) == len(samples)


def test_every_class_in_every_split():
    samples = _make_samples()
    train, val, test = stratified_split(samples, val_ratio=0.2, test_ratio=0.2, seed=0)
    for split in (train, val, test):
        assert set(class_counts(split)) == {0, 1, 2}


def test_split_is_deterministic():
    samples = _make_samples()
    a = stratified_split(samples, 0.2, 0.2, seed=7)
    b = stratified_split(samples, 0.2, 0.2, seed=7)
    assert [s.path for s in a[0]] == [s.path for s in b[0]]
