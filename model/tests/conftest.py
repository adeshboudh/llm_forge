"""Shared fixtures for model tests."""

from __future__ import annotations

import jax
import pytest


@pytest.fixture
def rng():
    """Deterministic JAX RNG key for tests."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def batch_shape():
    """Standard (B, T, D) shape for module tests."""
    return (2, 16, 64)
