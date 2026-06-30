"""TPU / device topology helpers.

Works on any device list — CPU (1 CPU) for smoke tests, Kaggle TPU v5e-8
(8 TPU devices) for real runs. Same code path, no branching.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def setup_devices() -> list[jax.Device]:
    """Return the list of devices visible to JAX (CPU or TPU)."""
    return list(jax.devices())


def make_mesh(devices: list[jax.Device] | None = None) -> Mesh:
    """1D mesh over the batch axis. JAX global array default style."""
    if devices is None:
        devices = setup_devices()
    return Mesh(np.asarray(devices), axis_names=("batch",))


def make_input_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding for input_ids / target_ids: split batch across cores, replicate seq."""
    return NamedSharding(mesh, P("batch", None))


def make_param_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding for params: fully replicated (data-parallel)."""
    return NamedSharding(mesh, P())


def make_loss_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding for scalar loss: replicated."""
    return NamedSharding(mesh, P())


@contextmanager
def tpu_context() -> Iterator[Mesh]:
    """Context manager: sets up devices + mesh, yields mesh, cleans up on exit."""
    devices = setup_devices()
    mesh = make_mesh(devices)
    with mesh:
        yield mesh
