"""Backend discovery and policy for accelerator experiments."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal, cast


MultipointEngine = Literal["dense", "tree"]


@dataclass(frozen=True)
class BackendInfo:
    """Availability and intended role for a compute backend."""

    name: str
    available: bool
    role: str


def list_backend_status() -> tuple[BackendInfo, ...]:
    """Report backend availability without importing heavyweight packages."""

    return (
        BackendInfo(
            "numpy",
            find_spec("numpy") is not None,
            "reference CPU array backend",
        ),
        BackendInfo(
            "cupy",
            find_spec("cupy") is not None,
            "primary CUDA float64 array prototype",
        ),
        BackendInfo(
            "jax",
            find_spec("jax") is not None,
            "experimental accelerator prototype; requires jax_enable_x64=True",
        ),
        BackendInfo(
            "numba-cuda-mlir",
            find_spec("numba_cuda_mlir") is not None,
            "custom deterministic CUDA kernel candidate",
        ),
    )


def executable_likelihood_backends() -> tuple[str, ...]:
    """Backends currently wired into likelihood and IBD calculations."""

    return ("numpy",)


def executable_multipoint_engines() -> tuple[MultipointEngine, ...]:
    """Return inheritance-state engines available for multipoint analysis."""

    return ("dense", "tree")


def validate_multipoint_engine(engine: str) -> MultipointEngine:
    """Validate and narrow one public multipoint engine name."""

    available_engines = executable_multipoint_engines()
    if engine not in available_engines:
        choices = ", ".join(available_engines)
        raise ValueError(
            f"Unknown multipoint engine {engine!r}; choose one of: {choices}."
        )
    return cast(MultipointEngine, engine)
