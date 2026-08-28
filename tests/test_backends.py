from pymerlin.backends import (
    executable_likelihood_backends,
    executable_multipoint_engines,
    list_backend_status,
)


def test_numpy_is_the_only_executable_likelihood_backend_for_now() -> None:
    assert executable_likelihood_backends() == ("numpy",)


def test_dense_and_tree_multipoint_engines_are_executable() -> None:
    assert executable_multipoint_engines() == ("dense", "tree")


def test_backend_status_reports_current_candidates() -> None:
    statuses = {backend.name: backend for backend in list_backend_status()}

    assert {"numpy", "cupy", "jax", "numba-cuda-mlir"} == statuses.keys()
    assert statuses["numpy"].available is True
    assert "float64" in statuses["cupy"].role
    assert "jax_enable_x64=True" in statuses["jax"].role
