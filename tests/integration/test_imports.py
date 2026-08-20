"""Gate G0's import clause, and the ADR-003 regression guard."""

from __future__ import annotations

import importlib.util

import pytest


def test_nixtla_stack_imports():
    import mlforecast
    import statsforecast
    import utilsforecast

    assert statsforecast.__version__ >= "2.0"
    assert mlforecast.__version__
    assert utilsforecast.__version__


def test_engine_declares_no_dependency_on_numba():
    """ADR-003 regression guard.

    statsforecast 2.0 dropped numba for compiled coreforecast. Asserted against package
    *metadata* rather than sys.modules, because both are environment-dependent in opposite
    directions and only metadata answers the question ADR-003 actually asks.

    Verified in Phase 0: a core install (no `explain` extra) contains neither numba nor
    llvmlite and the engine imports fine. In a full dev install numba IS present -- shap
    requires it -- and `fugue` then imports it opportunistically. That is a property of the
    explain layer, not of the engine, and it must not be allowed to migrate.
    """
    from importlib.metadata import requires

    for package in ("statsforecast", "mlforecast", "utilsforecast", "fugue", "triad"):
        declared = requires(package) or []
        hard = [r for r in declared if "extra ==" not in r]
        offenders = [r for r in hard if r.split()[0].lower() in {"numba", "llvmlite"}]
        assert not offenders, f"{package} now hard-depends on {offenders}"


def test_llvmlite_is_floored_above_the_cp311_boundary():
    """shap declares numba and llvmlite with NO lower bound, so a resolver backtracks to
    llvmlite 0.36 -- which predates cp311 wheels and then tries to build LLVM from source on
    Python 3.11. This is the numba/llvmlite friction the risk register warned about; it is
    real, and it lives in the explain extra rather than the engine. The pyproject floors are
    the mitigation and this is their regression guard."""
    if importlib.util.find_spec("llvmlite") is None:
        pytest.skip("explain extra not installed")
    from importlib.metadata import version

    from packaging.version import Version

    assert Version(version("llvmlite")) >= Version("0.43")
    assert Version(version("numba")) >= Version("0.60")


def test_schemas_import_without_the_heavy_stack():
    """schemas/ must not reach into statsforecast: the API and CLI validate requests long
    before any model is constructed, and the add-in path should not pay for a 300MB import."""
    import xlforecast.schemas  # noqa: F401
    from xlforecast.schemas.registry import MODEL_REGISTRY

    assert len(MODEL_REGISTRY) == 17


def test_utilsforecast_has_no_plain_crps():
    """FR-208's naming decision, pinned. What exists is scaled_crps (quantile-grid,
    normalised by the sum of actuals), which is why the leaderboard field is not called crps."""
    from utilsforecast import losses

    assert not hasattr(losses, "crps")
    assert hasattr(losses, "scaled_crps")
