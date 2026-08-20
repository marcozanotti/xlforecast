# syntax=docker/dockerfile:1.7
# Multi-stage. Base is bookworm (glibc 2.36) because coreforecast and statsforecast publish
# manylinux_2_27 / manylinux_2_28 wheels ONLY -- Alpine/musl cannot install this stack and has
# no sdist path worth taking. See docs/03-BUILD-PLAN.md risk register.

FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer, cached independently of source.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-extras --no-dev

COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --no-dev

# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# libgomp is required by LightGBM and XGBoost at import time; the slim image omits it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 xlf
WORKDIR /app

COPY --from=builder --chown=xlf:xlf /app/.venv /app/.venv
COPY --from=builder --chown=xlf:xlf /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# NFR-02: byte-identical leaderboards require a pinned thread configuration, because float
# reductions reorder under thread count. Recorded into Manifest.thread_config at runtime.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

USER xlf

# Gate G0: the stack must import cleanly in the container, and the engine must declare no
# hard dependency on numba (ADR-003).
#
# Asserted against package METADATA, not `find_spec`. This image installs --all-extras, so
# numba IS present -- shap requires it -- and an earlier version of this check asserted its
# absence and failed the build. Presence is environment-dependent and answers the wrong
# question; what ADR-003 actually claims is that the *engine* does not need numba, which is
# a property of the dependency declarations.
RUN python -c "\
import statsforecast, mlforecast, utilsforecast, lightgbm, xgboost; \
from importlib.metadata import requires; \
bad = {p: [r for r in (requires(p) or []) if 'extra ==' not in r \
           and r.split()[0].lower() in ('numba', 'llvmlite')] \
       for p in ('statsforecast', 'mlforecast', 'utilsforecast', 'fugue', 'triad')}; \
bad = {k: v for k, v in bad.items() if v}; \
assert not bad, f'engine now hard-depends on numba/llvmlite: {bad}'; \
print('G0 import gate OK -', statsforecast.__version__, mlforecast.__version__, utilsforecast.__version__)"

CMD ["python", "-c", "import xlforecast; print(xlforecast.__version__)"]
