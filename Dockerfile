FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

LABEL org.opencontainers.image.description = "Grug is a self-hosted Discord AI Agent designed to assist groups who play TTRPGs."

# Install the project into /app
WORKDIR /app

# ensure that all uv commands within the Dockerfile compile bytecode:
ENV UV_COMPILE_BYTECODE=1
# silences warnings about not being able to use hard links
ENV UV_LINK_MODE=copy
# Disable automatic downloads of Python
ENV UV_PYTHON_DOWNLOADS=never

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

CMD ["start-grug"]

# TODO: Add a healthcheck
