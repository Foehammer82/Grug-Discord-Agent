# Stage 1: Build
FROM postgres:17-alpine AS build

LABEL org.opencontainers.image.description "A PostgreSQL image with pgvector and pgmq extensions pre-installed."

ENV PGUSER=postgres

RUN apk --no-cache add python3 pipx make gcc musl-dev clang19 llvm

# INSTALL: pgvector
RUN pipx run pgxnclient install vector

# INSTALL: pgmq
RUN pipx run pgxnclient install pgmq

# Stage 2: Runtime
FROM postgres:17-alpine

ENV PGUSER=postgres

# Copy the necessary files from the build stage
COPY --from=build /usr/local /usr/local

# Enable Container Healthcheck
HEALTHCHECK --interval=30s --timeout=60s --retries=5 --start-period=80s \
  CMD pg_isready -d db_prod || exit 1
