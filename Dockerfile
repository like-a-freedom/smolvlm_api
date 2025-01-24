FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY pyproject.toml .
RUN uv pip compile pyproject.toml -o requirements.txt \
    && uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

FROM python:3.12-slim-bookworm

COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder --chown=app:app /app /app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV MODEL_CACHE_DIR=/app/models

RUN useradd -m appuser \
    && chown -R appuser:appuser /app \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER appuser
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]