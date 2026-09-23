FROM ghcr.io/astral-sh/uv:0.12.18@sha256:3adc3706091ce7c2fe595e669628caedd6d951551b92b258b7e7dbe06d9440bc AS uv
FROM python:3.12-alpine@sha256:4c47124a8391cb7a9f571164147d154777cf012a4ece5f86097130d7a4478111 AS builder

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1 \
    UV_LINK_MODE=copy

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY optimizer ./optimizer
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-alpine@sha256:4c47124a8391cb7a9f571164147d154777cf012a4ece5f86097130d7a4478111 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    OPTIMIZER_CONFIG=/config/optimizer.yaml

RUN addgroup -S optimizer && adduser -S -G optimizer -h /app optimizer
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

RUN mkdir -p /config /data && chown -R optimizer:optimizer /app /config /data
USER optimizer

EXPOSE 4000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4000/health/ready', timeout=2)"]

ENTRYPOINT ["optimizer"]
CMD ["serve"]
