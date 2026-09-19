FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=5003

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser

COPY requirements.txt .
# This service uses CPU inference. Install a CPU-only Torch version compatible
# with current transformers/sentence-transformers releases. Keep this pinned
# because the broad torch>=2.2 requirement can otherwise leave an old Torch
# version in the image while the other ML packages continue to upgrade.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.5.1" \
    && python -m pip install --no-cache-dir --requirement requirements.txt

COPY --chown=appuser:appgroup . .

# Runtime directories must remain writable when no external data volume is mounted.
RUN mkdir -p /app/data/cache /app/data/documents /app/data/logs \
    && chown -R appuser:appgroup /app/data
USER appuser

EXPOSE 5003

STOPSIGNAL SIGTERM

# One worker is intentional: conversation memory and the BM25 index are process-local.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://127.0.0.1:5003/ready || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "5003", "--workers", "1", "--proxy-headers"]
