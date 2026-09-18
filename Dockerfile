FROM python:3.12-slim-bookworm

WORKDIR /app

# Create non-root user early so we can own /app
RUN useradd --create-home appuser && chown appuser:appuser /app

# Install uv without retaining a pip cache in the image
RUN pip install --no-cache-dir uv

# Copy dependency files first (Docker layer caching)
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Switch to non-root user before installing deps (so .venv is owned by appuser)
USER appuser

# Install dependencies (production only)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY --chown=appuser:appuser app/ app/
# Include the source documents used to build the retrieval index.
COPY --chown=appuser:appuser docs/ docs/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run uvicorn directly from venv (avoids uv re-syncing at runtime)
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]