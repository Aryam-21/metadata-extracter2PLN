FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install .

FROM python:3.12-slim

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system app && useradd --system --gid app --home /app app
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
USER app
EXPOSE 8080
CMD ["uvicorn", "metadata_extractor2pln.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--no-server-header"]
