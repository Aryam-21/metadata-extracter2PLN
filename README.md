# metadata-extractor2PLN

A headless, schema-adaptive metadata extraction service. It inspects JSON records, builds a typed extraction plan, applies deterministic and model-backed extractors, records evidence, and compiles accepted values into facts understood by PeTTaChainer.

The service does **not** execute MeTTa supplied by callers and does not write directly to a knowledge base. Its trust boundary ends at validated fact strings such as:

```metta
(: news_abc123 (engagement news_article-42 "High") (STV 0.75 0.9))
```

## What is implemented

- typed plan discovery with a deterministic fallback
- mandatory-property enforcement after model planning
- deterministic structured, numeric, date, length, reading-time, and engagement extraction
- Bedrock schema-constrained output for semantic properties, batched across each extraction request
- allowed-value and source-evidence validation
- PeTTaChainer-compatible fact compilation with idempotency keys
- authenticated bulk HTTP endpoints with request, concurrency, timeout, and rate limits

The first release accepts bounded inline JSON batches. Durable jobs, source connectors, plan storage, and delivery to downstream PeTTaChainer servers are deliberately left outside this initial trust boundary.

Engagement is calculated from weighted interactions: comments count twice,
shares count three times, and other reactions count once. When views are
available the service classifies the resulting engagement rate; otherwise it
classifies the weighted interaction count. Compiled facts expose both the complete PeTTaChainer
statement and its validated `atom`/truth-value fields for downstream adapters.

Plan discovery sends only bounded samples to the configured model. If it is unavailable or
returns output that fails the service contract, discovery falls back to the
deterministic planner. Semantic properties for all records in one `/v1/extract`
request are classified in one model request; each source text is capped at
12,000 characters before it crosses the model boundary.

## Run locally

Python 3.11 or newer and `uv` are recommended.

```bash
cp .env.example .env
# Edit .env, set a long API secret, choose a model provider, and set its key.
uv sync --extra dev
set -a; source .env; set +a
uv run uvicorn metadata_extractor2pln.api:app --host 127.0.0.1 --port 8080
```

`METADATA_API_KEYS` is a comma-separated list so multiple clients can be rotated independently. Each entry is `owner-id:secret`; callers send only the secret as the bearer token. It is unrelated to model-provider authentication.

Bedrock is the default provider. Choose a model that supports Bedrock structured
outputs, such as DeepSeek V3.2:

```dotenv
METADATA_BEDROCK_MODEL_ID=deepseek.v3.2
AWS_REGION=us-east-1
```

```bash
curl -s http://127.0.0.1:8080/health

curl -s http://127.0.0.1:8080/v1/run \
  -H 'Authorization: Bearer replace-with-at-least-32-random-characters' \
  -H 'Content-Type: application/json' \
  -d '{
    "namespace": "demo",
    "source_name": "articles",
    "records": [{
      "id": "article-42",
      "content": "A concise technical introduction for experienced engineers.",
      "likes": 18,
      "comments": 4
    }]
  }'
```

For a completely offline smoke test, set `"use_model": false` and `"required_properties": ["engagement"]`. Omit `required_properties` to let plan discovery choose properties from the supplied records; provide it only when specific properties must be present.

## API

- `GET /health` — process liveness, no authentication
- `GET /ready` — service and model-backend readiness, no authentication
- `POST /v1/plans/discover` — inspect samples and return a sanitized plan
- `POST /v1/plans/validate` — verify a pinned plan and its fingerprint
- `POST /v1/extract` — apply a previously generated plan to up to 100 records
- `POST /v1/run` — plan and extract one bounded batch

Generated plans contain a fingerprint. Changing a plan without regenerating its fingerprint causes `/v1/extract` to reject it. This prevents an audited plan from silently changing in transit.

## Development

```bash
uv run pytest
```

The optional integration test validates compiled facts with the sibling PeTTaChainer checkout when that package is available.

## Current security boundary

- JSON only; no arbitrary URL fetching or user-supplied code
- strict Pydantic request and model-output schemas
- post-model enforcement of required properties and known source paths
- exact enum matching and verification that evidence quotes occur in source text
- bearer authentication with per-owner rate limiting
- bounded record counts, body size, concurrency, and processing time
- no persistence and no automatic downstream mutation in v0.1

For public deployment, terminate TLS at a reverse proxy, enforce its own body/rate limits, inject secrets through the deployment platform, and run more than one replica behind a shared rate limiter. The in-process limiter is intentionally only a single-instance safeguard.
