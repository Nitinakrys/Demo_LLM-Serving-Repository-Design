# LLM Serving Repository Design

## Your Three Models

| Model | Use Case | Approx Size | GPU Needs |
|-------|----------|-------------|-----------|
| Gemma 3 1B IT | Lightweight inference, edge | ~2 GB | 1× GPU |
| MedGemma 1.5 4B | Medical Q&A, translation, OCR | ~8 GB | 1× GPU |
| MedGemma 27B | Advanced medical reasoning | ~54 GB | 2–4× GPU |

---

## Repository Structure

```
llm-serving/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Lint, unit tests, Docker build on every PR
│       ├── deploy-dev.yml            # Auto-deploy to DEV on merge to dev
│       ├── deploy-qa.yml             # Auto-deploy to QA on merge to qa
│       └── deploy-prod.yml           # Deploy to PROD on merge to main (with approval)
│
├── models/
│   ├── gemma-1b-it/
│   │   ├── config.env                # MODEL_NAME, MODEL_PATH, TENSOR_PARALLEL_SIZE, MAX_MODEL_LEN
│   │   └── README.md                 # Model card: what it does, limits, versions
│   │
│   ├── medgemma-4b/
│   │   ├── config.env
│   │   └── README.md
│   │
│   └── medgemma-27b/
│       ├── config.env
│       └── README.md
│
├── deployment/
│   ├── dev/
│   │   ├── gemma-1b-it.env           # DEV overrides: LOG_LEVEL=DEBUG, GPU_COUNT=1
│   │   ├── medgemma-4b.env
│   │   └── medgemma-27b.env
│   │
│   ├── qa/
│   │   ├── gemma-1b-it.env           # QA overrides: LOG_LEVEL=INFO
│   │   ├── medgemma-4b.env
│   │   └── medgemma-27b.env
│   │
│   └── prod/
│       ├── gemma-1b-it.env           # PROD overrides: LOG_LEVEL=WARNING
│       ├── medgemma-4b.env
│       └── medgemma-27b.env
│
├── benchmarks/
│   ├── datasets/
│   │   ├── medical_qa.json           # Standard medical Q&A test set
│   │   ├── translation_pairs.json    # Hindi/Tamil/etc. translation pairs
│   │   └── ocr_samples/              # Sample images for OCR testing
│   │
│   ├── gemma-1b-it/
│   │   └── results/                  # Versioned benchmark results
│   │
│   ├── medgemma-4b/
│   │   └── results/
│   │
│   └── medgemma-27b/
│       └── results/
│
├── tests/
│   ├── unit/
│   │   ├── test_config_loader.py
│   │   └── test_logger.py
│   │
│   ├── integration/
│   │   ├── test_health_endpoint.py
│   │   └── test_api_contract.py
│   │
│   └── inference/
│       ├── test_model_loading.py     # Does the model load? GPU detected?
│       ├── test_chat_completion.py   # Basic chat/completion works?
│       └── test_medical_qa.py        # Domain-specific sanity checks
│
├── scripts/
│   ├── benchmark.py                  # Run accuracy/latency/VRAM benchmarks
│   ├── health_check.py               # Hit /health endpoint, verify GPU status
│   ├── model_test.py                 # Quick inference smoke test
│   ├── deploy.sh                     # Pull image + restart container on target server
│   └── rollback.sh                   # Revert to previous image tag
│
├── src/
│   ├── custom_vllm_logger.py         # Shared across all models
│   └── start.sh                      # Generic entrypoint, reads config.env
│
├── docker/
│   ├── Dockerfile                    # Default: latest stable vLLM
│   └── Dockerfile.vllm028            # Pinned version if a model needs it
│
├── docker-compose.dev.yml            # Local dev: spin up one model for testing
├── .env.example                      # Template — never commit real secrets
├── .gitignore
└── README.md
```

---

## Branch Strategy

```
feature/*          ← developers work here
     │
     │  PR (code review)
     ▼
    dev            ← auto-deploys to DEV environment
     │
     │  PR (CI must pass: unit + integration + inference smoke test)
     ▼
    qa             ← auto-deploys to QA environment
     │
     │  PR (QA benchmarks must pass + manual approval)
     ▼
   main            ← auto-deploys to PRODUCTION
     │
     ▼
    TAG            ← e.g. medgemma-4b-v1.2.0
```

### Branch Rules

| Branch | Who merges | Required checks | Deploy target |
|--------|-----------|-----------------|---------------|
| `dev` | Any developer via PR | Unit tests, Docker build, lint | DEV GPU server |
| `qa` | Dev lead via PR from dev | All CI + inference smoke test | QA GPU server |
| `main` | Tech lead via PR from qa | All CI + QA benchmarks + approval | PROD servers |

---

## Model Configuration Files

### models/gemma-1b-it/config.env

```bash
MODEL_NAME=gemma-1b-it
MODEL_PATH=/models/gemma-4-31b-it
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192
DTYPE=float16
GPU_MEMORY_UTILIZATION=0.85
```

### models/medgemma-4b/config.env

```bash
MODEL_NAME=medgemma-4b
MODEL_PATH=/models/medgemma-v1.5-4b
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192
DTYPE=float16
GPU_MEMORY_UTILIZATION=0.90
```

### models/medgemma-27b/config.env

```bash
MODEL_NAME=medgemma-27b
MODEL_PATH=/models/medgemma-27b
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=4096
DTYPE=bfloat16
GPU_MEMORY_UTILIZATION=0.92
```

---

## Environment Overrides

### deployment/dev/medgemma-4b.env

```bash
LOG_LEVEL=DEBUG
ENVIRONMENT=dev
SERVER_HOST=dev-gpu-01
GPU_COUNT=1
```

### deployment/qa/medgemma-4b.env

```bash
LOG_LEVEL=INFO
ENVIRONMENT=qa
SERVER_HOST=qa-gpu-01
GPU_COUNT=1
```

### deployment/prod/medgemma-4b.env

```bash
LOG_LEVEL=WARNING
ENVIRONMENT=prod
SERVER_HOST=server-130
GPU_COUNT=1
```

---

## Generic Dockerfile

```dockerfile
FROM vllm/vllm-openai:v0.28.0

WORKDIR /app

# Common code — same for every model
COPY src/start.sh src/custom_vllm_logger.py ./

RUN chmod +x start.sh

# No model-specific config baked in
# Everything comes from config.env at runtime

ENTRYPOINT []
CMD ["./start.sh"]
```

---

## Generic start.sh

```bash
#!/bin/bash
set -euo pipefail

# These come from config.env + environment overrides
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "========================================"
echo "  Model:    ${MODEL_NAME}"
echo "  Path:     ${MODEL_PATH}"
echo "  TP:       ${TENSOR_PARALLEL_SIZE}"
echo "  MaxLen:   ${MAX_MODEL_LEN}"
echo "  Dtype:    ${DTYPE}"
echo "  Env:      ${ENVIRONMENT:-unknown}"
echo "========================================"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --host 0.0.0.0 \
    --port 8000
```

---

## Server Mapping (Production)

```
                    Docker Image
                  llm-serving:v1.2.0
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Server A       Server B       Server C
       1× GPU         1× GPU         2–4× GPU
       Gemma 1B IT    MedGemma 4B    MedGemma 27B
       config.env     config.env     config.env
          │              │              │
         vLLM           vLLM           vLLM
          │              │              │
        :8000          :8000          :8000
```

---

## Tagging Strategy

Every production release gets a tag:

```
gemma-1b-it-v1.0.0
gemma-1b-it-v1.1.0

medgemma-4b-v1.0.0
medgemma-4b-v1.1.0
medgemma-4b-v2.0.0      ← model version upgrade

medgemma-27b-v1.0.0
```

Tag format: `{model}-v{major}.{minor}.{patch}`

- **major** → model version change (e.g. MedGemma v1.5 → v2.0)
- **minor** → config change, vLLM upgrade, new features
- **patch** → bug fix, logging change, minor tweak

### Docker image tags

```
registry/llm-serving:v1.2.0              ← code version
registry/llm-serving:v1.2.0-sha-abc123   ← with git commit
```

---

## CI/CD Workflows

### .github/workflows/ci.yml — Runs on every PR

```yaml
name: CI

on:
  pull_request:
    branches: [dev, qa, main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/ -v

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -f docker/Dockerfile -t llm-serving:test .

  config-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate all model configs
        run: |
          for config in models/*/config.env; do
            echo "Validating $config"
            source "$config"
            [ -z "$MODEL_NAME" ] && echo "FAIL: MODEL_NAME missing in $config" && exit 1
            [ -z "$MODEL_PATH" ] && echo "FAIL: MODEL_PATH missing in $config" && exit 1
          done
          echo "All configs valid"
```

### .github/workflows/deploy-dev.yml

```yaml
name: Deploy to DEV

on:
  push:
    branches: [dev]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      gemma-1b: ${{ steps.changes.outputs.gemma-1b }}
      medgemma-4b: ${{ steps.changes.outputs.medgemma-4b }}
      medgemma-27b: ${{ steps.changes.outputs.medgemma-27b }}
      common: ${{ steps.changes.outputs.common }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: changes
        with:
          filters: |
            gemma-1b:
              - 'models/gemma-1b-it/**'
              - 'deployment/dev/gemma-1b-it.env'
            medgemma-4b:
              - 'models/medgemma-4b/**'
              - 'deployment/dev/medgemma-4b.env'
            medgemma-27b:
              - 'models/medgemma-27b/**'
              - 'deployment/dev/medgemma-27b.env'
            common:
              - 'src/**'
              - 'docker/**'
              - 'Dockerfile'

  build:
    runs-on: ubuntu-latest
    needs: detect-changes
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker image
        run: |
          IMAGE_TAG="dev-$(git rev-parse --short HEAD)"
          docker build -f docker/Dockerfile -t registry/llm-serving:${IMAGE_TAG} .
          docker push registry/llm-serving:${IMAGE_TAG}

  deploy-gemma-1b:
    needs: [detect-changes, build]
    if: needs.detect-changes.outputs.gemma-1b == 'true' || needs.detect-changes.outputs.common == 'true'
    runs-on: ubuntu-latest
    environment: dev
    steps:
      - name: Deploy Gemma 1B IT to DEV
        run: echo "SSH to dev server → pull image → restart gemma-1b-it container"

  deploy-medgemma-4b:
    needs: [detect-changes, build]
    if: needs.detect-changes.outputs.medgemma-4b == 'true' || needs.detect-changes.outputs.common == 'true'
    runs-on: ubuntu-latest
    environment: dev
    steps:
      - name: Deploy MedGemma 4B to DEV
        run: echo "SSH to dev server → pull image → restart medgemma-4b container"

  deploy-medgemma-27b:
    needs: [detect-changes, build]
    if: needs.detect-changes.outputs.medgemma-27b == 'true' || needs.detect-changes.outputs.common == 'true'
    runs-on: ubuntu-latest
    environment: dev
    steps:
      - name: Deploy MedGemma 27B to DEV
        run: echo "SSH to dev server → pull image → restart medgemma-27b container"
```

### .github/workflows/deploy-prod.yml

```yaml
name: Deploy to PROD

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production    # ← requires approval in GitHub settings
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to production servers
        run: |
          echo "Deploying approved image to production"
          # Same pattern as DEV but targets prod servers
```

---

## QA Evaluation Matrix

When a PR reaches `qa`, run these checks per model:

### Model Loading

| Check | Gemma 1B IT | MedGemma 4B | MedGemma 27B |
|-------|:-----------:|:-----------:|:------------:|
| Downloads from cache | ✓ | ✓ | ✓ |
| Loads into GPU | ✓ | ✓ | ✓ |
| Tokenizer works | ✓ | ✓ | ✓ |
| CUDA detected | ✓ | ✓ | ✓ |

### Functional

| Check | Gemma 1B IT | MedGemma 4B | MedGemma 27B |
|-------|:-----------:|:-----------:|:------------:|
| Chat completion | ✓ | ✓ | ✓ |
| Medical Q&A | — | ✓ | ✓ |
| Translation (Indian langs) | — | ✓ | ✓ |
| OCR | — | ✓ | — |
| General reasoning | ✓ | — | ✓ |

### Performance (compare against previous version)

| Metric | Gemma 1B IT | MedGemma 4B | MedGemma 27B |
|--------|-------------|-------------|--------------|
| TTFT | < 200ms | < 500ms | < 1s |
| Tokens/sec | > 80 | > 40 | > 20 |
| Latency (p95) | < 1s | < 3s | < 5s |
| VRAM | < 4 GB | < 12 GB | < 48 GB |

### Regression Report (auto-generated)

```
                     Old         New        Verdict
Accuracy (med QA)    84.2%       87.1%      ✅ improved
Latency (p95)        2.1s        2.3s       ⚠️ 9.5% slower
VRAM                 10.2 GB     10.5 GB    ✅ within budget
Tokens/sec           42          39         ⚠️ 7% slower

Overall: PASS (accuracy improved, perf within tolerance)
```

---

## Rollback Procedure

```
Production problem detected
         │
         ▼
Check current tag
   medgemma-4b-v1.2.0
         │
         ▼
Rollback command
   ./scripts/rollback.sh medgemma-4b v1.1.0
         │
         ▼
Server pulls previous image
   registry/llm-serving:v1.1.0
         │
         ▼
Restart container with old config
         │
         ▼
Verify health check
         │
         ▼
Done — investigate the issue on dev
```

---

## Key Principles

1. **One repo, one Docker image, three configs.** The same `llm-serving` image runs on all servers. Only `config.env` differs.

2. **Branches = release flow, not models.** `dev → qa → main` is about code maturity. Models are deployment targets configured via env files.

3. **Smart deployments.** If only `models/medgemma-4b/` changed, only the MedGemma 4B server gets redeployed. Gemma 1B IT and MedGemma 27B stay untouched.

4. **Model weights stay out of Git.** Weights live in HuggingFace / object storage / local cache on the GPU server. Git tracks code + config only.

5. **Tag every production release.** `medgemma-4b-v1.2.0` tells you exactly what's running and lets you rollback in seconds.

6. **Secrets never in the repo.** Use GitHub Secrets + environment variables on the server. Commit `.env.example`, never `.env`.
