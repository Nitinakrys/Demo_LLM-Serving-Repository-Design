# LLM Serving Repository Design

## Your Three Models

| Model | `MODEL_NAME` | Use Case | Approx Size | GPU Needs (dev/qa) | GPU Needs (prod) |
|-------|-------------|----------|-------------|---------------------|-------------------|
| Gemma 4 31B IT | `google/gemma-4-31B-it` | General text-only inference | ~62 GB (bf16) | TP=2 (2× L40S 48GB) | TP=1 (1× H200 141GB) |
| MedGemma 1.5 4B | `google/medgemma-1.5-4b-it` | Medical Q&A, translation, OCR | ~8 GB | TP=1 (1× L40S) | TP=1 (1× H200) |
| MedGemma 27B | `google/medgemma-27b-it` | Advanced medical reasoning | ~54 GB (bf16) | TP=2 (2× L40S 48GB) | TP=1 (1× H200 141GB) |

All three are multimodal except `gemma-4-31b-it`, which is text-only. All three are gated on Hugging Face.

---

## Repository Structure

Unlike a single-branch repo, **each branch here carries a different tree** — only its own environment folder and workflow file, not all three side by side. This is the actual layout on the `Dev` branch:

```
Demo_LLM-Serving-Repository-Design/
│
├── .github/
│   └── workflows/
│       └── deploy-dev.yml            # THIS BRANCH ONLY — Qa has deploy-qa.yml, main has deploy-prod.yml
│
├── models/
│   ├── gemma-4-31b-it/
│   │   └── config.env                # MODEL_NAME, GATED, MULTIMODAL — identical on every branch
│   ├── medgemma-1.5-4b-it/
│   │   └── config.env
│   └── medgemma-27b-it/
│       └── config.env
│
├── dev/                               # THIS BRANCH ONLY — Qa has qa/, main has main/
│   ├── gemma-4-31b-it.env             # GPU/environment tuning: TP, dtype, ctx len, GPU_COUNT, LOG_LEVEL
│   ├── medgemma-1.5-4b-it.env
│   └── medgemma-27b-it.env
│
├── scripts/
│   └── deploy.sh                      # merges models/<model>/config.env + <env-folder>/<model>.env, builds + runs the container
│
├── src/
│   ├── custom_vllm_logger.py          # shared JSON logging wrapper, same on every branch
│   └── start.sh                       # generic entrypoint, reads config from env vars, same on every branch
│
├── docker/
│   └── Dockerfile                     # one image for every model, same on every branch
│
├── .env                                # tracked with empty placeholder values (see below)
├── .env.example                        # VLLM_API_KEY / HUGGING_FACE_HUB_TOKEN template
├── .gitignore                          # ignores .secrets/ (deploy.sh's runtime secret mount, never .env itself)
└── README.md
```

There is no `benchmarks/`, `tests/`, `ci.yml`, `docker-compose.dev.yml`, `rollback.sh`, or per-model `README.md` in this repo today — see [Not Yet Implemented](#not-yet-implemented) if you want to add any of these later.

---

## Branch Strategy

```
Dev            ← auto-deploys to DEV environment (3 GPU VMs, one per model)
 │
 │  PR
 ▼
Qa             ← auto-deploys to QA environment (shared 2× L40S per model)
 │
 │  PR
 ▼
main           ← auto-deploys to PRODUCTION (1× H200 per model)
```

### Branch Rules

| Branch | Env folder on that branch | Workflow | GPU |
|--------|---------------------------|----------|-----|
| `Dev` | `dev/` | `deploy-dev.yml` | 3 VMs: 1× L40S (4B), 2× L40S (Gemma 31B), 2× L40S (MedGemma 27B) |
| `Qa` | `qa/` | `deploy-qa.yml` | 2× L40S per model |
| `main` | `main/` | `deploy-prod.yml` | 1× H200 per model |

> **Caveat:** because each branch's environment folder is named differently (`dev/` vs `qa/` vs `main/`), a plain `git merge Dev → Qa → main` will **not** cleanly promote environment config — the promoted branch would need to keep renaming the incoming folder, and would also pull in the source branch's workflow file. In practice, promote via `git cherry-pick` of the specific commits, then manually reconcile the destination branch's own env folder and workflow file. Shared code (`docker/`, `src/`, `models/`, `scripts/`) merges/cherry-picks cleanly since it's identical on every branch.

---

## Model Configuration Files

### models/gemma-4-31b-it/config.env

```bash
# Identity + model-level defaults only. GPU/environment-specific tuning
# (tensor-parallel-size, batch sizes, context length) lives in
# deployment/<env>/gemma-4-31b-it.env instead, since it varies by which GPU
# this actually runs on, not by the model itself.

MODEL_NAME=google/gemma-4-31B-it
# GATED on Hugging Face — accept Google's license on the model page with the
# account that owns HUGGING_FACE_HUB_TOKEN before deploying.
GATED=true
# Text-only — no vision encoder, so LIMIT_MM_PER_PROMPT is never set for
# this model in any environment.
MULTIMODAL=false
```

### models/medgemma-1.5-4b-it/config.env

```bash
# Identity + model-level defaults only. GPU/environment-specific tuning
# lives in deployment/<env>/medgemma-1.5-4b-it.env instead.

MODEL_NAME=google/medgemma-1.5-4b-it
# GATED on Hugging Face — accept Google's license/usage terms on the model
# page with the account that owns HUGGING_FACE_HUB_TOKEN before deploying.
GATED=true
# Multimodal (text + image) — used for medical Q&A, translation, OCR.
MULTIMODAL=true
```

### models/medgemma-27b-it/config.env

```bash
# Identity + model-level defaults only. GPU/environment-specific tuning
# lives in deployment/<env>/medgemma-27b-it.env instead — notably
# TENSOR_PARALLEL_SIZE, which drops from 2 (dev/qa, 2x L40S 48GB) to 1
# (prod, 1x H200 141GB) since the ~54GB of weights fits on a single H200.

MODEL_NAME=google/medgemma-27b-it
# GATED on Hugging Face — accept Google's license/usage terms on the model
# page with the account that owns HUGGING_FACE_HUB_TOKEN before deploying.
# If you specifically need the text-only variant, use
# google/medgemma-27b-text-it instead and set MULTIMODAL=false below.
GATED=true
# Multimodal (text + image) — advanced medical reasoning.
MULTIMODAL=true
```

---

## Environment Overrides (Dev branch, `dev/`)

### dev/gemma-4-31b-it.env

```bash
ENVIRONMENT=dev
GPU_TYPE=L40S
GPU_COUNT=2
LOG_LEVEL=DEBUG

TENSOR_PARALLEL_SIZE=2
DTYPE=float16
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.85
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=8192
```

### dev/medgemma-1.5-4b-it.env

```bash
ENVIRONMENT=dev
GPU_TYPE=L40S
GPU_COUNT=1
LOG_LEVEL=DEBUG

TENSOR_PARALLEL_SIZE=1
DTYPE=float16
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=8192
LIMIT_MM_PER_PROMPT={"image": 4}
```

### dev/medgemma-27b-it.env

```bash
ENVIRONMENT=dev
GPU_TYPE=L40S
GPU_COUNT=2
LOG_LEVEL=DEBUG

# ~54GB of BF16 weights don't fit on a single 48GB L40S, so TP=2 is
# required here, not optional (sharded to ~27GB/GPU).
TENSOR_PARALLEL_SIZE=2
DTYPE=bfloat16
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=8192
LIMIT_MM_PER_PROMPT={"image": 4}
```

`Qa`'s `qa/*.env` files are identical to these except `ENVIRONMENT=qa` and `LOG_LEVEL=INFO`. `main`'s `main/*.env` files run on a single H200 each — `TENSOR_PARALLEL_SIZE=1` even for `medgemma-27b-it` (fits on one H200's 141GB), with much higher `MAX_MODEL_LEN`/`MAX_NUM_SEQS` since there's more headroom.

---

## Generic Dockerfile

```dockerfile
# One image for every model — no model-specific code baked in. Everything
# comes from env vars at runtime (see src/start.sh). Built FROM the official
# vLLM image rather than a bare CUDA base + hand-pinned torch/transformers,
# since that image ships the exact tested combination for this vLLM release
# and already includes Gemma 3 / MedGemma (`Gemma3ForConditionalGeneration`)
# support.
#
# If `vllm serve` fails with an unrecognized model_type error, this tag
# predates that support — bump the tag below (see
# hub.docker.com/r/vllm/vllm-openai/tags) or use vllm/vllm-openai:latest.
FROM vllm/vllm-openai:v0.28.0

WORKDIR /app

# Common code — same for every model, every environment.
COPY src/start.sh src/custom_vllm_logger.py ./
RUN chmod +x start.sh

# The base image sets ENTRYPOINT ["vllm", "serve"]; reset it so CMD below
# runs our wrapper directly instead of being appended as arguments to
# `vllm serve`.
ENTRYPOINT []
CMD ["./start.sh"]
```

---

## Generic start.sh

```bash
#!/bin/bash
set -euo pipefail

# Generic entrypoint shared by every model. Nothing here is model- or
# GPU-specific — all of that comes from environment variables assembled by
# scripts/deploy.sh from models/<model>/config.env (identity) +
# deployment/<env>/<model>.env (GPU/environment-specific tuning).
#
# This is what lets the SAME image run gemma-4-31b-it and medgemma-27b-it
# tensor-parallel across 2 L40S in dev/qa, or as a single process on one
# H200 in prod — only the env vars change.

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required (set in models/<model>/config.env)}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# Optional — only set for multimodal models (e.g. '{"image": 4}'); omitted
# entirely for text-only models like gemma-4-31b-it.
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"

# Secrets are mounted as files (see scripts/deploy.sh), not passed as
# plaintext env vars, so they don't show up in `docker inspect`.
if [ -f /run/secrets/vllm_api_key ]; then
  VLLM_API_KEY="$(cat /run/secrets/vllm_api_key)"
fi
if [ -f /run/secrets/hf_token ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat /run/secrets/hf_token)"
fi

if [ -z "${VLLM_API_KEY:-}" ]; then
  echo "ERROR: VLLM_API_KEY not found at /run/secrets/vllm_api_key" >&2
  exit 1
fi
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "ERROR: HUGGING_FACE_HUB_TOKEN not found at /run/secrets/hf_token (required — all three models are gated on Hugging Face)" >&2
  exit 1
fi

echo "========================================"
echo "  Model:       ${MODEL_NAME}"
echo "  Env:         ${ENVIRONMENT:-unknown}"
echo "  GPU type:    ${GPU_TYPE:-unknown}"
echo "  TP:          ${TENSOR_PARALLEL_SIZE}"
echo "  MaxLen:      ${MAX_MODEL_LEN}"
echo "  Dtype:       ${DTYPE}"
echo "  MaxNumSeqs:  ${MAX_NUM_SEQS}"
echo "========================================"

ARGS=(
  --model "${MODEL_NAME}"
  --download-dir /var/lib/vllm_models/model
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --enable-chunked-prefill
  --enable-prefix-caching
  --block-size 16
  --no-enable-log-requests
  --port 8000
  --api-key "${VLLM_API_KEY}"
)

if [ -n "${LIMIT_MM_PER_PROMPT}" ]; then
  ARGS+=(--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}")
fi

exec python3 custom_vllm_logger.py "${ARGS[@]}"
```

---

## scripts/deploy.sh

Run on the target VM, from the repo root, on the branch matching that environment:

```bash
cp .env.example .env   # fill in VLLM_API_KEY and HUGGING_FACE_HUB_TOKEN
./scripts/deploy.sh <model> <env-folder>
# e.g. on the Dev branch:
./scripts/deploy.sh medgemma-27b-it dev
```

It resolves `models/<model>/config.env` + `<env-folder>/<model>.env`, then:
1. Writes `VLLM_API_KEY`/`HUGGING_FACE_HUB_TOKEN` to files under `.secrets/<model>-<env>/` (mounted read-only into the container) instead of passing them as plaintext `-e` env vars, so they don't show up in `docker inspect`.
2. Stops/removes any existing container named `<model>-<env>`.
3. `docker build`s the shared image and `docker run --gpus all`s it, binding port 8000.

---

## Server Mapping — Dev Environment

Dev is split across **three separate VMs**, one per model, since they don't share the same GPU count. Each VM needs its own self-hosted GitHub Actions runner label so `deploy-dev.yml` routes each model's deploy job to the right machine:

```
                    Docker Image (same image, different env vars)
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
       VM A                VM B                VM C
       1× L40S              2× L40S              2× L40S
       MedGemma 1.5 4B      Gemma 4 31B          MedGemma 27B
       runner: dev-4b       runner: dev-gemma    runner: dev-27b
          │                   │                   │
         vLLM                vLLM                vLLM
          │                   │                   │
        :8000               :8000               :8000
```

Register each VM's runner with its matching label, e.g. on VM A:
```bash
./config.sh --url https://github.com/Nitinakrys/Demo_LLM-Serving-Repository-Design --token <TOKEN> --labels self-hosted,dev-4b
```

`Qa` and `main` each deploy all three models to a single VM (tagged `qa-gpu` / `prod-gpu` respectively) — only `Dev` needs per-model runner labels, since it's the only environment split across multiple physical machines.

---

## CI/CD Workflow — .github/workflows/deploy-dev.yml (actual, current)

```yaml
name: Deploy to DEV

on:
  push:
    branches: [Dev]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      gemma-4-31b-it: ${{ steps.changes.outputs.gemma-4-31b-it }}
      medgemma-1-5-4b-it: ${{ steps.changes.outputs.medgemma-1-5-4b-it }}
      medgemma-27b-it: ${{ steps.changes.outputs.medgemma-27b-it }}
      common: ${{ steps.changes.outputs.common }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: changes
        with:
          filters: |
            gemma-4-31b-it:
              - 'models/gemma-4-31b-it/**'
              - 'dev/gemma-4-31b-it.env'
            medgemma-1-5-4b-it:
              - 'models/medgemma-1.5-4b-it/**'
              - 'dev/medgemma-1.5-4b-it.env'
            medgemma-27b-it:
              - 'models/medgemma-27b-it/**'
              - 'dev/medgemma-27b-it.env'
            common:
              - 'src/**'
              - 'docker/**'
              - 'scripts/**'

  deploy-gemma-4-31b-it:
    needs: detect-changes
    if: needs.detect-changes.outputs.gemma-4-31b-it == 'true' || needs.detect-changes.outputs.common == 'true'
    runs-on: [self-hosted, dev-gemma]
    environment: dev
    steps:
      - uses: actions/checkout@v4
      - name: Deploy gemma-4-31b-it to DEV
        run: ./scripts/deploy.sh gemma-4-31b-it dev

  # Job id can't contain dots (GitHub Actions requirement) — the model
  # identifier itself (medgemma-1.5-4b-it, matching models/ and dev/ paths)
  # keeps the dot; only the job id / output key swap it for a hyphen.
  deploy-medgemma-1-5-4b-it:
    needs: detect-changes
    if: needs.detect-changes.outputs.medgemma-1-5-4b-it == 'true' || needs.detect-changes.outputs.common == 'true'
    runs-on: [self-hosted, dev-4b]
    environment: dev
    steps:
      - uses: actions/checkout@v4
      - name: Deploy medgemma-1.5-4b-it to DEV
        run: ./scripts/deploy.sh medgemma-1.5-4b-it dev

  deploy-medgemma-27b-it:
    needs: detect-changes
    if: needs.detect-changes.outputs.medgemma-27b-it == 'true' || needs.detect-changes.outputs.common == 'true'
    runs-on: [self-hosted, dev-27b]
    environment: dev
    steps:
      - uses: actions/checkout@v4
      - name: Deploy medgemma-27b-it to DEV
        run: ./scripts/deploy.sh medgemma-27b-it dev
```

`deploy-qa.yml` and `deploy-prod.yml` follow the identical pattern — only the branch trigger (`Qa`/`main`), runner label (`qa-gpu`/`prod-gpu`), `environment:` name (`qa`/`production`), and the `deploy.sh` env-folder argument (`qa`/`main`) differ. `deploy-prod.yml`'s jobs also set `environment: production`, which requires manual approval if configured under **Settings → Environments** in GitHub.

Unlike an earlier draft of this doc, there is **no separate `build` job** pushing to an image registry — `deploy.sh` builds the image locally on the target VM as part of the same run, since each VM only ever runs its own model(s) and there's no shared registry in this design yet.

---

## Not Yet Implemented

These were ideas from an earlier draft of this doc — not present in the repo today. Worth reconsidering if the project grows:

- **`benchmarks/`, `tests/`** — no unit/integration/inference test suite or benchmark datasets exist yet.
- **`ci.yml`** — no lint/test/build-validation workflow runs on PRs yet; only the three deploy-on-push workflows exist.
- **Tagging strategy / rollback** — no `{model}-v{major}.{minor}.{patch}` git tags or `scripts/rollback.sh` exist; rolling back today means re-deploying an older commit by hand.
- **QA Evaluation Matrix / regression reports** — no automated accuracy/latency/VRAM comparison step exists in `deploy-qa.yml`.
- **Docker image registry** — images are built locally on each VM by `deploy.sh`, not pushed to/pulled from a shared registry.

---

## Key Principles

1. **One repo, one Docker image, three model configs.** The same image (built from `docker/Dockerfile`) runs every model — only the env vars passed in at `docker run` differ.

2. **Branches = environments, not code-maturity stages alone.** `Dev`/`Qa`/`main` map 1:1 to `dev`/`qa`/`prod` GPU infrastructure — each branch carries only its own environment folder and workflow file, not all three.

3. **Smart deployments.** `detect-changes` + `dorny/paths-filter` mean only the model(s) whose config actually changed get redeployed; a change to shared code (`src/`, `docker/`, `scripts/`) redeploys all three.

4. **Model weights stay out of Git.** Weights are downloaded by vLLM from Hugging Face at container start into `/var/lib/vllm_models` on the host — Git tracks code + config only.

5. **Secrets never committed in plaintext.** `.env` is tracked with empty placeholder values only; real secrets go in an untracked local `.env` copy on each VM, and `deploy.sh` mounts them into the container as files under `/run/secrets`, not as plaintext env vars.

6. **Promotion between branches is a manual cherry-pick, not a merge** — see the [Branch Strategy](#branch-strategy) caveat above.
