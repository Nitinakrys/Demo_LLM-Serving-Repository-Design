# Demo_LLM-Serving-Repository-Design

One repo, one Docker image, three models, three environments. See [`llm-serving-repo-design.md`](llm-serving-repo-design.md) for the full design rationale.

## Layout

```
docker/Dockerfile              # one image for every model
src/start.sh                   # generic entrypoint, reads config from env vars
src/custom_vllm_logger.py      # shared JSON logging wrapper
models/<model>/config.env      # per-model identity (MODEL_NAME, GATED, MULTIMODAL)
deployment/<env>/<model>.env   # per-(model,environment) GPU tuning (TP size, batch, context)
scripts/deploy.sh              # merges the two configs above, builds + runs the container
.env.example                   # VLLM_API_KEY / HUGGING_FACE_HUB_TOKEN template
```

## Models × environments

| Model | dev / qa (2× L40S, 48GB each) | prod (H200, 141GB) |
|---|---|---|
| `gemma-3-1b-it` | TP=1, 8K ctx | TP=1, 32K ctx, higher concurrency |
| `medgemma-4b` | TP=1, 8K ctx | TP=1, 32K ctx, higher concurrency |
| `medgemma-27b` | **TP=2** (54GB weights need 2 GPUs) | **TP=1** (fits on one H200) |

dev and qa run identical GPU tuning (both L40S) — they only differ in `LOG_LEVEL`. Prod's H200 has enough headroom to change tensor-parallel size for `medgemma-27b`, not just batch/context knobs.

## Deploy

On the target VM, from the repo root:

```bash
cp .env.example .env   # fill in VLLM_API_KEY and HUGGING_FACE_HUB_TOKEN
./scripts/deploy.sh <model> <environment>
# e.g.
./scripts/deploy.sh medgemma-27b dev
./scripts/deploy.sh medgemma-27b prod
```

All three models are gated on Hugging Face — accept Google's license on each model's page with the account that owns `HUGGING_FACE_HUB_TOKEN` before deploying.

## Branch strategy

`dev → qa → main`, matching the deploy environments of the same name — see the design doc's [Branch Strategy](llm-serving-repo-design.md#branch-strategy) section.
