# Demo_LLM-Serving-Repository-Design

One repo, one Docker image, three models, three environments. See [`llm-serving-repo-design.md`](llm-serving-repo-design.md) for the full design rationale.

## Layout

**Each branch carries only its own environment folder, named after the branch** — `Dev` has `dev/`, `Qa` has `qa/`, `main` has `main/`. You will not see all three side by side on any one branch; check out the branch for the environment you want.

```
docker/Dockerfile              # one image for every model (same on every branch)
src/start.sh                   # generic entrypoint, reads config from env vars
src/custom_vllm_logger.py      # shared JSON logging wrapper
models/<model>/config.env      # per-model identity (MODEL_NAME, GATED, MULTIMODAL) — same on every branch
qa/<model>.env                 # THIS BRANCH (Qa) ONLY — per-model GPU tuning for the qa environment (2x L40S)
scripts/deploy.sh              # merges models/<model>/config.env + <env-folder>/<model>.env, builds + runs the container
.env.example                   # VLLM_API_KEY / HUGGING_FACE_HUB_TOKEN template
```

`Dev` branch has the equivalent `dev/gemma-4-31b-it.env`, `dev/medgemma-4b.env`, `dev/medgemma-27b.env` instead of `qa/`. `main` branch has `main/` with the same three files, tuned for its H200 GPU.

## Models × environments

| Model | dev / qa (2× L40S, 48GB each) | main (1× H200, 141GB) |
|---|---|---|
| `gemma-4-31b-it` | **TP=2**, 8K ctx | TP=1, 32K ctx, higher concurrency |
| `medgemma-4b` | TP=1, 8K ctx | TP=1, 32K ctx, higher concurrency |
| `medgemma-27b` | **TP=2** (54GB weights need 2 GPUs) | **TP=1** (fits on one H200) |

dev and qa run identical GPU tuning (both L40S) — they only differ in `LOG_LEVEL`. The H200 on `main` has enough headroom to change tensor-parallel size for `medgemma-27b`, not just batch/context knobs.

## Deploy

On the target VM, on the branch matching that environment, from the repo root:

```bash
cp .env.example .env   # fill in VLLM_API_KEY and HUGGING_FACE_HUB_TOKEN
./scripts/deploy.sh <model> <env-folder>
# on this branch (Qa):
./scripts/deploy.sh medgemma-27b qa
```

All three models are gated on Hugging Face — accept Google's license on each model's page with the account that owns `HUGGING_FACE_HUB_TOKEN` before deploying.

## Branch strategy

| Branch | Env folder on that branch | Pushes trigger | GPU |
|---|---|---|---|
| `Dev` | `dev/` | `deploy-dev.yml` | 2× L40S |
| `Qa` | `qa/` | `deploy-qa.yml` | 2× L40S |
| `main` | `main/` | `deploy-prod.yml` | 1× H200 |

Each workflow requires a self-hosted GitHub Actions runner on the corresponding GPU VM, tagged `dev-gpu` / `qa-gpu` / `prod-gpu` respectively. See the design doc's [Branch Strategy](llm-serving-repo-design.md#branch-strategy) section for the full PR/review flow.

> **Caveat:** because each branch's environment folder is named differently (`dev/` vs `qa/` vs `main/`), a plain `git merge` of `Dev → Qa → main` will not cleanly promote environment config the way the design doc describes — the promoted branch would need to keep renaming the incoming folder. Shared code (`docker/`, `src/`, `models/`, `scripts/`) merges fine since it's identical on every branch; only the environment folder needs manual reconciliation on merge.
