#!/bin/bash
#
# Usage: ./scripts/deploy.sh <model> <env-folder>
#   e.g. ./scripts/deploy.sh medgemma-27b dev
#
# Run this ON the target VM, from the repo root, on the branch matching
# that environment (Dev branch has a dev/ folder, Qa branch has qa/, main
# branch has main/ — each branch carries only its own environment folder,
# not all three). It merges models/<model>/config.env (identity) with
# <env-folder>/<model>.env (GPU/environment-specific tuning), then builds
# and (re)starts the container.

set -e

MODEL="${1:?Usage: $0 <model> <env-folder>}"
ENVIRONMENT="${2:?Usage: $0 <model> <env-folder>}"

MODEL_CONFIG="models/${MODEL}/config.env"
ENV_CONFIG="${ENVIRONMENT}/${MODEL}.env"

for f in "$MODEL_CONFIG" "$ENV_CONFIG"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found (unknown model, or wrong branch checked out for this environment?)" >&2
    exit 1
  fi
done

echo "Loading config: $MODEL_CONFIG + $ENV_CONFIG"
set -a
source "$MODEL_CONFIG"
source "$ENV_CONFIG"
set +a

CONTAINER_NAME="${MODEL}-${ENVIRONMENT}"
IMAGE_NAME="llm-serving-${MODEL}:${ENVIRONMENT}"

# --- Secrets (loaded from .env at repo root, never committed) ---

if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo "ERROR: .env file not found. Copy .env.example to .env and fill in VLLM_API_KEY and HUGGING_FACE_HUB_TOKEN." >&2
  exit 1
fi

if [ -z "${VLLM_API_KEY:-}" ] || [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "ERROR: VLLM_API_KEY and HUGGING_FACE_HUB_TOKEN must be set in .env" >&2
  exit 1
fi
if [ "${GATED:-false}" = "true" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "ERROR: ${MODEL_NAME} is gated on Hugging Face — HUGGING_FACE_HUB_TOKEN is required." >&2
  exit 1
fi

# --- Write secrets to files instead of passing them as plaintext env vars ---
# `docker run -e` bakes secrets into the container config in plaintext,
# visible to anyone who can run `docker inspect` on the container. This has
# to be a stable path (not a mktemp dir cleaned up on script exit) since the
# container uses `--restart unless-stopped` and needs the mount to survive
# past this script's lifetime. Keyed by model+env so multiple deployments on
# the same host don't clobber each other's secrets.
SECRETS_DIR="$(pwd)/.secrets/${MODEL}-${ENVIRONMENT}"
mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"
printf '%s' "$VLLM_API_KEY" > "$SECRETS_DIR/vllm_api_key"
printf '%s' "$HUGGING_FACE_HUB_TOKEN" > "$SECRETS_DIR/hf_token"
chmod 600 "$SECRETS_DIR"/*

echo "Checking for existing container..."
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
  echo "Stopping and removing existing container..."
  docker stop "$CONTAINER_NAME"
  docker rm "$CONTAINER_NAME"
else
  echo "No existing container found."
fi

echo "Building image ${IMAGE_NAME}..."
docker build -f docker/Dockerfile -t "$IMAGE_NAME" .

echo "Starting container ${CONTAINER_NAME}..."
docker run -d --name "$CONTAINER_NAME" --gpus all --ipc=host -p 8000:8000 \
  --restart unless-stopped \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /var/lib/vllm_models:/var/lib/vllm_models \
  -v "$SECRETS_DIR":/run/secrets:ro \
  -e MODEL_NAME \
  -e ENVIRONMENT \
  -e GPU_TYPE \
  -e TENSOR_PARALLEL_SIZE \
  -e DTYPE \
  -e MAX_MODEL_LEN \
  -e GPU_MEMORY_UTILIZATION \
  -e MAX_NUM_SEQS \
  -e MAX_NUM_BATCHED_TOKENS \
  -e LIMIT_MM_PER_PROMPT \
  -e LOG_LEVEL \
  -e LOG_ENVIRONMENT="${ENVIRONMENT}" \
  "$IMAGE_NAME"

echo "Done. Container status:"
docker ps | grep "$CONTAINER_NAME"
