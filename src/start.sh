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
