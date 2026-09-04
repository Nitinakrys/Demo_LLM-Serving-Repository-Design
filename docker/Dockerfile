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
