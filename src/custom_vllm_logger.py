import os
import sys
import json
import socket
import logging
import runpy
from datetime import datetime, timezone

# --- 1. Preemptively disable vLLM/Uvicorn default logging ---
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

import uvicorn.config
# We must provide a valid dictConfig version or it throws ValueError
uvicorn.config.LOGGING_CONFIG.clear()
uvicorn.config.LOGGING_CONFIG.update({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {},
    "handlers": {},
    "loggers": {
        "uvicorn": {"propagate": True},
        "uvicorn.error": {"propagate": True},
        "uvicorn.access": {"propagate": True},
    }
})

# --- 2. Define the Custom JSON Formatter ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Service name comes from MODEL_NAME (set by start.sh from config.env)
        # so one shared logger works for every model without editing code.
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": os.environ.get("MODEL_NAME", "llm-serving"),
            "environment": os.environ.get("LOG_ENVIRONMENT", "development"),
            "host": socket.gethostname(),
            "message": record.getMessage(),
        }

        # Attach 'extra_data' if it was passed via extra={"extra_data": {...}}
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)

        # Handle Exceptions/Stack Traces
        if record.exc_info:
            log_record["error_type"] = str(record.exc_info[0].__name__)
            log_record["stack_trace"] = self.formatException(record.exc_info)

        return json.dumps(log_record)

# --- 3. Global Logger Setup ---
def setup_global_logger():
    # Force the root logger to use our JsonFormatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any default handlers (like the ones vLLM might try to add late)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(handler)

    # Ensure vLLM internal loggers propagate to root
    for name in ["vllm", "vllm.engine.async_llm_engine", "aioprometheus"]:
        l = logging.getLogger(name)
        l.propagate = True
        l.handlers = []

# --- 4. Execution ---
if __name__ == "__main__":
    setup_global_logger()

    # This runs the vLLM API server using the arguments passed to the script
    # e.g., --model, --port, etc. from start.sh
    try:
        runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
    except Exception as e:
        # Final safety net for startup crashes
        logging.error(f"Failed to start vLLM server: {e}", exc_info=True)
        sys.exit(1)
