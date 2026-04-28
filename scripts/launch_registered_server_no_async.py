#!/usr/bin/env python3

import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BUNDLE_DIR = str(ROOT_DIR / "bundle")


def main() -> None:
    os.chdir(BUNDLE_DIR)

    if BUNDLE_DIR not in sys.path:
        sys.path.insert(0, BUNDLE_DIR)

    import vllm_phase2_plugin

    vllm_phase2_plugin.register()

    model_dir = os.environ.get("MODEL_DIR", "/mnt/sfs_turbo/models/Qwen3.5-4B")
    allowed_local_media_path = os.environ.get(
        "ALLOWED_LOCAL_MEDIA_PATH",
        str(ROOT_DIR / "data"),
    )
    host = os.environ.get("HOST", "127.0.0.1")
    port = os.environ.get("PORT", "8000")
    tensor_parallel_size = os.environ.get("TENSOR_PARALLEL_SIZE", "4")
    mm_processor_cache_type = os.environ.get("MM_PROCESSOR_CACHE_TYPE", "shm")
    mm_processor_cache_gb = os.environ.get("MM_PROCESSOR_CACHE_GB", "20")
    mm_encoder_tp_mode = os.environ.get("MM_ENCODER_TP_MODE", "data")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "36864")
    enable_cpu_binding = os.environ.get("ENABLE_CPU_BINDING", "true")

    sys.argv = [
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_dir,
        "--host",
        host,
        "--port",
        port,
        "--enforce-eager",
        "--tensor-parallel-size",
        tensor_parallel_size,
        "--mm-processor-cache-type",
        mm_processor_cache_type,
        "--mm-processor-cache-gb",
        mm_processor_cache_gb,
        "--mm-encoder-tp-mode",
        mm_encoder_tp_mode,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--max-model-len",
        max_model_len,
        "--allowed-local-media-path",
        allowed_local_media_path,
        "--additional-config",
        f'{{"enable_cpu_binding": {enable_cpu_binding}}}',
        "--no-async-scheduling",
    ]

    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
