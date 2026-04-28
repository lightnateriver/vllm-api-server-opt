import os
from pathlib import Path


TRACE_FILE = Path(
    os.environ.get(
        "VLLM_ASCEND_MM_FILE_MAP",
        "/tmp/vllm_ascend_mm_file_map_phase2.jsonl",
    )
)
VALIDATE_FILE = Path(
    os.environ.get(
        "VLLM_ASCEND_PHASE2_VALIDATE_FILE",
        "/tmp/vllm_ascend_phase2_validate_phase2.jsonl",
    )
)


def runtime_component(default: str = "api_server") -> str:
    import multiprocessing

    process_name = multiprocessing.current_process().name.lower()
    if "enginecore" in process_name or "engine_core" in process_name:
        return "engine_core"
    if "worker" in process_name:
        return "tp_worker"

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return "tp_worker"
    except Exception:
        pass

    return default
