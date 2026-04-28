from .perf import apply_perf_patches
from .phase1 import apply_phase1_patch, stable_local_file_uuid
from .phase2 import apply_phase23_patches

__all__ = [
    "apply_perf_patches",
    "apply_phase1_patch",
    "apply_phase23_patches",
    "stable_local_file_uuid",
]
