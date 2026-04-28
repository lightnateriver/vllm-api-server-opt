from .common.phase import phase_at_least
from .patches import (
    apply_perf_patches,
    apply_phase1_patch,
    apply_phase23_patches,
)


_PLUGIN_PATCHED = False


def register() -> None:
    global _PLUGIN_PATCHED
    if _PLUGIN_PATCHED:
        return

    apply_phase1_patch()
    if phase_at_least(2):
        apply_phase23_patches(enable_phase3=phase_at_least(3))
    apply_perf_patches()

    _PLUGIN_PATCHED = True
