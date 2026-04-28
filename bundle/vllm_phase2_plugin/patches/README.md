Layout for plugin-managed runtime hooks:

- `phase1.py`: local image UUID/hash and API-side multimodal path collection
- `phase2.py`: phase2/phase3 request plumbing and worker-side image processing
- `perf.py`: perf probe wrappers for API server, engine core, and tp worker

Top-level compatibility modules in `vllm_phase2_plugin/` now only re-export these
submodules so the plugin entrypoint and older local imports keep working.
