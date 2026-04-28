import json
import math
import os
import time
import traceback
from pathlib import Path

from .runtime import VALIDATE_FILE


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        fp.write("\n")


def tensor_equal(a, b) -> bool:
    import torch
    from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalKwargsItem

    if isinstance(a, MultiModalKwargsItem) and isinstance(b, MultiModalKwargsItem):
        return set(a.keys()) == set(b.keys()) and all(
            tensor_equal(a[k], b[k]) for k in a.keys()
        )
    if isinstance(a, MultiModalFieldElem) and isinstance(b, MultiModalFieldElem):
        return type(a.field) is type(b.field) and tensor_equal(a.data, b.data)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a.keys()) == set(b.keys()) and all(
            tensor_equal(a[k], b[k]) for k in a.keys()
        )
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tensor_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if torch.is_floating_point(a):
            return torch.allclose(a.cpu(), b.cpu(), atol=1e-3, rtol=1e-3)
        return torch.equal(a.cpu(), b.cpu())
    return a == b


def summarize_tensor(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return {"type": type(value).__name__}


def summarize_mm_item(item):
    from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalKwargsItem

    if item is None:
        return None
    if isinstance(item, MultiModalKwargsItem):
        return {key: summarize_mm_item(value) for key, value in item.items()}
    if isinstance(item, MultiModalFieldElem):
        summary = summarize_tensor(item.data)
        summary["field"] = type(item.field).__name__
        return summary
    if isinstance(item, dict):
        return {key: summarize_mm_item(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [summarize_mm_item(value) for value in item]
    return {"type": type(item).__name__}


def estimate_item_size(mm_item: dict) -> int:
    grid = mm_item.get("image_grid_thw")
    if hasattr(grid, "data"):
        grid = grid.data
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().tolist()
    if isinstance(grid, list) and grid and isinstance(grid[0], list):
        grid = grid[0]
    if isinstance(grid, list) and len(grid) == 3:
        return int(math.prod(int(v) for v in grid))

    pixel_values = mm_item.get("pixel_values")
    if hasattr(pixel_values, "data"):
        pixel_values = pixel_values.data
    if hasattr(pixel_values, "shape") and len(pixel_values.shape) > 0:
        return int(pixel_values.shape[0])
    return 1


def normalize_local_mm_paths(value) -> dict[str, list[str | None]] | None:
    if not isinstance(value, dict):
        return None

    image_paths = value.get("image")
    if not isinstance(image_paths, list):
        return None

    normalized: list[str | None] = []
    for path in image_paths:
        normalized.append(path if isinstance(path, str) else None)
    return {"image": normalized}


def normalize_image_source_metadata(value) -> dict[str, list[dict | None]] | None:
    if not isinstance(value, dict):
        return None

    image_items = value.get("image")
    if not isinstance(image_items, list):
        return None

    normalized: list[dict | None] = []
    has_entry = False
    for item in image_items:
        if not isinstance(item, dict):
            normalized.append(None)
            continue

        media_mode = item.get("media_mode")
        local_path = item.get("local_path")
        media_uuid = item.get("media_uuid")
        request_scope_key = item.get("request_scope_key")
        source_key = item.get("source_key")
        image_url = item.get("image_url")

        entry = {
            "media_mode": str(media_mode) if isinstance(media_mode, str) else None,
            "local_path": str(local_path) if isinstance(local_path, str) else None,
            "media_uuid": str(media_uuid) if isinstance(media_uuid, str) else None,
            "request_scope_key": (
                str(request_scope_key) if isinstance(request_scope_key, str) else None
            ),
            "source_key": str(source_key) if isinstance(source_key, str) else None,
            "image_url": str(image_url) if isinstance(image_url, str) else None,
        }
        if any(value is not None for value in entry.values()):
            has_entry = True
            normalized.append(entry)
        else:
            normalized.append(None)

    if not has_entry:
        return None

    return {"image": normalized}


def get_image_count_from_mm_data(mm_data) -> int:
    if not isinstance(mm_data, dict):
        return 0
    image_items = mm_data.get("image")
    if not isinstance(image_items, list):
        return 0
    return len(image_items)


def are_all_image_paths_local(local_mm_paths, *, expected_count: int | None = None) -> bool:
    normalized = normalize_local_mm_paths(local_mm_paths)
    if normalized is None:
        return False

    image_paths = normalized["image"]
    if expected_count is not None and len(image_paths) != expected_count:
        return False
    if not image_paths:
        return False
    return all(isinstance(path, str) for path in image_paths)


def extract_local_mm_paths_from_mm_data(mm_data) -> dict[str, list[str | None]] | None:
    if not isinstance(mm_data, dict):
        return None

    image_items = mm_data.get("image")
    if not isinstance(image_items, list):
        return None

    local_paths: list[str | None] = []
    has_local = False
    for item in image_items:
        path = getattr(item, "_phase2_local_path", None)
        if isinstance(path, str):
            has_local = True
            local_paths.append(path)
        else:
            local_paths.append(None)

    if not has_local:
        return None

    return {"image": local_paths}


def extract_image_source_metadata_from_mm_data(mm_data) -> dict[str, list[dict | None]] | None:
    if not isinstance(mm_data, dict):
        return None

    image_items = mm_data.get("image")
    if not isinstance(image_items, list):
        return None

    entries: list[dict | None] = []
    has_entry = False
    for item in image_items:
        entry = {
            "media_mode": getattr(item, "_phase2_media_mode", None),
            "local_path": getattr(item, "_phase2_local_path", None),
            "media_uuid": getattr(item, "_phase2_media_uuid", None),
            "request_scope_key": getattr(item, "_phase2_request_scope_key", None),
            "source_key": getattr(item, "_phase2_source_key", None),
        }
        if any(isinstance(value, str) for value in entry.values()):
            has_entry = True
            entries.append(entry)
        else:
            entries.append(None)

    if not has_entry:
        return None

    return {"image": entries}


def phase3_request_uses_local_media(mm_data) -> bool:
    image_count = get_image_count_from_mm_data(mm_data)
    if image_count <= 0:
        return False

    return are_all_image_paths_local(
        extract_local_mm_paths_from_mm_data(mm_data),
        expected_count=image_count,
    )


def extract_phase3_image_refs(mm_data) -> list[object] | None:
    if not isinstance(mm_data, dict):
        return None

    image_items = mm_data.get("image")
    if not isinstance(image_items, list) or not image_items:
        return None

    refs: list[object] = []
    for item in image_items:
        width = getattr(item, "_phase3_width", None)
        height = getattr(item, "_phase3_height", None)
        if width is None or height is None:
            return None
        refs.append(item)

    return refs


def build_phase3_mm_data(renderer, mm_data, mm_processor_kwargs):
    refs = extract_phase3_image_refs(mm_data)
    if not refs:
        return None

    try:
        import torch

        mm_kwargs = dict(mm_processor_kwargs or {})
        mm_processor = renderer.get_mm_processor()
        info = mm_processor.info
        image_processor = info.get_image_processor(**mm_kwargs)
        patch_size = int(info.get_hf_config().vision_config.patch_size)

        grid_rows: list[list[int]] = []
        for ref in refs:
            image_width = int(getattr(ref, "_phase3_width"))
            image_height = int(getattr(ref, "_phase3_height"))
            preprocessed_size, _ = info._get_vision_info(
                image_width=image_width,
                image_height=image_height,
                num_frames=1,
                image_processor=image_processor,
                mm_kwargs=mm_kwargs,
            )
            grid_rows.append(
                [
                    1,
                    int(preprocessed_size.height) // patch_size,
                    int(preprocessed_size.width) // patch_size,
                ]
            )

        return {
            "image": {
                "image_grid_thw": torch.tensor(grid_rows, dtype=torch.int64),
            }
        }
    except Exception as exc:
        append_jsonl(
            VALIDATE_FILE,
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "status": "phase3_mm_data_build_error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return None


def extract_image_paths_for_request(req_state) -> list[str | None]:
    sampling_params = getattr(req_state, "sampling_params", None)
    if sampling_params is None:
        return []

    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, dict):
        return []

    local_mm_paths = normalize_local_mm_paths(extra_args.get("phase2_local_mm_paths"))
    if local_mm_paths is None:
        return []

    image_paths = local_mm_paths["image"]
    out: list[str | None] = []
    image_idx = 0
    for feature in req_state.mm_features:
        if feature.modality != "image":
            continue
        out.append(image_paths[image_idx] if image_idx < len(image_paths) else None)
        image_idx += 1
    return out


def extract_image_source_metadata_for_request(req_state) -> list[dict | None]:
    sampling_params = getattr(req_state, "sampling_params", None)
    if sampling_params is None:
        return []

    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, dict):
        return []

    source_metadata = normalize_image_source_metadata(
        extra_args.get("phase2_image_source_metadata")
    )
    if source_metadata is None:
        return []

    image_entries = source_metadata["image"]
    out: list[dict | None] = []
    image_idx = 0
    for feature in req_state.mm_features:
        if feature.modality != "image":
            continue
        out.append(image_entries[image_idx] if image_idx < len(image_entries) else None)
        image_idx += 1
    return out


def phase3_request_uses_local_media_for_req_state(req_state) -> bool:
    sampling_params = getattr(req_state, "sampling_params", None)
    if sampling_params is None:
        return False

    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, dict):
        return False

    return bool(extra_args.get("phase3_local_mm_ready"))


def get_grid_thw_tensor(mm_item):
    grid = mm_item.get("image_grid_thw") if mm_item is not None else None
    if hasattr(grid, "data"):
        grid = grid.data
    return grid


def has_pixel_values_tensor(mm_item) -> bool:
    if mm_item is None:
        return False
    pixel_values = mm_item.get("pixel_values")
    if pixel_values is None:
        return False
    if hasattr(pixel_values, "data"):
        pixel_values = pixel_values.data
    return pixel_values is not None


def infer_phase3_patch_dim(processor) -> int:
    hf_config = processor.info.get_hf_config()
    vision_cfg = hf_config.vision_config
    in_channels = int(getattr(vision_cfg, "in_channels", 3))
    patch_size = int(getattr(vision_cfg, "patch_size", 14))
    temporal_patch_size = int(getattr(vision_cfg, "temporal_patch_size", 2))
    return in_channels * patch_size * patch_size * temporal_patch_size


def build_phase3_full_items(
    processor,
    image_features,
    local_indices,
    local_items,
):
    import torch
    from vllm.multimodal.inputs import MultiModalKwargsItems

    local_item_by_idx = {
        int(global_idx): item
        for global_idx, item in zip(local_indices, local_items)
    }
    sample_item = local_items[0] if local_items else None

    if sample_item is not None:
        sample_pixel_values = sample_item["pixel_values"].data
        patch_dim = int(sample_pixel_values.shape[1])
        pixel_dtype = sample_pixel_values.dtype
        pixel_device = sample_pixel_values.device
    else:
        patch_dim = infer_phase3_patch_dim(processor)
        pixel_dtype = torch.float32
        pixel_device = torch.device("cpu")

    pixel_chunks = []
    grid_rows = []
    for image_idx, feature in enumerate(image_features):
        base_item = feature.data
        grid_tensor = get_grid_thw_tensor(base_item)
        if grid_tensor is None:
            raise RuntimeError(f"missing image_grid_thw at image_idx={image_idx}")

        if hasattr(grid_tensor, "detach"):
            grid_tensor = grid_tensor.detach()
        grid_tensor = grid_tensor.to(device=pixel_device, dtype=torch.int64).reshape(-1)
        if grid_tensor.numel() != 3:
            raise RuntimeError(
                f"invalid image_grid_thw shape at image_idx={image_idx}: "
                f"{tuple(grid_tensor.shape)}"
            )
        grid_rows.append(grid_tensor)

        local_item = local_item_by_idx.get(image_idx)
        if local_item is not None:
            pixel_values = local_item["pixel_values"].data
            pixel_values = pixel_values.to(device=pixel_device, dtype=pixel_dtype)
        else:
            patch_count = int(torch.prod(grid_tensor).item())
            pixel_values = torch.zeros(
                (patch_count, patch_dim),
                dtype=pixel_dtype,
                device=pixel_device,
            )
        pixel_chunks.append(pixel_values)

    if pixel_chunks:
        pixel_values_batch = torch.cat(pixel_chunks, dim=0)
        image_grid_thw_batch = torch.stack(grid_rows, dim=0)
    else:
        pixel_values_batch = torch.zeros(
            (0, patch_dim),
            dtype=pixel_dtype,
            device=pixel_device,
        )
        image_grid_thw_batch = torch.zeros(
            (0, 3),
            dtype=torch.int64,
            device=pixel_device,
        )

    hf_inputs = {
        "pixel_values": pixel_values_batch,
        "image_grid_thw": image_grid_thw_batch,
    }
    mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
        hf_inputs,
        processor._get_mm_fields_config(hf_inputs, {}),
    )
    return list(mm_kwargs.get("image", []))
