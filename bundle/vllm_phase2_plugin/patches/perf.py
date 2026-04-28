import json
import os
import time
from pathlib import Path
from urllib3.util import parse_url

from ..common.runtime import runtime_component
from ..probes.perf import perf_probe_multi_span, perf_probe_record


_PERF_PATCHED = False
_MM_FEATURE_DEBUG_ENV = "VLLM_ASCEND_MM_FEATURE_DEBUG"
_MM_FEATURE_DEBUG_FILE_ENV = "VLLM_ASCEND_MM_FEATURE_DEBUG_FILE"
_MM_FEATURE_DEBUG_DEFAULT_FILE = "/tmp/vllm_ascend_mm_feature_debug.jsonl"
_MM_INPUT_ORDER_DEBUG_ENV = "VLLM_ASCEND_MM_INPUT_ORDER_DEBUG"
_MM_INPUT_ORDER_DEBUG_FILE_ENV = "VLLM_ASCEND_MM_INPUT_ORDER_DEBUG_FILE"
_MM_INPUT_ORDER_DEBUG_DEFAULT_FILE = "/tmp/vllm_ascend_mm_input_order_debug.jsonl"
_MM_API_PROMPT_DEBUG_ENV = "VLLM_ASCEND_MM_API_PROMPT_DEBUG"
_MM_API_PROMPT_DEBUG_FILE_ENV = "VLLM_ASCEND_MM_API_PROMPT_DEBUG_FILE"
_MM_API_PROMPT_DEBUG_DEFAULT_FILE = "/tmp/vllm_ascend_mm_api_prompt_debug.jsonl"


def _scheduled_encoder_request_count(scheduler_output) -> int:
    scheduled = getattr(scheduler_output, "scheduled_encoder_inputs", None)
    return len(scheduled or {})


def _is_decode_only_scheduler_step(scheduler_output) -> bool:
    return bool(
        getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0
        and _scheduled_encoder_request_count(scheduler_output) == 0
    )


def _mm_feature_debug_enabled() -> bool:
    value = os.environ.get(_MM_FEATURE_DEBUG_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _mm_feature_debug_file() -> Path:
    return Path(
        os.environ.get(
            _MM_FEATURE_DEBUG_FILE_ENV,
            _MM_FEATURE_DEBUG_DEFAULT_FILE,
        )
    )


def _append_mm_feature_debug(record: dict) -> None:
    if not _mm_feature_debug_enabled():
        return

    path = _mm_feature_debug_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        fp.write("\n")


def _tensor_summary(value) -> dict:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    numel = int(value.numel()) if hasattr(value, "numel") else None
    elem_size = int(value.element_size()) if hasattr(value, "element_size") else None
    bytes_size = numel * elem_size if numel is not None and elem_size is not None else None
    return {
        "shape": list(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
        "device": str(device) if device is not None else None,
        "numel": numel,
        "bytes": bytes_size,
    }


def _summarize_mm_data(data) -> dict:
    try:
        from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalKwargsItem
    except Exception:
        MultiModalFieldElem = tuple()  # type: ignore[assignment]
        MultiModalKwargsItem = dict  # type: ignore[assignment]

    if data is None:
        return {"type": "None", "bytes": 0}
    if hasattr(data, "shape") and hasattr(data, "dtype"):
        summary = _tensor_summary(data)
        summary["type"] = type(data).__name__
        return summary
    if isinstance(data, MultiModalFieldElem):
        summary = _summarize_mm_data(data.data)
        summary["field"] = type(data.field).__name__
        summary["type"] = "MultiModalFieldElem"
        return summary
    if isinstance(data, (MultiModalKwargsItem, dict)):
        fields = {}
        total_bytes = 0
        for key, value in data.items():
            field_summary = _summarize_mm_data(value)
            fields[key] = field_summary
            total_bytes += int(field_summary.get("bytes") or 0)
        return {
            "type": type(data).__name__,
            "fields": fields,
            "bytes": total_bytes,
        }
    if isinstance(data, (list, tuple)):
        items = [_summarize_mm_data(item) for item in data]
        return {
            "type": type(data).__name__,
            "items": items,
            "bytes": sum(int(item.get("bytes") or 0) for item in items),
        }
    return {
        "type": type(data).__name__,
        "repr": repr(data)[:256],
        "bytes": 0,
    }


def _record_mm_feature_state(
    stage: str,
    req_id: str,
    req_state,
    *,
    decode_only: bool,
    extra: dict | None = None,
) -> None:
    if not _mm_feature_debug_enabled():
        return

    mm_features = getattr(req_state, "mm_features", None) or []
    features_summary = []
    total_payload_bytes = 0
    for idx, feature in enumerate(mm_features):
        data_summary = _summarize_mm_data(getattr(feature, "data", None))
        total_payload_bytes += int(data_summary.get("bytes") or 0)
        mm_position = getattr(feature, "mm_position", None)
        features_summary.append(
            {
                "index": idx,
                "modality": getattr(feature, "modality", None),
                "identifier": getattr(feature, "identifier", None),
                "mm_position": {
                    "offset": int(getattr(mm_position, "offset", 0)),
                    "length": int(getattr(mm_position, "length", 0)),
                }
                if mm_position is not None
                else None,
                "data": data_summary,
            }
        )

    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "phase": os.environ.get("VLLM_ASCEND_API_OPT_PHASE", ""),
        "stage": stage,
        "request_id": req_id,
        "decode_only": bool(decode_only),
        "num_computed_tokens": int(getattr(req_state, "num_computed_tokens", 0)),
        "num_prompt_tokens": int(getattr(req_state, "num_prompt_tokens", 0)),
        "feature_count": len(mm_features),
        "total_payload_bytes": total_payload_bytes,
        "features": features_summary,
    }
    if extra:
        record["extra"] = extra
    _append_mm_feature_debug(record)


def _mm_input_order_debug_enabled() -> bool:
    value = os.environ.get(_MM_INPUT_ORDER_DEBUG_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _mm_input_order_debug_file() -> Path:
    return Path(
        os.environ.get(
            _MM_INPUT_ORDER_DEBUG_FILE_ENV,
            _MM_INPUT_ORDER_DEBUG_DEFAULT_FILE,
        )
    )


def _append_mm_input_order_debug(record: dict) -> None:
    if not _mm_input_order_debug_enabled():
        return

    path = _mm_input_order_debug_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        fp.write("\n")


def _mm_api_prompt_debug_enabled() -> bool:
    value = os.environ.get(_MM_API_PROMPT_DEBUG_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _mm_api_prompt_debug_file() -> Path:
    return Path(
        os.environ.get(
            _MM_API_PROMPT_DEBUG_FILE_ENV,
            _MM_API_PROMPT_DEBUG_DEFAULT_FILE,
        )
    )


def _append_mm_api_prompt_debug(record: dict) -> None:
    if not _mm_api_prompt_debug_enabled():
        return

    path = _mm_api_prompt_debug_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        fp.write("\n")


def _mask_spans(mask: list[bool]) -> list[dict]:
    spans: list[dict] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            spans.append(
                {
                    "start": start,
                    "end": idx,
                    "length": idx - start,
                }
            )
            start = None
    if start is not None:
        spans.append(
            {
                "start": start,
                "end": len(mask),
                "length": len(mask) - start,
            }
        )
    return spans


def _mask_preview(mask: list[bool], limit: int = 128) -> str:
    preview = "".join("M" if value else "T" for value in mask[:limit])
    if len(mask) > limit:
        preview += f"...(+{len(mask) - limit})"
    return preview


def _text_prefix_len(mask: list[bool]) -> int:
    for idx, value in enumerate(mask):
        if value:
            return idx
    return len(mask)


def _text_suffix_len(mask: list[bool]) -> int:
    for idx in range(len(mask) - 1, -1, -1):
        if mask[idx]:
            return len(mask) - idx - 1
    return len(mask)


def _summarize_req_mm_positions(req_state) -> list[dict]:
    positions = []
    for idx, feature in enumerate(getattr(req_state, "mm_features", None) or []):
        mm_position = getattr(feature, "mm_position", None)
        positions.append(
            {
                "index": idx,
                "modality": getattr(feature, "modality", None),
                "identifier": getattr(feature, "identifier", None),
                "offset": int(getattr(mm_position, "offset", 0))
                if mm_position is not None
                else None,
                "length": int(getattr(mm_position, "length", 0))
                if mm_position is not None
                else None,
            }
        )
    return positions


def _truncate_text(value, limit: int = 160) -> str | None:
    if value is None:
        return None

    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(+{len(text) - limit})"


def _extract_attr_or_key(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _summarize_message_content(content) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [
            {
                "type": "text",
                "text_len": len(content),
                "text_preview": _truncate_text(content),
            }
        ]
    if not isinstance(content, list):
        return [{"type": type(content).__name__, "repr": _truncate_text(repr(content), 120)}]

    summary = []
    for item in content:
        item_type = _extract_attr_or_key(item, "type") or type(item).__name__
        entry = {"type": str(item_type)}
        text_value = _extract_attr_or_key(item, "text")
        if text_value is not None:
            entry["text_len"] = len(str(text_value))
            entry["text_preview"] = _truncate_text(text_value)

        image_url = _extract_attr_or_key(item, "image_url")
        if image_url is not None:
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            elif hasattr(image_url, "url"):
                image_url = getattr(image_url, "url", None)
            entry["image_url"] = _truncate_text(image_url, 120)

        input_image = _extract_attr_or_key(item, "input_image")
        if input_image is not None:
            image_value = input_image
            if isinstance(input_image, dict):
                image_value = input_image.get("image_url") or input_image.get("url")
            entry["input_image"] = _truncate_text(image_value, 120)

        summary.append(entry)

    return summary


def _summarize_chat_messages(messages) -> list[dict]:
    summaries = []
    for idx, message in enumerate(messages or []):
        summaries.append(
            {
                "index": idx,
                "role": _extract_attr_or_key(message, "role"),
                "content": _summarize_message_content(
                    _extract_attr_or_key(message, "content")
                ),
            }
        )
    return summaries


def _normalize_mm_hashes(mm_hashes) -> dict:
    if not isinstance(mm_hashes, dict):
        return {}
    return {
        str(modality): [str(item) for item in items]
        for modality, items in mm_hashes.items()
    }


def _normalize_mm_placeholders(mm_placeholders) -> dict:
    if not isinstance(mm_placeholders, dict):
        return {}

    normalized = {}
    for modality, ranges in mm_placeholders.items():
        normalized[str(modality)] = [
            {
                "offset": int(_extract_attr_or_key(item, "offset") or 0),
                "length": int(_extract_attr_or_key(item, "length") or 0),
            }
            for item in ranges
        ]
    return normalized


def _flatten_placeholder_ranges(mm_placeholders: dict) -> list[dict]:
    ranges = []
    for modality, items in mm_placeholders.items():
        for idx, item in enumerate(items):
            ranges.append(
                {
                    "modality": modality,
                    "index": idx,
                    "offset": int(item.get("offset", 0)),
                    "length": int(item.get("length", 0)),
                }
            )
    return sorted(ranges, key=lambda item: (item["offset"], item["index"], item["modality"]))


def _mask_from_placeholders(prompt_len: int, mm_placeholders: dict) -> list[bool]:
    mask = [False] * max(prompt_len, 0)
    for item in _flatten_placeholder_ranges(mm_placeholders):
        start = max(0, min(prompt_len, int(item["offset"])))
        end = max(start, min(prompt_len, start + int(item["length"])))
        for idx in range(start, end):
            mask[idx] = True
    return mask


def _summarize_single_engine_prompt(prompt: dict) -> dict:
    prompt_type = prompt.get("type")
    prompt_token_ids = list(prompt.get("prompt_token_ids") or [])
    mm_hashes = _normalize_mm_hashes(prompt.get("mm_hashes"))
    mm_placeholders = _normalize_mm_placeholders(prompt.get("mm_placeholders"))
    mask = _mask_from_placeholders(len(prompt_token_ids), mm_placeholders)

    return {
        "type": prompt_type,
        "prompt_token_count": len(prompt_token_ids),
        "prompt_text_preview": _truncate_text(prompt.get("prompt")),
        "prompt_token_ids_head": prompt_token_ids[:32],
        "prompt_token_ids_tail": (
            prompt_token_ids[-16:] if len(prompt_token_ids) > 16 else []
        ),
        "mm_modalities": sorted(mm_placeholders.keys()),
        "mm_hashes": mm_hashes,
        "mm_hash_count": sum(len(items) for items in mm_hashes.values()),
        "mm_placeholders": mm_placeholders,
        "mm_placeholders_flat": _flatten_placeholder_ranges(mm_placeholders),
        "mm_token_count": sum(1 for value in mask if value),
        "starts_with_multimodal": bool(mask[0]) if mask else False,
        "ends_with_multimodal": bool(mask[-1]) if mask else False,
        "text_prefix_len": _text_prefix_len(mask),
        "text_suffix_len": _text_suffix_len(mask),
        "mm_mask_spans": _mask_spans(mask),
        "mm_mask_preview": _mask_preview(mask),
    }


def _summarize_engine_prompt(prompt) -> dict:
    if not isinstance(prompt, dict):
        return {"repr": _truncate_text(repr(prompt), 200), "type": type(prompt).__name__}

    prompt_type = prompt.get("type")
    if prompt_type == "enc_dec":
        return {
            "type": prompt_type,
            "encoder_prompt": _summarize_engine_prompt(prompt.get("encoder_prompt")),
            "decoder_prompt": _summarize_engine_prompt(prompt.get("decoder_prompt")),
        }

    return _summarize_single_engine_prompt(prompt)


def _record_api_prompt_state(
    stage: str,
    request,
    messages,
    conversation,
    engine_prompt,
) -> None:
    if not _mm_api_prompt_debug_enabled():
        return

    body_request_id = getattr(request, "request_id", None)
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "phase": os.environ.get("VLLM_ASCEND_API_OPT_PHASE", ""),
        "stage": stage,
        "body_request_id": body_request_id,
        "engine_request_id_hint": (
            f"chatcmpl-{body_request_id}" if body_request_id else None
        ),
        "request_messages": _summarize_chat_messages(messages),
        "conversation": _truncate_text(repr(conversation), 800),
        "engine_prompt": _summarize_engine_prompt(engine_prompt),
    }
    _append_mm_api_prompt_debug(record)


def _record_qwen35_mm_input_order(
    model,
    input_ids,
    is_multimodal,
    multimodal_embeddings,
) -> None:
    if not _mm_input_order_debug_enabled():
        return
    if input_ids is None or is_multimodal is None:
        return
    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return

    context = getattr(model, "_ascend_mm_input_order_debug_context", None)
    if not context:
        return

    try:
        input_ids_list = input_ids.detach().cpu().tolist()
        mask_list = is_multimodal.detach().to("cpu").tolist()
    except Exception as exc:
        _append_mm_input_order_debug(
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "phase": os.environ.get("VLLM_ASCEND_API_OPT_PHASE", ""),
                "stage": "qwen35_embed_input_ids_error",
                "error": repr(exc),
            }
        )
        setattr(model, "_ascend_mm_input_order_debug_context", None)
        return

    if isinstance(mask_list, bool):
        mask_list = [mask_list]

    requests = []
    cursor = 0
    req_ids = list(context.get("req_ids") or [])
    scheduled_counts = dict(context.get("scheduled_counts") or {})
    req_mm_positions = dict(context.get("req_mm_positions") or {})
    req_num_computed_tokens = dict(context.get("req_num_computed_tokens") or {})

    for req_id in req_ids:
        count = int(scheduled_counts.get(req_id, 0))
        if count <= 0:
            continue

        req_input_ids = input_ids_list[cursor : cursor + count]
        req_mask = [bool(v) for v in mask_list[cursor : cursor + count]]
        spans = _mask_spans(req_mask)
        requests.append(
            {
                "request_id": req_id,
                "num_computed_tokens_before_step": int(
                    req_num_computed_tokens.get(req_id, 0)
                ),
                "scheduled_token_count": count,
                "mm_token_count": sum(1 for v in req_mask if v),
                "starts_with_multimodal": bool(req_mask[0]) if req_mask else False,
                "ends_with_multimodal": bool(req_mask[-1]) if req_mask else False,
                "text_prefix_len": _text_prefix_len(req_mask),
                "text_suffix_len": _text_suffix_len(req_mask),
                "mm_mask_spans": spans,
                "mm_mask_preview": _mask_preview(req_mask),
                "input_ids_head": req_input_ids[:32],
                "input_ids_tail": req_input_ids[-16:] if len(req_input_ids) > 16 else [],
                "mm_feature_positions": req_mm_positions.get(req_id, []),
            }
        )
        cursor += count

    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "phase": os.environ.get("VLLM_ASCEND_API_OPT_PHASE", ""),
        "stage": "qwen35_embed_input_ids",
        "batch_total_tokens": len(input_ids_list),
        "batch_mm_token_count": sum(1 for v in mask_list if v),
        "batch_mask_preview": _mask_preview([bool(v) for v in mask_list]),
        "request_count": len(requests),
        "requests": requests,
    }
    _append_mm_input_order_debug(record)
    setattr(model, "_ascend_mm_input_order_debug_context", None)


def apply_perf_patches() -> None:
    global _PERF_PATCHED
    if _PERF_PATCHED:
        return

    from vllm.distributed.device_communicators.shm_object_storage import (
        MsgpackSerde,
        SingleWriterShmObjectStorage,
    )
    from vllm.entrypoints.chat_utils import AsyncMultiModalContentParser
    from vllm.entrypoints.openai.engine.serving import OpenAIServing
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
    from vllm.multimodal.cache import (
        ShmObjectStoreReceiverCache,
        ShmObjectStoreSenderCache,
    )
    from vllm.multimodal.media.connector import MediaConnector
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.renderers.base import BaseRenderer
    from vllm.utils.async_utils import AsyncMicrobatchTokenizer
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if (
        getattr(AsyncMultiModalContentParser._image_with_uuid_async, "__name__", "")
        != "_requested_image_with_uuid_async"
    ):
        original_image_with_uuid_async = (
            AsyncMultiModalContentParser._image_with_uuid_async
        )

        async def _requested_image_with_uuid_async(self, image_url, uuid):
            with perf_probe_multi_span(
                "api_server",
                ("api_parse_image_url",),
                extra={"has_image_url": image_url is not None},
            ):
                return await original_image_with_uuid_async(self, image_url, uuid)

        AsyncMultiModalContentParser._image_with_uuid_async = (
            _requested_image_with_uuid_async
        )

    if getattr(MediaConnector.fetch_image_async, "__name__", "") != "_requested_fetch_image_async":
        original_fetch_image_async = MediaConnector.fetch_image_async

        async def _requested_fetch_image_async(self, image_url: str, *, image_mode: str = "RGB"):
            def _extra():
                try:
                    url_scheme = parse_url(image_url).scheme if isinstance(image_url, str) else None
                except Exception:
                    url_scheme = None
                return {
                    "image_url": image_url[:256] if isinstance(image_url, str) else None,
                    "url_scheme": url_scheme,
                }

            with perf_probe_multi_span(
                "api_server",
                ("MediaConnector.fetch_image_async", "api_fetch_image_async"),
                extra=_extra,
            ):
                return await original_fetch_image_async(
                    self,
                    image_url,
                    image_mode=image_mode,
                )

        MediaConnector.fetch_image_async = _requested_fetch_image_async

    if getattr(AsyncMicrobatchTokenizer.encode, "__name__", "") != "_requested_async_tokenizer_encode":
        original_async_encode = AsyncMicrobatchTokenizer.encode

        async def _requested_async_tokenizer_encode(self, prompt, **kwargs):
            extra = {
                "prompt_type": type(prompt).__name__,
            }
            with perf_probe_multi_span(
                "api_server",
                ("AsyncMicrobatchTokenizer.encode",),
                extra=extra,
            ):
                return await original_async_encode(self, prompt, **kwargs)

        AsyncMicrobatchTokenizer.encode = _requested_async_tokenizer_encode

    original_tokenize_prompt_async = getattr(BaseRenderer, "tokenize_prompt_async", None)
    if (
        original_tokenize_prompt_async is not None
        and getattr(original_tokenize_prompt_async, "__name__", "")
        != "_requested_tokenize_prompt_async"
    ):

        async def _requested_tokenize_prompt_async(self, prompt, params):
            extra = {
                "prompt_type": type(prompt).__name__,
            }
            with perf_probe_multi_span(
                "api_server",
                ("api_tokenize_prompt_async",),
                extra=extra,
            ):
                return await original_tokenize_prompt_async(self, prompt, params)

        BaseRenderer.tokenize_prompt_async = _requested_tokenize_prompt_async

    if getattr(BaseRenderer._process_multimodal, "__name__", "") != "_requested_process_multimodal":
        original_process_multimodal = BaseRenderer._process_multimodal

        def _requested_process_multimodal(
            self,
            prompt,
            mm_data,
            mm_uuids,
            mm_processor_kwargs,
            tokenization_kwargs,
        ):
            extra = {
                "prompt_type": type(prompt).__name__,
                "modalities": sorted(mm_data.keys()) if isinstance(mm_data, dict) else [],
            }
            with perf_probe_multi_span(
                "api_server",
                ("api_process_multimodal_total",),
                extra=extra,
            ):
                return original_process_multimodal(
                    self,
                    prompt,
                    mm_data,
                    mm_uuids,
                    mm_processor_kwargs,
                    tokenization_kwargs,
                )

        BaseRenderer._process_multimodal = _requested_process_multimodal

    if getattr(BaseRenderer.process_for_engine, "__name__", "") != "_requested_process_for_engine":
        original_process_for_engine = BaseRenderer.process_for_engine

        def _requested_process_for_engine(self, prompt, arrival_time: float):
            with perf_probe_multi_span(
                "api_server",
                ("api_process_for_engine_total",),
            ):
                return original_process_for_engine(self, prompt, arrival_time)

        BaseRenderer.process_for_engine = _requested_process_for_engine

    original_process_for_engine_async = getattr(
        BaseRenderer,
        "process_for_engine_async",
        None,
    )
    if (
        original_process_for_engine_async is not None
        and getattr(original_process_for_engine_async, "__name__", "")
        != "_requested_process_for_engine_async"
    ):

        async def _requested_process_for_engine_async(self, prompt, arrival_time: float):
            with perf_probe_multi_span(
                "api_server",
                ("api_process_for_engine_async_total",),
            ):
                return await original_process_for_engine_async(self, prompt, arrival_time)

        BaseRenderer.process_for_engine_async = _requested_process_for_engine_async

    if getattr(ProcessorInputs.get_mm_hashes, "__name__", "") != "_requested_get_mm_hashes":
        original_get_mm_hashes = ProcessorInputs.get_mm_hashes

        def _requested_get_mm_hashes(self, model_id: str):
            mm_data_items = getattr(self, "mm_data_items", None)
            extra = {
                "model_id": model_id,
                "modalities": sorted(mm_data_items.keys()) if mm_data_items else [],
            }
            with perf_probe_multi_span(
                "api_server",
                ("ProcessorInputs.get_mm_hashes",),
                extra=extra,
            ):
                return original_get_mm_hashes(self, model_id)

        ProcessorInputs.get_mm_hashes = _requested_get_mm_hashes

    if getattr(OpenAIServing._preprocess_chat, "__name__", "") != "_requested_openai_preprocess_chat":
        original_openai_preprocess_chat = OpenAIServing._preprocess_chat

        async def _requested_openai_preprocess_chat(
            self,
            request,
            messages,
            default_template,
            default_template_content_format,
            default_template_kwargs,
            tool_dicts=None,
            tool_parser=None,
        ):
            conversation, engine_prompts = await original_openai_preprocess_chat(
                self,
                request,
                messages,
                default_template,
                default_template_content_format,
                default_template_kwargs,
                tool_dicts=tool_dicts,
                tool_parser=tool_parser,
            )
            if engine_prompts:
                _record_api_prompt_state(
                    "openai_preprocess_chat",
                    request,
                    messages,
                    conversation,
                    engine_prompts[0],
                )
            return conversation, engine_prompts

        OpenAIServing._preprocess_chat = _requested_openai_preprocess_chat

    if getattr(OpenAIServingRender._preprocess_chat, "__name__", "") != "_requested_openai_render_preprocess_chat":
        original_openai_render_preprocess_chat = OpenAIServingRender._preprocess_chat

        async def _requested_openai_render_preprocess_chat(
            self,
            request,
            messages,
            default_template,
            default_template_content_format,
            default_template_kwargs,
            tool_dicts=None,
            tool_parser=None,
        ):
            conversation, engine_prompts = await original_openai_render_preprocess_chat(
                self,
                request,
                messages,
                default_template,
                default_template_content_format,
                default_template_kwargs,
                tool_dicts=tool_dicts,
                tool_parser=tool_parser,
            )
            if engine_prompts:
                _record_api_prompt_state(
                    "openai_render_preprocess_chat",
                    request,
                    messages,
                    conversation,
                    engine_prompts[0],
                )
            return conversation, engine_prompts

        OpenAIServingRender._preprocess_chat = _requested_openai_render_preprocess_chat

    if getattr(Qwen3VLMultiModalProcessor._call_hf_processor, "__name__", "") != "_requested_qwen3_call_hf_processor":
        original_qwen3_call_hf_processor = Qwen3VLMultiModalProcessor._call_hf_processor

        def _requested_qwen3_call_hf_processor(
            self,
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        ):
            component = runtime_component("api_server")
            extra = {
                "prompt_type": type(prompt).__name__,
                "modalities": sorted(mm_data.keys()) if isinstance(mm_data, dict) else [],
            }
            with perf_probe_multi_span(
                component,
                ("Qwen3VLMultiModalProcessor._call_hf_processor",),
                extra=extra,
            ):
                return original_qwen3_call_hf_processor(
                    self,
                    prompt,
                    mm_data,
                    mm_kwargs,
                    tok_kwargs,
                )

        Qwen3VLMultiModalProcessor._call_hf_processor = _requested_qwen3_call_hf_processor

    if getattr(ShmObjectStoreSenderCache.get_and_update_item, "__name__", "") != "_requested_shm_sender_get_and_update_item":
        original_get_and_update_item = ShmObjectStoreSenderCache.get_and_update_item

        def _requested_shm_sender_get_and_update_item(self, mm_item, mm_hash: str):
            with perf_probe_multi_span(
                "api_server",
                (
                    "ShmObjectStoreSenderCache.get_and_update_items_with_callback",
                    "api_mm_cache_put_shm",
                ),
                extra={"mm_hash": mm_hash},
            ):
                return original_get_and_update_item(self, mm_item, mm_hash)

        ShmObjectStoreSenderCache.get_and_update_item = _requested_shm_sender_get_and_update_item

    if getattr(SingleWriterShmObjectStorage.copy_to_buffer, "__name__", "") != "_requested_copy_to_buffer":
        original_copy_to_buffer = SingleWriterShmObjectStorage.copy_to_buffer

        def _requested_copy_to_buffer(
            self,
            data,
            data_bytes: int,
            metadata: bytes,
            md_bytes: int,
            data_view,
        ) -> None:
            extra = {
                "data_bytes": int(data_bytes),
                "metadata_bytes": int(md_bytes),
                "chunk_count": len(data) if isinstance(data, list) else 1,
            }
            with perf_probe_multi_span(
                "api_server",
                ("SingleWriterShmObjectStorage.batch_copy_to_buffer",),
                extra=extra,
            ):
                return original_copy_to_buffer(
                    self,
                    data,
                    data_bytes,
                    metadata,
                    md_bytes,
                    data_view,
                )

        SingleWriterShmObjectStorage.copy_to_buffer = _requested_copy_to_buffer

    if getattr(ShmObjectStoreReceiverCache.get_and_update_item, "__name__", "") != "_requested_shm_receiver_get_and_update_item":
        original_receiver_get_and_update_item = (
            ShmObjectStoreReceiverCache.get_and_update_item
        )

        def _requested_shm_receiver_get_and_update_item(self, mm_item, mm_hash: str):
            if mm_item is not None and "address" in mm_item:
                with perf_probe_multi_span(
                    "tp_worker",
                    ("tp_worker_mm_cache_get_shm",),
                    extra={"mm_hash": mm_hash},
                ):
                    return original_receiver_get_and_update_item(self, mm_item, mm_hash)
            return original_receiver_get_and_update_item(self, mm_item, mm_hash)

        ShmObjectStoreReceiverCache.get_and_update_item = (
            _requested_shm_receiver_get_and_update_item
        )

    if getattr(EngineCore.__init__, "__name__", "") != "_requested_engine_core_init":
        original_engine_core_init = EngineCore.__init__

        def _requested_engine_core_init(self, *args, **kwargs):
            original_engine_core_init(self, *args, **kwargs)
            self.step_fn = (
                self.step
                if getattr(self, "batch_queue", None) is None
                else self.step_with_batch_queue
            )

        EngineCore.__init__ = _requested_engine_core_init

    if getattr(EngineCore.step, "__name__", "") != "_requested_engine_core_step":
        def _requested_engine_core_step(self):
            start = time.perf_counter()
            scheduler_output = None
            extra = None
            try:
                if not self.scheduler.has_requests():
                    return {}, False
                scheduler_output = self.scheduler.schedule()
                extra = {
                    "total_num_scheduled_tokens": int(
                        getattr(scheduler_output, "total_num_scheduled_tokens", 0)
                    ),
                    "scheduled_encoder_request_count": _scheduled_encoder_request_count(
                        scheduler_output
                    ),
                }
                future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
                grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
                with (
                    self.log_error_detail(scheduler_output),
                    self.log_iteration_details(scheduler_output),
                ):
                    model_output = future.result()
                    if model_output is None:
                        model_output = self.model_executor.sample_tokens(grammar_output)

                self._process_aborts_queue()
                engine_core_outputs = self.scheduler.update_from_output(
                    scheduler_output, model_output
                )

                return (
                    engine_core_outputs,
                    scheduler_output.total_num_scheduled_tokens > 0,
                )
            finally:
                elapsed_s = time.perf_counter() - start
                perf_probe_record(
                    "engine_core",
                    "engine_core_time",
                    elapsed_s,
                    extra=extra,
                )
                perf_probe_record(
                    "engine_core",
                    "engine_core_step_total",
                    elapsed_s,
                    extra=extra,
                )
                if scheduler_output is not None and _is_decode_only_scheduler_step(
                    scheduler_output
                ):
                    perf_probe_record(
                        "engine_core",
                        "engine_core_decode_only_time",
                        elapsed_s,
                        extra=extra,
                    )

        EngineCore.step = _requested_engine_core_step

    if getattr(EngineCore.step_with_batch_queue, "__name__", "") != "_requested_engine_core_step_with_batch_queue":
        def _requested_engine_core_step_with_batch_queue(self):
            start = time.perf_counter()
            completed_scheduler_output = None
            extra = None
            try:
                batch_queue = self.batch_queue
                assert batch_queue is not None
                assert len(batch_queue) < self.batch_queue_size

                model_executed = False
                deferred_scheduler_output = None
                exec_future = None
                if self.scheduler.has_requests():
                    scheduler_output = self.scheduler.schedule()
                    with self.log_error_detail(scheduler_output):
                        exec_future = self.model_executor.execute_model(
                            scheduler_output, non_block=True
                        )
                    if self.is_ec_consumer:
                        model_executed = (
                            scheduler_output.total_num_scheduled_tokens > 0
                        )

                    if self.is_pooling_model or not model_executed:
                        future = exec_future
                    else:
                        if not scheduler_output.pending_structured_output_tokens:
                            grammar_output = self.scheduler.get_grammar_bitmask(
                                scheduler_output
                            )
                            future = self.model_executor.sample_tokens(
                                grammar_output, non_block=True
                            )
                        else:
                            deferred_scheduler_output = scheduler_output

                    if not deferred_scheduler_output:
                        batch_queue.appendleft((future, scheduler_output, exec_future))
                        if (
                            model_executed
                            and len(batch_queue) < self.batch_queue_size
                            and not batch_queue[-1][0].done()
                        ):
                            return None, True

                elif not batch_queue:
                    return None, False

                future, scheduler_output, exec_model_fut = batch_queue.pop()
                completed_scheduler_output = scheduler_output
                extra = {
                    "total_num_scheduled_tokens": int(
                        getattr(scheduler_output, "total_num_scheduled_tokens", 0)
                    ),
                    "scheduled_encoder_request_count": _scheduled_encoder_request_count(
                        scheduler_output
                    ),
                }
                with (
                    self.log_error_detail(scheduler_output),
                    self.log_iteration_details(scheduler_output),
                ):
                    model_output = future.result()
                    if model_output is None:
                        exec_model_fut.result()
                        raise RuntimeError("unexpected error")

                self._process_aborts_queue()
                engine_core_outputs = self.scheduler.update_from_output(
                    scheduler_output, model_output
                )

                if deferred_scheduler_output:
                    if self.use_spec_decode:
                        draft_token_ids = self.model_executor.take_draft_token_ids()
                        assert draft_token_ids is not None
                        self.scheduler.update_draft_token_ids_in_output(
                            draft_token_ids, deferred_scheduler_output
                        )
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        deferred_scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                    batch_queue.appendleft(
                        (future, deferred_scheduler_output, exec_future)
                    )

                return engine_core_outputs, model_executed
            finally:
                elapsed_s = time.perf_counter() - start
                perf_probe_record(
                    "engine_core",
                    "engine_core_time",
                    elapsed_s,
                    extra=extra,
                )
                perf_probe_record(
                    "engine_core",
                    "engine_core_step_total",
                    elapsed_s,
                    extra=extra,
                )
                if completed_scheduler_output is not None and _is_decode_only_scheduler_step(
                    completed_scheduler_output
                ):
                    perf_probe_record(
                        "engine_core",
                        "engine_core_decode_only_time",
                        elapsed_s,
                        extra=extra,
                    )

        EngineCore.step_with_batch_queue = _requested_engine_core_step_with_batch_queue

    if getattr(ImageMediaIO.load_file, "__name__", "") != "_requested_image_load_file":
        original_load_file = ImageMediaIO.load_file

        def _requested_image_load_file(self, filepath):
            component = runtime_component("api_server")
            extra = {
                "filepath": str(filepath)[:256],
            }
            with perf_probe_multi_span(
                component,
                ("ImageMediaIO.load_file",),
                extra=extra,
            ):
                return original_load_file(self, filepath)

        ImageMediaIO.load_file = _requested_image_load_file

    if getattr(MsgpackSerde.deserialize, "__name__", "") != "_requested_msgpack_deserialize":
        original_msgpack_deserialize = MsgpackSerde.deserialize

        def _requested_msgpack_deserialize(self, data_view):
            component = runtime_component("tp_worker")
            with perf_probe_multi_span(
                component,
                ("MsgpackSerde.deserialize",),
            ):
                return original_msgpack_deserialize(self, data_view)

        MsgpackSerde.deserialize = _requested_msgpack_deserialize

    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        from vllm_ascend.worker.worker import NPUWorker
    except Exception:
        NPUModelRunner = None
        NPUWorker = None

    if (
        NPUModelRunner is not None
        and getattr(NPUModelRunner._prepare_inputs, "__name__", "")
        != "_requested_prepare_inputs"
    ):
        original_prepare_inputs = NPUModelRunner._prepare_inputs

        def _requested_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
            extra = {
                "total_num_scheduled_tokens": int(
                    getattr(scheduler_output, "total_num_scheduled_tokens", 0)
                ),
            }
            with perf_probe_multi_span(
                "tp_worker",
                ("tp_worker_prepare_inputs",),
                extra=extra,
            ):
                return original_prepare_inputs(
                    self,
                    scheduler_output,
                    num_scheduled_tokens,
                )

        NPUModelRunner._prepare_inputs = _requested_prepare_inputs

    if (
        NPUModelRunner is not None
        and getattr(NPUModelRunner._execute_mm_encoder, "__name__", "")
        != "_requested_execute_mm_encoder"
    ):
        original_execute_mm_encoder = NPUModelRunner._execute_mm_encoder

        def _requested_execute_mm_encoder(self, scheduler_output):
            extra = {
                "scheduled_request_count": len(
                    getattr(scheduler_output, "scheduled_encoder_inputs", {}) or {}
                ),
            }
            with perf_probe_multi_span(
                "tp_worker",
                ("tp_worker_vit_inference",),
                extra=extra,
            ):
                return original_execute_mm_encoder(self, scheduler_output)

        NPUModelRunner._execute_mm_encoder = _requested_execute_mm_encoder

    if (
        NPUWorker is not None
        and getattr(NPUWorker.execute_model, "__name__", "")
        != "_requested_npu_worker_execute_model"
    ):
        original_worker_execute_model = NPUWorker.execute_model

        def _requested_npu_worker_execute_model(self, scheduler_output):
            extra = {
                "total_num_scheduled_tokens": int(
                    getattr(scheduler_output, "total_num_scheduled_tokens", 0)
                ),
                "scheduled_encoder_request_count": _scheduled_encoder_request_count(
                    scheduler_output
                ),
            }
            stage_names = ["tp_worker_execute_model_total"]
            if _is_decode_only_scheduler_step(scheduler_output):
                stage_names.append("tp_worker_execute_model_decode_only")
            with perf_probe_multi_span(
                "tp_worker",
                tuple(stage_names),
                extra=extra,
            ):
                return original_worker_execute_model(self, scheduler_output)

        NPUWorker.execute_model = _requested_npu_worker_execute_model

    if getattr(GPUModelRunner._execute_mm_encoder, "__name__", "") != "_requested_gpu_model_runner_execute_mm_encoder":
        original_gpu_execute_mm_encoder = GPUModelRunner._execute_mm_encoder

        def _requested_gpu_model_runner_execute_mm_encoder(self, scheduler_output):
            result = original_gpu_execute_mm_encoder(self, scheduler_output)
            scheduled = getattr(scheduler_output, "scheduled_encoder_inputs", None) or {}
            for req_id in scheduled:
                req_state = self.requests.get(req_id)
                if req_state is None or getattr(req_state, "_mm_feature_debug_prefill_logged", False):
                    continue
                _record_mm_feature_state(
                    "post_mm_encoder",
                    req_id,
                    req_state,
                    decode_only=False,
                    extra={
                        "scheduled_mm_input_count": len(scheduled.get(req_id) or []),
                    },
                )
                setattr(req_state, "_mm_feature_debug_prefill_logged", True)
            return result

        GPUModelRunner._execute_mm_encoder = _requested_gpu_model_runner_execute_mm_encoder

    if getattr(GPUModelRunner._gather_mm_embeddings, "__name__", "") != "_requested_gpu_model_runner_gather_mm_embeddings":
        original_gather_mm_embeddings = GPUModelRunner._gather_mm_embeddings

        def _requested_gpu_model_runner_gather_mm_embeddings(
            self,
            scheduler_output,
            shift_computed_tokens: int = 0,
        ):
            if _is_decode_only_scheduler_step(scheduler_output):
                for req_id in self.input_batch.req_ids:
                    req_state = self.requests.get(req_id)
                    if req_state is None or getattr(req_state, "_mm_feature_debug_decode_logged", False):
                        continue
                    _record_mm_feature_state(
                        "decode_only_before_gather",
                        req_id,
                        req_state,
                        decode_only=True,
                        extra={
                            "shift_computed_tokens": int(shift_computed_tokens),
                            "scheduled_tokens": int(
                                getattr(scheduler_output, "num_scheduled_tokens", {}).get(req_id, 0)
                            ),
                        },
                    )
                    setattr(req_state, "_mm_feature_debug_decode_logged", True)
            result = original_gather_mm_embeddings(
                self,
                scheduler_output,
                shift_computed_tokens=shift_computed_tokens,
            )
            if _mm_input_order_debug_enabled():
                scheduled_counts = {}
                req_ids = []
                req_mm_positions = {}
                req_num_computed_tokens = {}
                for req_id in self.input_batch.req_ids:
                    scheduled_count = int(
                        getattr(scheduler_output, "num_scheduled_tokens", {}).get(
                            req_id, 0
                        )
                    )
                    if scheduled_count <= 0:
                        continue
                    req_state = self.requests.get(req_id)
                    req_ids.append(req_id)
                    scheduled_counts[req_id] = scheduled_count
                    if req_state is not None:
                        req_mm_positions[req_id] = _summarize_req_mm_positions(
                            req_state
                        )
                        req_num_computed_tokens[req_id] = int(
                            getattr(req_state, "num_computed_tokens", 0)
                        )

                setattr(
                    self.model,
                    "_ascend_mm_input_order_debug_context",
                    {
                        "req_ids": req_ids,
                        "scheduled_counts": scheduled_counts,
                        "req_mm_positions": req_mm_positions,
                        "req_num_computed_tokens": req_num_computed_tokens,
                    },
                )
            return result

        GPUModelRunner._gather_mm_embeddings = _requested_gpu_model_runner_gather_mm_embeddings

    if getattr(Qwen3_5ForConditionalGeneration.embed_input_ids, "__name__", "") != "_requested_qwen35_embed_input_ids":
        original_qwen35_embed_input_ids = Qwen3_5ForConditionalGeneration.embed_input_ids

        def _requested_qwen35_embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings=None,
            *,
            is_multimodal=None,
        ):
            result = original_qwen35_embed_input_ids(
                self,
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
            _record_qwen35_mm_input_order(
                self,
                input_ids,
                is_multimodal,
                multimodal_embeddings,
            )
            return result

        Qwen3_5ForConditionalGeneration.embed_input_ids = (
            _requested_qwen35_embed_input_ids
        )

    _PERF_PATCHED = True
