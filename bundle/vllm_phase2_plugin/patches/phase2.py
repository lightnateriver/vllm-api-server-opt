import os
import time
import traceback
from pathlib import Path

from ..common.mm import (
    are_all_image_paths_local,
    append_jsonl,
    build_phase3_mm_data,
    estimate_item_size,
    extract_image_paths_for_request,
    extract_image_source_metadata_for_request,
    extract_image_source_metadata_from_mm_data,
    extract_local_mm_paths_from_mm_data,
    get_image_count_from_mm_data,
    normalize_image_source_metadata,
    normalize_local_mm_paths,
    phase3_request_uses_local_media,
    phase3_request_uses_local_media_for_req_state,
    summarize_mm_item,
    tensor_equal,
)
from ..common.phase import env_flag, phase_at_least
from ..common.runtime import TRACE_FILE, VALIDATE_FILE
from ..probes.perf import perf_probe_span


_INPUT_PROCESSOR_PATCHED = False
_WORKER_PATCHED = False
_PARSER_PATCHED = False
_RENDERER_PATCHED = False
_OPENAI_CHAT_PATCHED = False


def _phase3_local_only_experiment_enabled() -> bool:
    value = os.environ.get("VLLM_ASCEND_PHASE3_LOCAL_ONLY_EXPERIMENT", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _phase3_disable_api_grid_only() -> bool:
    return env_flag("VLLM_ASCEND_PHASE3_DISABLE_API_GRID_ONLY")


def _phase3_disable_worker_rebuild() -> bool:
    return env_flag("VLLM_ASCEND_PHASE3_DISABLE_WORKER_REBUILD")


def patch_phase2_hf_renderer() -> None:
    global _RENDERER_PATCHED
    if _RENDERER_PATCHED:
        return

    from vllm.renderers.base import BaseRenderer
    from vllm.renderers.hf import HfRenderer

    original_render_messages = HfRenderer.render_messages
    original_render_messages_async = HfRenderer.render_messages_async
    original_process_multimodal = BaseRenderer._process_multimodal
    original_process_tokens = BaseRenderer._process_tokens

    def _phase2_render_messages(self, messages, params):
        conversation, prompt = original_render_messages(self, messages, params)
        phase3_local_ready = phase3_request_uses_local_media(
            prompt.get("multi_modal_data") if isinstance(prompt, dict) else None
        )
        local_mm_paths = extract_local_mm_paths_from_mm_data(
            prompt.get("multi_modal_data") if isinstance(prompt, dict) else None
        )
        image_source_metadata = extract_image_source_metadata_from_mm_data(
            prompt.get("multi_modal_data") if isinstance(prompt, dict) else None
        )
        if isinstance(prompt, dict):
            prompt["phase3_local_mm_ready"] = phase3_local_ready
        if local_mm_paths and isinstance(prompt, dict):
            prompt["phase2_local_mm_paths"] = local_mm_paths
        if image_source_metadata and isinstance(prompt, dict):
            prompt["phase2_image_source_metadata"] = image_source_metadata
        if (local_mm_paths or image_source_metadata) and isinstance(prompt, dict):
            append_jsonl(
                TRACE_FILE,
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "renderer_paths",
                    "local_mm_paths": local_mm_paths,
                    "image_source_metadata": image_source_metadata,
                    "phase3_local_mm_ready": phase3_local_ready,
                },
            )
        return conversation, prompt

    async def _phase2_render_messages_async(self, messages, params):
        conversation, prompt = await original_render_messages_async(self, messages, params)
        phase3_local_ready = phase3_request_uses_local_media(
            prompt.get("multi_modal_data") if isinstance(prompt, dict) else None
        )
        local_mm_paths = extract_local_mm_paths_from_mm_data(
            prompt.get("multi_modal_data") if isinstance(prompt, dict) else None
        )
        image_source_metadata = extract_image_source_metadata_from_mm_data(
            prompt.get("multi_modal_data") if isinstance(prompt, dict) else None
        )
        if isinstance(prompt, dict):
            prompt["phase3_local_mm_ready"] = phase3_local_ready
        if local_mm_paths and isinstance(prompt, dict):
            prompt["phase2_local_mm_paths"] = local_mm_paths
        if image_source_metadata and isinstance(prompt, dict):
            prompt["phase2_image_source_metadata"] = image_source_metadata
        if (local_mm_paths or image_source_metadata) and isinstance(prompt, dict):
            append_jsonl(
                TRACE_FILE,
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "renderer_paths_async",
                    "local_mm_paths": local_mm_paths,
                    "image_source_metadata": image_source_metadata,
                    "phase3_local_mm_ready": phase3_local_ready,
                },
            )
        return conversation, prompt

    def _phase2_process_multimodal(
        self,
        prompt,
        mm_data,
        mm_uuids,
        mm_processor_kwargs,
        tokenization_kwargs,
    ):
        mm_data_to_use = mm_data
        phase3_local_ready = phase3_request_uses_local_media(mm_data)
        if (
            phase_at_least(3)
            and phase3_local_ready
            and not _phase3_disable_api_grid_only()
        ):
            phase3_mm_data = build_phase3_mm_data(
                self,
                mm_data=mm_data,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            if phase3_mm_data is not None:
                mm_data_to_use = phase3_mm_data
                append_jsonl(
                    TRACE_FILE,
                    {
                        "ts": time.time(),
                        "pid": os.getpid(),
                        "stage": "renderer_phase3_mm_data",
                        "image_count": len(
                            phase3_mm_data.get("image", {}).get("image_grid_thw", [])
                        ),
                    },
                )
        elif phase_at_least(3):
            append_jsonl(
                TRACE_FILE,
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": (
                        "renderer_phase3_mm_data_disabled"
                        if phase3_local_ready
                        else "renderer_phase3_stock_fallback_nonlocal"
                    ),
                    "image_count": get_image_count_from_mm_data(mm_data),
                    "phase3_local_mm_ready": phase3_local_ready,
                },
            )

        return original_process_multimodal(
            self,
            prompt,
            mm_data_to_use,
            mm_uuids=mm_uuids,
            mm_processor_kwargs=mm_processor_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

    def _phase2_process_tokens(self, prompt):
        inputs = original_process_tokens(self, prompt)
        local_mm_paths = normalize_local_mm_paths(
            prompt.get("phase2_local_mm_paths") if isinstance(prompt, dict) else None
        )
        image_source_metadata = normalize_image_source_metadata(
            prompt.get("phase2_image_source_metadata") if isinstance(prompt, dict) else None
        )
        phase3_local_ready = bool(
            prompt.get("phase3_local_mm_ready") if isinstance(prompt, dict) else False
        )
        if local_mm_paths and isinstance(inputs, dict):
            inputs["phase2_local_mm_paths"] = local_mm_paths
        if image_source_metadata and isinstance(inputs, dict):
            inputs["phase2_image_source_metadata"] = image_source_metadata
        if (local_mm_paths or image_source_metadata) and isinstance(inputs, dict):
            append_jsonl(
                TRACE_FILE,
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "renderer_process_tokens",
                    "local_mm_paths": local_mm_paths,
                    "image_source_metadata": image_source_metadata,
                    "phase3_local_mm_ready": phase3_local_ready,
                },
            )
        if isinstance(inputs, dict):
            inputs["phase3_local_mm_ready"] = phase3_local_ready
        return inputs

    HfRenderer.render_messages = _phase2_render_messages
    HfRenderer.render_messages_async = _phase2_render_messages_async
    BaseRenderer._process_multimodal = _phase2_process_multimodal
    BaseRenderer._process_tokens = _phase2_process_tokens
    _RENDERER_PATCHED = True


def patch_phase2_openai_chat() -> None:
    global _OPENAI_CHAT_PATCHED
    if _OPENAI_CHAT_PATCHED:
        return

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    original_render_chat_request = OpenAIServingChat.render_chat_request
    original_to_sampling_params = ChatCompletionRequest.to_sampling_params

    async def _phase2_render_chat_request(self, request):
        result = await original_render_chat_request(self, request)
        if isinstance(result, tuple):
            _, engine_prompts = result
            queue = []
            phase3_ready_queue = []
            source_metadata_queue = []
            for prompt in engine_prompts:
                queue.append(
                    normalize_local_mm_paths(
                        prompt.get("phase2_local_mm_paths")
                        if isinstance(prompt, dict)
                        else None
                    )
                )
                phase3_ready_queue.append(
                    bool(
                        prompt.get("phase3_local_mm_ready")
                        if isinstance(prompt, dict)
                        else False
                    )
                )
                source_metadata_queue.append(
                    normalize_image_source_metadata(
                        prompt.get("phase2_image_source_metadata")
                        if isinstance(prompt, dict)
                        else None
                    )
                )

            request.__dict__["_phase2_local_mm_paths_queue"] = queue
            request.__dict__["_phase3_local_mm_ready_queue"] = phase3_ready_queue
            request.__dict__["_phase2_image_source_metadata_queue"] = source_metadata_queue
            request.__dict__["_phase2_base_vllm_xargs"] = dict(request.vllm_xargs or {})
            append_jsonl(
                TRACE_FILE,
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "chat_request_queue",
                    "queue": queue,
                    "source_metadata_queue": source_metadata_queue,
                    "phase3_local_mm_ready_queue": phase3_ready_queue,
                },
            )
        return result

    def _phase2_to_sampling_params(self, max_tokens, default_sampling_params):
        queue = self.__dict__.get("_phase2_local_mm_paths_queue")
        phase3_ready_queue = self.__dict__.get("_phase3_local_mm_ready_queue")
        source_metadata_queue = self.__dict__.get("_phase2_image_source_metadata_queue")
        base_vllm_xargs = dict(self.__dict__.get("_phase2_base_vllm_xargs") or {})

        current_paths = None
        if isinstance(queue, list) and queue:
            current_paths = queue.pop(0)
            self.__dict__["_phase2_local_mm_paths_queue"] = queue

        current_phase3_ready = False
        if isinstance(phase3_ready_queue, list) and phase3_ready_queue:
            current_phase3_ready = bool(phase3_ready_queue.pop(0))
            self.__dict__["_phase3_local_mm_ready_queue"] = phase3_ready_queue

        current_source_metadata = None
        if isinstance(source_metadata_queue, list) and source_metadata_queue:
            current_source_metadata = source_metadata_queue.pop(0)
            self.__dict__["_phase2_image_source_metadata_queue"] = source_metadata_queue

        if current_paths is not None:
            merged_xargs = dict(base_vllm_xargs)
            merged_xargs["phase2_local_mm_paths"] = current_paths
            merged_xargs["phase3_local_mm_ready"] = current_phase3_ready
            if current_source_metadata is not None:
                merged_xargs["phase2_image_source_metadata"] = current_source_metadata
            self.vllm_xargs = merged_xargs
            append_jsonl(
                TRACE_FILE,
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "chat_sampling_params",
                    "phase2_local_mm_paths": current_paths,
                    "phase2_image_source_metadata": current_source_metadata,
                    "phase3_local_mm_ready": current_phase3_ready,
                },
            )
        else:
            merged_xargs = dict(base_vllm_xargs)
            if current_phase3_ready:
                merged_xargs["phase3_local_mm_ready"] = current_phase3_ready
            if current_source_metadata is not None:
                merged_xargs["phase2_image_source_metadata"] = current_source_metadata
            self.vllm_xargs = merged_xargs or None

        return original_to_sampling_params(self, max_tokens, default_sampling_params)

    OpenAIServingChat.render_chat_request = _phase2_render_chat_request
    ChatCompletionRequest.to_sampling_params = _phase2_to_sampling_params
    _OPENAI_CHAT_PATCHED = True


def patch_phase3_qwen_parser() -> None:
    global _PARSER_PATCHED
    if _PARSER_PATCHED:
        return

    from vllm.model_executor.models.qwen2_vl import (
        Qwen2VLMultiModalDataParser,
        _create_qwen2vl_field_factory,
    )
    from vllm.multimodal.inputs import MultiModalKwargsItems
    from vllm.multimodal.parse import ModalityDataItems

    original_parse_image_data = Qwen2VLMultiModalDataParser._parse_image_data

    class _Phase3GridItems(ModalityDataItems[dict, dict]):
        def __init__(self, data: dict, modality: str, fields_factory):
            from transformers.feature_extraction_utils import BatchFeature

            super().__init__(data, modality)
            self._kwargs = MultiModalKwargsItems.from_hf_inputs(
                BatchFeature(dict(data)),
                fields_factory(data),
            )

        def get_count(self) -> int:
            return len(self._kwargs[self.modality])

        def get(self, index: int):
            return self._kwargs[self.modality][index].get_data()

        def get_processor_data(self):
            return {}

        def get_passthrough_data(self):
            return self.data

    def _phase3_parse_image_data(self, data):
        if (
            phase_at_least(3)
            and isinstance(data, dict)
            and "image_grid_thw" in data
            and "image_embeds" not in data
        ):
            return _Phase3GridItems(
                data,
                modality="image",
                fields_factory=_create_qwen2vl_field_factory(self._spatial_merge_size),
            )

        return original_parse_image_data(self, data)

    Qwen2VLMultiModalDataParser._parse_image_data = _phase3_parse_image_data
    _PARSER_PATCHED = True


def patch_phase2_input_processor() -> None:
    global _INPUT_PROCESSOR_PATCHED
    if _INPUT_PROCESSOR_PATCHED:
        return

    from vllm.v1.engine.input_processor import InputProcessor

    original_process_inputs = InputProcessor.process_inputs

    def _phase2_process_inputs(
        self,
        request_id,
        prompt,
        params,
        supported_tasks,
        arrival_time=None,
        lora_request=None,
        tokenization_kwargs=None,
        trace_headers=None,
        priority=0,
        data_parallel_rank=None,
        resumable=False,
    ):
        raw_local_mm_paths = None
        raw_image_source_metadata = None
        raw_phase3_local_ready = False
        if isinstance(prompt, dict):
            raw_local_mm_paths = normalize_local_mm_paths(
                prompt.get("phase2_local_mm_paths")
            )
            raw_image_source_metadata = normalize_image_source_metadata(
                prompt.get("phase2_image_source_metadata")
            )
            raw_phase3_local_ready = bool(prompt.get("phase3_local_mm_ready"))

        processed_prompt = prompt
        processed_tokenization_kwargs = tokenization_kwargs

        if not (isinstance(prompt, dict) and "type" in prompt):
            processed_prompt = self.input_preprocessor.preprocess(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
            )
            processed_tokenization_kwargs = None

        request = original_process_inputs(
            self,
            request_id,
            processed_prompt,
            params,
            supported_tasks,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=processed_tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            resumable=resumable,
        )

        local_mm_paths = None
        image_source_metadata = None
        phase3_local_ready = raw_phase3_local_ready
        if isinstance(processed_prompt, dict):
            local_mm_paths = normalize_local_mm_paths(
                processed_prompt.get("phase2_local_mm_paths")
            )
            image_source_metadata = normalize_image_source_metadata(
                processed_prompt.get("phase2_image_source_metadata")
            )
            phase3_local_ready = bool(
                processed_prompt.get("phase3_local_mm_ready", raw_phase3_local_ready)
            )
        if local_mm_paths is None:
            local_mm_paths = raw_local_mm_paths
        if image_source_metadata is None:
            image_source_metadata = raw_image_source_metadata

        if local_mm_paths and request.sampling_params is not None:
            extra_args = dict(request.sampling_params.extra_args or {})
            extra_args["phase2_local_mm_paths"] = local_mm_paths
            if image_source_metadata is not None:
                extra_args["phase2_image_source_metadata"] = image_source_metadata
            extra_args["phase3_local_mm_ready"] = phase3_local_ready and are_all_image_paths_local(
                local_mm_paths,
                expected_count=len(local_mm_paths["image"]),
            )
            request.sampling_params.extra_args = extra_args
            append_jsonl(
                TRACE_FILE,
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "engine_input",
                    "request_id": request.request_id,
                    "local_mm_paths": local_mm_paths,
                    "image_source_metadata": image_source_metadata,
                    "phase3_local_mm_ready": extra_args["phase3_local_mm_ready"],
                },
            )
        elif request.sampling_params is not None:
            extra_args = dict(request.sampling_params.extra_args or {})
            if image_source_metadata is not None:
                extra_args["phase2_image_source_metadata"] = image_source_metadata
            extra_args["phase3_local_mm_ready"] = phase3_local_ready
            request.sampling_params.extra_args = extra_args

        return request

    InputProcessor.process_inputs = _phase2_process_inputs
    _INPUT_PROCESSOR_PATCHED = True


def patch_phase2_worker_validation() -> None:
    global _WORKER_PATCHED
    if _WORKER_PATCHED:
        return

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.model_executor.models.vision import get_load_balance_assignment
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import MultiModalKwargsItems
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.multimodal.utils import group_and_batch_mm_kwargs
    from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_batch_mm_inputs = GPUModelRunner._batch_mm_inputs_from_scheduler
    original_execute = GPUModelRunner._execute_mm_encoder

    def _phase3_encode_local_items_and_gather(
        self,
        *,
        image_features,
        local_indices,
        local_items,
        order,
        counts,
    ):
        import torch

        model = self.model
        local_mm_kwargs = [("image", item) for item in local_items]
        local_outputs = []

        original_use_data_parallel = getattr(model, "use_data_parallel", None)
        if original_use_data_parallel is not None:
            model.use_data_parallel = False

        try:
            for _, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
                local_mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
            ):
                batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
                sanity_check_mm_encoder_outputs(
                    batch_outputs,
                    expected_num_items=num_items,
                )
                local_outputs.extend(batch_outputs)
        finally:
            if original_use_data_parallel is not None:
                model.use_data_parallel = original_use_data_parallel

        tp_size = get_tensor_model_parallel_world_size()
        prefix = [0]
        for count in counts:
            prefix.append(prefix[-1] + count)

        output_sizes = [int(feature.mm_position.get_num_embeds()) for feature in image_features]
        grouped_output_lens = []
        for rank in range(tp_size):
            rank_indices = order[prefix[rank] : prefix[rank + 1]]
            grouped_output_lens.append(sum(output_sizes[idx] for idx in rank_indices))

        hidden_size = int(getattr(model.visual, "out_hidden_size", 0))
        if getattr(model, "is_multimodal_pruning_enabled", False):
            hidden_size += 5

        if local_outputs:
            local_cat = torch.cat(local_outputs, dim=0)
            hidden_size = int(local_cat.shape[1])
        else:
            local_cat = torch.empty(
                (0, hidden_size),
                device=self.device,
                dtype=getattr(model.visual, "dtype", torch.float32),
            )

        max_len_per_rank = max(grouped_output_lens) if grouped_output_lens else 0
        if local_cat.shape[0] < max_len_per_rank:
            pad = torch.empty(
                (max_len_per_rank - local_cat.shape[0], hidden_size),
                device=local_cat.device,
                dtype=local_cat.dtype,
            )
            local_padded = torch.cat([local_cat, pad], dim=0)
        else:
            local_padded = local_cat

        gathered = tensor_model_parallel_all_gather(local_padded.contiguous(), dim=0)

        rank_embeddings = []
        for rank in range(tp_size):
            start = rank * max_len_per_rank
            end = start + grouped_output_lens[rank]
            rank_embeddings.append(gathered[start:end])

        original_order_embeddings = [None] * len(image_features)
        current_idx = 0
        for rank in range(tp_size):
            count = counts[rank]
            if count <= 0:
                continue
            rank_indices = order[current_idx : current_idx + count]
            rank_embed = rank_embeddings[rank]
            embed_start = 0
            for image_idx in rank_indices:
                image_len = output_sizes[image_idx]
                original_order_embeddings[image_idx] = rank_embed[
                    embed_start : embed_start + image_len
                ]
                embed_start += image_len
            current_idx += count

        assert all(embed is not None for embed in original_order_embeddings)
        return local_outputs, original_order_embeddings

    def _phase2_batch_mm_inputs_from_scheduler(self, scheduler_output):
        scheduled = scheduler_output.scheduled_encoder_inputs
        extra = {
            "request_count": len(scheduled or {}),
            "scheduled_item_count": sum(
                len(input_ids) for input_ids in (scheduled or {}).values()
            ),
        }
        with perf_probe_span(
            "tp_worker",
            "tp_worker_prepare_mm_inputs",
            extra=extra,
        ):
            return original_batch_mm_inputs(self, scheduler_output)

    def _phase2_execute_mm_encoder(self, scheduler_output):
        phase3_fallback_scheduled: dict[str, list[int]] = {}
        try:
            scheduled = scheduler_output.scheduled_encoder_inputs
            if scheduled:
                for req_id, input_ids in scheduled.items():
                    with perf_probe_span(
                        "tp_worker",
                        "tp_worker_local_mm_prepare_total",
                        extra={"request_id": req_id},
                    ):
                        with perf_probe_span(
                            "tp_worker",
                            "tp_worker_local_mm_assign",
                            extra={"request_id": req_id},
                        ):
                            req_state = self.requests[req_id]
                            tp_size = get_tensor_model_parallel_world_size()
                            tp_rank = get_tensor_model_parallel_rank()

                            image_features = []
                            image_paths = []
                            image_source_metadata = []
                            all_image_paths = extract_image_paths_for_request(req_state)
                            all_image_source_metadata = extract_image_source_metadata_for_request(
                                req_state
                            )
                            scheduled_modalities: list[str] = []

                            image_seen = 0
                            image_path_by_input_id: dict[int, str | None] = {}
                            image_source_by_input_id: dict[int, dict | None] = {}
                            for feature_idx, feature in enumerate(req_state.mm_features):
                                if feature.modality == "image":
                                    image_path_by_input_id[feature_idx] = (
                                        all_image_paths[image_seen]
                                        if image_seen < len(all_image_paths)
                                        else None
                                    )
                                    image_source_by_input_id[feature_idx] = (
                                        all_image_source_metadata[image_seen]
                                        if image_seen < len(all_image_source_metadata)
                                        else None
                                    )
                                    image_seen += 1

                            for mm_input_id in input_ids:
                                feature = req_state.mm_features[mm_input_id]
                                scheduled_modalities.append(str(feature.modality))
                                if feature.modality == "image" and feature.data is not None:
                                    image_features.append(feature)
                                    image_paths.append(
                                        image_path_by_input_id.get(mm_input_id)
                                    )
                                    image_source_metadata.append(
                                        image_source_by_input_id.get(mm_input_id)
                                    )

                            if any(modality != "image" for modality in scheduled_modalities):
                                phase3_fallback_scheduled[req_id] = list(input_ids)
                                append_jsonl(
                                    VALIDATE_FILE,
                                    {
                                        "ts": time.time(),
                                        "pid": os.getpid(),
                                        "request_id": req_id,
                                        "tp_rank": tp_rank,
                                        "tp_size": tp_size,
                                        "scheduled_modalities": scheduled_modalities,
                                        "status": "stock-fallback-unsupported-modality",
                                    },
                                )
                                append_jsonl(
                                    TRACE_FILE,
                                    {
                                        "ts": time.time(),
                                        "pid": os.getpid(),
                                        "stage": "worker_stock_fallback_unsupported_modality",
                                        "request_id": req_id,
                                        "tp_rank": tp_rank,
                                        "scheduled_modalities": scheduled_modalities,
                                    },
                                )
                                continue

                            if not image_features:
                                phase3_fallback_scheduled[req_id] = list(input_ids)
                                append_jsonl(
                                    VALIDATE_FILE,
                                    {
                                        "ts": time.time(),
                                        "pid": os.getpid(),
                                        "request_id": req_id,
                                        "tp_rank": tp_rank,
                                        "tp_size": tp_size,
                                        "scheduled_modalities": scheduled_modalities,
                                        "status": "stock-fallback-empty-image-features",
                                    },
                                )
                                continue

                            sizes = [
                                estimate_item_size(feature.data)
                                for feature in image_features
                            ]
                            order, counts, loads = get_load_balance_assignment(
                                sizes, tp_size
                            )
                            prefix = [0]
                            for count in counts:
                                prefix.append(prefix[-1] + count)
                            local_indices = order[prefix[tp_rank] : prefix[tp_rank + 1]]

                            record = {
                                "ts": time.time(),
                                "pid": os.getpid(),
                                "request_id": req_id,
                                "tp_rank": tp_rank,
                                "tp_size": tp_size,
                                "total_images": len(image_features),
                                "gpu_sample_counts": counts,
                                "gpu_loads": loads,
                                "local_global_indices": local_indices,
                                "sizes": sizes,
                            }
                            phase3_local_ready = phase3_request_uses_local_media_for_req_state(
                                req_state
                            )
                            record["phase3_local_mm_ready"] = phase3_local_ready

                            if phase_at_least(3) and not phase3_local_ready:
                                record["status"] = "phase3-stock-fallback-nonlocal"
                                phase3_fallback_scheduled[req_id] = list(input_ids)
                                append_jsonl(VALIDATE_FILE, record)
                                continue

                            if not local_indices and not phase_at_least(3):
                                record["status"] = "ok-empty"
                                append_jsonl(VALIDATE_FILE, record)
                                continue

                            local_paths = [image_paths[idx] for idx in local_indices]

                        append_jsonl(
                            TRACE_FILE,
                            {
                                "ts": time.time(),
                                "pid": os.getpid(),
                                "stage": "worker_request_paths",
                                "request_id": req_id,
                                "tp_rank": tp_rank,
                                "local_indices": list(local_indices),
                                "local_paths": local_paths,
                                "image_source_metadata": image_source_metadata,
                            },
                        )
                        if not all(isinstance(path, str) for path in local_paths):
                            record["status"] = "missing_path"
                            record["local_paths"] = local_paths
                            if phase_at_least(3):
                                phase3_fallback_scheduled[req_id] = list(input_ids)
                            append_jsonl(VALIDATE_FILE, record)
                            continue

                        try:
                            processor = MULTIMODAL_REGISTRY.create_processor(
                                self.model_config,
                                cache=None,
                            )
                            image_io = ImageMediaIO(image_mode="RGB")

                            stage_extra = {
                                "request_id": req_id,
                                "tp_rank": tp_rank,
                                "local_image_count": len(local_paths),
                                "total_image_count": len(image_features),
                            }
                            with perf_probe_span(
                                "tp_worker",
                                "tp_worker_local_mm_load_images",
                                extra=stage_extra,
                            ):
                                local_images = [
                                    image_io.load_file(Path(path)) for path in local_paths
                                ]

                            with perf_probe_span(
                                "tp_worker",
                                "tp_worker_local_mm_parse_data",
                                extra=stage_extra,
                            ):
                                mm_data_items = processor.info.parse_mm_data(
                                    {"image": local_images},
                                    validate=False,
                                )

                            with perf_probe_span(
                                "tp_worker",
                                "tp_worker_local_mm_hf_process",
                                extra=stage_extra,
                            ):
                                mm_processed_data = processor._apply_hf_processor_mm_only(
                                    mm_data_items,
                                    hf_processor_mm_kwargs={},
                                    tokenization_kwargs={},
                                )

                            with perf_probe_span(
                                "tp_worker",
                                "tp_worker_local_mm_build_kwargs",
                                extra=stage_extra,
                            ):
                                mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
                                    mm_processed_data,
                                    processor._get_mm_fields_config(mm_processed_data, {}),
                                )

                            local_items = list(mm_kwargs.get("image", []))
                        except Exception as exc:
                            record["status"] = "phase3-stock-fallback-error"
                            record["local_paths"] = local_paths
                            record["error"] = repr(exc)
                            record["traceback"] = traceback.format_exc()
                            if phase_at_least(3):
                                phase3_fallback_scheduled[req_id] = list(input_ids)
                                append_jsonl(VALIDATE_FILE, record)
                                continue
                            raise

                        if phase_at_least(3):
                            with perf_probe_span(
                                "tp_worker",
                                "tp_worker_local_mm_manual_encode",
                                extra=stage_extra,
                            ):
                                _local_outputs, gathered_outputs = (
                                    _phase3_encode_local_items_and_gather(
                                        self,
                                        image_features=image_features,
                                        local_indices=local_indices,
                                        local_items=local_items,
                                        order=order,
                                        counts=counts,
                                    )
                                )
                                for feature, encoder_output in zip(
                                    image_features, gathered_outputs
                                ):
                                    self.encoder_cache[feature.identifier] = (
                                        encoder_output
                                    )
                                    self.maybe_save_ec_to_connector(
                                        self.encoder_cache, feature.identifier
                                    )

                            record["status"] = "phase3-direct-encode-ok"
                            record["local_paths"] = local_paths
                            record["local_items"] = len(local_items)
                            record["rebuilt_items"] = 0
                            append_jsonl(
                                TRACE_FILE,
                                {
                                    "ts": time.time(),
                                    "pid": os.getpid(),
                                    "stage": "worker_phase3_direct_encode",
                                    "request_id": req_id,
                                    "tp_rank": tp_rank,
                                    "local_indices": list(local_indices),
                                    "local_paths": local_paths,
                                    "total_images": len(image_features),
                                },
                            )
                        else:
                            expected = [image_features[idx].data for idx in local_indices]
                            actual = local_items

                            record["local_paths"] = local_paths
                            record["expected_summary"] = [
                                summarize_mm_item(item) for item in expected
                            ]
                            record["actual_summary"] = [
                                summarize_mm_item(item) for item in actual
                            ]
                            record["status"] = (
                                "ok"
                                if (
                                    len(expected) == len(actual)
                                    and all(
                                        tensor_equal(exp, act)
                                        for exp, act in zip(expected, actual)
                                    )
                                )
                                else "mismatch"
                            )

                            if record["status"] == "ok":
                                with perf_probe_span(
                                    "tp_worker",
                                    "tp_worker_local_mm_apply_back",
                                    extra=stage_extra,
                                ):
                                    for local_feature_idx, local_item in zip(
                                        local_indices, actual
                                    ):
                                        image_features[local_feature_idx].data = local_item
                                record["injected_local_items"] = len(actual)
                                append_jsonl(
                                    TRACE_FILE,
                                    {
                                        "ts": time.time(),
                                        "pid": os.getpid(),
                                        "stage": "worker_local_inject",
                                        "request_id": req_id,
                                        "tp_rank": tp_rank,
                                        "local_indices": list(local_indices),
                                        "local_paths": local_paths,
                                    },
                                )

                        append_jsonl(
                            TRACE_FILE,
                            {
                                "ts": time.time(),
                                "pid": os.getpid(),
                                "stage": "worker_local_load",
                                "request_id": req_id,
                                "tp_rank": tp_rank,
                                "local_paths": local_paths,
                            },
                        )
                        append_jsonl(VALIDATE_FILE, record)
        except Exception as exc:
            append_jsonl(
                VALIDATE_FILE,
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )

        if phase_at_least(3):
            if not phase3_fallback_scheduled:
                with perf_probe_span("tp_worker", "tp_worker_mm_encoder_core"):
                    return []

            original_scheduled = scheduler_output.scheduled_encoder_inputs
            scheduler_output.scheduled_encoder_inputs = phase3_fallback_scheduled
            try:
                with perf_probe_span("tp_worker", "tp_worker_mm_encoder_core"):
                    return original_execute(self, scheduler_output)
            finally:
                scheduler_output.scheduled_encoder_inputs = original_scheduled

        with perf_probe_span("tp_worker", "tp_worker_mm_encoder_core"):
            return original_execute(self, scheduler_output)

    GPUModelRunner._batch_mm_inputs_from_scheduler = _phase2_batch_mm_inputs_from_scheduler
    GPUModelRunner._execute_mm_encoder = _phase2_execute_mm_encoder
    _WORKER_PATCHED = True


def apply_phase23_patches(*, enable_phase3: bool) -> None:
    patch_phase2_hf_renderer()
    patch_phase2_openai_chat()
    patch_phase2_input_processor()
    if enable_phase3:
        patch_phase3_qwen_parser()
    patch_phase2_worker_validation()
