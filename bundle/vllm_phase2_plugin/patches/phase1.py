import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse


_PATCHED = False
_TRACE_FILE = Path(
    os.environ.get(
        "VLLM_ASCEND_MM_FILE_MAP",
        "/tmp/vllm_ascend_mm_file_map_phase2.jsonl",
    )
)


class _Phase3LocalImageRef:
    """Lightweight local image reference used by phase3 API planning path."""

    __slots__ = (
        "_phase2_local_path",
        "_phase2_media_mode",
        "_phase2_media_uuid",
        "_phase2_request_scope_key",
        "_phase2_source_key",
        "_phase3_height",
        "_phase3_width",
    )

    def __init__(self, path: str, width: int, height: int) -> None:
        self._phase2_local_path = path
        self._phase2_media_mode = "local_path"
        self._phase2_media_uuid = None
        self._phase2_request_scope_key = None
        self._phase2_source_key = None
        self._phase3_width = int(width)
        self._phase3_height = int(height)


def _phase_at_least(target: int) -> bool:
    raw = os.environ.get("VLLM_ASCEND_API_OPT_PHASE", "").strip()
    if not raw:
        return False
    try:
        return int(raw) >= target
    except ValueError:
        return False


def stable_local_file_uuid(filepath: Path) -> uuid.UUID:
    stat = filepath.stat()
    identity = f"{filepath.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    digest = hashlib.md5(identity.encode("utf-8")).hexdigest()
    return uuid.UUID(digest)


def _stable_source_uuid(prefix: str, identity: str) -> str:
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return f"phase1-{prefix}-{digest}"


def _local_path_from_url(url: str | None) -> Path | None:
    if not url:
        return None

    parsed = urlparse(url)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path)
    elif parsed.scheme == "":
        raw_path = url
    else:
        return None

    path = Path(raw_path)
    if not path.exists():
        return None
    return path


def _local_uuid_str_from_url(url: str | None) -> str | None:
    path = _local_path_from_url(url)
    if path is None:
        return None
    return str(stable_local_file_uuid(path))


def _canonical_http_url(url: str | None) -> str | None:
    if not isinstance(url, str) or not url:
        return None

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None

    return parsed._replace(fragment="").geturl()


def _is_base64_image_url(url: str | None) -> bool:
    if not isinstance(url, str):
        return False

    parsed = urlparse(url)
    if parsed.scheme != "data":
        return False

    lowered = url.lower()
    return lowered.startswith("data:image/") and ";base64," in lowered


def _get_parser_request_scope(parser) -> str:
    request_scope = getattr(parser, "_phase2_request_scope_key", None)
    if not isinstance(request_scope, str):
        request_scope = uuid.uuid4().hex
        setattr(parser, "_phase2_request_scope_key", request_scope)
    return request_scope


def _next_parser_media_index(parser, modality: str) -> int:
    counter_attr = "_phase2_media_indices"
    counters = getattr(parser, counter_attr, None)
    if not isinstance(counters, dict):
        counters = {}
        setattr(parser, counter_attr, counters)

    next_index = int(counters.get(modality, 0))
    counters[modality] = next_index + 1
    return next_index


def _resolve_image_source_info(parser, image_url: str | None, uuid_str: str | None) -> dict:
    user_uuid = str(uuid_str) if uuid_str is not None else None

    local_path = _local_path_from_url(image_url)
    if local_path is not None:
        resolved = str(local_path.resolve())
        stat = local_path.stat()
        source_key = f"local_path:{resolved}|{stat.st_mtime_ns}|{stat.st_size}"
        return {
            "image_url": image_url,
            "local_path": resolved,
            "media_mode": "local_path",
            "media_uuid": user_uuid or str(stable_local_file_uuid(local_path)),
            "request_scope_key": None,
            "source_key": source_key,
        }

    canonical_http_url = _canonical_http_url(image_url)
    if canonical_http_url is not None:
        source_key = f"http:{canonical_http_url}"
        return {
            "image_url": image_url,
            "local_path": None,
            "media_mode": "http",
            "media_uuid": user_uuid or _stable_source_uuid("http", source_key),
            "request_scope_key": None,
            "source_key": source_key,
        }

    if _is_base64_image_url(image_url):
        request_scope = _get_parser_request_scope(parser)
        image_idx = _next_parser_media_index(parser, "image")
        source_key = f"base64:req:{request_scope}:image:{image_idx}"
        return {
            "image_url": image_url,
            "local_path": None,
            "media_mode": "base64",
            "media_uuid": user_uuid or source_key,
            "request_scope_key": f"req:{request_scope}",
            "source_key": source_key,
        }

    raw_value = image_url if isinstance(image_url, str) else None
    source_key = f"raw:{raw_value}" if raw_value else None
    return {
        "image_url": image_url,
        "local_path": None,
        "media_mode": "unknown",
        "media_uuid": user_uuid or (_stable_source_uuid("raw", source_key) if source_key else None),
        "request_scope_key": _get_parser_request_scope(parser),
        "source_key": source_key,
    }


def _resolve_phase3_local_source_info(image_url: str | None) -> dict | None:
    local_path = _local_path_from_url(image_url)
    if local_path is None:
        return None

    resolved = str(local_path.resolve())
    stat = local_path.stat()
    return {
        "image_url": image_url,
        "local_path": resolved,
        "media_mode": "local_path",
        "media_uuid": str(stable_local_file_uuid(local_path)),
        "request_scope_key": None,
        "source_key": f"local_path:{resolved}|{stat.st_mtime_ns}|{stat.st_size}",
    }


def _append_trace(record: dict) -> None:
    if not _phase_at_least(2):
        return
    _TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_TRACE_FILE, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        fp.write("\n")


def _apply_image_source_metadata(media, source_info: dict | None):
    if media is None or not isinstance(source_info, dict):
        return media

    attrs = {
        "_phase2_local_path": source_info.get("local_path"),
        "_phase2_media_mode": source_info.get("media_mode"),
        "_phase2_media_uuid": source_info.get("media_uuid"),
        "_phase2_request_scope_key": source_info.get("request_scope_key"),
        "_phase2_source_key": source_info.get("source_key"),
    }
    for attr, value in attrs.items():
        if value is None:
            continue
        try:
            setattr(media, attr, value)
        except Exception:
            pass
    return media


def _phase3_local_image_from_url(url: str | None, source_info: dict | None = None):
    path = _local_path_from_url(url)
    if path is None:
        return None

    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            width, height = im.size
    except Exception:
        return None

    return _apply_image_source_metadata(
        _Phase3LocalImageRef(str(path.resolve()), width=width, height=height),
        source_info,
    )


def _attach_source_metadata(media, source_info: dict | None):
    return _apply_image_source_metadata(media, source_info)


def _extract_phase2_local_mm_paths(mm_data) -> dict[str, list[str | None]] | None:
    if not isinstance(mm_data, dict):
        return None

    image_items = mm_data.get("image")
    if not isinstance(image_items, list):
        return None

    paths: list[str | None] = []
    has_local = False
    for item in image_items:
        path = getattr(item, "_phase2_local_path", None)
        if isinstance(path, str):
            has_local = True
            paths.append(path)
        else:
            paths.append(None)

    if not has_local:
        return None

    return {"image": paths}


def apply_phase1_patch() -> None:
    global _PATCHED
    if _PATCHED or not _phase_at_least(1):
        return

    from PIL import Image
    from vllm.entrypoints.chat_utils import (
        AsyncMultiModalContentParser,
        MultiModalContentParser,
    )
    from vllm.inputs.data import token_inputs
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.multimodal.media.base import MediaWithBytes
    from vllm.multimodal.media.connector import MediaConnector
    from vllm.multimodal.media.image import ImageMediaIO

    if getattr(ImageMediaIO.load_file, "__name__", "") == "_phase1_load_file":
        _PATCHED = True
        return

    original_load_file = ImageMediaIO.load_file
    original_parse_image = MultiModalContentParser.parse_image
    original_async_image_with_uuid = AsyncMultiModalContentParser._image_with_uuid_async
    original_fetch_image = MediaConnector.fetch_image
    original_fetch_image_async = MediaConnector.fetch_image_async
    original_process_tokens = InputPreprocessor._process_tokens

    def _phase1_load_file(self, filepath: Path):
        media = original_load_file(self, filepath)
        if not isinstance(media, MediaWithBytes):
            return media

        exif = media.media.getexif()
        exif[Image.ExifTags.Base.ImageID] = stable_local_file_uuid(Path(filepath))
        return media

    def _phase1_parse_image(self, image_url: str | None, uuid: str | None = None) -> None:
        source_info = _resolve_image_source_info(self, image_url, uuid)
        uuid = source_info.get("media_uuid")
        if _phase_at_least(3):
            image = (
                _phase3_local_image_from_url(image_url, source_info)
                if image_url
                else None
            )
            if image_url and image is None:
                return original_parse_image(self, image_url, uuid=uuid)

            placeholder = self._tracker.add("image", (image, uuid))
            self._add_placeholder("image", placeholder)
            _append_trace(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "api_parse_image_phase3_ref",
                    "image_url": image_url,
                    "uuid": uuid,
                    "media_mode": source_info.get("media_mode"),
                    "local_path": getattr(image, "_phase2_local_path", None),
                    "request_scope_key": source_info.get("request_scope_key"),
                    "source_key": source_info.get("source_key"),
                    "width": getattr(image, "_phase3_width", None),
                    "height": getattr(image, "_phase3_height", None),
                }
            )
            return None
        return original_parse_image(self, image_url, uuid=uuid)

    async def _phase1_image_with_uuid_async(
        self,
        image_url: str | None,
        uuid: str | None,
    ):
        source_info = _resolve_image_source_info(self, image_url, uuid)
        uuid = source_info.get("media_uuid")
        if _phase_at_least(3):
            image = (
                _phase3_local_image_from_url(image_url, source_info)
                if image_url
                else None
            )
            if image_url and image is None:
                return await original_async_image_with_uuid(self, image_url, uuid)

            _append_trace(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "api_parse_image_async_phase3_ref",
                    "image_url": image_url,
                    "uuid": uuid,
                    "media_mode": source_info.get("media_mode"),
                    "local_path": getattr(image, "_phase2_local_path", None),
                    "request_scope_key": source_info.get("request_scope_key"),
                    "source_key": source_info.get("source_key"),
                    "width": getattr(image, "_phase3_width", None),
                    "height": getattr(image, "_phase3_height", None),
                }
            )
            return image, uuid
        return await original_async_image_with_uuid(self, image_url, uuid)

    def _phase1_fetch_image(self, image_url: str, *, image_mode: str = "RGB"):
        if _phase_at_least(3):
            image = _phase3_local_image_from_url(
                image_url,
                _resolve_phase3_local_source_info(image_url),
            )
            if image is not None:
                _append_trace(
                    {
                        "ts": time.time(),
                        "pid": os.getpid(),
                        "stage": "api_fetch_image_phase3_ref",
                        "image_url": image_url,
                        "media_mode": "local_path",
                        "local_path": image._phase2_local_path,
                        "source_key": getattr(image, "_phase2_source_key", None),
                        "width": image._phase3_width,
                        "height": image._phase3_height,
                    }
                )
                return image

        image = original_fetch_image(self, image_url, image_mode=image_mode)
        image = _attach_source_metadata(
            image,
            _resolve_image_source_info(self, image_url, None),
        )
        path = getattr(image, "_phase2_local_path", None)
        _append_trace(
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "stage": "api_fetch_image",
                "image_url": image_url,
                "media_mode": getattr(image, "_phase2_media_mode", None),
                "local_path": path,
                "request_scope_key": getattr(image, "_phase2_request_scope_key", None),
                "source_key": getattr(image, "_phase2_source_key", None),
            }
        )
        return image

    async def _phase1_fetch_image_async(self, image_url: str, *, image_mode: str = "RGB"):
        if _phase_at_least(3):
            image = _phase3_local_image_from_url(
                image_url,
                _resolve_phase3_local_source_info(image_url),
            )
            if image is not None:
                _append_trace(
                    {
                        "ts": time.time(),
                        "pid": os.getpid(),
                        "stage": "api_fetch_image_async_phase3_ref",
                        "image_url": image_url,
                        "media_mode": "local_path",
                        "local_path": image._phase2_local_path,
                        "source_key": getattr(image, "_phase2_source_key", None),
                        "width": image._phase3_width,
                        "height": image._phase3_height,
                    }
                )
                return image

        image = await original_fetch_image_async(self, image_url, image_mode=image_mode)
        image = _attach_source_metadata(
            image,
            _resolve_image_source_info(self, image_url, None),
        )
        path = getattr(image, "_phase2_local_path", None)
        _append_trace(
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "stage": "api_fetch_image_async",
                "image_url": image_url,
                "media_mode": getattr(image, "_phase2_media_mode", None),
                "local_path": path,
                "request_scope_key": getattr(image, "_phase2_request_scope_key", None),
                "source_key": getattr(image, "_phase2_source_key", None),
            }
        )
        return image

    def _phase1_process_text(self, parsed_content, tokenization_kwargs=None):
        prompt_text = parsed_content["prompt"]

        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=parsed_content.get("multi_modal_uuids"),
            )
            if _phase_at_least(2):
                local_mm_paths = _extract_phase2_local_mm_paths(multi_modal_data)
                _append_trace(
                    {
                        "ts": time.time(),
                        "pid": os.getpid(),
                        "stage": "api_collect_local_paths_text",
                        "local_mm_paths": local_mm_paths,
                    }
                )
                if local_mm_paths:
                    inputs["phase2_local_mm_paths"] = local_mm_paths
        else:
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            inputs = token_inputs(prompt_token_ids)

        inputs["prompt"] = prompt_text

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _phase1_process_tokens(self, parsed_content, tokenization_kwargs=None):
        inputs = original_process_tokens(self, parsed_content, tokenization_kwargs)
        if _phase_at_least(2) and inputs.get("type") == "multimodal":
            local_mm_paths = _extract_phase2_local_mm_paths(
                parsed_content.get("multi_modal_data")
            )
            _append_trace(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "api_collect_local_paths_tokens",
                    "local_mm_paths": local_mm_paths,
                }
            )
            if local_mm_paths:
                inputs["phase2_local_mm_paths"] = local_mm_paths
        return inputs

    ImageMediaIO.load_file = _phase1_load_file
    MultiModalContentParser.parse_image = _phase1_parse_image
    AsyncMultiModalContentParser._image_with_uuid_async = _phase1_image_with_uuid_async
    MediaConnector.fetch_image = _phase1_fetch_image
    MediaConnector.fetch_image_async = _phase1_fetch_image_async
    InputPreprocessor._process_text = _phase1_process_text
    InputPreprocessor._process_tokens = _phase1_process_tokens
    _PATCHED = True
