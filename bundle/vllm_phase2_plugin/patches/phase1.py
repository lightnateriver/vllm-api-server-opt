import hashlib
import json
import os
import time
import uuid
import fcntl
import mimetypes
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


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
        "_phase2_image_url",
        "_phase3_height",
        "_phase3_width",
    )

    def __init__(self, path: str, width: int, height: int) -> None:
        self._phase2_local_path = path
        self._phase2_media_mode = "local_path"
        self._phase2_media_uuid = None
        self._phase2_request_scope_key = None
        self._phase2_source_key = None
        self._phase2_image_url = None
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


def _phase4_enabled() -> bool:
    return _phase_at_least(4)


def stable_local_file_uuid(filepath: Path) -> uuid.UUID:
    stat = filepath.stat()
    identity = f"{filepath.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    digest = hashlib.md5(identity.encode("utf-8")).hexdigest()
    return uuid.UUID(digest)


def _stable_source_uuid(prefix: str, identity: str) -> str:
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return f"phase1-{prefix}-{digest}"


def _phase4_http_cache_dir() -> Path:
    return Path(
        os.environ.get(
            "VLLM_ASCEND_HTTP_CACHE_DIR",
            "/tmp/vllm_ascend_http_cache",
        )
    )


def _phase4_http_cache_ttl_s() -> int:
    raw = os.environ.get("VLLM_ASCEND_HTTP_CACHE_TTL_S", "86400").strip() or "86400"
    try:
        return int(raw)
    except ValueError:
        return 86400


def _phase4_http_timeout_s() -> float:
    raw = os.environ.get("VLLM_ASCEND_HTTP_TIMEOUT_S", "30").strip() or "30"
    try:
        return float(raw)
    except ValueError:
        return 30.0


def _phase4_http_max_file_bytes() -> int:
    raw = os.environ.get(
        "VLLM_ASCEND_HTTP_MAX_FILE_BYTES",
        str(64 * 1024 * 1024),
    ).strip() or str(64 * 1024 * 1024)
    try:
        return int(raw)
    except ValueError:
        return 64 * 1024 * 1024


def _phase4_http_cache_max_bytes() -> int:
    raw = os.environ.get("VLLM_ASCEND_HTTP_CACHE_MAX_GB", "0").strip() or "0"
    try:
        max_gb = float(raw)
    except ValueError:
        return 0
    if max_gb <= 0:
        return 0
    return int(max_gb * 1024 * 1024 * 1024)


def _phase4_http_chunk_bytes() -> int:
    raw = os.environ.get(
        "VLLM_ASCEND_HTTP_CHUNK_BYTES",
        str(1024 * 1024),
    ).strip() or str(1024 * 1024)
    try:
        return max(4096, int(raw))
    except ValueError:
        return 1024 * 1024


def _phase4_http_user_agent() -> str:
    return os.environ.get(
        "VLLM_ASCEND_HTTP_USER_AGENT",
        "vllm-ascend-phase4-http-cache/1.0",
    )


def _phase4_http_cache_key(canonical_http_url: str) -> str:
    return hashlib.sha1(canonical_http_url.encode("utf-8")).hexdigest()


def _phase4_http_meta_path(cache_key: str) -> Path:
    return _phase4_http_cache_dir() / f"{cache_key}.json"


def _phase4_http_lock_path(cache_key: str) -> Path:
    return _phase4_http_cache_dir() / f"{cache_key}.lock"


def _phase4_http_gc_lock_path() -> Path:
    return _phase4_http_cache_dir() / ".phase4_gc.lock"


def _phase4_guess_http_suffix(
    canonical_http_url: str,
    content_type: str | None = None,
) -> str:
    path_suffix = Path(urlparse(canonical_http_url).path).suffix.lower()
    if path_suffix:
        return path_suffix

    if content_type:
        guessed = mimetypes.guess_extension(content_type.partition(";")[0].strip())
        if guessed:
            return guessed

    return ".img"


def _phase4_load_http_meta(meta_path: Path) -> dict | None:
    try:
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _phase4_http_meta_is_fresh(meta: dict, *, ttl_s: int) -> bool:
    local_path = meta.get("local_path")
    downloaded_at = meta.get("downloaded_at")
    if not isinstance(local_path, str):
        return False
    if not Path(local_path).exists():
        return False
    if ttl_s < 0:
        return True
    if not isinstance(downloaded_at, (int, float)):
        return False
    return (time.time() - float(downloaded_at)) <= ttl_s


def _phase4_try_lock_file(lock_path: Path, *, blocking: bool):
    lock_fp = open(lock_path, "a+", encoding="utf-8")
    flags = fcntl.LOCK_EX
    if not blocking:
        flags |= fcntl.LOCK_NB
    try:
        fcntl.flock(lock_fp.fileno(), flags)
        return lock_fp
    except OSError:
        lock_fp.close()
        return None


def _phase4_unlock_file(lock_fp) -> None:
    if lock_fp is None:
        return
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock_fp.close()
    except Exception:
        pass


def _phase4_cache_file_candidates(
    cache_key: str,
    *,
    explicit_local_path: Path | None = None,
) -> list[Path]:
    cache_dir = _phase4_http_cache_dir()
    lock_path = _phase4_http_lock_path(cache_key).resolve()
    candidates: list[Path] = []

    if explicit_local_path is not None:
        candidates.append(explicit_local_path.resolve())

    for path in cache_dir.glob(f"{cache_key}*"):
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved == lock_path:
            continue
        candidates.append(resolved)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _phase4_remove_cache_files(paths: list[Path]) -> list[str]:
    removed: list[str] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
            removed.append(str(path))
        except Exception:
            continue
    return removed


def _phase4_invalidate_http_cache(
    canonical_http_url: str | None,
    *,
    image_url: str | None,
    local_path: str | None,
    reason: str,
    error: str | None = None,
) -> dict:
    if not isinstance(canonical_http_url, str) or not canonical_http_url:
        return {"removed_files": [], "cache_key": None}

    cache_key = _phase4_http_cache_key(canonical_http_url)
    explicit_path = Path(local_path) if isinstance(local_path, str) and local_path else None
    meta_path = _phase4_http_meta_path(cache_key)
    removed_files: list[str] = []

    lock_fp = _phase4_try_lock_file(_phase4_http_lock_path(cache_key), blocking=True)
    try:
        if lock_fp is not None:
            removed_files = _phase4_remove_cache_files(
                _phase4_cache_file_candidates(
                    cache_key,
                    explicit_local_path=explicit_path,
                )
            )
    finally:
        _phase4_unlock_file(lock_fp)

    _append_trace(
        {
            "ts": time.time(),
            "pid": os.getpid(),
            "stage": "phase4_http_cache_invalid",
            "image_url": image_url,
            "canonical_http_url": canonical_http_url,
            "cache_key": cache_key,
            "local_path": local_path,
            "meta_path": str(meta_path.resolve()),
            "source_key": f"http:{canonical_http_url}",
            "invalid_reason": reason,
            "error": error,
            "removed_files": removed_files,
        }
    )
    return {"removed_files": removed_files, "cache_key": cache_key}


def _phase4_collect_cache_entries(cache_dir: Path, *, ttl_s: int) -> list[dict]:
    entries: list[dict] = []
    for meta_path in cache_dir.glob("*.json"):
        cache_key = meta_path.stem
        meta = _phase4_load_http_meta(meta_path)
        local_path = meta.get("local_path") if isinstance(meta, dict) else None
        local_path_obj = Path(local_path) if isinstance(local_path, str) else None
        downloaded_at = meta.get("downloaded_at") if isinstance(meta, dict) else None
        if isinstance(downloaded_at, (int, float)):
            downloaded_at_value = float(downloaded_at)
        else:
            downloaded_at_value = None
        exists = bool(local_path_obj and local_path_obj.exists())
        fresh = bool(isinstance(meta, dict) and _phase4_http_meta_is_fresh(meta, ttl_s=ttl_s))
        size_bytes = 0
        if exists and local_path_obj is not None:
            try:
                size_bytes = int(local_path_obj.stat().st_size)
            except OSError:
                size_bytes = 0
        entries.append(
            {
                "cache_key": cache_key,
                "meta_path": meta_path,
                "meta": meta,
                "local_path": local_path_obj,
                "canonical_http_url": (
                    meta.get("canonical_http_url") if isinstance(meta, dict) else None
                ),
                "downloaded_at": downloaded_at_value,
                "exists": exists,
                "fresh": fresh,
                "size_bytes": size_bytes,
            }
        )
    return entries


def _phase4_evict_http_cache_entry(
    entry: dict,
    *,
    reason: str,
    image_url: str | None = None,
) -> dict:
    cache_key = entry.get("cache_key")
    if not isinstance(cache_key, str) or not cache_key:
        return {"removed": False, "removed_files": [], "busy": False}

    lock_fp = _phase4_try_lock_file(_phase4_http_lock_path(cache_key), blocking=False)
    if lock_fp is None:
        return {"removed": False, "removed_files": [], "busy": True}

    try:
        explicit_local_path = entry.get("local_path")
        if not isinstance(explicit_local_path, Path):
            explicit_local_path = None
        removed_files = _phase4_remove_cache_files(
            _phase4_cache_file_candidates(
                cache_key,
                explicit_local_path=explicit_local_path,
            )
        )
    finally:
        _phase4_unlock_file(lock_fp)

    canonical_http_url = entry.get("canonical_http_url")
    _append_trace(
        {
            "ts": time.time(),
            "pid": os.getpid(),
            "stage": "phase4_http_cache_evict",
            "image_url": image_url,
            "canonical_http_url": canonical_http_url,
            "cache_key": cache_key,
            "source_key": (
                f"http:{canonical_http_url}"
                if isinstance(canonical_http_url, str) and canonical_http_url
                else None
            ),
            "evict_reason": reason,
            "removed_files": removed_files,
        }
    )
    return {"removed": bool(removed_files), "removed_files": removed_files, "busy": False}


def _phase4_prune_http_cache(
    *,
    cache_dir: Path,
    image_url: str | None,
    current_cache_key: str | None,
) -> None:
    ttl_s = _phase4_http_cache_ttl_s()
    max_bytes = _phase4_http_cache_max_bytes()
    if ttl_s < 0 and max_bytes <= 0:
        return

    gc_lock_fp = _phase4_try_lock_file(_phase4_http_gc_lock_path(), blocking=False)
    if gc_lock_fp is None:
        return

    try:
        entries = _phase4_collect_cache_entries(cache_dir, ttl_s=ttl_s)
        initial_total_bytes = sum(
            entry["size_bytes"]
            for entry in entries
            if entry.get("fresh") and isinstance(entry.get("size_bytes"), int)
        )
        expired_removed = 0
        capacity_removed = 0
        skipped_busy = 0

        live_entries: list[dict] = []
        for entry in entries:
            if entry.get("fresh"):
                live_entries.append(entry)
                continue
            result = _phase4_evict_http_cache_entry(
                entry,
                reason="expired_or_missing",
                image_url=image_url,
            )
            if result.get("busy"):
                skipped_busy += 1
            elif result.get("removed"):
                expired_removed += 1

        if max_bytes > 0:
            live_entries = [
                entry
                for entry in live_entries
                if entry.get("exists")
                and isinstance(entry.get("size_bytes"), int)
            ]
            total_live_bytes = sum(int(entry["size_bytes"]) for entry in live_entries)
            eviction_candidates = sorted(
                (
                    entry
                    for entry in live_entries
                    if entry.get("cache_key") != current_cache_key
                ),
                key=lambda item: (
                    item.get("downloaded_at")
                    if isinstance(item.get("downloaded_at"), float)
                    else 0.0,
                    str(item.get("meta_path")),
                ),
            )
            for entry in eviction_candidates:
                if total_live_bytes <= max_bytes:
                    break
                result = _phase4_evict_http_cache_entry(
                    entry,
                    reason="capacity_limit",
                    image_url=image_url,
                )
                if result.get("busy"):
                    skipped_busy += 1
                    continue
                if result.get("removed"):
                    total_live_bytes -= int(entry.get("size_bytes") or 0)
                    capacity_removed += 1

        remaining_entries = _phase4_collect_cache_entries(cache_dir, ttl_s=ttl_s)
        remaining_total_bytes = sum(
            entry["size_bytes"]
            for entry in remaining_entries
            if entry.get("fresh") and isinstance(entry.get("size_bytes"), int)
        )
        _append_trace(
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "stage": "phase4_http_cache_gc",
                "image_url": image_url,
                "current_cache_key": current_cache_key,
                "cache_dir": str(cache_dir.resolve()),
                "ttl_s": ttl_s,
                "max_bytes": max_bytes,
                "initial_total_bytes": initial_total_bytes,
                "remaining_total_bytes": remaining_total_bytes,
                "expired_removed": expired_removed,
                "capacity_removed": capacity_removed,
                "skipped_busy": skipped_busy,
            }
        )
    finally:
        _phase4_unlock_file(gc_lock_fp)


def _phase4_materialize_http_url(image_url: str | None) -> tuple[str | None, dict | None]:
    canonical_http_url = _canonical_http_url(image_url)
    if canonical_http_url is None:
        return None, None

    cache_dir = _phase4_http_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _phase4_http_cache_key(canonical_http_url)
    meta_path = _phase4_http_meta_path(cache_key)
    lock_path = _phase4_http_lock_path(cache_key)
    ttl_s = _phase4_http_cache_ttl_s()

    lock_fp = _phase4_try_lock_file(lock_path, blocking=True)
    try:
        if lock_fp is None:
            _append_trace(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "phase4_http_cache_error",
                    "image_url": image_url,
                    "canonical_http_url": canonical_http_url,
                    "source_key": f"http:{canonical_http_url}",
                    "error": "lock_acquire_failed",
                }
            )
            return None, None

        cached_meta = _phase4_load_http_meta(meta_path)
        if cached_meta and _phase4_http_meta_is_fresh(cached_meta, ttl_s=ttl_s):
            local_path = str(Path(cached_meta["local_path"]).resolve())
            _append_trace(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "phase4_http_cache_hit",
                    "image_url": image_url,
                    "canonical_http_url": canonical_http_url,
                    "local_path": local_path,
                    "source_key": f"http:{canonical_http_url}",
                }
            )
            return local_path, cached_meta

        timeout_s = _phase4_http_timeout_s()
        max_file_bytes = _phase4_http_max_file_bytes()
        chunk_bytes = _phase4_http_chunk_bytes()

        request_obj = Request(
            canonical_http_url,
            headers={
                "User-Agent": _phase4_http_user_agent(),
                "Accept": "image/*,*/*;q=0.8",
            },
        )

        tmp_path: Path | None = None
        try:
            with urlopen(request_obj, timeout=timeout_s) as resp:
                content_type = resp.headers.get("Content-Type")
                etag = resp.headers.get("ETag")
                last_modified = resp.headers.get("Last-Modified")
                content_length_header = resp.headers.get("Content-Length")
                suffix = _phase4_guess_http_suffix(
                    canonical_http_url,
                    content_type=content_type,
                )
                with tempfile.NamedTemporaryFile(
                    dir=cache_dir,
                    prefix=f"{cache_key}.",
                    suffix=".download",
                    delete=False,
                ) as tmp_fp:
                    total_bytes = 0
                    while True:
                        chunk = resp.read(chunk_bytes)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > max_file_bytes:
                            raise ValueError(
                                f"phase4 http image exceeds max bytes: "
                                f"{total_bytes} > {max_file_bytes}"
                            )
                        tmp_fp.write(chunk)
                    tmp_path = Path(tmp_fp.name)

            final_path = cache_dir / f"{cache_key}{suffix}"
            os.replace(tmp_path, final_path)
            tmp_path = None

            meta = {
                "canonical_http_url": canonical_http_url,
                "local_path": str(final_path.resolve()),
                "downloaded_at": time.time(),
                "content_type": content_type,
                "content_length": (
                    int(content_length_header)
                    if isinstance(content_length_header, str)
                    and content_length_header.isdigit()
                    else None
                ),
                "etag": etag,
                "last_modified": last_modified,
            }
            with open(meta_path, "w", encoding="utf-8") as meta_fp:
                json.dump(meta, meta_fp, ensure_ascii=True, sort_keys=True)

            _append_trace(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "phase4_http_cache_fill",
                    "image_url": image_url,
                    "canonical_http_url": canonical_http_url,
                    "local_path": meta["local_path"],
                    "content_type": content_type,
                    "content_length": meta["content_length"],
                    "source_key": f"http:{canonical_http_url}",
                }
            )
            _phase4_prune_http_cache(
                cache_dir=cache_dir,
                image_url=image_url,
                current_cache_key=cache_key,
            )
            return str(final_path.resolve()), meta
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            _append_trace(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": "phase4_http_cache_error",
                    "image_url": image_url,
                    "canonical_http_url": canonical_http_url,
                    "source_key": f"http:{canonical_http_url}",
                    "error": repr(exc),
                }
            )
            return None, None
    finally:
        _phase4_unlock_file(lock_fp)


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


def _phase3_local_image_from_path(path: Path, source_info: dict | None = None):
    width, height, _ = _phase3_probe_local_image(path)
    if width is None or height is None:
        return None

    return _apply_image_source_metadata(
        _Phase3LocalImageRef(str(path.resolve()), width=width, height=height),
        source_info,
    )


def _phase3_probe_local_image(path: Path) -> tuple[int | None, int | None, str | None]:
    if not path.exists():
        return None, None, "file_missing"

    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            width, height = im.size
        return int(width), int(height), None
    except Exception as exc:
        return None, None, repr(exc)


def _resolve_phase4_http_source_info(
    image_url: str | None,
    source_info: dict | None = None,
) -> dict | None:
    local_path, cache_meta = _phase4_materialize_http_url(image_url)
    if local_path is None:
        return None

    base = dict(source_info or {})
    canonical_http_url = _canonical_http_url(image_url)
    source_key = base.get("source_key") or (
        f"http:{canonical_http_url}" if canonical_http_url else None
    )
    return {
        "image_url": image_url,
        "local_path": local_path,
        "media_mode": "http",
        "media_uuid": base.get("media_uuid")
        or (
            _stable_source_uuid("http", source_key)
            if isinstance(source_key, str)
            else None
        ),
        "request_scope_key": None,
        "source_key": source_key,
        "canonical_http_url": canonical_http_url,
        "cache_key": (
            _phase4_http_cache_key(canonical_http_url)
            if isinstance(canonical_http_url, str)
            else None
        ),
        "content_type": cache_meta.get("content_type") if isinstance(cache_meta, dict) else None,
        "etag": cache_meta.get("etag") if isinstance(cache_meta, dict) else None,
        "last_modified": cache_meta.get("last_modified") if isinstance(cache_meta, dict) else None,
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
        "_phase2_image_url": source_info.get("image_url"),
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

    return _phase3_local_image_from_path(path, source_info)


def _phase4_http_image_from_url(url: str | None, source_info: dict | None = None):
    if not _phase4_enabled():
        return None

    http_source_info = _resolve_phase4_http_source_info(url, source_info)
    if http_source_info is None:
        canonical_http_url = _canonical_http_url(url)
        _append_trace(
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "stage": "phase4_http_fallback_stock",
                "image_url": url,
                "canonical_http_url": canonical_http_url,
                "source_key": (
                    f"http:{canonical_http_url}"
                    if isinstance(canonical_http_url, str) and canonical_http_url
                    else None
                ),
                "fallback_reason": "materialize_failed",
            }
        )
        return None

    local_path = http_source_info.get("local_path")
    if not isinstance(local_path, str):
        canonical_http_url = http_source_info.get("canonical_http_url")
        _append_trace(
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "stage": "phase4_http_fallback_stock",
                "image_url": url,
                "canonical_http_url": canonical_http_url,
                "source_key": http_source_info.get("source_key"),
                "fallback_reason": "materialized_local_path_missing",
            }
        )
        return None

    local_path_obj = Path(local_path)
    width, height, probe_error = _phase3_probe_local_image(local_path_obj)
    if width is None or height is None:
        invalidation = _phase4_invalidate_http_cache(
            http_source_info.get("canonical_http_url"),
            image_url=url,
            local_path=local_path,
            reason="local_ref_invalid",
            error=probe_error,
        )
        _append_trace(
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "stage": "phase4_http_fallback_stock",
                "image_url": url,
                "canonical_http_url": http_source_info.get("canonical_http_url"),
                "cache_key": invalidation.get("cache_key"),
                "local_path": local_path,
                "source_key": http_source_info.get("source_key"),
                "fallback_reason": "cache_invalid_local_ref",
                "error": probe_error,
                "removed_files": invalidation.get("removed_files"),
            }
        )
        return None

    return _apply_image_source_metadata(
        _Phase3LocalImageRef(str(local_path_obj.resolve()), width=width, height=height),
        http_source_info,
    )


def _phase_direct_image_from_url(url: str | None, source_info: dict | None = None):
    image = _phase3_local_image_from_url(url, source_info)
    if image is not None:
        return image

    if _phase4_enabled():
        return _phase4_http_image_from_url(url, source_info)

    return None


def _phase_direct_trace_stage(base: str, media_mode: str | None) -> str:
    if media_mode == "http" and _phase4_enabled():
        return f"{base}_phase4_http_ref"
    return f"{base}_phase3_ref"


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
            image = _phase_direct_image_from_url(image_url, source_info) if image_url else None
            if image_url and image is None:
                return original_parse_image(self, image_url, uuid=uuid)

            placeholder = self._tracker.add("image", (image, uuid))
            self._add_placeholder("image", placeholder)
            _append_trace(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": _phase_direct_trace_stage(
                        "api_parse_image",
                        getattr(image, "_phase2_media_mode", None),
                    ),
                    "image_url": image_url,
                    "uuid": uuid,
                    "media_mode": getattr(image, "_phase2_media_mode", None),
                    "local_path": getattr(image, "_phase2_local_path", None),
                    "request_scope_key": getattr(image, "_phase2_request_scope_key", None),
                    "source_key": getattr(image, "_phase2_source_key", None),
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
            image = _phase_direct_image_from_url(image_url, source_info) if image_url else None
            if image_url and image is None:
                return await original_async_image_with_uuid(self, image_url, uuid)

            _append_trace(
                {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "stage": _phase_direct_trace_stage(
                        "api_parse_image_async",
                        getattr(image, "_phase2_media_mode", None),
                    ),
                    "image_url": image_url,
                    "uuid": uuid,
                    "media_mode": getattr(image, "_phase2_media_mode", None),
                    "local_path": getattr(image, "_phase2_local_path", None),
                    "request_scope_key": getattr(image, "_phase2_request_scope_key", None),
                    "source_key": getattr(image, "_phase2_source_key", None),
                    "width": getattr(image, "_phase3_width", None),
                    "height": getattr(image, "_phase3_height", None),
                }
            )
            return image, uuid
        return await original_async_image_with_uuid(self, image_url, uuid)

    def _phase1_fetch_image(self, image_url: str, *, image_mode: str = "RGB"):
        if _phase_at_least(3):
            image = _phase_direct_image_from_url(
                image_url,
                _resolve_image_source_info(self, image_url, None),
            )
            if image is not None:
                _append_trace(
                    {
                        "ts": time.time(),
                        "pid": os.getpid(),
                        "stage": _phase_direct_trace_stage(
                            "api_fetch_image",
                            getattr(image, "_phase2_media_mode", None),
                        ),
                        "image_url": image_url,
                        "media_mode": getattr(image, "_phase2_media_mode", None),
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
            image = _phase_direct_image_from_url(
                image_url,
                _resolve_image_source_info(self, image_url, None),
            )
            if image is not None:
                _append_trace(
                    {
                        "ts": time.time(),
                        "pid": os.getpid(),
                        "stage": _phase_direct_trace_stage(
                            "api_fetch_image_async",
                            getattr(image, "_phase2_media_mode", None),
                        ),
                        "image_url": image_url,
                        "media_mode": getattr(image, "_phase2_media_mode", None),
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
