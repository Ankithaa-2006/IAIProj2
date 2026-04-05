from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import logging
import os
import atexit
import json
from pathlib import Path
import random
import threading
import time
from collections import deque

from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download

ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.txt",
    "*.md",
    "*.py",
    "*.jinja",
    "*.safetensors",
]

REQUIRED_FILE_NAMES = [
    "config.json",
    "tokenizer.model",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def _matches_allowed(filename: str) -> bool:
    return any(fnmatch.fnmatch(filename, pattern) for pattern in ALLOW_PATTERNS)


def _directory_size_bytes(directory: Path) -> int:
    if not directory.exists():
        return 0
    total = 0
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _combined_size_bytes(directories: list[Path]) -> int:
    return sum(_directory_size_bytes(directory) for directory in directories)


def _list_target_files(model_id: str, token: str | None) -> list[tuple[str, int | None]]:
    try:
        api = HfApi()
        info = api.model_info(model_id, token=token)
        targets: list[tuple[str, int | None]] = []
        for sibling in info.siblings:
            if _matches_allowed(sibling.rfilename):
                size = getattr(sibling, "size", None)
                targets.append((sibling.rfilename, size if isinstance(size, int) else None))

        def sort_key(item: tuple[str, int | None]) -> tuple[int, int]:
            name, size = item
            priority = 0 if name.endswith(".safetensors") else 1
            return (priority, -(size or 0))

        targets.sort(key=sort_key)
        return targets
    except Exception:
        return []


def _expected_download_size_bytes(target_files: list[tuple[str, int | None]]) -> int | None:
    total = 0
    for _, size in target_files:
        if isinstance(size, int):
            total += size
    return total if total > 0 else None


def _expected_size_from_safetensors_index(model_id: str, cache_dir: Path, token: str | None) -> int | None:
    try:
        index_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            cache_dir=str(cache_dir),
            token=token,
            local_files_only=True,
        )
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        total_size = metadata.get("total_size")
        return int(total_size) if isinstance(total_size, int) and total_size > 0 else None
    except Exception:
        return None


class DownloadProgressLogger:
    def __init__(
        self,
        logger: logging.Logger,
        watch_dirs: list[Path],
        expected_bytes: int | None,
        interval_seconds: float = 15.0,
    ) -> None:
        self.logger = logger
        self.watch_dirs = watch_dirs
        self.expected_bytes = expected_bytes
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_size = 0
        self._last_time = time.time()
        self._recent_nonzero_speeds: list[float] = []
        self._zero_streak = 0
        self._samples: deque[tuple[float, int]] = deque(maxlen=240)

    def start(self) -> None:
        self._last_size = _combined_size_bytes(self.watch_dirs)
        self._last_time = time.time()
        self._samples.append((self._last_time, self._last_size))
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            now = time.time()
            current_size = _combined_size_bytes(self.watch_dirs)
            elapsed = max(now - self._last_time, 1e-6)
            delta = max(current_size - self._last_size, 0)
            speed_mb_s = (delta / (1024 * 1024)) / elapsed
            self._samples.append((now, current_size))
            rolling_60 = self._rolling_speed(window_seconds=60.0)
            rolling_180 = self._rolling_speed(window_seconds=180.0)

            eta_text = "n/a"
            progress_text = "n/a"
            if self.expected_bytes and self.expected_bytes > 0:
                ratio = min(max(current_size / self.expected_bytes, 0.0), 1.0)
                progress_text = f"{ratio * 100:.2f}%"
                if speed_mb_s > 0:
                    remaining_mb = max(self.expected_bytes - current_size, 0) / (1024 * 1024)
                    eta_minutes = remaining_mb / speed_mb_s / 60
                    eta_text = f"{eta_minutes:.1f} min"

            self.logger.info(
                "progress=%s downloaded=%.3f GB speed=%.2f MB/s speed60=%.2f MB/s speed180=%.2f MB/s eta=%s",
                progress_text,
                current_size / (1024 * 1024 * 1024),
                speed_mb_s,
                rolling_60,
                rolling_180,
                eta_text,
            )

            self._log_dip_if_needed(speed_mb_s, delta, current_size)

            self._last_size = current_size
            self._last_time = now

    def _rolling_speed(self, window_seconds: float) -> float:
        if len(self._samples) < 2:
            return 0.0

        now, current_size = self._samples[-1]
        start_time = now - window_seconds

        first_time = self._samples[0][0]
        first_size = self._samples[0][1]
        for timestamp, size in self._samples:
            if timestamp >= start_time:
                first_time = timestamp
                first_size = size
                break

        elapsed = max(now - first_time, 1e-6)
        delta = max(current_size - first_size, 0)
        return (delta / (1024 * 1024)) / elapsed

    def _log_dip_if_needed(self, speed_mb_s: float, delta_bytes: int, current_size: int) -> None:
        if speed_mb_s <= 0.01 and delta_bytes == 0:
            self._zero_streak += 1
            if self._zero_streak >= 2:
                self.logger.warning(
                    "dip_detected type=stall streak=%d reason=%s downloaded=%.3f GB",
                    self._zero_streak,
                    "No new bytes this interval. Likely shard switch, retry/backoff after timeout, or transient network stall.",
                    current_size / (1024 * 1024 * 1024),
                )
            return

        if speed_mb_s > 0.05:
            self._recent_nonzero_speeds.append(speed_mb_s)
            if len(self._recent_nonzero_speeds) > 12:
                self._recent_nonzero_speeds = self._recent_nonzero_speeds[-12:]

        self._zero_streak = 0

        if len(self._recent_nonzero_speeds) < 4:
            return

        baseline = sum(self._recent_nonzero_speeds[-4:]) / 4
        if baseline < 0.5:
            return

        drop_ratio = speed_mb_s / baseline if baseline > 0 else 1.0
        if speed_mb_s > 0.01 and drop_ratio < 0.35:
            self.logger.warning(
                "dip_detected type=throughput_drop speed=%.2f MB/s baseline=%.2f MB/s drop_ratio=%.2f reason=%s downloaded=%.3f GB",
                speed_mb_s,
                baseline,
                drop_ratio,
                "Speed dropped sharply versus recent baseline. Common causes: congestion, CDN route switch, worker contention, or temporary packet loss.",
                current_size / (1024 * 1024 * 1024),
            )


def _download_one_file(
    logger: logging.Logger,
    model_id: str,
    filename: str,
    cache_dir: Path,
    token: str | None,
    max_attempts: int,
) -> None:
    etag_timeout = float(os.getenv("HF_HUB_ETAG_TIMEOUT", "30"))
    for attempt in range(1, max_attempts + 1):
        try:
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir=str(cache_dir),
                token=token,
                etag_timeout=etag_timeout,
            )
            logger.info("file_complete filename=%s attempt=%d", filename, attempt)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == max_attempts:
                logger.error("file_failed filename=%s error=%s", filename, exc.__class__.__name__)
                raise
            sleep_seconds = min(20.0, 2.0 * attempt) + random.uniform(0.0, 1.0)
            logger.warning(
                "file_retry filename=%s attempt=%d/%d reason=%s backoff=%.1fs",
                filename,
                attempt,
                max_attempts,
                exc.__class__.__name__,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)


def _download_files_parallel(
    logger: logging.Logger,
    model_id: str,
    target_files: list[str],
    cache_dir: Path,
    token: str | None,
    max_workers: int,
    max_attempts: int,
) -> None:
    worker_count = max(1, min(max_workers, len(target_files)))
    logger.info("parallel_download_start files=%d workers=%d", len(target_files), worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _download_one_file,
                logger,
                model_id,
                filename,
                cache_dir,
                token,
                max_attempts,
            ): filename
            for filename in target_files
        }

        for future in as_completed(futures):
            filename = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("parallel_download_error filename=%s error=%s", filename, exc.__class__.__name__)
                raise


def _required_files_ready(local_dir: Path) -> bool:
    return all((local_dir / name).exists() for name in REQUIRED_FILE_NAMES)


def _required_files_ready_in_cache(model_id: str, cache_dir: Path, token: str | None) -> bool:
    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            token=token,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.md"],
            local_files_only=True,
        )
        snapshot_dir = Path(snapshot_path)
        if not all((snapshot_dir / name).exists() for name in REQUIRED_FILE_NAMES):
            return False
        return not any(snapshot_dir.glob("*.incomplete"))
    except Exception:
        return False


def _build_staged_file_lists(target_files: list[str]) -> tuple[list[str], list[str]]:
    metadata_files = [name for name in target_files if not name.endswith(".safetensors")]
    safetensor_files = [name for name in target_files if name.endswith(".safetensors")]
    return metadata_files, safetensor_files


def _acquire_single_instance_lock(lock_path: Path) -> bool:
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False

    def _read_lock_pid() -> int | None:
        try:
            payload = lock_path.read_text(encoding="utf-8").strip()
            return int(payload) if payload else None
        except Exception:
            return None

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
            lock_file.write(str(os.getpid()))
        return True
    except FileExistsError:
        existing_pid = _read_lock_pid()
        if existing_pid is not None and not _pid_is_alive(existing_pid):
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                return False
            return _acquire_single_instance_lock(lock_path)
        return False


def _release_single_instance_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        return


def _build_logger(log_file_path: Path) -> logging.Logger:
    logger = logging.getLogger("translategemma_download")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / ".hf-cache"
    model_repo_dir = cache_dir / "models--google--translategemma-4b-it"
    model_blob_dir = model_repo_dir / "blobs"
    model_snapshot_dir = model_repo_dir / "snapshots"
    lock_path = cache_dir / ".locks" / "translategemma-download.lock"
    log_dir = project_root / "backend" / "logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_repo_dir.mkdir(parents=True, exist_ok=True)
    model_blob_dir.mkdir(parents=True, exist_ok=True)
    model_snapshot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = _build_logger(log_dir / "translategemma_download.log")

    if not _acquire_single_instance_lock(lock_path):
        logger.error("another_downloader_running lock=%s", lock_path)
        raise RuntimeError("Another translategemma downloader instance is already running")

    atexit.register(_release_single_instance_lock, lock_path)

    backend_mode = os.getenv("HF_DOWNLOAD_BACKEND", "transfer").strip().lower()
    if backend_mode == "xet":
        os.environ["HF_HUB_DISABLE_XET"] = "0"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    else:
        backend_mode = "transfer"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))

    token = os.getenv("HF_TOKEN") or get_token()
    model_id = "google/translategemma-4b-it"
    max_workers = int(os.getenv("HF_DOWNLOAD_WORKERS", "6"))
    safetensor_workers = int(os.getenv("HF_DOWNLOAD_SAFETENSOR_WORKERS", "1"))
    max_attempts = int(os.getenv("HF_DOWNLOAD_MAX_ATTEMPTS", "6"))

    target_file_info = _list_target_files(model_id, token)
    target_files = [name for (name, _) in target_file_info]
    expected_bytes = _expected_download_size_bytes(target_file_info)
    if not expected_bytes:
        expected_bytes = _expected_size_from_safetensors_index(model_id=model_id, cache_dir=cache_dir, token=token)

    logger.info("download_start model=%s", model_id)
    logger.info("backend_mode=%s", backend_mode)
    logger.info("hf_transfer=%s", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1")
    logger.info("token_present=%s", bool(token))
    logger.info("cache_repo_dir=%s", model_repo_dir)
    logger.info("blob_watch_dir=%s", model_blob_dir)
    logger.info("snapshot_watch_dir=%s", model_snapshot_dir)
    logger.info("download_workers=%d", max_workers)
    logger.info("safetensor_workers=%d", safetensor_workers)
    logger.info("download_max_attempts=%d", max_attempts)
    logger.info("target_file_count=%d", len(target_files))
    if expected_bytes:
        logger.info("expected_size_gb=%.3f", expected_bytes / (1024 * 1024 * 1024))
    else:
        logger.info("expected_size_gb=unknown")

    print("Downloading model:", model_id)
    print("Backend mode:", backend_mode)
    print("Using HF Transfer:", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1")
    print("Using token:", bool(token))

    monitor = DownloadProgressLogger(
        logger=logger,
        watch_dirs=[model_blob_dir, model_snapshot_dir],
        expected_bytes=expected_bytes,
        interval_seconds=15.0,
    )
    monitor.start()

    start_time = time.time()

    try:
        metadata_files, safetensor_files = _build_staged_file_lists(target_files)

        if metadata_files:
            logger.info("stage_start stage=metadata files=%d workers=%d", len(metadata_files), max_workers)
            _download_files_parallel(
                logger=logger,
                model_id=model_id,
                target_files=metadata_files,
                cache_dir=cache_dir,
                token=token,
                max_workers=max_workers,
                max_attempts=max_attempts,
            )

        if safetensor_files:
            logger.info(
                "stage_start stage=safetensors files=%d workers=%d",
                len(safetensor_files),
                safetensor_workers,
            )
            _download_files_parallel(
                logger=logger,
                model_id=model_id,
                target_files=safetensor_files,
                cache_dir=cache_dir,
                token=token,
                max_workers=safetensor_workers,
                max_attempts=max_attempts,
            )
        local_path = str(model_repo_dir)
    except Exception as exc:
        logger.exception("download_failed error=%s", exc.__class__.__name__)
        monitor.stop()
        raise
    finally:
        monitor.stop()
        _release_single_instance_lock(lock_path)

    elapsed_minutes = (time.time() - start_time) / 60
    final_size = _directory_size_bytes(model_repo_dir) / (1024 * 1024 * 1024)
    ready = _required_files_ready_in_cache(model_id=model_id, cache_dir=cache_dir, token=token)
    logger.info("required_files_ready=%s", ready)
    logger.info("download_complete elapsed_min=%.2f size_gb=%.3f path=%s", elapsed_minutes, final_size, local_path)

    print("Download completed at:", local_path)


if __name__ == "__main__":
    main()
