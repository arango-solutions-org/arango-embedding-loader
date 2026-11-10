"""Document sources: in-memory lists, uploaded (temp) files and remote URLs.

Each source wraps a synchronous document iterator; the pipeline pulls batches in a worker
thread so parsing and network I/O never block the event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import IO, Any
from urllib.parse import urlparse

import httpx

from app.errors import AppError
from app.ingest.parsers import CountingReader, CsvOptions, FileFormat, iter_documents


class DocumentSource:
    def __init__(
        self,
        iterator: Iterator[dict[str, Any]],
        *,
        bytes_read: Any = None,
        bytes_total: int | None = None,
        total: int | None = None,
        close: Any = None,
        reopen: Callable[[], DocumentSource] | None = None,
        spool: Callable[[], DocumentSource] | None = None,
    ):
        self._it = iterator
        self._bytes_read = bytes_read
        self.bytes_total = bytes_total
        self.total = total
        self._close = close
        #: Opens a fresh copy of the same documents, for a second pass. ``None`` when the
        #: source can only be read once (streamed URLs).
        self.reopen = reopen
        #: For a source that can only be read once: saves it to disk and returns a source over
        #: the saved copy, which can be reopened. Blocking; run it in a worker thread.
        self.spool = spool

    @property
    def bytes_read(self) -> int | None:
        return self._bytes_read() if self._bytes_read else None

    async def next_batch(self, size: int) -> list[dict[str, Any]]:
        return await asyncio.to_thread(lambda: list(islice(self._it, size)))

    def close(self) -> None:
        if self._close:
            self._close()


def list_source(documents: Iterable[dict[str, Any]]) -> DocumentSource:
    docs = list(documents)
    return DocumentSource(
        iter(docs),
        total=len(docs),
        reopen=lambda: list_source(docs),  # the scan does not modify documents
    )


def file_source(
    path: str,
    *,
    filename: str | None,
    fmt: FileFormat = "auto",
    csv_options: CsvOptions | None = None,
    json_path: str | None = None,
    total: int | None = None,
) -> DocumentSource:
    fh = open(path, "rb")  # noqa: SIM115 - closed by DocumentSource.close
    counter = CountingReader(fh)
    iterator = iter_documents(
        io.BufferedReader(counter),
        filename=filename,
        fmt=fmt,
        csv_options=csv_options,
        json_path=json_path,
    )
    return DocumentSource(
        iterator,
        bytes_read=lambda: counter.bytes_read,
        bytes_total=os.path.getsize(path),
        total=total,
        close=fh.close,
        reopen=lambda: file_source(
            path,
            filename=filename,
            fmt=fmt,
            csv_options=csv_options,
            json_path=json_path,
            total=total,
        ),
    )


def files_source(
    files: list[tuple[str, str | None]],
    *,
    fmt: FileFormat = "auto",
    csv_options: CsvOptions | None = None,
    json_path: str | None = None,
    on_close: Callable[[], None] | None = None,
) -> DocumentSource:
    """Several local files (``(path, filename)``) read one after the other."""
    opened: list[DocumentSource] = []

    def generate() -> Iterator[dict[str, Any]]:
        for path, filename in files:
            src = file_source(
                path, filename=filename, fmt=fmt, csv_options=csv_options, json_path=json_path
            )
            opened.append(src)
            try:
                yield from src._it
            finally:
                src.close()

    def close() -> None:
        for src in opened:
            src.close()
        if on_close:
            on_close()

    return DocumentSource(
        generate(),
        bytes_read=lambda: sum(src.bytes_read or 0 for src in opened),
        bytes_total=sum(os.path.getsize(p) for p, _ in files),
        close=close,
        reopen=lambda: files_source(files, fmt=fmt, csv_options=csv_options, json_path=json_path),
    )


# ----------------------------------------------------------------------------- URLs


def normalize_url(url: str) -> str:
    if url.startswith("gs://"):
        bucket_and_path = url[len("gs://") :]
        return f"https://storage.googleapis.com/{bucket_and_path}"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise AppError(400, "unsupported_url", f"Only http(s) and gs:// URLs are supported: {url}")
    return url


def check_url_allowed(url: str, allowed_hosts: list[str]) -> None:
    if not allowed_hosts:
        return
    host = urlparse(url).hostname or ""
    if not any(host == h or host.endswith("." + h) for h in allowed_hosts):
        raise AppError(403, "url_host_not_allowed", f"Host '{host}' is not in URL_ALLOWED_HOSTS")


class _IterReader(io.RawIOBase):
    def __init__(self, chunks: Iterator[bytes], counter: list[int]):
        self._chunks = chunks
        self._buf = b""
        self._counter = counter

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        while not self._buf:
            try:
                self._buf = next(self._chunks)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        self._counter[0] += n
        return n


def head_sizes(urls: list[str], timeout: float = 15) -> int | None:
    total = 0
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        for url in urls:
            try:
                resp = client.head(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise AppError(400, "url_unreachable", f"{url}: {exc}") from exc
            length = resp.headers.get("content-length")
            if length is None or resp.headers.get("content-encoding"):
                return None
            total += int(length)
    return total


def url_source(
    urls: list[str],
    *,
    fmt: FileFormat = "auto",
    csv_options: CsvOptions | None = None,
    json_path: str | None = None,
    bytes_total: int | None = None,
    timeout: float = 120,
) -> DocumentSource:
    counter = [0]
    client = httpx.Client(follow_redirects=True, timeout=timeout)

    def generate() -> Iterator[dict[str, Any]]:
        for url in urls:
            with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise AppError(
                        400, "url_fetch_failed", f"GET {url} returned HTTP {resp.status_code}"
                    )
                filename = urlparse(url).path.rsplit("/", 1)[-1]
                raw: IO[bytes] = io.BufferedReader(_IterReader(resp.iter_bytes(), counter))
                yield from iter_documents(
                    raw, filename=filename, fmt=fmt, csv_options=csv_options, json_path=json_path
                )

    def spool() -> DocumentSource:
        """Download every URL to a temporary file (counted in ``bytes_read`` as it goes)."""
        from app.config import get_settings

        settings = get_settings()
        directory = settings.upload_dir
        cap = settings.max_url_download_mb * 1024 * 1024
        saved = 0
        files: list[tuple[str, str | None]] = []

        def remove() -> None:
            for path, _ in files:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path)

        try:
            for url in urls:
                filename = urlparse(url).path.rsplit("/", 1)[-1]
                fd, path = tempfile.mkstemp(
                    prefix="ael-url-",
                    suffix=os.path.splitext(filename)[1],
                    dir=str(directory) if directory else None,
                )
                files.append((path, filename))
                with os.fdopen(fd, "wb") as out, client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        raise AppError(
                            400, "url_fetch_failed", f"GET {url} returned HTTP {resp.status_code}"
                        )
                    for chunk in resp.iter_bytes():
                        saved += len(chunk)
                        if saved > cap:
                            raise AppError(
                                413,
                                "url_download_too_large",
                                f"The URL load is over {settings.max_url_download_mb} MB "
                                "(MAX_URL_DOWNLOAD_MB). With long_input: error it is saved to "
                                "disk to check lengths before embedding; use another long_input "
                                "mode to stream it instead, or raise the limit.",
                            )
                        out.write(chunk)
                        counter[0] += len(chunk)
        except BaseException:
            remove()
            raise
        finally:
            client.close()
        return files_source(
            files, fmt=fmt, csv_options=csv_options, json_path=json_path, on_close=remove
        )

    return DocumentSource(
        generate(),
        bytes_read=lambda: counter[0],
        bytes_total=bytes_total,
        close=client.close,
        spool=spool,
    )
