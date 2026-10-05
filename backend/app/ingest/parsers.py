"""Streaming document parsers for JSON arrays, JSON Lines and CSV.

Parsers are synchronous generators over binary file-like objects so they can stream very large
inputs (local temp files or HTTP responses) with constant memory. The async pipeline pulls
batches from them via ``asyncio.to_thread``.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
from collections.abc import Iterator
from typing import IO, Any, Literal, cast

import ijson
from pydantic import BaseModel, Field

FileFormat = Literal["auto", "json", "jsonl", "csv"]


class CsvOptions(BaseModel):
    delimiter: str = Field(",", min_length=1, max_length=1, description="Field delimiter.")
    infer_types: bool = Field(
        True, description="Convert numeric / boolean / empty cells to JSON types."
    )


class ParseError(ValueError):
    pass


class CountingReader(io.RawIOBase):
    """Wraps a binary stream and counts bytes read (used for progress reporting)."""

    def __init__(self, raw: IO[bytes]):
        self._raw = raw
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        data = self._raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.bytes_read += n
        return n


def as_buffered(stream: IO[bytes]) -> io.BufferedReader:
    if isinstance(stream, io.BufferedReader):
        return stream
    return io.BufferedReader(cast(Any, stream))


def detect_format(filename: str | None, head: bytes) -> Literal["json", "jsonl", "csv"]:
    name = (filename or "").lower().removesuffix(".gz")
    if name.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if name.endswith((".csv", ".tsv")):
        return "csv"
    if name.endswith(".json"):
        stripped = head.lstrip()
        if stripped.startswith(b"{"):
            first_line = stripped.split(b"\n", 1)[0]
            try:
                json.loads(first_line)
                if b"\n" in stripped.strip():
                    return "jsonl"
            except ValueError:
                pass
        return "json"
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"["):
        return "json"
    if stripped.startswith(b"{"):
        first_line = stripped.split(b"\n", 1)[0]
        try:
            json.loads(first_line)
            return "jsonl"
        except ValueError:
            return "json"
    return "csv"


def maybe_decompress(stream: IO[bytes], filename: str | None) -> IO[bytes]:
    buffered = as_buffered(stream)
    magic = buffered.peek(2)[:2]
    if magic == b"\x1f\x8b" or (filename or "").lower().endswith(".gz"):
        return gzip.GzipFile(fileobj=buffered)  # type: ignore[return-value]
    return buffered


_INT_RE = re.compile(r"-?(0|[1-9][0-9]*)")
_FLOAT_RE = re.compile(r"-?(0|[1-9][0-9]*)?(\.[0-9]+)?([eE][-+]?[0-9]+)?")


def _infer(value: str | None) -> Any:
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return None
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null":
        return None
    if _INT_RE.fullmatch(v):
        return int(v)
    if any(c in v for c in ".eE") and _FLOAT_RE.fullmatch(v) and v not in ("-", "."):
        try:
            return float(v)
        except ValueError:
            return value
    return value


def iter_jsonl(stream: IO[bytes]) -> Iterator[dict[str, Any]]:
    for lineno, raw in enumerate(stream, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except ValueError as exc:
            raise ParseError(f"Invalid JSON on line {lineno}: {exc}") from exc
        if not isinstance(doc, dict):
            raise ParseError(f"Line {lineno} is not a JSON object")
        yield doc


def iter_json(stream: IO[bytes], json_path: str | None = None) -> Iterator[dict[str, Any]]:
    buffered = as_buffered(stream)
    head = buffered.peek(64).lstrip(b"\xef\xbb\xbf \t\r\n")
    if json_path is None and head.startswith(b"{"):
        # A single JSON object document.
        doc = json.load(buffered)
        if not isinstance(doc, dict):
            raise ParseError("JSON root is not an object")
        yield doc
        return
    prefix = json_path or "item"
    try:
        for i, doc in enumerate(ijson.items(buffered, prefix, use_float=True)):
            if not isinstance(doc, dict):
                raise ParseError(f"Element {i} at '{prefix}' is not a JSON object")
            yield doc
    except ijson.JSONError as exc:
        raise ParseError(f"Invalid JSON: {exc}") from exc


def iter_csv(stream: IO[bytes], options: CsvOptions) -> Iterator[dict[str, Any]]:
    text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(text, delimiter=options.delimiter)
    for row in reader:
        doc: dict[str, Any] = {}
        for key, value in row.items():
            if key is None:
                continue
            doc[key] = _infer(value) if options.infer_types else value
        yield doc


def iter_documents(
    stream: IO[bytes],
    *,
    filename: str | None = None,
    fmt: FileFormat = "auto",
    csv_options: CsvOptions | None = None,
    json_path: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield documents from ``stream`` (optionally gzip-compressed)."""
    data = maybe_decompress(stream, filename)
    buffered = as_buffered(data)
    if fmt != "auto":
        resolved = fmt
    elif json_path:
        resolved = "json"
    else:
        resolved = detect_format(filename, buffered.peek(4096))
    if resolved == "jsonl":
        yield from iter_jsonl(buffered)
    elif resolved == "json":
        yield from iter_json(buffered, json_path)
    else:
        if fmt == "auto" and (filename or "").lower().removesuffix(".gz").endswith(".tsv"):
            csv_options = (csv_options or CsvOptions()).model_copy(update={"delimiter": "\t"})
        yield from iter_csv(buffered, csv_options or CsvOptions())


def count_records(path: str, filename: str | None, fmt: FileFormat = "auto") -> int | None:
    """Cheap record count for line-oriented formats (JSONL/CSV). ``None`` for JSON arrays."""
    with open(path, "rb") as fh:
        data = maybe_decompress(fh, filename)
        buffered = as_buffered(data)
        resolved = detect_format(filename, buffered.peek(4096)) if fmt == "auto" else fmt
        if resolved == "json":
            return None
        count = 0
        for line in buffered:
            if line.strip():
                count += 1
        return max(count - 1, 0) if resolved == "csv" else count
