"""Inputs longer than an embedding model's token limit (``long_input`` on the embedding spec).

Tokens are counted with the model's own tokenizer when one is available (local Hugging Face
models, OpenAI via tiktoken, Voyage via its published tokenizer) and estimated from the text
length otherwise. A text over the limit is then truncated, chunked, skipped or rejected according
to the spec's ``long_input`` mode.
"""

from __future__ import annotations

import bisect
import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

LongInputMode = Literal[
    "truncate_end",
    "truncate_start",
    "truncate_middle",
    "chunk_mean",
    "chunk_docs",
    "skip",
    "error",
]
TRUNCATE_MODES = ("truncate_end", "truncate_start", "truncate_middle")
DEFAULT_TOKEN_CHARS = 4.0

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class LongInput:
    """The spec's long-input settings, as passed to :meth:`EmbeddingProvider.plan_inputs`."""

    mode: LongInputMode = "truncate_end"
    chunk_overlap_tokens: int = 25
    max_tokens: int | None = None
    token_chars: float = DEFAULT_TOKEN_CHARS
    warn: bool = True


@dataclass
class TextPlan:
    """How one input text is sent to the model.

    ``combine`` says how the vectors of ``pieces`` become the result: ``single`` (one piece, one
    vector), ``mean`` (one vector, the weighted mean of the pieces) or ``split`` (one vector per
    piece, each stored as its own document). ``action`` is ``skip`` or ``error`` when the text
    must not be embedded at all. ``stats`` is recorded in the document's ``<target>_meta``.
    """

    pieces: list[str]
    combine: Literal["single", "mean", "split"] = "single"
    action: Literal["embed", "skip", "error"] = "embed"
    weights: list[int] | None = None
    #: The text is longer than the model's limit.
    overrun: bool = False
    warn: bool = False
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tokens:
    """One text's tokens: the character span of each, and whether each starts a word."""

    spans: list[tuple[int, int]]
    word_start: list[bool]

    def __len__(self) -> int:
        return len(self.spans)


class Tokenizer:
    #: Counts with the model's own tokenizer (``False``: an estimate).
    exact: bool = True

    def tokenize(self, texts: list[str]) -> list[Tokens]:
        raise NotImplementedError


def _wordwise(text: str, count: Callable[[str], int]) -> Tokens:
    """Tokens per whitespace-separated word; every token of a word carries the word's span."""
    spans: list[tuple[int, int]] = []
    starts: list[bool] = []
    for m in _WORD.finditer(text):
        n = max(1, count(m.group()))
        spans.extend([m.span()] * n)
        starts.extend([True] + [False] * (n - 1))
    return Tokens(spans, starts)


class EstimateTokenizer(Tokenizer):
    """No tokenizer for the model: about ``token_chars`` characters per token, per word."""

    exact = False

    def __init__(self, token_chars: float = DEFAULT_TOKEN_CHARS):
        self.token_chars = token_chars

    def tokenize(self, texts: list[str]) -> list[Tokens]:
        return [_wordwise(t, lambda w: math.ceil(len(w) / self.token_chars)) for t in texts]


class HFTokenizer(Tokenizer):
    """A Hugging Face tokenizer: ``transformers`` (fast or slow) or a ``tokenizers.Tokenizer``."""

    def __init__(self, tokenizer: Any, lock: threading.Lock | None = None):
        self.tok = tokenizer
        self.lock = lock or threading.Lock()  # fast tokenizers are not thread-safe

    def tokenize(self, texts: list[str]) -> list[Tokens]:
        with self.lock:
            if hasattr(self.tok, "encode_batch"):  # tokenizers.Tokenizer
                encs = self.tok.encode_batch(texts, add_special_tokens=False)
                return [_from_offsets(t, e.offsets) for t, e in zip(texts, encs, strict=True)]
            if getattr(self.tok, "is_fast", False):
                enc = self.tok(
                    texts,
                    add_special_tokens=False,
                    truncation=False,
                    return_offsets_mapping=True,
                    verbose=False,
                )
                return [_from_offsets(t, enc["offset_mapping"][i]) for i, t in enumerate(texts)]
            # Slow tokenizers have no offsets; their pre-tokenizers split on whitespace, so
            # counting word by word gives the same total.
            return [_wordwise(t, lambda w: len(self.tok.tokenize(w))) for t in texts]


def _starts_word(text: str, a: int) -> bool:
    """A token starting at ``a`` begins a whitespace-separated word.

    Tokenizers' own word ids are not used: many split digits or punctuation into separate
    "words", and cutting there would split ``word116`` into ``word`` and ``116``.
    """
    return a == 0 or (a < len(text) and text[a].isspace()) or text[a - 1].isspace()


def _from_offsets(text: str, offsets: Any) -> Tokens:
    spans = [(int(a), int(b)) for a, b in offsets]
    return Tokens(spans, [_starts_word(text, a) for a, _ in spans])


class TiktokenTokenizer(Tokenizer):
    """OpenAI models, counted with ``tiktoken``."""

    def __init__(self, encoding: Any):
        self.enc = encoding

    def tokenize(self, texts: list[str]) -> list[Tokens]:
        out = []
        for text in texts:
            ids = self.enc.encode(text, disallowed_special=())
            _, starts = self.enc.decode_with_offsets(ids)
            ends = [*starts[1:], len(text)]
            spans = list(zip(starts, ends, strict=True))
            out.append(Tokens(spans, [_starts_word(text, a) for a, _ in spans]))
        return out


# --------------------------------------------------------------------------- planning


def plan_text(
    text: str,
    tokens: Tokens,
    limit: int | None,
    settings: LongInput,
    *,
    exact: bool,
    native: frozenset[str] = frozenset(),
) -> TextPlan:
    """Decide how ``text`` is sent to the model under ``settings``.

    ``native`` lists the truncate modes the provider applies itself: the whole text is sent and
    the provider truncates it, which is exact even when the count here is an estimate.
    """
    n = len(tokens)
    mode = settings.mode
    stats: dict[str, Any] = {
        "input_tokens": n,
        "max_tokens": limit,
        "truncated": False,
        "tokens_dropped": 0,
        "chunks": 1,
        "long_input": mode,
        "token_count": "exact" if exact else "estimated",
    }
    if limit is None or n <= limit:
        return TextPlan(pieces=[text], stats=stats)

    plan = TextPlan(pieces=[], overrun=True, warn=settings.warn, stats=stats)
    if mode in ("error", "skip"):
        plan.action = mode
        stats["chunks"] = 0
        stats["tokens_dropped"] = n
        return plan
    if mode in native:
        plan.pieces = [text]
        stats.update(truncated=True, tokens_dropped=n - limit)
        return plan

    # Token positions where a piece may start or end without cutting a word in half.
    cuts = [j for j in range(n) if tokens.word_start[j]] + [n]

    def at_most(pos: int, floor: int) -> int:
        """Largest cut in (floor, pos], else ``pos``."""
        i = bisect.bisect_right(cuts, pos) - 1
        return cuts[i] if i >= 0 and cuts[i] > floor else pos

    def at_least(pos: int, ceiling: int) -> int:
        """Smallest cut in [pos, ceiling), else ``pos``."""
        i = bisect.bisect_left(cuts, pos)
        return cuts[i] if i < len(cuts) and cuts[i] < ceiling else pos

    def span(a: int, b: int) -> str:
        return text[tokens.spans[a][0] : tokens.spans[b - 1][1]].strip()

    if mode == "truncate_end":
        end = at_most(limit, 0)
        plan.pieces = [span(0, end)]
        stats.update(truncated=True, tokens_dropped=n - end)
    elif mode == "truncate_start":
        start = at_least(n - limit, n)
        plan.pieces = [span(start, n)]
        stats.update(truncated=True, tokens_dropped=start)
    elif mode == "truncate_middle":
        head = at_most(max(1, limit // 2), 0)
        # One token of the budget goes to the line break that joins the two halves.
        tail = at_least(n - max(1, limit - head - 1), n)
        plan.pieces = [span(0, head) + "\n" + span(tail, n)]
        stats.update(truncated=True, tokens_dropped=tail - head)
    else:  # chunk_mean / chunk_docs
        overlap = min(max(0, settings.chunk_overlap_tokens), limit // 2)
        bounds: list[tuple[int, int]] = []
        start = 0
        while True:
            end = at_most(min(n, start + limit), start)
            bounds.append((start, end))
            if end >= n:
                break
            nxt = end
            if overlap:
                # The cut nearest to ``end - overlap`` that still moves forward.
                target = end - overlap
                i = bisect.bisect_left(cuts, target)
                options = [c for c in cuts[max(0, i - 1) : i + 1] if start < c <= end]
                if options:
                    nxt = min(options, key=lambda c: (abs(c - target), -c))
            start = nxt
        plan.pieces = [span(a, b) for a, b in bounds]
        plan.weights = [b - a for a, b in bounds]
        plan.combine = "mean" if mode == "chunk_mean" else "split"
        stats["chunks"] = len(bounds)
    return plan
