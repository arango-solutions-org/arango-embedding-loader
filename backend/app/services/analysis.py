"""Token-length analysis: find texts longer than the model's limit without embedding anything.

Used by ``analyze: true`` (report only) and by the pre-scan that runs before a load or re-embed
whose spec has ``long_input: error``, so a long text fails the job before any compute is spent.
"""

from __future__ import annotations

import asyncio
import heapq
from typing import TYPE_CHECKING, Any

from app.ingest.text import get_path

if TYPE_CHECKING:
    from app.providers.long_input import TextPlan
    from app.services.pipeline import ResolvedSpec

#: Over-limit documents listed per field in a report.
MAX_LISTED = 20


def doc_label(doc: dict[str, Any], key_field: str | None, index: int) -> str:
    """The document's key (``_key`` or ``key_field``), else its position in the input."""
    key = doc.get("_key")
    if key is None and key_field:
        key = get_path(doc, key_field)
    return str(key) if key is not None else f"#{index}"


def long_input_of(spec: ResolvedSpec) -> str:
    return spec.spec.long_input


def _percentile(values: list[int], pct: float) -> int:
    """Nearest-rank percentile of sorted ``values``."""
    rank = max(1, -(-len(values) * pct // 100))  # ceil
    return values[int(rank) - 1]


class FieldReport:
    """Token statistics for one embedding field."""

    def __init__(self, spec: ResolvedSpec):
        self.spec = spec
        self.tokens: list[int] = []
        self.missing = 0
        self.uncounted = 0
        self.max_tokens: int | None = None
        self.over_limit = 0
        self.estimated = False
        self.tokens_over = 0
        self._longest: list[tuple[int, str]] = []  # min-heap of (tokens, document)

    def add(self, label: str, plan: TextPlan) -> None:
        stats = plan.stats
        n = stats.get("input_tokens")
        if n is None:
            self.uncounted += 1
            return
        self.tokens.append(n)
        self.max_tokens = stats.get("max_tokens", self.max_tokens)
        self.estimated = self.estimated or stats.get("token_count") == "estimated"
        if plan.overrun:
            self.over_limit += 1
            self.tokens_over += n - (self.max_tokens or n)
            item = (n, label)
            if len(self._longest) < MAX_LISTED:
                heapq.heappush(self._longest, item)
            else:
                heapq.heappushpop(self._longest, item)

    def longest(self) -> list[dict[str, Any]]:
        return [{"document": d, "tokens": n} for n, d in sorted(self._longest, reverse=True)]

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "provider": self.spec.provider.name,
            "model": self.spec.model,
            "long_input": long_input_of(self.spec),
            "max_tokens": self.max_tokens,
            "token_count": "estimated" if self.estimated else "exact",
            "documents": len(self.tokens) + self.uncounted,
            "missing_text": self.missing,
        }
        if self.max_tokens is None:
            out["note"] = "The model's token limit is unknown: set `max_tokens` on the spec."
        if not self.tokens:
            return out
        ordered = sorted(self.tokens)
        out.update(
            over_limit=self.over_limit,
            over_limit_pct=round(100 * self.over_limit / len(ordered), 2),
            tokens_over_limit=self.tokens_over,
            tokens={
                "total": sum(ordered),
                "median": _percentile(ordered, 50),
                "p90": _percentile(ordered, 90),
                "p99": _percentile(ordered, 99),
                "max": ordered[-1],
            },
            longest=self.longest(),
        )
        return out


class Analysis:
    def __init__(self, specs: list[ResolvedSpec], key_field: str | None):
        self.specs = specs
        self.key_field = key_field
        self.fields = {s.target: FieldReport(s) for s in specs}
        self.documents = 0
        #: Documents with at least one over-limit text.
        self.over_limit_documents = 0

    async def add_batch(self, docs: list[dict[str, Any]], start: int) -> int:
        """Count tokens for a batch. Returns how many of its documents are over a limit."""
        over: set[int] = set()
        for spec in self.specs:
            report = self.fields[spec.target]
            texts: list[str] = []
            positions: list[int] = []
            for i, doc in enumerate(docs):
                text = spec.text_for(doc)
                if text is None:
                    report.missing += 1
                else:
                    texts.append(text)
                    positions.append(i)
            if not texts:
                continue
            plans = await asyncio.to_thread(
                spec.provider.plan_inputs,
                texts,
                model=spec.model,
                options=spec.options,
                input_type="document",
                long_input=spec.spec.long_input_settings,
            )
            for pos, plan in zip(positions, plans, strict=True):
                report.add(doc_label(docs[pos], self.key_field, start + pos), plan)
                if plan.overrun:
                    over.add(pos)
        self.documents += len(docs)
        self.over_limit_documents += len(over)
        return len(over)

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "over_limit_documents": self.over_limit_documents,
            "fields": {t: r.as_dict() for t, r in self.fields.items()},
        }

    def error_message(self) -> str | None:
        """Why a pre-scan fails: fields with ``long_input: error`` that have long texts."""
        parts = []
        for report in self.fields.values():
            if long_input_of(report.spec) != "error" or not report.over_limit:
                continue
            listed = ", ".join(f"{d['document']} ({d['tokens']})" for d in report.longest()[:5])
            more = report.over_limit - min(5, report.over_limit)
            parts.append(
                f"{report.over_limit} of {len(report.tokens)} documents are over the "
                f"{report.max_tokens}-token limit for '{report.spec.target}' "
                f"({report.spec.provider.name}/{report.spec.model}). Longest: {listed}"
                + (f" and {more} more" if more else "")
                + "."
            )
        if not parts:
            return None
        return (
            " ".join(parts)
            + " Nothing was embedded (long_input: error). See result.analysis for the list, or "
            "choose another long_input."
        )
