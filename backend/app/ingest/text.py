"""Helpers for reading/writing dot-path fields and turning field values into embeddable text."""

from __future__ import annotations

import json
import re
from typing import Any

_MISSING = object()
_TEMPLATE_RE = re.compile(r"\{([A-Za-z0-9_\-.$@]+)\}")


def split_path(path: str) -> list[str]:
    return [part for part in path.split(".") if part]


def get_path(doc: dict[str, Any], path: str, default: Any = None) -> Any:
    """Return the value at dot-path ``path`` (``"a.b.c"``) or ``default`` when absent."""
    current: Any = doc
    for part in split_path(path):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return default
    return current


def has_path(doc: dict[str, Any], path: str) -> bool:
    return get_path(doc, path, _MISSING) is not _MISSING


def set_path(doc: dict[str, Any], path: str, value: Any) -> None:
    """Set ``value`` at dot-path ``path``, creating intermediate objects as needed."""
    parts = split_path(path)
    current = doc
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value


def value_to_text(value: Any) -> str | None:
    """Convert a field value to text; ``None``/empty values return ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, bool | int | float):
        text = str(value)
    elif isinstance(value, list):
        parts = [value_to_text(v) for v in value]
        text = "\n".join(p for p in parts if p)
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = text.strip()
    return text or None


def template_fields(template: str) -> list[str]:
    return _TEMPLATE_RE.findall(template)


def render_template(template: str, doc: dict[str, Any]) -> str | None:
    """Render ``{field.path}`` placeholders. Returns ``None`` if every placeholder is empty."""
    found_any = False

    def repl(match: re.Match[str]) -> str:
        nonlocal found_any
        text = value_to_text(get_path(doc, match.group(1)))
        if text:
            found_any = True
            return text
        return ""

    rendered = _TEMPLATE_RE.sub(repl, template).strip()
    return rendered if found_any and rendered else None


def meta_field_for(target_field: str) -> str:
    return f"{target_field}_meta"
