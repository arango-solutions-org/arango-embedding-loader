"""Export the OpenAPI (Swagger) document to docs/openapi.yaml and docs/openapi.json.

Usage (from the repo root):  cd backend && uv run python ../scripts/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import yaml  # noqa: E402

from app.main import create_app  # noqa: E402


def main() -> None:
    spec = create_app().openapi()
    spec["servers"] = [{"url": "http://localhost:8000", "description": "Local development"}]
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "openapi.json").write_text(json.dumps(spec, indent=2) + "\n")
    (docs / "openapi.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100)
    )
    try:
        from openapi_spec_validator import validate

        validate(spec)  # type: ignore[arg-type]
        print("OpenAPI spec is valid.")
    except ImportError:
        pass
    print(f"Wrote {docs / 'openapi.yaml'} and {docs / 'openapi.json'}")
    print(f"{len(spec['paths'])} paths")


if __name__ == "__main__":
    main()
