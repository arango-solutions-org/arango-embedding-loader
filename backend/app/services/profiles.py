"""Persistence for collection embedding profiles.

Profiles live in the ``embedding_loader_profiles`` collection of the same database as the
target collection, keyed by collection name, so they travel with the data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from arango.database import StandardDatabase

from app.config import get_settings
from app.models.profiles import Profile, ProfileConfig


def _collection_name() -> str:
    return get_settings().profiles_collection


def _ensure_store(db: StandardDatabase) -> None:
    name = _collection_name()
    if not db.has_collection(name):
        db.create_collection(name)


def get_profile(db: StandardDatabase, collection: str) -> Profile | None:
    name = _collection_name()
    if not db.has_collection(name):
        return None
    doc = db.collection(name).get(collection)
    if not doc:
        return None
    return _from_doc(doc)  # type: ignore[arg-type]


def list_profiles(db: StandardDatabase) -> list[Profile]:
    name = _collection_name()
    if not db.has_collection(name):
        return []
    return [_from_doc(d) for d in db.collection(name).all()]  # type: ignore[union-attr]


def save_profile(db: StandardDatabase, collection: str, config: ProfileConfig) -> Profile:
    _ensure_store(db)
    store = db.collection(_collection_name())
    existing = store.get(collection)
    now = datetime.now(UTC)
    data: dict[str, Any] = config.model_dump(mode="json", exclude={"embeddings"})
    # Secrets in provider_options are never persisted.
    data["embeddings"] = [spec.public_dict() for spec in config.embeddings]
    data["collection"] = collection
    data["updated_at"] = now.isoformat()
    data["created_at"] = (existing or {}).get("created_at") or now.isoformat()  # type: ignore[union-attr]
    data["_key"] = collection
    store.insert(data, overwrite=True)
    return _from_doc(data)


def delete_profile(db: StandardDatabase, collection: str) -> bool:
    name = _collection_name()
    if not db.has_collection(name):
        return False
    return bool(db.collection(name).delete(collection, ignore_missing=True))


def _from_doc(doc: dict[str, Any]) -> Profile:
    data = {k: v for k, v in doc.items() if not k.startswith("_")}
    return Profile.model_validate(data)
