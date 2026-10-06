import pytest

from app.models.embedding import EmbeddingFieldSpec
from app.providers.base import EmbeddingProvider, ProviderError
from app.providers.presets import get_preset
from app.providers.registry import get_provider, list_providers, register_provider
from app.services.pipeline import embed_documents, resolve_specs


def test_registry_has_builtins():
    names = {p.name for p in list_providers()}
    assert {
        "openai",
        "voyage",
        "cohere",
        "bedrock",
        "ollama",
        "sentence_transformers",
        "fake",
    } <= names
    assert "anthropic" not in names


def test_unknown_provider():
    with pytest.raises(ProviderError):
        get_provider("does-not-exist")


def test_pubmedbert_preset():
    preset = get_preset("pubmedbert")
    assert preset.provider == "sentence_transformers"
    assert preset.model == "NeuML/pubmedbert-base-embeddings"
    assert preset.dimensions == 768


async def test_fake_provider_deterministic():
    p = get_provider("fake")
    opts = p.parse_options(None)
    a = await p.embed(["x", "y"], model="fake-384", options=opts)
    b = await p.embed(["x"], model="fake-384", options=opts)
    assert len(a[0]) == 384 and a[0] == b[0] and a[0] != a[1]
    c = await p.embed(["x"], model="fake-384", options=opts, dimensions=16)
    assert len(c[0]) == 16


def test_invalid_options():
    with pytest.raises(ProviderError):
        get_provider("fake").parse_options({"nope": 1})


async def test_custom_provider_plugin():
    class Constant(EmbeddingProvider):
        name = "constant_test"
        label = "Constant"
        default_model = "c"

        async def embed_batch(self, texts, *, model, options, dimensions, input_type):
            return [[1.0, 0.0] for _ in texts]

    register_provider(Constant)
    specs = resolve_specs([EmbeddingFieldSpec(source_field="t", provider="constant_test")])
    docs = [{"t": "a"}]
    await embed_documents(docs, specs, store_metadata=False)
    assert docs[0]["t_embedding"] == [1.0, 0.0]


async def test_embed_documents_on_missing():
    specs = resolve_specs(
        [
            EmbeddingFieldSpec(source_field="a", target_field="va", preset="fake"),
            EmbeddingFieldSpec(
                source_field="b", target_field="vb", preset="fake", on_missing="null"
            ),
            EmbeddingFieldSpec(
                source_field="c", target_field="vc", preset="fake", on_missing="error"
            ),
        ]
    )
    docs = [{"a": "x", "c": "z"}, {"a": "y"}]
    outcomes = await embed_documents(docs, specs, store_metadata=True)
    assert outcomes[0].error is None and outcomes[0].embedded
    assert docs[0]["vb"] is None and docs[0]["vb_meta"]["status"] == "skipped"
    assert len(docs[0]["vc"]) == 384
    assert outcomes[1].error and "vc" in outcomes[1].error
    assert docs[1]["vc_meta"]["status"] == "error"


def test_duplicate_targets_rejected():
    from app.errors import AppError

    with pytest.raises(AppError):
        resolve_specs(
            [
                EmbeddingFieldSpec(source_field="a", target_field="v", preset="fake"),
                EmbeddingFieldSpec(source_field="b", target_field="v", preset="fake"),
            ]
        )


def test_spec_validation_and_secrets():
    with pytest.raises(ValueError):
        EmbeddingFieldSpec(source_field="a")
    spec = EmbeddingFieldSpec(
        source_field="content.body", provider="openai", provider_options={"api_key": "s", "x": 1}
    )
    assert spec.resolved_target == "content_body_embedding"
    assert spec.public_dict()["provider_options"] == {"x": 1}


def test_baked_model_path(tmp_path, monkeypatch):
    from app.providers.sentence_transformers_provider import baked_model_path

    monkeypatch.setenv("ST_MODELS_DIR", str(tmp_path))
    model_dir = tmp_path / "NeuML" / "pubmedbert-base-embeddings"
    model_dir.mkdir(parents=True)
    assert baked_model_path("NeuML/pubmedbert-base-embeddings") is None  # no modules.json yet

    (model_dir / "modules.json").write_text("[]")
    assert baked_model_path("NeuML/pubmedbert-base-embeddings") == str(model_dir)
    assert baked_model_path("sentence-transformers/all-MiniLM-L6-v2") is None


def test_baked_model_path_stays_inside_models_dir(tmp_path, monkeypatch):
    from app.providers.sentence_transformers_provider import baked_model_path

    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "evil"
    outside.mkdir()
    (outside / "modules.json").write_text("[]")
    monkeypatch.setenv("ST_MODELS_DIR", str(root))
    assert baked_model_path("../evil") is None
