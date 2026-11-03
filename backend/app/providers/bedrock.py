"""Amazon Bedrock embeddings (Titan Text Embeddings and Cohere Embed on Bedrock)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import Field, SecretStr

from app.providers.base import (
    Availability,
    EmbeddingProvider,
    InputType,
    ModelInfo,
    NativeTruncationOptions,
    ProviderOptions,
)
from app.providers.cohere_provider import cohere_truncate


class BedrockOptions(NativeTruncationOptions):
    region: str | None = Field(
        default_factory=lambda: os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        description="AWS region. Defaults to $AWS_REGION.",
    )
    profile_name: str | None = Field(None, description="AWS named profile.")
    aws_access_key_id: SecretStr | None = Field(None, description="Overrides the AWS chain.")
    aws_secret_access_key: SecretStr | None = None
    aws_session_token: SecretStr | None = None
    normalize: bool = Field(True, description="Titan v2: return unit-length vectors.")


class BedrockProvider(EmbeddingProvider):
    name = "bedrock"
    label = "Amazon Bedrock"
    description = "Titan Text Embeddings and Cohere Embed models via Amazon Bedrock."
    Options = BedrockOptions
    requires = ("boto3",)
    install_hint = "uv sync --extra bedrock"
    max_batch_size = 96
    max_concurrency = 8
    default_model = "amazon.titan-embed-text-v2:0"
    supports_dimensions = True
    known_models = (
        ModelInfo(id="amazon.titan-embed-text-v2:0", dimension=1024, max_tokens=8192),
        ModelInfo(id="amazon.titan-embed-text-v1", dimension=1536, max_tokens=8192),
        ModelInfo(id="cohere.embed-english-v3", dimension=1024, max_tokens=512),
        ModelInfo(id="cohere.embed-multilingual-v3", dimension=1024, max_tokens=512),
    )

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def native_long_input(self, model: str) -> frozenset[str]:
        if model.startswith("cohere."):
            return frozenset({"truncate_end", "truncate_start"})
        return frozenset()  # Titan has no truncation option

    def check_configuration(self, options: ProviderOptions) -> Availability:
        assert isinstance(options, BedrockOptions)
        if not options.region:
            return Availability(available=False, reason="No region: set AWS_REGION or region.")
        return Availability(available=True)

    def _client(self, options: BedrockOptions) -> Any:
        def secret(v: SecretStr | None) -> str | None:
            return v.get_secret_value() if v else None

        key = "|".join(
            str(x)
            for x in (
                options.region,
                options.profile_name,
                secret(options.aws_access_key_id),
                secret(options.aws_secret_access_key),
                secret(options.aws_session_token),
            )
        )
        client = self._clients.get(key)
        if client is None:
            import boto3

            session = boto3.Session(
                profile_name=options.profile_name,
                aws_access_key_id=secret(options.aws_access_key_id),
                aws_secret_access_key=secret(options.aws_secret_access_key),
                aws_session_token=secret(options.aws_session_token),
                region_name=options.region,
            )
            client = session.client("bedrock-runtime")
            self._clients[key] = client
        return client

    def _invoke(self, client: Any, model: str, body: dict[str, Any]) -> dict[str, Any]:
        response = client.invoke_model(
            modelId=model,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(response["body"].read())

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        options: ProviderOptions,
        dimensions: int | None,
        input_type: InputType,
    ) -> list[list[float]]:
        assert isinstance(options, BedrockOptions)
        client = self._client(options)
        if model.startswith("cohere."):
            body: dict[str, Any] = {
                "texts": texts,
                "input_type": "search_document" if input_type == "document" else "search_query",
                "truncate": cohere_truncate(options.long_input_mode),
            }
            result = await asyncio.to_thread(self._invoke, client, model, body)
            embeddings = result["embeddings"]
            if isinstance(embeddings, dict):
                embeddings = embeddings.get("float", [])
            return [list(v) for v in embeddings]

        # Titan accepts one text per request.
        sem = asyncio.Semaphore(8)

        async def one(text: str) -> list[float]:
            payload: dict[str, Any] = {"inputText": text}
            if "v2" in model:
                payload["normalize"] = options.normalize
                if dimensions:
                    payload["dimensions"] = dimensions
            async with sem:
                result = await asyncio.to_thread(self._invoke, client, model, payload)
            return list(result["embedding"])

        return list(await asyncio.gather(*(one(t) for t in texts)))
