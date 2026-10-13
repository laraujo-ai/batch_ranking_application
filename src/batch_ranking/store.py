"""Milvus collections of dataset embeddings, one per model (``YOLOX_military_assets`` -> ``YOLOX_military_assets``)."""

from __future__ import annotations

import logging
import re

import numpy as np
from pymilvus import DataType, MilvusClient

_MAX_LEN = 2048


def collection_name(model_id: str) -> str:
    """Milvus names allow only letters, digits and underscores, and must not start with a digit."""
    name = re.sub(r"[^0-9A-Za-z_]", "_", model_id)
    if not name:
        raise ValueError("model_id must not be empty")
    return name if not name[0].isdigit() else f"_{name}"


class Store:
    def __init__(self, uri: str):
        if not uri.startswith(("http://", "https://", "tcp://")):
            # Milvus Lite logs a harmless traceback for an RPC newer pymilvus clients probe.
            logging.getLogger("grpc._server").setLevel(logging.CRITICAL)
        self.client = MilvusClient(uri=uri)

    def ensure_collection(self, name: str, dim: int, embedder: str) -> None:
        if self.client.has_collection(name):
            desc = self.client.describe_collection(name)
            stored = (desc.get("properties") or {}).get("embedder")
            stored_dim = next(f["params"].get("dim") for f in desc["fields"] if f["name"] == "embedding")
            if (stored and stored != embedder) or stored_dim != dim:
                raise ValueError(
                    f"collection {name!r} holds {stored} embeddings (dim {stored_dim}), "
                    f"not {embedder} (dim {dim}); use --recreate to rebuild it"
                )
            return
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("path", DataType.VARCHAR, is_primary=True, max_length=_MAX_LEN)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        index = self.client.prepare_index_params()
        index.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(name, schema=schema, index_params=index, properties={"embedder": embedder})

    def upsert(self, name: str, paths: list[str], embeddings: np.ndarray) -> None:
        if paths:
            self.client.upsert(name, [{"path": p, "embedding": e.tolist()} for p, e in zip(paths, embeddings)])

    def count(self, name: str) -> int:
        return int(self.client.query(name, output_fields=["count(*)"])[0]["count(*)"])

    def drop(self, name: str) -> None:
        self.client.drop_collection(name)
