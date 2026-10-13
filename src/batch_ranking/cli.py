"""``embed-dataset DATASET --model-id YOLOX_military_assets``: embed a dataset into the model's own collection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="embed-dataset",
        description="Embed every image under DATASET and upsert it into a Milvus collection named after the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="local image directory or s3://bucket/prefix")
    parser.add_argument("--model-id", required=True, help="detector the dataset belongs to, e.g. YOLOX_military_assets")
    parser.add_argument(
        "--milvus-uri", default=os.environ.get("MILVUS_URI", "data/milvus.db"),
        help="a file path runs Milvus Lite; or http://host:19530 (env: MILVUS_URI)",
    )
    parser.add_argument("--embedder", default="efficientnet_b0", help="timm model used for the embeddings")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recreate", action="store_true", help="drop the model's collection first")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from batch_ranking.embedder import Embedder
    from batch_ranking.images import list_images, load_images
    from batch_ranking.store import Store, collection_name

    collection = collection_name(args.model_id)
    paths = list_images(args.dataset)
    if not paths:
        sys.exit(f"no images found under {args.dataset}")
    print(f"{len(paths)} images -> collection {collection!r}", file=sys.stderr)

    if args.milvus_uri.endswith(".db"):
        Path(args.milvus_uri).parent.mkdir(parents=True, exist_ok=True)
    store = Store(args.milvus_uri)
    if args.recreate and store.client.has_collection(collection):
        store.drop(collection)
    model = Embedder(args.embedder, args.device)
    store.ensure_collection(collection, model.dim, args.embedder)

    failed: list[str] = []
    for start in range(0, len(paths), args.batch_size):
        chunk = paths[start : start + args.batch_size]
        images = load_images(chunk)
        ok = [(p, img) for p, img in zip(chunk, images) if img is not None]
        failed += [p for p, img in zip(chunk, images) if img is None]
        store.upsert(collection, [p for p, _ in ok], model.embed([img for _, img in ok]))
        print(f"\rembedded {start + len(chunk)}/{len(paths)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    for p in failed:
        print(f"skipped unreadable image: {p}", file=sys.stderr)
    print(
        f"done: {len(paths) - len(failed)} embedded, {len(failed)} skipped; "
        f"{collection!r} now has {store.count(collection)} rows",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
