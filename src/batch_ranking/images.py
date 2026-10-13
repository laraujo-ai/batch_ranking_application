"""List and load dataset images from a local directory or an S3 prefix."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@lru_cache
def _s3():
    import boto3

    return boto3.client("s3")


def _split_s3(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def list_images(dataset: str) -> list[str]:
    """All image paths under a local directory or ``s3://bucket/prefix``, sorted."""
    if dataset.startswith("s3://"):
        bucket, prefix = _split_s3(dataset)
        refs = []
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            refs += [f"s3://{bucket}/{o['Key']}" for o in page.get("Contents", [])
                     if Path(o["Key"]).suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(refs)
    root = Path(dataset)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset}")
    return sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _load(ref: str) -> np.ndarray | None:
    try:
        if ref.startswith("s3://"):
            bucket, key = _split_s3(ref)
            data = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
        else:
            data = Path(ref).read_bytes()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001 - a bad image is skipped, not fatal
        return None


def load_images(refs: list[str], threads: int = 16) -> list[np.ndarray | None]:
    """BGR uint8 arrays in the order of ``refs``; ``None`` where an image could not be read."""
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(_load, refs))
