import cv2
import numpy as np
import pytest

from batch_ranking.cli import main
from batch_ranking.store import Store, collection_name


@pytest.mark.parametrize(
    "model_id, name",
    [("YOLOX_military_assets", "YOLOX_military_assets"), ("YOLOX_military-assets.v2", "YOLOX_military_assets_v2"), ("9yolo", "_9yolo")],
)
def test_collection_name(model_id, name):
    assert collection_name(model_id) == name


def test_one_collection_per_model(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        for i in range(3):
            cv2.imwrite(str(d / f"{i}.jpg"), rng.integers(0, 255, (64, 80, 3), dtype=np.uint8))
    (tmp_path / "a" / "broken.jpg").write_bytes(b"not an image")
    db = str(tmp_path / "milvus.db")

    for model_id, ds in [("YOLOX_military_assets", "a"), ("YOLOX_person_detector", "b"), ("YOLOX_military_assets", "a")]:
        main([str(tmp_path / ds), "--model-id", model_id, "--milvus-uri", db, "--device", "cpu", "--batch-size", "2"])

    store = Store(db)
    assert sorted(store.client.list_collections()) == ["YOLOX_military_assets", "YOLOX_person_detector"]
    assert store.count("YOLOX_military_assets") == 3  # re-run is idempotent, broken image skipped
    assert store.count("YOLOX_person_detector") == 3
