from pathlib import Path

import pytest

from src.data.validation import validate_yolo_dataset


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"

    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    (image_dir / "image_001.jpg").write_bytes(b"image")
    (label_dir / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.3\n",
        encoding="utf-8",
    )

    return tmp_path


def test_valid_yolo_annotation(dataset_dir: Path) -> None:
    reports = validate_yolo_dataset(
        dataset_dir,
        class_names=["person"],
        splits=("train",),
    )

    assert reports[0].is_valid
    assert reports[0].image_count == 1
    assert reports[0].label_count == 1


def test_missing_label_is_reported(dataset_dir: Path) -> None:
    (dataset_dir / "images" / "train" / "image_002.jpg").write_bytes(b"image")

    reports = validate_yolo_dataset(
        dataset_dir,
        class_names=["person"],
        splits=("train",),
    )

    assert not reports[0].is_valid
    assert any("Missing label" in issue for issue in reports[0].issues)


def test_invalid_class_id_is_reported(dataset_dir: Path) -> None:
    label_path = dataset_dir / "labels" / "train" / "image_001.txt"
    label_path.write_text("2 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    reports = validate_yolo_dataset(
        dataset_dir,
        class_names=["person"],
        splits=("train",),
    )

    assert any("invalid class ID" in issue for issue in reports[0].issues)


def test_out_of_bounds_box_is_reported(dataset_dir: Path) -> None:
    label_path = dataset_dir / "labels" / "train" / "image_001.txt"
    label_path.write_text("0 0.1 0.5 0.5 0.3\n", encoding="utf-8")

    reports = validate_yolo_dataset(
        dataset_dir,
        class_names=["person"],
        splits=("train",),
    )

    assert any("extends outside" in issue for issue in reports[0].issues)


def test_missing_dataset_directory_raises_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_yolo_dataset(
            tmp_path / "missing",
            class_names=["person"],
            splits=("train",),
        )
