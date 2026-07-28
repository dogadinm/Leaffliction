from pathlib import Path

import numpy as np
import pytest

from leaffliction.io import save_image

import predict


def test_main_errors_without_a_traceback_on_an_unreadable_image(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")

    code = predict.main([str(corrupt)])

    assert code == 1
    assert "cannot read image" in capsys.readouterr().err


def test_main_errors_when_no_trained_model_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    image = tmp_path / "leaf.jpg"
    save_image(np.zeros((16, 16, 3), dtype=np.uint8), image)
    missing_model_dir = tmp_path / "model"

    code = predict.main([str(image), "--model-dir", str(missing_model_dir)])

    assert code == 1
    assert "run train.py first" in capsys.readouterr().err


def test_main_rejects_an_unknown_transform_kind_via_argparse(tmp_path: Path) -> None:
    image = tmp_path / "leaf.jpg"
    save_image(np.zeros((16, 16, 3), dtype=np.uint8), image)

    with pytest.raises(SystemExit):
        predict.main([str(image), "--transform", "NotARealTransform"])


def test_predict_picks_the_highest_probability_label() -> None:
    labels = {"0": "Apple_healthy", "1": "Apple_scab"}
    config = {"img_size": 4}

    class FakeModel:
        def predict(self, batch, verbose=0):
            return np.array([[0.1, 0.9]])

    img = np.zeros((4, 4, 3), dtype=np.uint8)

    label, confidence = predict.predict(FakeModel(), labels, config, img)

    assert label == "Apple_scab"
    assert confidence == pytest.approx(0.9)
