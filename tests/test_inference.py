from __future__ import annotations

from melanoma_ia.schemas import PredictionResult


def test_predict_returns_valid_result(engine, sample_image):
    result = engine.predict(sample_image)

    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.probability <= 1.0
    assert result.label in {"malignant", "benign"}
    assert result.threshold == engine.settings.threshold


def test_predict_label_matches_threshold(engine, sample_image):
    result = engine.predict(sample_image)
    expected_label = "malignant" if result.probability >= result.threshold else "benign"
    assert result.label == expected_label


def test_preprocess_output_is_batched_tensor(engine, sample_image):
    tensor = engine.preprocess(sample_image)
    assert tensor.dim() == 4 
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3


def test_to_dict_is_json_serializable(engine, sample_image):
    import json

    result = engine.predict(sample_image)
    json.dumps(result.to_dict()) 
