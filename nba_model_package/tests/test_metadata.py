from nba_model import metadata


def test_load_metadata_without_file(tmp_path, monkeypatch):
    # Point artifact directory to an empty tmp path
    monkeypatch.setattr(metadata, "ARTIFACT_DIR", tmp_path)
    meta = metadata.load_metadata()
    assert meta.version == "0.0.0"
    assert meta.extra == {}


def test_require_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, "ARTIFACT_DIR", tmp_path)
    try:
        metadata.require_artifact("missing.joblib")
    except FileNotFoundError as exc:
        assert "missing.joblib" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected FileNotFoundError")

