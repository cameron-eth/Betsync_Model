# Betsync NBA Model Package

This directory houses the distributable bundle for the NBA hybrid model. The package will be built and published whenever a new model version is cut, allowing the inference workers and scheduled actions to simply pin a version such as `betsync-nba-model==2025.11.0`.

## Layout

```
nba_model_package/
  pyproject.toml          # Packaging metadata
  src/nba_model/
    __init__.py
    inference.py          # load_bundle(), prepare_features(), predict()
    metadata.py           # helpers for version + artifact lookup
    artifacts/            # Serialized model + transformers (ignored in git)
      README.md
  tests/                  # Unit tests run before publishing
```

## Updating Artifacts

1. Place the freshly trained artifacts into `src/nba_model/artifacts/`:
   - `ml_model.joblib`
   - `spread_model.joblib`
   - `preprocessor.joblib` (optional, if using pipelines)
   - `feature_list.json`
   - `metadata.json`
2. Bump the version in `pyproject.toml`.
3. Run `pytest` to confirm the bundle loads and predicts.
4. Build and publish with `python -m build` followed by `twine upload` or equivalent registry push.

Model artifacts are intentionally excluded from source control; CI should source them from the training workflow or artifact storage.

