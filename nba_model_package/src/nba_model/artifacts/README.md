# NBA Model Artifacts

Place the serialized artifacts for the current NBA model in this directory before
building the package. The build step will include everything inside this folder.

Required files:

- `ml_model.joblib`
- `spread_model.joblib`
- `feature_list.json`
- `metadata.json`

Optional:

- `preprocessor.joblib` (if you use a fitted scaler/encoder pipeline)

These files are deliberately omitted from source control — CI should fetch the
latest bundle from the training workflow or artifact storage.

