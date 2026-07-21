# :yum: CHANGELOG :yum:

This file tracks the major updates of the [base_dl_project](https://github.com/yui-mhcp/base_dl_project) project. For the global, cross-project overview of all the `yui-mhcp` repositories, see the [central CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) :smile:

## [v1.0.0] - 21/07/2026 - Major refactoring release ! :yum:

First tagged release (`v1.0.0`) following a large refactoring of the `BaseModel` foundation, built on top of the shared `utils` / `loggers` socle (see the [data_processing CHANGELOG](https://github.com/yui-mhcp/data_processing/blob/main/CHANGELOG.md)) !

### Major updates

- The installation now relies on `pyproject.toml` instead of `requirements.txt` : install with `pip install -e .` and pick a keras backend extra (`tf` or `torch`), plus the modality / dataset helpers you need (`image`, `datasets`, `dev`)
- The backend is no longer hard-coded : it is selected at runtime through the `KERAS_BACKEND` environment variable, and the base install ships no backend
- The test suite has been migrated to `pytest`, mirroring the `utils` / `loggers` and model tree, with auto-skipped markers (`tensorflow`, `torch`, `keras`, `cv2`, `gpu`, `slow`, ...)
- The version is single-sourced in `__version__.py`
- **[BREAKING CHANGE]** the model interfaces have been fully restructured : the old `models/interfaces/base_*_model.py` classes are gone. `BaseModel` now lives in `models/core/base_model.py` and is composed from small, reusable **mixins** in `models/core/mixins/` (`audio`, `image`, `text`, `classification`, `processing`, `checkpoint`, `training`)
- **[BREAKING CHANGE]** model-core internals (saving / loading, legacy migration, description) have moved to `models/core/utils/{saving, migration, describe}.py`

### Minor updates

- New `models/example/mnist_classifier.py` (`MNISTClassifier`) and a matching `pretrained_models/mnist_classifier` checkpoint, loadable via `models.get_pretrained`
- The `example_classifier.ipynb` notebook has been reworked around the new `BaseModel` interface
- `custom_train_objects` cleanup : callbacks (`checkpoint_callback`, `history_callback`), losses, metrics, optimizers (`lr_schedulers`), generators and the `CheckpointManager` / `History` classes
- The `architectures` module keeps the generic layers, transformer architectures (`bart`, `bert`, `gpt2`, `mistral`, `t5`, `whisper`, ...) and the `simple_models` helpers
