# :yum: Base Deep Learning project

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/licence-AGPL--3.0-green.svg)
![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)

This repository provides a simple and comprehensive example to use the `BaseModel` interface. This allows to easily create new models to experiment, train, and use the trained model. The primary objective is to facilitate the learning and comprehension of Deep Learning (DL), and provide convenient methods to make your project easier!

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers                   : custom layer implementations
│   ├── transformers             : transformer architecture implementations
│   │   ├── *_arch.py                : concrete transformer models (bart, bert, gpt2, mistral, t5, whisper, ...)
│   │   ├── text_transformer_arch.py : base blocks for text-based Transformers
│   │   └── transformer_arch.py      : generic Transformer blocks
│   ├── current_blocks.py        : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── generation_utils.py      : utilities for text and sequence generation
│   ├── hparams.py               : hyperparameter management
│   └── simple_models.py         : defines classical models such as CNN / RNN / MLP and siamese
├── custom_train_objects    : custom objects used in training / testing
│   ├── callbacks                : callbacks loading and implementations
│   │   ├── checkpoint_callback.py   : custom ModelCheckpoint working with the `CheckpointManager`
│   │   └── history_callback.py      : callback for tracking training history
│   ├── generators               : custom data generators
│   ├── losses                   : loss functions
│   │   └── loss_with_multiple_outputs.py : utility class allowing losses to output additional metrics
│   ├── metrics                  : metrics loading and implementations
│   │   └── text_accuracy.py         : accuracy metrics for text evaluation
│   ├── optimizers               : optimizer loading
│   │   └── lr_schedulers.py         : custom learning-rate schedulers
│   ├── checkpoint_manager.py    : handle model checkpoint management (inspired from `tf.train.CheckpointManager`)
│   └── history.py               : main History class to compute training statistics / track config
├── loggers                 : custom utilities for the `logging` module*
├── models                  : main directory for model classes
│   ├── core                     : the `BaseModel` foundation
│   │   ├── mixins                   : composable features mixed into `BaseModel`
│   │   │   ├── audio_mixin.py           : audio processing functions
│   │   │   ├── checkpoint_mixin.py      : checkpoint / weights management
│   │   │   ├── classification_mixin.py  : classification utilities
│   │   │   ├── image_mixin.py           : image processing functions
│   │   │   ├── processing_mixin.py      : generic input / output processing
│   │   │   ├── text_mixin.py            : text encoding / decoding / processing
│   │   │   └── training_mixin.py        : training loop / `fit` utilities
│   │   ├── utils                    : internal helpers for the model core
│   │   │   ├── describe.py              : model description / summary helpers
│   │   │   ├── migration.py             : legacy config / weights migration
│   │   │   └── saving.py                : model saving / loading helpers
│   │   └── base_model.py            : main `BaseModel` class assembling the mixins
│   ├── example                  : example implementations
│   │   └── mnist_classifier.py      : example MNIST classifier
│   └── weights_converter.py     : utilities to convert weights between different models
├── pretrained_models       : pretrained checkpoints (the example MNIST classifier)
├── tests                   : `pytest` suite mirroring the `utils` / `loggers` / model tree*
├── utils                   : shared data-processing / keras utilities*
├── example_classifier.ipynb    : jupyter notebook with a basic classifier example
├── CHANGELOG.md                : project-specific changelog
├── CONTRIBUTING.md             : contribution guidelines
├── INSTALLATION.md             : shared GPU environment installation guide
├── LICENCE                     : project license file
├── pyproject.toml              : project metadata, dependencies and tooling config
└── README.md                   : this file
```

\* The `loggers`, `utils` and most of the `tests` tree are the shared foundation maintained in [the data_processing repository](https://github.com/yui-mhcp/data_processing) — check it for more information on these modules.

All projects also contain a `README.md` file, that provides general information on the project features/usage and links (tutorials/projects) related to the topic, and some `example_*.ipynb` notebooks for practical usage examples.

## Installation and usage

See [the installation guide](INSTALLATION.md) for a step-by-step setup of the shared GPU environment (NVIDIA driver, CUDA, `mamba` and the deep-learning backends) :smile:

Here is a summary of the installation procedure, once your python environment is ready :
1. Clone this repository : `git clone https://github.com/yui-mhcp/base_dl_project.git`
2. Go to the root of this repository : `cd base_dl_project`
3. Install the package with a backend : `pip install -e .[tf]` (or `pip install -e .[torch]`)
4. Open an example notebook and follow the instructions !

Unlike the pure-`numpy` `data_processing` socle, this project trains real models, so it needs a **keras backend**. Pick the one matching your setup — the extras pull `keras>=3` plus the backend :

```bash
pip install -e .[tf]         # keras + tensorflow
pip install -e .[torch]      # keras + torch
pip install -e .[keras]      # alias of [tf]

# modality / dataset helpers used by the example classifier :
pip install -e .[image]      # utils/image
pip install -e .[datasets]   # utils/datasets (tensorflow-datasets, pandas)
pip install -e .[dev]        # test tooling (pytest & plugins)

# extras can be combined :
pip install -e .[tf,image,datasets]
```

The backend is selected at runtime through the `KERAS_BACKEND` environment variable (e.g., `os.environ['KERAS_BACKEND'] = 'tensorflow'`), as shown at the top of the example notebook.

### Quickstart

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from models import get_pretrained
from models.example import MNISTClassifier
from utils.datasets import get_dataset

# 1. Build a fresh model from the `BaseModel` interface ...
model = MNISTClassifier(input_size = (28, 28, 1), labels = list(range(10)), name = 'my_classifier')
print(model)

# 2. ... or load the provided pretrained checkpoint
model = get_pretrained('mnist_classifier')
model.compile()

# 3. Train on a dataset (here the keras MNIST dataset)
train, valid = get_dataset('mnist', source = 'keras')
model.fit(train, validation_data = valid, epochs = 5, batch_size = 64)
```

See the `example_classifier.ipynb` notebook for the full walkthrough (dataset loading, training, evaluation and prediction). :smile:

### Testing

The tests use [`pytest`](https://docs.pytest.org/) and live in the `tests/` directory, mirroring the `utils` / `loggers` and model tree.

```bash
pip install -e .[dev]        # pytest, pytest-cov, pytest-xdist, pytest-timeout

pytest                       # run the whole suite
pytest -n auto               # run in parallel (pytest-xdist)
pytest -m "not slow"         # skip the slow / heavy tests
pytest --cov                 # run with coverage report
pytest tests/models          # run a single subpackage
```

Tests are annotated with markers (declared in `pyproject.toml`) so the suite adapts to your environment : the `tensorflow`, `torch`, `keras`, `cv2` and `gpu` markers are **auto-skipped** when the corresponding dependency (or hardware) is missing, meaning you can run the tests for the backend you installed without pulling every dependency.

## TO-DO list

- [x] Make the TO-DO list.
- [x] Make examples for dataset loading/processing (moved in the `data_processing` project).
- [x] Make examples for `BaseModel` subclassing.
- [x] Comment the code.
- [ ] Multi-GPU support.
- [x] Support `keras3`
- [ ] Make clear tutorials to extend the base project (e.g., add new architectures/layers/losses/...)

## Tutorials and Learning Resources

### Getting Started with Deep Learning

1. **Online Courses**
   - [Introduction to Deep Learning](https://www.tensorflow.org/resources/learn-ml) by TensorFlow
   - [Fast.ai Practical Deep Learning for Coders](https://course.fast.ai/) - Hands-on approach

2. **Books**
   - [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
   - [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron
   - [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by François Chollet

### Framework-Specific Resources

1. **TensorFlow/Keras**
   - [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
   - [Keras Documentation](https://keras.io/guides/)
   - [TensorFlow Developer Certificate](https://www.tensorflow.org/certificate)

2. **PyTorch**
   - [PyTorch Tutorials](https://pytorch.org/tutorials/)
   - [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) : High-level interface for PyTorch

### Advanced Topics

1. **Computer Vision**
   - [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
   - [PyImageSearch](https://pyimagesearch.com/blog/) : Practical computer vision tutorials

2. **Natural Language Processing**
   - [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
   - [Hugging Face Course](https://huggingface.co/course)

3. **Reinforcement Learning**
   - [Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) by Hugging Face
   - [Spinning Up in Deep RL](https://spinningup.openai.com/) by OpenAI

## Contacts and licence

Contacts:
- **Mail**: `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)**: yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENCE](LICENCE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp,
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```
