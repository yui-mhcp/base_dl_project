# :yum: Base Deep Learning project

This repository provides a simple and comprehensive example to use the `BaseModel` interface. This allows to easily create new models to experiment, train, and use the trained model. The primary objective is to facilitate the learning and comprehension of Deep Learning (DL), and provide convenient methods to make your project easier !

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

## Project structure

```bash
├── custom_architectures    : utilities to load custom architectures
│   ├── transformers_arch       : main directory defining the Transformers architectures
│   │   ├── {...}_arch.py           : concrete transformer model, built upon the generic transformer blocks
│   │   ├── text_transformer_arch.py    : defines the main Transformers blocks
│   │   └── transformer_arch.py         : defines features for text-based Transformers
│   ├── current_blocks.py   : defines some blocks that are currently used (e.g., Conv + BN + ReLU)
│   └── simple_models.py    : defines some classical models such as CNN / RNN / MLP and siamese
├── custom_layers       : directory for the custom layers (such as MHA / custom activations)
├── custom_train_objects    : custom objects used in training / testing
│   ├── callbacks       : callbacks loading
│   │   └── checkpoint_callback.py  : custom ModelCheckpoint sub-class working with the `CheckpointManager`
│   ├── generators      : custom data generators (used in some projects)
│   ├── losses          : losses loading
│   │   └── loss_with_multiple_outputs.py   : utlity class allowing losses to output additional metrics
│   ├── metrics         : metrics loading
│   ├── optimizers      : optimizer loading
│   │   └── lr_schedulers.py    : custom learning-rate schedulers
│   └── checkpoint_manager.py   : handle model checkpoint management (inspired from `tf.train.CheckpointManager`)
│   └── history.py      : main History class to compute training statistics / track config
├── docker              : directory for the `docker` scripts
├── loggers             : *
├── models              : main directory for model's classes'
│   ├── classification      : example for BaseModel subclassing
│   │   └── mnist_classifier.py : example MNIST classifier
│   ├── interfaces          : directories for interfaces' classes'
│   │   ├── runtime             : an experimental module to support custom inference runtimes
│   │   ├── base_audio_model.py     : defines many audio processing functions
│   │   ├── base_classification_model.py    : defines classification utilities
│   │   ├── base_embedding_model.py : defines embedding-based utilities
│   │   ├── base_image_model.py     : defines functions for image processing
│   │   ├── base_model.py           : main BaseModel class
│   │   └── base_text_model.py      : defines functions for text encoding / decoding / processing
│   ├── saving.py           : utilities functions on models
│   ├── model_utils.py      : utilities functions on models
│   └── weights_converter.py    : utilities to convert weights from 2 different models
├── pretrained_models   : main directory where all trained models are saved
├── unitests        : *
├── utils           : *
├── example_classifier.ipynb
└── example_classifier_2.ipynb
```

\* Check [the data processing repository](https://github.com/yui-mhcp/data_processing) for more information on these modules. 

All projects also contain a `README.md` file, that provides general information on the project features / usage, some links (tutorials / projects) related to the topic, and some `example_*.jpynb` notebooks for practical usage examples. 

## Installation and usage

Check [this installagion guide](https://github.com/yui-mhcp/yui-mhcp/blob/main/INSTALLATION.md) for the step-by-step instructions !

## TO-DO list

- [x] Make the TO-DO list.
- [x] Make examples for dataset loading / processing (moved in the `data_processing` project).
- [x] Make examples for `BaseModel` subclassing.
- [x] Comment the code.
- [ ] Multi-GPU support.
- [ ] Make tutorials to extend the project
    - [x] Add new architectures
    - [x] Add new training objects
    - [x] Add new datasets
    - [ ] Add new models' classes
- [x] Add `Dockerfile` scripts (experimental)
- [x] Support `keras3`
    - [x] Migrate the `train` custom features to be handled by the `fit` method
    - [x] Clean the code
    - [x] Make a custom `CheckpointManager` to support all backends
    - [x] Make the `History` and `CheckpointManager` classes compatible with `fit`
    - [x] Improve the processing pipeline methods
    - [x] Support all keras losses / optimizers / callbacks / metrics in the `compile` and `fit` methods
    - [ ] Add the `PredictorCalback` feature
    - [ ] Convert the available datasets to be compatible with the new `utils/datasets` module
    - [ ] Make the `evaluate` method compatible with the processing pipeline / `History` class etc.
- [ ] Make clear tutorials to extend the base project (e.g., add new architectures / layers / losses / ...)
- [ ] Add more references to start learning ML / DL

## Contacts and licence

Contacts :
- **Mail** : `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)** : yui0732

### Terms of use

The goal of these projects is to support and advance education and research in Deep Learning technology. To facilitate this, all associated code is made available under the [GNU Affero General Public License (AGPL) v3](AGPLv3.licence), supplemented by a clause that prohibits commercial use (cf the [LICENCE](LICENCE) file).

These projects are released as "free software", allowing you to freely use, modify, deploy, and share the software, provided you adhere to the terms of the license. While the software is freely available, it is not public domain and retains copyright protection. The license conditions are designed to ensure that every user can utilize and modify any version of the code for their own educational and research projects.

If you wish to use this project in a proprietary commercial endeavor, you must obtain a separate license. For further details on this process, please contact me directly.

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project, or make a Pull Request to solve it :smile: 

### Citation

If you find this project useful in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references 

Tutorials :
- [tensorflow's tutorials](https://tensorflow.org/tutorials) : list of tensorflow's tutorials covering multiple topics. Some of the proposed projects / features are inspired from these tutorials (such as the `tf.data.Dataset` pipeline builder `prepare_dataset`). 

Papers :
- [Attention is all you need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) : original paper introducing the `WarmupScheduler` and `Transformer` architecture. 
