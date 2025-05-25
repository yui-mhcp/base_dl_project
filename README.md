# :yum: Base Deep Learning project

This repository provides a simple and comprehensive example to use the `BaseModel` interface. This allows to easily create new models to experiment, train, and use the trained model. The primary objective is to facilitate the learning and comprehension of Deep Learning (DL), and provide convenient methods to make your project easier!

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers                   : custom layer implementations
│   ├── transformers             : transformer architecture implementations
│   │   ├── *_arch.py                : concrete transformer models built upon generic transformer blocks
│   │   ├── text_transformer_arch.py : defines the main Transformers blocks
│   │   └── transformer_arch.py      : defines features for text-based Transformers
│   ├── current_blocks.py        : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── generation_utils.py      : utilities for text and sequence generation
│   ├── hparams.py               : hyperparameter management
│   └── simple_models.py         : defines classical models such as CNN / RNN / MLP and siamese
├── custom_train_objects    : custom objects used in training / testing
│   ├── callbacks               : callbacks loading and implementations
│   │   ├── checkpoint_callback.py  : custom ModelCheckpoint sub-class working with the `CheckpointManager`
│   │   └── history_callback.py     : callback for tracking training history
│   ├── generators              : custom data generators
│   ├── losses                  : loss functions
│   │   └── loss_with_multiple_outputs.py   : utility class allowing losses to output additional metrics
│   ├── metrics                 : metrics loading and implementations
│   │   └── text_accuracy.py         : accuracy metrics for text evaluation
│   ├── optimizers              : optimizer loading
│   │   └── lr_schedulers.py         : custom learning-rate schedulers
│   ├── checkpoint_manager.py   : handle model checkpoint management (inspired from `tf.train.CheckpointManager`)
│   └── history.py              : main History class to compute training statistics / track config
├── loggers                 : logging utilities*
├── models                  : main directory for model classes
│   ├── example                 : example implementations
│   │   └── mnist_classifier.py     : example MNIST classifier
│   ├── interfaces              : directories for interfaces classes
│   │   ├── base_audio_model.py        : defines audio processing functions
│   │   ├── base_classification_model.py : defines classification utilities
│   │   ├── base_image_model.py        : defines functions for image processing
│   │   ├── base_model.py              : main BaseModel class
│   │   └── base_text_model.py         : defines functions for text encoding/decoding/processing
│   └── weights_converter.py    : utilities to convert weights between different models
├── tests                   : unit and integration tests*
├── utils                   : utility functions*
├── example_classifier.ipynb    : jupyter notebook with basic classifier example
├── LICENCE                     : project license file
├── README.md                   : this file
└── requirements.txt            : required packages
```

\* Check [the data processing repository](https://github.com/yui-mhcp/data_processing) for more information on these modules.

All projects also contain a `README.md` file, that provides general information on the project features/usage and links (tutorials/projects) related to the topic, and some `example_*.ipynb` notebooks for practical usage examples.

## Installation and usage

See [the installation guide](https://github.com/yui-mhcp/blob/master/INSTALLATION.md) for a step-by-step installation :smile:

Here is a summary of the installation procedure, if you have a working python environment :
1. Clone this repository: `git clone https://github.com/yui-mhcp/base_dl_project.git`
2. Go to the root of this repository: `cd base_dl_project`
3. Install requirements\*: `pip install -r requirements.txt`
4. Open an example notebook and follow the instructions!

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

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```