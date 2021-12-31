# :yum: Create a new project

## Structure of a model folder

When you initialize a `BaseModel` subclass, it automatically creates some folders and configuration files : 
```bash
pretrained_models/mnist_classifier/ : the main directory of your model
├── config.json             : the configuration of your model (from `get_config()`)
├── eval/                   : where you can store output or evaluation
├── outputs/                : where you can store output of prediction
├── saving/                 : where all models and config are stored
│   ├── checkpoint          : checkpoint file used by `tensorflow SavedModel`
│   ├── ckpt-1.data-00000-of-00001
│   ├── ckpt-1.index
│   ├── config_models.json      : configuration of models
│   ├── historique.json         : `History` data
│   └── classifier.json         : your model configuration (`classifier` is its variable name)
└── training-logs/
    ├── checkpoints/    : training checkpoints
    └── eval/           : where you can store the `predict_with_target` results
```

## Subclassing `BaseModel`

I will not show an example of subclassing here (you can check the [BaseClassifier](models/classification/base_classifier.py) example for that) but will describe the principal methods you can define / override.

### Properties

#### Main properties

- `input_signature` / `output_signature`    : signatures for input / output. They are used to compile functions in `tensorflow graph` mode.
- `training_hparams`    : if you have custom training hparams to define (do not forget to define the `init_train_config` function).

#### Secondary properties

- `run_eagetly` (default `False`) : whether to run in Eager or Graph mode.

### Methods

#### Main methods 

- `__init__` and `get_config` : 

You can override them in order to add more hyperparameters to your model. Do not forget to add them in the `get_config` in order to allow saving and restoring !

You must also call their respective `super()` method to benefit from all base configuration and builds.

- `_build_model` :

General function to define configuration for model architectures.

The typical structure is to define model's configuration and call `super().build_model()` with keywargs where keys are the model's variable name and value is its configuration (passed to the `get_architecture()` method). Configuration must therefore contain a `architecture_name` field.

You can either define a single model (as in the classifier : `classifier = classifier_config`) or multiple models (for instance in `GAN`'s : `generator = generator_config, discrimintator = discriminator_config` which will create 2 instance variables `generator` and `discriminator`).

- Data processing methods `encode_data`, `filter_data`, `augment_data`, `preprocess_data` and `memory_consuming_fn`

These methods are not necessary but if they are defined, they will be passed at the dataset construction time (see dataset processing for more information on the order of application). 

- `call` : 

You can define it if you need a specific call function (default behaviour is to pass inputs to the `call` method of your main model variable).

In order to be compatible with graph mode and the `input_signature`, you need to pass inputs as a list of tensors (or individual tensor) but not pass them as multiple arguments.

If the `input_signature` is not the same for the `call` inputs, you can define the property `call_signature`.

Note that if your model contains multiple sub models (such as `GAN`'s), you **must** redefine it.

- `train_step`, `eval_step` : 

These methods takes as input a `batch (input, targets)` with `[input_signature, output_signature]` signatures and you cannot change this signature in order to run in graph mode. It means that the output of your latest processing function **must** be a tuple of 2 elements : `inputs` and `targets` with right shapes and types.

Note that inside the `{train / eval}_step` you can make whathever you want so it is not a limitation ! :smile:

- `predict_with_target` : method called by the `PredictorCallback` in order to mnotor the evolution of prediction during trainings.

#### Secondary methods

- `__str__` : if you want to show more information (on custom configuration) in model descriptions.
- `_init_folders`   : only necessary if you want to create new subfolders to save specific elements.
- `init_train_config`  / `update_train_config` : if you have custom training hyper-parameters to define.
- `get_dataset_config` : can be useful if you need padded batch for instance.
- `compile` : if you want to modify the default loss / metrics.
- `predict` : if you want to have more powefrul features for prediction.

### Classical subclassing procedure

#### Define your model configuration

The first most important part is to define the hyper parameters you want in your model. 

For this purpose you should redefine the `__init__` method and call `super().__init__(** kwargs)` at the end.

Note that if you need some directories such as the `save_dir`, you can call the `super().__init__()` before using them because they are created in the `super` function.

Next you can add these custom configuration in the `get_config()` and in the `__str__` one in order to have a good overview when printing your model.

Note that these configuration are more general and are not training hyperparameters. 
To add custom training parameters, you should redefine the `training_hparams` property :
```python
    @property
    def training_hparams(self):
        return super().training_hparams(
            max_input_length    = 150,
            max_output_length   = 1024
        )
```
And define these variables in the `init_train_config()` method. This procedure allows the `History` callback to track them. 

#### Define the model architecture

Now that you have defined your general model configuration, you can define your model architecture.

Quite simple : just redefine the `_build_model()` method and call `super()._build_model(...)` with *kwargs* where keys are your models' variable name and values are their configuration ! (do not forget the `architecture_name` in their config).

The `super` method will create them as instance variable, add them in the `Checkpoint` and save their configuration / weights in the `saving` directory !

Optionnally you can override the `compile` method to specify new default parameters for loss / metrics.

Example (from `BaseClassifier`) :
```python
    def _build_model(self, ** kwargs):
        super()._build_model(classifier = {
            'architecture_name' : 'simple_cnn',
            'input_shape'   : self.input_size,
            'output_shape'  : 1 if self.nb_class <= 2 else self.nb_class,
            
            'final_activation'  : final_activation,
            ** kwargs
        })
```

This method will : 
1. Call `get_architecture()` with value as parameters. 
2. Create a simple `simple_cnn` architecture (which is a `CNN` with single input)
3. Associate it to the `instance variable "classifier"`

#### Define the data processing pipeline

Now that your model is configured and created, you need to define the processing pipeline.

The default behaviour of many dataset processing is to give as output a `pandas.DataFrame` which will be converted to a list of dict in the `prepare_dataset` method. It means that your first method in the processing pipeline will mostly receive a dictionary where keys are columns names.

The final function in the pipeline **must** return a tuple `(inputs, targets)` matching respectivelly `input_signature` and `output_signature` signatures. 

The methods you can define : 
- `encode_data` : often use to convert your data dictionary to a tuple of encoded item (loading the image / audio from its filename, encode text to their index, ...).
- `filter_data` : can be useful to filter too long data that will raises `Out Of Memory errors` (for instance audios longer than a given time)\*.
- `preprocess_data` : usefull to apply time-consuming processing such as normalization.
- `memory_consuming_fn` : usefull for memory-intensive operation which will not be cached.
- `augment_data` : useful for personnalized data augmentation (a `augment_prct` variable is by default created if you want to augment a given proportion of your batches).

\* A good practice is to define these thresholds as training hyper-parameters so that the `History` will keep track of each specified value for each training phase

#### (Optional) Define a prediction monitoring function

Now that your pipeline is done, you can define a `predict_with_target` method in order to make predictions every n steps during trainings. It can be useful to see evolution of prediction during training !

#### (Optional) Define your training step

This step is optional but if you need a custom training step, you can override the `train_step` and `eval_step` methods which take a single `batch` as argument and returns metrics.

Currently metrics should be computed as follow but it will be improved in future version :
```python
    def train_step(self, batch):
        inputs, target = batch

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training = True)

            loss = # Loss computing

        # Gradient application

        return self.update_metrics(target, y_pred)
    
    def eval_step(self, batch):
        inputs, target = batch

        y_pred = self(inputs, training = True)

        return self.update_metrics(target, y_pred)
```

Note that you do not have to return the loss as metric, it is automatically computed and added to metrics !

#### (Optional) Define a funny `predict` method !

Now that all your model is define, it's time to use it for funny applications ! You can define the `predict` function and whathever other function you want to make your model as easy to use as possible ! :smile:

## Instanciate an existing model

This is a really powerful feature of the `BaseModel` class : if a model already exists, just specify its name and it will be loaded !

In practice, you can just type : 
`model = BaseModel(nom = 'mnist_classifier')`
and the model will magically be loaded with all its configuration and models will be restored. :smile:

This even if you have arguments in the constructor without default value, you do not need to specify them !

Furthermore, all models are treated as `Singleton` so that if the model is loaded, you can re-instanciate it the same way and it will just give you the exact same instance so a big gain of time and memory.

## Training

Now that your model is created, you can train it with the `fit` or `train` method and all will be done automatically (dataset preparation, configuration tracking, ...).

The only thing you have to do is to load the dataset and pass it to the function. You can also pass it all training configuration you want such as `epochs`, `batch_size`, ... 

You can even pass different `batch_size` for training and testing if you want to speed up evaluation (because evaluation uses less memory so `valid_batch_size` can be greater than `batch_size`).

## Testing / evaluation

Same as training you have the `evaluate` method which calls the `tensorflow Model.evaluate` method or the `test` method which calls your custom `eval_step` function

Note : currently `evaluate` does not handle correctly the configuration tracking but it will be solved soon !

Note 2 : `test` is called like this because it is for `testing` which is different of `evaluation` !
- `Evaluation` is for evaluating model during its training and evaluate its generalization. However, as you try to fit your `hyperparameters` to have a good score on this particular dataset, it **cannot** be considered as *totally independant* of your model !
- `Testing` is to test your trained model on a different dataset in order to see its generalization on new data (after fine-tuning parameters on the `validation` set).

## Prediction

For prediction you have the `predict` method you defined ! By thefault it just calls the `predict` method of `Model` class but it is your choice to give it a more funny behaviour !

## Analyzing

Now that your model is trained, you can analyze its training with all features of the `History` class. 

For more example, see the [example_classifier](example_classifier.ipynb) notebook to see all available features !