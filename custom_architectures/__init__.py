# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import json
import keras

from utils import import_objects, get_object, print_objects, partial, dispatch_wrapper
from .simple_models import classifier, perceptron, simple_cnn

_architectures = {
    'perceptron'    : perceptron,
    'simple_cnn'    : simple_cnn,
    ** import_objects(
        __package__.replace('.', os.path.sep),
        filters = lambda name, val: name[0].isupper() and 'current_blocks' not in val.__module__,
        classes = keras.Model
    ),
    ** {
        k : partial(classifier, v)
        for k, v in import_objects(keras.applications, types = type).items()
    }
}
globals().update(_architectures)

@dispatch_wrapper(_architectures, 'architecture')
def get_architecture(architecture, * args, ** kwargs):
    return get_object(
        _architectures, architecture, * args, print_name = 'architecture', ** kwargs
    )

def print_architectures():
    print_objects(_architectures, 'model architectures')

"""custom_objects = _activations.copy()
_custom_architectures = {}

#__load()

_keras_architectures = {
    'densenet121'       : partial(classifier, keras.applications.DenseNet121),
    'densenet169'       : partial(classifier, keras.applications.DenseNet169),
    'densenet201'       : partial(classifier, keras.applications.DenseNet201),
    'inceptionresnetv2' : partial(classifier, keras.applications.InceptionResNetV2),
    'inceptionv3'       : partial(classifier, keras.applications.InceptionV3),
    'mobilenet'         : partial(classifier, keras.applications.MobileNet),
    'mobilenetv2'       : partial(classifier, keras.applications.MobileNetV2),
    'nasnetlarge'       : partial(classifier, keras.applications.NASNetLarge),
    'resnet50'          : partial(classifier, keras.applications.ResNet50),
    'resnet50v2'        : partial(classifier, keras.applications.ResNet50V2),
    'resnet101'         : partial(classifier, keras.applications.ResNet101),
    'resnet101v2'       : partial(classifier, keras.applications.ResNet101V2),
    'resnet152'         : partial(classifier, keras.applications.ResNet152),
    'resnet152v2'       : partial(classifier, keras.applications.ResNet152V2),
    'vgg16'             : partial(classifier, keras.applications.VGG16),
    'vgg19'             : partial(classifier, keras.applications.VGG19),
    'xception'          : partial(classifier, keras.applications.Xception)
}

architectures = {**_keras_architectures, **_custom_architectures}"""