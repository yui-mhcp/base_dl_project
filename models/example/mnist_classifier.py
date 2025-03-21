# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import numpy as np

from ..interfaces import BaseModel
from utils.keras import TensorSpec, ops
from utils import plot_multiple, plot_embedding
from utils.image import display_image, build_gif, load_image

class MNISTClassifier(BaseModel):
    _default_loss   = 'crossentropy'
    _default_metrics    = ['accuracy']
    
    def __init__(self, input_size, labels, ** kwargs):
        self.input_size = tuple(input_size)
        self.labels   = list(labels)
        
        super().__init__(** kwargs)
    
    def build(self, model = None, ** kwargs):
        if model is None:
            model = {
                'architecture'  : 'simple_cnn',
                'input_shape'   : self.input_size,
                'output_shape'  : len(self.labels),

                'n_conv'    : 2,
                'filters'   : 16,
                'kernel_size'   : 3,
                'strides'       : 1,
                'pooling'       : 'max',

                'dense_as_final'    : True,

                'n_dense'   : 1,
                'dense_size'    : 64,

                'final_activation'  : 'softmax',
                'final_name'        : 'classification_layer',

                'name'  : 'cnn_classifier',
                ** kwargs
            }
            
        super().build(model = model)
        
    @property
    def output_signature(self):
        return TensorSpec(shape = (None,), dtype = 'int32')
    
    @property
    def encoder(self):
        import keras
        
        if not isinstance(self.model, keras.Sequential):
            return keras.Model(self.inputs, self.layers[-1].output, name = 'feature_extractor')
        
        return keras.Sequential(self.layers[:-1], name = 'feature_extractor')
    
    def __str__(self):
        return super().__str__() + '- Labels : {}\n'.format(self.labels)
    
    def prepare_input(self, filename, ** kwargs):
        return load_image(
            filename, size = self.input_size[:2], dtype = 'float32', channels = 1, ** kwargs
        )
    
    def prepare_output(self, data, ** _):
        if isinstance(data, dict): data = data['label']
        return ops.cast(data, 'int32')
        
    def predict(self, images):
        if ops.is_array(images):
            if ops.rank(images) == 2:
                images = images[None, :, :, None]
            elif ops.rank(images) == 3:
                images = images[None]
        
        images = ops.stack([self.get_input(image) for image in images], axis = 0)
        
        probs = self(images, training = False)
        preds = ops.argmax(probs, axis = -1)
        
        return [
            (self.labels[c], prob[c], {l : prob[i] for i, l in enumerate(self.labels)})
            for c, prob in zip(ops.convert_to_numpy(preds), ops.convert_to_numpy(probs))
        ]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'labels'    : self.labels,
            'input_size'    : self.input_size
        })
        return config
        