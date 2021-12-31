import os
import math
import tensorflow as tf

from models.base_model import BaseModel
from utils import plot_multiple, plot_embedding
from utils.image import display_image, build_gif, load_image

class BaseClassifier(BaseModel):
    def __init__(self,
                 input_size,
                 
                 labels,
                 nb_class   = None,
                 multi_class    = False,

                 **kwargs
                ):
        self.input_size = tuple(input_size)
        self.labels   = list(labels)
        self.nb_class = nb_class if nb_class and nb_class >= len(self.labels) else len(self.labels)
        
        self.multi_class    = multi_class
        
        super().__init__(** kwargs)
        
    def _build_model(self, ** kwargs):
        final_activation = 'sigmoid' if self.nb_class <= 2 or self.multi_class else 'softmax'
        config = {
            'architecture_name' : 'simple_cnn',
            'input_shape'   : self.input_size,
            'output_shape'  : 1 if self.nb_class <= 2 else self.nb_class,
            
            'n_conv'    : 2,
            'filters'   : 16,
            'kernel_size'   : 3,
            'strides'       : 1,
            'pooling'       : 'max',
                
            'dense_as_final'    : True,
            
            'n_dense'   : 1,
            'dense_size'    : 64,
            
            'final_activation'  : final_activation,
            'final_name'        : 'classification_layer',
            
            'name'  : 'cnn_classifier',
            ** kwargs
        }
            
        super()._build_model(classifier = config)
        
    @property
    def output_signature(self):
        return tf.TensorSpec(shape = (None,), dtype = tf.int32)
    
    @property
    def encoder(self):
        if not isinstance(self.classifier, tf.keras.Sequential):
            return None
        
        encoder = tf.keras.Sequential(name = 'feature_extractor')
        for l in self.classifier.layers[:-1]:
            encoder.add(l)
        return encoder
    
    def __str__(self):
        des = super().__str__()
        des += "Labels : {}\n".format(self.labels)
        des += "Multi-class : {}\n".format(self.multi_class)
        return des
    
    def compile(self, **kwargs):
        loss = 'binary_crossentropy' if self.multi_class or self.nb_class == 1 else 'sparse_categorical_crossentropy'
        metric = 'binary_accuracy' if self.multi_class or self.nb_class == 1 else 'sparse_categorical_accuracy'
        
        kwargs.setdefault('loss', loss)
        kwargs.setdefault('metrics', [metric])
        
        super().compile(** kwargs)
    
    def preprocess_data(self, data):
        image = load_image(data['image'], target_shape = self.input_size, dtype = tf.float32)
        return image, tf.cast(data['label'], tf.int32)
    
    def _get_train_config(self, * args, test_size = 1, test_batch_size = 256, ** kwargs):
        return super()._get_train_config(
            * args, test_size = test_size, test_batch_size = test_batch_size, ** kwargs
        )
    
    def predict_with_target(self, batch, step, prefix, directory = None,
                            max_pred = 16, **kwargs):
        if directory is None: directory = self.train_test_dir
        prefix = os.path.join(directory, prefix)
        
        inputs, target = batch
        
        pred = self(inputs[:max_pred], training = False)
        
        pred_class = tf.argmax(pred, axis = -1)
        
        kwargs.setdefault('size', 5)
        kwargs.setdefault('show', False)
        kwargs.setdefault('ncols', math.sqrt(max_pred))
        
        titles = [
            'Classs {}\npred {} ({:.2f} %)'.format(t, p_c, 100 * pred[i,p_c])
            for i, (t, p_c) in enumerate(zip(target[:max_pred], pred_class))
        ]
        
        datas = {
            'pred_{}'.format(i) : {
                'x' : inputs[i],
                'title' : titles[i]
            } for i in range(len(titles))
        }
        
        plot_multiple(
            ** datas, ** kwargs, title = 'Prediction at step {}'.format(step),
            filename = prefix + '_pred.png', plot_type = 'imshow', figsize = (5,5)
        )
        
        encoder = self.encoder
        if encoder is not None:
            embedded = encoder(inputs)
            plot_embedding(
                embedded, ids = target, title = 'Feature space at step {}'.format(step),
                filename = prefix + '_embed.png', show = kwargs.get('show', False)
            )
    
    def build_gif(self, show = True, ** kwargs):
        embeddings_gif = build_gif(
            self.train_test_dir, img_name = '*_embed.png', ** kwargs,
            filename = os.path.join(self.train_test_dir, 'embeddings.gif')
        )
        
        prediction_gif = build_gif(
            self.train_test_dir, img_name = '*_pred.png', ** kwargs,
            filename = os.path.join(self.train_test_dir, 'predictions.gif')
        )
        
        if show:
            display_image(embeddings_gif)
            display_image(prediction_gif)
        
        return embeddings_gif, prediction_gif
        
    def predict(self, datas):
        datas = tf.cast(datas, tf.float32)
        if tf.reduce_max(datas) > 1.:
            datas = datas / 255.
        if len(tf.shape(datas)) == 2: datas = tf.expand_dims(datas, axis = -1)
        if len(tf.shape(datas)) == 3: datas = tf.expand_dims(datas, axis = 0)
        pred = self(datas, training = False)
        
        pred_class = tf.argmax(pred, axis = -1)
        
        return [(self.labels[c], pred[i,c]) for i, c in enumerate(pred_class)]
    
    def get_config(self, *args, **kwargs):
        config = super().get_config(*args, **kwargs)
        config['input_size']    = self.input_size
        config['labels']        = self.labels
        config['nb_class']      = self.nb_class
        config['multi_class'] = self.multi_class
        
        return config
        