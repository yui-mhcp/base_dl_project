
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import tensorflow as tf

from datasets import set_dataset_dir, get_dataset
from utils import parse_args, get_object, set_display_options
from models import _models, get_pretrained, get_models
from models.model_utils import is_model_name, compare_models, remove_training_checkpoint
from loggers import add_handler, set_level

logger  = logging.getLogger(__name__)

TOKEN = None

_default_args = ['mode']

set_level('info')

def _get_config(* args, ** kwargs):
    args = [a for a in args if a not in kwargs]
    
    config = parse_args(* _default_args + args, add_unknown = True)
    for a in _default_args: config.pop(a)
    
    for k, v in kwargs.items(): config.setdefault(k, v)
    
    logger.info('Config : {}'.format(config))
    
    return config

def get_strategy(multi_gpu, assert_gpu = False):
    gpus = tf.config.list_physical_devices('GPU')
    
    logger.info('tensorflow version : {}'.format(tf.__version__))
    logger.info('# of GPU : {}'.format(len(gpus)))
    
    if assert_gpu:
        assert len(gpus) > 0, 'No GPU detected !'
    
    if multi_gpu == -1:
        return tf.distribute.MirroredStrategy() if multi_gpu and len(gpus) > 1 else tf.distribute.get_strategy()
    
    if not isinstance(multi_gpu, list): multi_gpu = [multi_gpu]
    tf.config.set_visible_devices([gpus[idx] for idx in multi_gpu], 'GPU')
    logger.info('# of visible GPU : {}'.format(len(tf.config.list_logical_devices('GPU'))))
    return tf.distribute.MirroredStrategy() if len(multi_gpu) > 1 else tf.distribute.get_strategy()

def build(** kwargs):
    config = _get_config('class', 'nom', multi_gpu = -1, ** kwargs)
    
    strategy = get_strategy(config.pop('multi_gpu'), assert_gpu = False)
    
    with strategy.scope():
        if 'pretrained_name' in config:
            if is_model_name(config['pretrained_name']):
                model = _models.get(config.pop('class')).from_pretrained(** config)
                logger.info(model)
                return model
            
            logger.warning('Pretrained model {} does not exists !'.format(config['pretrained_name']))
        model = get_object(
            _models, config.pop('class'), err = True, print_name = 'Model class', ** config
        )
        logger.info(model)
        return model

def train(** kwargs):
    from datasets import prepare_dataset, test_dataset_time
    
    config      = _get_config('nom', dataset = None, multi_gpu = -1, dataset_dir = None, ** kwargs)
    strategy    = get_strategy(config.pop('multi_gpu'), assert_gpu = True)

    ds_dir = config.pop('dataset_dir', None)
    if ds_dir: set_dataset_dir(ds_dir)
    dataset = get_dataset(config.pop('dataset'), ** config.pop('dataset_config', {}))
    
    with strategy.scope():
        model   = get_pretrained(config.pop('nom'))

        model.compile(** config.pop('compile_config', {}))

        if TOKEN: add_handler('telegram', token = TOKEN)
        
        hist = model.train(dataset, ** config)
    logger.info(hist)
    
    return hist

def test(** kwargs):
    config      = _get_config(
        'nom', dataset = None, multi_gpu = -1, dataset_dir = None, metrics = None, ** kwargs
    )
    strategy    = get_strategy(config.pop('multi_gpu'), assert_gpu = True)
    
    ds_dir      = config.pop('dataset_dir', None)
    if ds_dir: set_dataset_dir(ds_dir)
    dataset     = get_dataset(
        config.pop('dataset'), modes = ['valid'], ** config.pop('dataset_config', {})
    )
    

    with strategy.scope():
        model   = get_pretrained(config.pop('nom'))
        
        if TOKEN: add_handler('telegram', token = TOKEN)

        hist = model.test(dataset, ** config)
    logger.info(hist)
    
    return hist

def predict(** kwargs):
    config      = _get_config(
        'nom', dataset = None, multi_gpu = -1, dataset_dir = None, metrics = None, ** kwargs
    )
    strategy    = get_strategy(config.pop('multi_gpu'), assert_gpu = True)

    ds_dir      = config.pop('dataset_dir', None)
    if ds_dir: set_dataset_dir(ds_dir)
    dataset     = get_dataset(
        config.pop('dataset'), modes = ['valid'], ** config.pop('dataset_config', {})
    )
    
    logging.info('Dataset length : {}'.format(len(dataset)))

    with strategy.scope():
        model   = get_pretrained(config.pop('nom'))
        
        if TOKEN: add_handler('telegram', token = TOKEN)

        pred = model.predict(dataset, ** config)
    
    return pred


def compare():
    pattern = _get_config(patterns = '').get('pattern', None)

    names   = get_models(pattern)

    infos = compare_models(names, True, True, epoch = 'best', add_training_config = True)
    
    set_display_options(columns = len(infos.columns), rows = len(infos))
    print(infos.sort_values('val_loss'))

def experiments():
    if TOKEN: add_handler('telegram', token = TOKEN)

    run_experiments()

def clean_checkpoints():
    pattern = _get_config(patterns = '').get('pattern', None)

    names = get_models(pattern)
    logger.info('Models corresponding to {} :\n- {}'.format(pattern, '\n- '.join(names)))
    for n in names:
        remove_training_checkpoint(n)
    
def run_unitests():
    import unitest.tests
    
    from unitest import run_tests
    
    config = parse_args(to_run = 'all')
    
    run_tests(** config).assert_succeed()

    
_modes  = {
    'build' : build,
    'train' : train,
    'test'  : test,
    'predict'   : predict,
    'compare'   : compare,
    'unitest'   : run_unitests,
    'clean'     : clean_checkpoints,
    'experiments'   : experiments
}

if __name__ == '__main__':
    mode   = parse_args(* _default_args)['mode']
    
    get_object(_modes, mode, err = True, print_name = 'Run mode')
