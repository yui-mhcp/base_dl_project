
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
"""
This script automates experiments based on model's names : the name should contain all the information to build the model (determine its configuration), train and test it. 
It allows to automate multiple training in a single command simply by passing multiple names. 

Example for a BART Q&A fine-tuning experiment : 
```
python3 main.py experiments --test --test_name test_nq_top5 --pred --pred_name pred_squad --names bart_nq bart_nq_coqa_newsqa
```

In this case, you can observe : 
    --test  : tells to perform model's testing
    --test_name test_nq_top5    : gives a name to the test and `nq` can represent the Natural Questions dataset while `top5` can represent the usage of Beam Search decoding
        Note that these information should be returned by the `testing_config_from_name()` method
    
    --pred  : tells to perform prediction
    --pred_name pred_squad  : gives a name to the prediction and `squad` can represent the SQUAD dataset
        These information should be returned by the `predict_config_frop_name` method
    
    --names : gives the model's names to build, train and test
    - bart_nq / bart_nq_coqa_newsqa : the model's names. The 1st one should be trained on NQ while the second one should be trained on [NQ, CoQA, newsQA] datasets
        The datasets' information should be returned in the `training_config_from_name` method
        The `bart_` prefix can tells to use a BART-based model (`AnswerGenerator` class) which should be returned in the `config_from_name` method
"""

import os
import glob
import json
import logging
import subprocess
import tensorflow as tf

from utils import parse_args
from models.model_utils import _pretrained_models_folder, get_model_history, get_models, is_model_name

logger  = logging.getLogger(__name__)

PRED_DIR    = os.path.join('__predictions')

def config_from_name(model_name, ** kwargs):
    raise NotImplementedError()

def training_config_from_name(model_name, retraining = False, ** kwargs):
    raise NotImplementedError()

def testing_config_from_name(model_name, test_name, ** kwargs):
    raise NotImplementedError()

def predict_config_from_name(model_name, pred_name, ** kwargs):
    raise NotImplementedError()

def _run_command(mode, * args, ** config):
    config = config_to_list(config)
    
    call_args   = ['python3', 'main.py', mode] + list(args) + config
    logger.info('Call arguments : `{}`'.format(' '.join(call_args)))
    
    return subprocess.run(call_args)

def config_to_list(config):
    config_list = []
    for k, v in config.items():
        config_list.append('--{}'.format(k))
        if not isinstance(v, (list, tuple)): v = [v]
        config_list.extend([json.dumps(vi) if not isinstance(vi, str) else vi for vi in v])
    
    return config_list

def run_experiments(names = [], ** kwargs):
    logger.info('tensorflow version : {}\n# GPU : {}'.format(
        tf.__version__, len(tf.config.list_physical_devices('GPU'))
    ))
    tf.config.set_visible_devices([], 'GPU')
    
    default_config = parse_args('mode', add_unknown = True)
    default_config.pop('mode')

    pred        = default_config.pop('pred', False)
    pred_name   = None if not pred else default_config.pop('pred_name', 'pred')

    testing     = default_config.pop('test', False)
    test_name   = None if not testing else default_config.pop('test_name', 'test')
    
    overwrite   = default_config.pop('overwrite', False)
    
    names       = default_config.pop('names', names)
    allow_retraining    = default_config.pop('retrain', False)
    if not isinstance(names, (list, tuple)):
        names = get_models(names) if '*' in names else [names]
    
    logger.info('Pred : {} ({}) - Test : {} ({})\nNames :\n{}\nConfig :\n{}'.format(
        pred, pred_name, testing, test_name, '\n'.join(names),
        '\n'.join(['- {}\t: {}'.format(k, v) for k, v in default_config.items()])
    ))
    for name in names:
        success = build_and_train(name, allow_retraining, ** default_config)
        
        if testing and success:
            success = test_model(name, test_name, overwrite = overwrite, ** default_config)
        
        if pred and success:
            success = pred_model(name, pred_name, overwrite = overwrite, ** default_config)
        
        if not success:
            break

    
def build_and_train(name, allow_retraining, ** default_config):
    hist = get_model_history(name)

    retraining = False
    if hist is not None and len(hist) > 0:
        logger.info('Model {} has already been trained, {}'.format(
            name, "retraining it for 1 epoch" if allow_retraining else "skipping it."
        ))
        if not allow_retraining: return True
        retraining = True
    
        
    if not is_model_name(name):
        err = _run_command('build', ** config_from_name(name, ** default_config))
    
        if err.returncode:
            logger.error('Error when building model {} (status {})'.format(name, err.returncode))
            return False
    
    err = _run_command('train', name, ** training_config_from_name(name, retraining, ** default_config))

    if err.returncode:
        logger.error('Error when training model {} (status code {})'.format(name, err.returncode))
        return False

    logger.info('Successfully built and trained {} !'.format(name))
    return True

def test_model(name, test_name, overwrite = False, ** default_config):
    hist = get_model_history(name)

    suffix = '_EM'
    if 'top5' in test_name: suffix += '-1'
    if hist is None:
        logging.warning('Model {} has not been trained yet, skip its test !'.format(name))
        return True
    elif not is_model_name(name):
        logging.warning('Model {} does not exist, skip its test !'.format(name))
        return True
    elif hist.contains(test_name + suffix):
        if not overwrite:
            logger.info('Test {} for {} already done !'.format(test_name, name))
            return True
        
        logger.info('Overwriting test {}'.format(test_name))
        hist.pop(test_name)
        hist.save()
    
    err = _run_command('test', name, ** testing_config_from_name(name, test_name, ** default_config))

    if err.returncode:
        logger.error('Error when testing model {} (status {})'.format(name, err.returncode))
        return False

    logger.info('Successfully tested {} !'.format(name))
    return True

def pred_model(name, pred_name, overwrite = False, ** default_config):
    hist = get_model_history(name)
    
    map_file    = os.path.join(PRED_DIR, name, pred_name + '.json')
    if hist is None:
        logger.warning('Model {} has not been trained yet, skip its prediction !'.format(name))
        return True
    elif not is_model_name(name):
        logger.warning('Model {} does not exist, skip its prediction !'.format(name))
        return True
    elif os.path.exists(map_file):
        if not overwrite:
            logger.info('Pred {} for {} already done !'.format(pred_name, name))
            return True
        
        logger.info('Overwriting prediction {}'.format(pred_name))
    
    err = _run_command('pred', name, ** predict_config_from_name(
        name, pred_name, overwrite = overwrite, ** default_config
    ))

    if err.returncode:
        logger.error('Error when making prediction for model {} (status {})'.format(
            name, err.returncode
        ))
        return False

    logger.info('Successfully predicted for {} !'.format(name))
    return True
