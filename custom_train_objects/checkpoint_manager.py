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
import copy
import logging

from functools import cache

from utils import load_json, dump_json

logger = logging.getLogger(__name__)

_default_ckpt_format    = 'ckpt'
_empty_checkpoint_infos = {
    'counter' : 0, 'loaded' : -1, 'checkpoints' : [], 'best_checkpoint' : {}
}

class CheckpointManager:
    """ This class manages the checkpoint for all the tracked models """
    def __init__(self, core, max_to_keep = 3, checkpoint_format = _default_ckpt_format):
        """
            Initializes a checkpoint manager
            
            Arguments :
                - core  : any data structure (typically a `BaseModel` instance)
                    The object must have the following attributes :
                        - save_dir  : the saving directory
                        - epochs    : the current number of training epochs
                        - steps     : the current number of training steps
                - max_to_keep   : the number of checkpoint files to keep (older are removed)
                - checkpoint_format : checkpoint filename format
                    The file format is formatted with the following keys :
                        - epochs    : the number of training epochs
                        - steps     : the number of training steps
                        - counter   : the number of already saved checkpoints
                        - model     : the model name (the `key` provided in the `add` call)
        """
        self.core   = core
        self.max_to_keep    = max_to_keep
        self.checkpoint_format  = checkpoint_format
        
        self._state = load_json(
            self.checkpoint_file, default = copy.deepcopy(_empty_checkpoint_infos)
        )
        self._models = {}
    
    step    = property(lambda self: self.core.steps)
    epoch   = property(lambda self: self.core.epochs)
    directory   = property(lambda self: self.core.save_dir)
    best_checkpoint_path    = property(
        lambda self: os.path.join(self.directory, 'best.weights.h5')
    )
    
    counter = property(lambda self: self._state['counter'])
    loaded  = property(lambda self: self._state['loaded'])
    checkpoints = property(lambda self: self._state['checkpoints'])
    latest_checkpoint_infos = property(lambda self: self[-1])
    
    track_multiple_models   = property(lambda self: len(self._models) > 1)
    
    @property
    def checkpoint_file(self):
        return os.path.join(self.directory, 'checkpoint.json')

    @property
    def infos(self):
        return {'epoch' : self.epoch, 'step' : self.step, 'counter' : self.counter}
    
    @property
    def latest_checkpoint(self):
        files = [self.get_filename(k, self[-1]) for k in self._models.keys()]
        return files if len(files) > 1 else files[0]
    
    @property
    def loaded_checkpoint(self):
        if self.loaded == 'best': return self.best_checkpoint
        files = [self.get_filename(k, self[self.loaded]) for k in self._models.keys()]
        return files if len(files) > 1 else files[0]
    
    @property
    def best_checkpoint(self):
        if os.path.exists(self.best_checkpoint_path):
            return self.best_checkpoint_path
        return self.latest_checkpoint
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        if not self.checkpoints:
            return '<CheckpointManager directory={} no checkpoint created>'.format(self.directory)
        loaded = self.loaded
        if isinstance(loaded, int):
            loaded, infos = loaded + 1, self[loaded]
        elif loaded == 'best':
            infos = self._state['best_checkpoint'] if self._state['best_checkpoint'] else self[-1]
        
        return '<CheckpointManager directory={} loaded={} ({} saved) (epoch {})>'.format(
            self.directory, loaded, len(self), infos['epoch']
        )
    
    def __len__(self):
        """ Return the number of tracked checkpoints """
        return len(self.checkpoints)
    
    def __getitem__(self, idx):
        """ Return the information of the checkpoint at the given index """
        return self.checkpoints[idx]
    
    def __update_state(self):
        if len(self) == self.max_to_keep: self.delete(force = True)
        
        self._state.update({
            'counter'   : self.counter + 1,
            'loaded'    : len(self),
            'checkpoints'   : self.checkpoints + [self.infos]
        })

    def set_best_checkpoint_infos(self, epoch, logs):
        self._state['best_checkpoint'] = {** logs, 'epoch' : epoch}
        self.save_state()

    def add(self, key, model):
        """ Add a new `model` (`keras.Model` instance) to track with the given name (`key`) """
        self._models[key] = model
        setattr(self, key, model)
    
    def get_filename(self, model, infos = None, ckpt_format = None):
        """ Return the effective checkpoint filename with the given configuration """
        if not ckpt_format:
            ckpt_format = standardize_checkpoint_format(
                self.checkpoint_format, self.track_multiple_models
            )
        
        if infos is None: infos = self.infos
        return os.path.join(self.directory, ckpt_format.format(model = model, ** infos))
    
    def save_state(self):
        dump_json(filename = self.checkpoint_file, data = self._state, indent = 4)

    def save(self, directory = None, *, ckpt_format = None, ** kwargs):
        if not directory and not self.checkpoints: os.makedirs(self.directory, exist_ok = True)
        
        files = []
        for name, model in self._models.items():
            if directory:
                filename = os.path.join(director, name + '.keras')
            else:
                filename = self.get_filename(name, ckpt_format = ckpt_format)
            
            logger.info('Save `{}` to {}'.format(name, filename))
            model.save(filename, ** kwargs)
            files.append(filename)
        
        if not directory and not ckpt_format:
            self.__update_state()
            self.save_state()
        return files if len(files) > 1 else files[0]
    
    def load(self, checkpoint = None, *, epoch = None, ** kwargs):
        if epoch is not None:
            for i, ckpt in enumerate(self):
                if ckpt['epoch'] == epoch:
                    checkpoint = i
                    break
            
            if checkpoint is None:
                raise ValueError('No checkpoint found for epoch {}'.format(epoch))
        
        if checkpoint is None:
            checkpoint = self.loaded_checkpoint
        elif isinstance(checkpoint, int):
            self._state['loaded'] = checkpoint
            checkpoint = [self.get_filename(k, self[checkpoint]) for k in self._models]
        elif checkpoint == 'best':
            self._state['loaded'] = 'best'
            checkpoint = self.best_checkpoint
        
        if not isinstance(checkpoint, list): checkpoint = [checkpoint]
        
        for filename, (name, model) in zip(checkpoint, self._models.items()):
            logger.debug('Restoring model `{}` from {}'.format(name, filename))
            if filename.endswith(('.keras', '.weights.h5')):
                model.load_weights(filename, ** kwargs)
            else:
                raise RuntimeError('Unsupported checkpoint file format : {}'.format(filename))
    
    def delete(self, index = 0, force = False):
        """ Deletes the checkpoint at the given `index` (default 0, the oldest checkpoint) """
        if index == self.loaded and self.max_to_keep != 1:
            if not force:
                raise RuntimeError('You are trying to delete the currently loaded checkpoint !')
        
        for k in self._models:
            weight_file = self.get_filename(k, self[index])
            logger.debug('Removing weight file {}'.format(weight_file))
            os.remove(weight_file)
        
        self._state['checkpoints'].pop(index)
        if isinstance(self._state['loaded'], int) and index <= self._state['loaded']:
            self._state['loaded'] -= 1

    def clear(self):
        for i in reversed(range(len(self))):
            if i != self.loaded_index: self.delete(i)
    
@cache
def standardize_checkpoint_format(ckpt_format, multi_models):
    ckpt_format, ext = os.path.splitext(ckpt_format)
    if not ext: ext = '.keras'
    if '{counter' not in ckpt_format: ckpt_format += '-{counter:04d}'
    if multi_models and '{model' not in ckpt_format: ckpt_format += '-{model}'
    return ckpt_format + ext