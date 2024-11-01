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

import time
import logging
import threading
import collections

from functools import wraps

from .utils import ContextManager, time_to_string, executing_eagerly

logger = logging.getLogger(__name__)

TIME_LEVEL      = 15
TIME_DEBUG_LEVEL    = 13
TIME_LOGGER_NAME    = 'timer'

_str_indent     = '  '

def create_timer(name):
    return {'name' : name, 'runs' : [], 'start' : -1, 'children' : collections.OrderedDict()}

def timer_to_str(timer, indent = 0, str_indent = _str_indent):
    _indent = indent * str_indent
    
    if len(timer['runs']) > 1:
        _infos = 'executed {} times : {} ({} / exec)'.format(
            len(timer['runs']), time_to_string(sum(timer['runs'])),
            time_to_string(sum(timer['runs']) / len(timer['runs']))
        )
    elif len(timer['runs']) == 0:
        _infos = 'not finished yet'
    else:
        _infos = ': {}'.format(time_to_string(timer['runs'][0]))
    
    des = '{}- {} {}'.format(_indent, timer['name'], _infos)
    for child_name, child_timer in timer['children'].items():
        des += '\n' + timer_to_str(child_timer, indent = indent + 1, str_indent = str_indent)
    return des

def _get_thread_id():
    thread  = threading.currentThread()
    return '{}-{}'.format(thread.name, thread.ident)

class RootTimer:
    """
        Main Timer class which stores recursively information on the execution time of functions / nested functions
        The timer is thread-safe and stores each thread's timer in a different dict
        
        `_timers` stores the different timers in a tree-like structured dict : 
        {thread : timer} wher `timer` is a dict {name :, runs : [], start :, children {timer...}}
        
        `_runnings` stores the running timer reference in a list : {thread : references...[]}
        
        Note that `_runnings` only contains references to the corresponding timer in the `_timers` structure, meaning that modifying the `_runnings`'s timers will automatically modify the timer in `_timers`
    """
    def __init__(self, name = 'timer'):
        self.name      = name
        self._timers   = collections.OrderedDict()
        self._runnings = {}
        self.mutex     = threading.Lock()
    
    @property
    def running(self):
        return any([len(runnings) > 0 for t, runnings in self._runnings.items()])
    
    def __str__(self, names = None):
        def thread_to_str(thread_timers, ** kwargs):
            des = ''
            for _, timer in thread_timers.items():
                des += '\n' + timer_to_str(timer, ** kwargs)
            return des
        
        if self.running: return "timer {} not stopped yet".format(self.name)
        
        with self.mutex:
            _timers, _runnings = self._timers, self._runnings
            self._timers, self._runnings = collections.OrderedDict(), {}
        
        des = 'Timers for logger {} :'.format(self.name)
        if len(_timers) == 1:
            des += thread_to_str(list(_timers.values())[0])
        else:
            for thread, timers in _timers.items():
                if len(timers) == 0: continue
                des += '\n- Timers in thread {} :{}'.format(
                    thread, thread_to_str(timers, indent = 1)
                )
        
        return des
    
    def get_thread_timers(self, thread_id = None):
        """
            This method is the only one required to be thread-safe as it (possibly creates) and returns the timers for a specific thread
            Once returned, the manipulation of these timers will be, by design, thread-safe as functions are executed in a single thread then will modify in a sequential order
        """
        with self.mutex:
            if thread_id is None:
                thread_id   = _get_thread_id()
            if thread_id not in self._timers:
                self._timers[thread_id]   = collections.OrderedDict()
                self._runnings[thread_id] = collections.deque()
            
            return self._timers[thread_id], self._runnings[thread_id]
    
    def get_current_timer(self, timers, runnings):
        return runnings[-1] if len(runnings) > 0 else timers
    
    def add_child(self, current, name):
        """ Add a new child to `current` and set `start` entry to current time """
        if name not in current: current[name] = create_timer(name)
        current[name]['start'] = time.time()
        return current[name]

    def stop_current_timer(self, runnings):
        """ Pop the running thread and add a new value for the `runs` list """
        timer = runnings.pop()
        timer['runs'].append(time.time() - timer['start'])
        timer['start'] = -1
        return timer
    
    def start_timer(self, name):
        timers, runnings = self.get_thread_timers()
        
        current   = self.get_current_timer(timers, runnings)
        new_timer = self.add_child(current.get('children', current), name)
        runnings.append(new_timer)
    
    def stop_timer(self, name):
        timers, runnings = self.get_thread_timers()
        if len(runnings) == 0:
            logger.error('empty runnings when stopping {} {}'.format(name, threading.currentThread().getName()))
            return
        timer = self.stop_current_timer(runnings)
        
        while len(runnings) > 0 and timer['name'] != name:
            timer = self.stop_current_timer(runnings)
        return timer

    
def timer(fn    = None,
          name  = None,
          debug = False,
          logger    = TIME_LOGGER_NAME,
          log_if_root   = True,
          force_logging = False
         ):
    """
        Decorator that will track execution time for the decorated function `fn`
        Arguments :
            - fn    : the function to decorate
            - name  : the timer's name (by default the function's name)
            - logger    : which logger to use (by default 'timer')
            - log_if_root   : whether to print if it is the root's timer
            - force_logging : logs even if it is not the main timer (not fully supported yet)
    """
    def wrapper(fn):
        @wraps(fn)
        def fn_with_timer(* args, ** kwargs):
            if not executing_eagerly():
                return fn(* args, ** kwargs)
            if not logger.isEnabledFor(level):
                return fn(* args, ** kwargs)

            logger.start_timer(timer_name, level = level)
            try:
                return fn(* args, ** kwargs)
            finally:
                logger.stop_timer(timer_name, level = level)

                if (log_if_root and not logger._timer.running) or (force_logging):
                    logger.log_time(timer_name, level = level)
        
        
        timer_name = name
        if not name:
            timer_name = fn.name if hasattr(fn, 'name') else fn.__name__
        return fn_with_timer
    
    level = TIME_LEVEL if not debug else TIME_DEBUG_LEVEL
    if isinstance(logger, str): logger = logging.getLogger(logger)
    elif logger is None:        logger = logging.getLogger()
    
    return wrapper if fn is None else wrapper(fn)

def with_timer(self, * args, ** kwargs):
    return ContextManager(
        enter   = lambda: self.start_timer(* args, ** kwargs),
        exit    = lambda *_: self.stop_timer(* args, ** kwargs)
    )

def start_timer(self, name, * args, level = -1, debug = False, ** kwargs):
    """ Starts a new timer with the given `name`. Do not forget to call `stop_timer(name)` ! """
    if not executing_eagerly(): return
    if level == -1: level = TIME_LEVEL if not debug else TIME_DEBUG_LEVEL
    if not self.isEnabledFor(level): return
    
    if not hasattr(self, '_timer'): self._timer = RootTimer(self.name)
    self._timer.start_timer(name)
    
def stop_timer(self, name, * args, level = -1, debug = False, ** kwargs):
    """ Stops the running timer with the given `name` """
    if not executing_eagerly(): return
    if level == -1: level = TIME_LEVEL if not debug else TIME_DEBUG_LEVEL
    if not self.isEnabledFor(level): return
    
    if not hasattr(self, '_timer'): self._timer = RootTimer(self.name)
    self._timer.stop_timer(name)

def log_time(self, names = None, * args, level = -1, debug = False, ** kwargs):
    if not executing_eagerly(): return
    if level == -1: level = TIME_LEVEL if not debug else TIME_DEBUG_LEVEL
    if not self.isEnabledFor(level): return
    if not hasattr(self, 'timer'): self._timer = RootTimer(self.name)
    des = self._timer.__str__(names)
    self.log(level if not debug else TIME_DEBUG_LEVEL, des, * args, ** kwargs)

logging.Logger.start_timer  = start_timer
logging.Logger.stop_timer   = stop_timer
logging.Logger.log_time     = log_time
logging.Logger.timer        = with_timer

time_logger = logging.getLogger(TIME_LOGGER_NAME)

logging.start_timer  = time_logger.start_timer
logging.stop_timer   = time_logger.stop_timer
logging.log_time     = time_logger.log_time
logging.timer        = time_logger.timer

