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
import time
import queue
import logging
import librosa
import threading
import collections
import numpy as np

from scipy.io.wavfile import write, read

from loggers import Timer, timer
from . import audio_processing
from .audio_player import AudioPlayer
from .audio_recorder import AudioRecorder
from ..keras import TensorSpec, ops, execute_eagerly
from ..wrappers import dispatch_wrapper 

logger = logging.getLogger(__name__)

MAX_DISPLAY_TIME = 600

_video_ext  = ('mp4', 'mov', 'ovg', 'avi')
_pydub_ext  = ('m4a', 'ogg')
_librosa_ext    = ('mp3', 'flac', 'opus')
_audiofile_ext  = ()
_ffmpeg_ext     = _video_ext

_write_pydub_ext    = ('mp3', )
_write_ffmpeg_ext   = ()

_load_fn    = {}
_write_fn   = {}

""" Streaming functions (microphone recording / speakers playing) """

_audio_player   = AudioPlayer()

def display_audio(filename, rate = None, play = False, ** kwargs):
    """
        Displays the audio with the `IPython.display.Audio` object, and returns `(rate, audio)`
        The function internally calls `read_audio`, meaning that all processing can be applied before display (i.e. `kwargs` are forwarded to `read_audio`)
    """
    from IPython.display import Audio, display
    
    rate, audio = read_audio(filename, target_rate = rate, rate = rate, ** kwargs)
    
    display(Audio(audio[: int(MAX_DISPLAY_TIME * rate)], rate = rate, autoplay = play))
    
    return rate, audio

def play_audio(audio, rate = None, blocking = True, raw = False, add_silence = True, ** kwargs):
    """ Plays `audio` on speakers """
    if isinstance(audio, str) or not raw:
        rate, audio = read_audio(audio, target_rate = rate, rate = rate, ** kwargs)
    
    event = _audio_player.append(
        audio, rate = rate, add_event = blocking, add_silence = add_silence
    )
    _audio_player.play()
    
    if blocking: event.wait()
    else: return _audio_player

def record_audio(blocking = True, ** kwargs):
    """ Plays `audio` on speakers """
    recorder = AudioRecorder(** kwargs).start()
    if not blocking: return recorder
    
    recorder.join()
    return recorder.rate, recorder.audio

def stream_audio(audio = None, rate = None, callback = None, real_time = True, ** kwargs):
    if audio is None: return record_audio(rate = rate, callback = callback, ** kwargs)
    
    rate, audio = read_audio(audio, target_rate = rate, rate = rate, ** kwargs)
    
    if callback is not None:
        chunk_size = rate // kwargs.get('fps', 10)
        for s in range(0, audio.shape[-1], chunk_size):
            callback(audio[..., s : s + chunk_size])
            if real_time: time.sleep(1. / kwargs.get('fps', 10))
    
    return rate, audio


""" Generic functions to load audio and mel """

def load_audio(data, rate, ** kwargs):
    """
        Load audio from different type of data :
            - str : filename of the audio file
            - np.ndarray / Tensor    : raw audio
            - dict : 
                'audio' : raw audio
                'wavs_{rate}'   : filename for audio of correct rate
                'filename'      : filename for audio (resample if needed)
        Return :
            - audio : `np.ndarray` or `Tensor` with shape [n_samples]
    """
    if isinstance(data, dict):
        if 'audio' in data:
            audio_key = 'audio'
        elif 'wavs_{}'.format(rate) in data:
            audio_key = 'wavs_{}'.format(rate)
        else:
            audio_key = 'filename' if 'filename' in data else 'audio_filename'
        
        if 'rate' in data: kwargs['rate'] = data['rate']
        data = data[audio_key]
    
    if not isinstance(data, (str, np.ndarray)) and not ops.is_tensor(data):
        raise ValueError("Unknown audio type : {}\n{}".format(type(data), data))

    if 'rate' not in kwargs: kwargs['rate'] = rate
    return read_audio(data, target_rate = rate, ** kwargs)[1]

def load_mel(data, stft_fn, ** kwargs):
    """
        Load mel from different type of data :
            - dict  : 
                'mel'   : raw mel
                stft.dir_name   : filename of mel
            - other : call load_audio(data) and apply stft_fn on audio
        Return : mel spectrogram (as 2D Tensor)
    """
    if isinstance(data, dict) and 'mel' in data:
        return data['mel']
    elif hasattr(data, 'shape') and len(data.shape) >= 2 and data.shape[-1] == stft_fn.n_mel_channels:
        return data
    else:
        mel = stft_fn(load_audio(data, stft_fn.rate, ** kwargs))
        return mel if len(mel.shape) == 2 else mel[0]

def resample_file(filename, new_rate, filename_out = None, normalize = False, ** kwargs):
    """
        Creates a copy of `filename` with the new expected rate (`new_rate`)
        Returns `None` if the initial audio format or expected output format is not supported
        `filename_out` is by default `{filename}_{new_rate}.{ext}`
    """
    if filename_out is None:
        base_name, ext = os.path.splitext(filename)
        filename_out = '{}_{}{}'.format(base_name, new_rate, ext)
    
    if os.path.exists(filename_out): return filename_out
    
    try:
        rate, audio = read_audio(
            filename, target_rate = new_rate, normalize = normalize, dtype = np.float32
        )
    except ValueError as e:
        logger.error("Error while loading file {} !\n{}".format(filename, e))
        return None
    
    try:
        write_audio(audio = audio, filename = filename_out, rate = rate)
    except ValueError as e:
        logger.error("Error while writing file {} !\n{}".format(filename, e))
        return None
    
    return filename_out

"""
    Methods for audio loading (with optional processing in `read_audio`)
    The generic signature for sub-methods is `read_<format>(filename)`
    The return values are `(rate, audio)`
"""

@dispatch_wrapper(_load_fn, 'File extension')
@timer
@execute_eagerly(signature = [
    TensorSpec(shape = (),       dtype = 'int32'),
    TensorSpec(shape = (None, ), dtype = 'float32')
], numpy = True)
def read_audio(filename,
               *,
               
               rate     = None,
               dtype    = None,
               target_rate  = None,
               
               # processing config
               normalize    = True, 
               reduce_noise = False,
               trim_silence = False,
               
               read_method  = None,
               
               ** kwargs
              ):
    """
        Generic method for audio loading : internally calls the loading function associated to the filename extension, then it applies the expected processing
        
        Arguments :
            - filename  : the audio filename or raw audio (if raw, `rate` must be provided)
            - target_rate   : the rate to resample to (if required) (resampled with `scipy.signal`)
            
            - offset    : the number of samples to skip at the start / end of the audio
            - normalize : whether to normalize or not the audio (in range [0., 1.])
                - if a `float` is provided, divides by the value
            - reduce_noise  : whether to reduce noise or not (see `reduce_noise` for more info)
            - trim_silence  : whether to trim silence or not (see `trim_silence` for more info)
            
            - start / end / time    : the time information to keep

            - rate  : the audio rate (only required if `filename` is the raw audio)
            - dtype : the expected output dtype
            - read_method   : string or callable, specific loading function to use
            
            - kwargs    : forwarded to the loading function, `reduce_noise` and `trim_silence`
        Returns : (rate, audio)
            - rate  : the audio rate (equals to `target_rate` if provided)
            - audio : 1D `np.ndarray`, the audio
        
        Note : if `normalize is True`, the audio will be normalized such that values are in the range [0, 1] and the maximal value is 1.
        A contrario, providing a `dtype` will normalize according to the maximal value of the current audio dtype, meaning that providing `dtype = np.float32` will outputs an audio in the range [0, 1] without any guarantee that the maximal value is 1.
    """
    if isinstance(filename, bytes): filename = filename.decode()
    if isinstance(filename, str):
        if read_method is None:
            read_method = filename.split('.')[-1]
        
        if isinstance(read_method, str):
            if read_method in _load_fn:
                read_method = _load_fn[read_method]
            elif read_method in globals():
                read_method = globals()[read_method]
            else:
                raise ValueError('Unsupported reading method !\n  Accepted : {}\n  Got : {}'.format(
                    tuple(_load_fn.keys()), read_method
                ))

        with Timer('read file'):
            rate, audio = read_method(filename, rate = target_rate)
    else:
        assert rate is not None, 'You must provide the `rate` when passing raw audio !'
        audio = filename
    
    if target_rate and target_rate != rate:
        audio, rate = audio_processing.resample_audio(audio, rate, target_rate)
    
    if normalize:
        if normalize is True:
            audio = audio_processing.normalize_audio(audio, max_val = 1.)
        elif normalize > 1 and np.issubdtype(audio.dtype, np.integer):
            audio = (audio / normalize).astype(np.float32)
    
    if reduce_noise:
        audio = audio_processing.reduce_noise(audio, rate = rate, ** kwargs)
        if normalize is True:
            audio = audio_processing.normalize_audio(audio, max_val = 1.)
    
    if trim_silence:
        audio = audio_processing.trim_silence(audio, rate = rate, ** kwargs)
    
    if dtype:
        audio = audio_processing.convert_audio_dtype(audio, dtype)
    
    return rate, audio

@read_audio.dispatch
def read_wav(filename, ** kwargs):
    """ Reads .wav audio with the `scipy.io.wavfile.read` method """
    return read(filename)

@read_audio.dispatch(_pydub_ext)
def read_pydub(filename, ** kwargs):
    """ Reads mp3 audio with the `pydub.AudioSegment.from_mp3()` function """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(filename)
    audio_np = np.array(audio.get_array_of_samples())
    if audio.channels > 1: audio_np = audio_np[::audio.channels]
    return audio.frame_rate, audio_np

@read_audio.dispatch(_librosa_ext)
def read_librosa(filename, ** kwargs):
    """ Reads an audio with the `librosa.load` function """
    audio, rate = librosa.load(filename, sr = None)
    return rate, audio

def read_audiofile(filename, ** kwargs):
    """ Reads an audio with the `librosa.load` function """
    import audiofile
    
    audio, rate = audiofile.read(filename)
    return rate, audio

@read_audio.dispatch(_ffmpeg_ext)
def read_ffmpeg(filename, rate = None):
    try:
        import ffmpeg
    except ImportError:
        logger.error("You must install ffmpeg : `pip install ffmpeg-python`")
        return None
    
    try:
        kw = {} if not rate else {'ar' : rate}
        out, _ = (
            ffmpeg.input(filename, threads = 0)
            .output("-", format = "s16le", acodec = "pcm_s16le", ac = 1, ** kw)
            .run(cmd = ["ffmpeg", "-nostdin"], capture_stdout = True, capture_stderr = True)
        )
        if not rate:
            infos   = [a for a in ffmpeg.probe(filename)['streams'] if a['codec_type'] == 'audio'][0]
            rate    = int(infos['sample_rate'])
    except ffmpeg.Error as e:
        raise RuntimeError("Failed to load audio : {}".format(e))

    return rate, np.frombuffer(out, np.int16).flatten()

def read_moviepy(filename, ** kwargs):
    """ Reads the audio of a video with the `moviepy` library """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        logger.error("You must install moviepy : `pip install moviepy`")
        return None

    with VideoFileClip(filename) as video:
        audio = video.audio

        fps     = audio.fps
        array   = audio.to_soundarray()
    
    if len(array.shape) > 1: array = array[:,0]
    
    return fps, array

""" 
    Methods for writing audio to file with the format given by the filename's extension
    The general signature for the sub-methods is `write_<method>(audio, filename, rate)`
"""

@dispatch_wrapper(_write_fn, 'Filename extension')
def write_audio(filename, audio, rate, normalize = True, factor = 32767, verbose = False):
    """
        Writes `audio` to `filename` with given `rate` and the format given by the filename extension
    """
    ext = filename.split('.')[-1]
    if ext not in _write_fn:
        raise ValueError("Unsupported file extension !\n  Accepted : {}\n  Got : {}".format(
            tuple(_write_fn.keys()), filename
        ))
    
    logger.log(logging.INFO if verbose else logging.DEBUG, "Saving audio to {}".format(filename))
    
    normalized = audio if isinstance(audio, np.ndarray) else ops.convert_to_numpy(audio)
    if normalize and len(audio) > 0:
        normalized = audio_processing.normalize_audio(audio, max_val = factor)
    
    _write_fn[ext](audio = normalized, filename = filename, rate = rate)
    return filename
    
@write_audio.dispatch
def write_wav(audio, filename, rate):
    """ Writes audio with `scipy.io.wavfile.write()` """
    write(filename, rate, audio)
    
@write_audio.dispatch(_write_pydub_ext)
def write_pydub(audio, filename, rate):
    """ Writes audio with `pydub.AudioSegment.export()` """
    from pydub import AudioSegment

    audio_segment = AudioSegment(
        audio.tobytes(), frame_rate = rate, sample_width = audio.dtype.itemsize, channels = 1
    )
    file = audio_segment.export(filename, format = filename.split('.')[-1])
    file.close()
    
def write_ffmpeg(audio, filename, rate):
    try:
        import ffmpeg

        format = 'f32le' if audio.dtype == 'float32' else 's16le'
        process = (
            ffmpeg
            .input('pipe:0', format = format, ac = 1, ar = rate)
            .output(filename, format = filename.split('.')[-1])
            .overwrite_output()
            .run_async(pipe_stdin = True)
        )

        process.stdin.write(audio.tobytes())
        process.stdin.close()
        process.wait()
    except ImportError:
        logger.error("You must install ffmpeg : `pip install ffmpeg-python`")
    except ffmpeg.Error as e:
        logger.error('Error while writing audio to {} : {}'.format(filename, e))
