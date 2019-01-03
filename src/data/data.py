import pandas as pd
import librosa
import numpy as np
import os
from pathlib import Path
import pdb

from tqdm import tqdm

def pad(x, hop_length):
  input_shape = x.shape[0]
  new_shape = (input_shape//hop_length)*hop_length + hop_length
  x = np.pad(array=x, pad_width=(0, new_shape-input_shape), mode='constant')
  return x, input_shape

def get_stft(x, n_fft=1024, hop_length=256, **kwargs):
  x, time_shape = pad(x, hop_length)
  stft = librosa.stft(x, n_fft, hop_length, **kwargs)
  return stft, time_shape
  
def get_mag(audio_file, n_fft=1024, hop_length=256, **kwargs):
  audio, sr = librosa.core.load(audio_file, sr=None, mono=True)
  stft, time_shape = get_stft(audio, n_fft=1024, hop_length=256, **kwargs)
  mag, phase = librosa.magphase(stft)
  return mag.T, time_shape, sr #, audio, stft

def get_istft(X, time_shape, hop_length=256, **kwargs):
  return librosa.istft(X, hop_length=hop_length, length=time_shape, **kwargs)

def griffin_lim(spectrogram, time_shape, n_fft, hop_length, win_length, num_iter=500):
    #spectrogram = np.power(10, spectrogram)
    p = 2 * np.pi * np.random.random_sample(spectrogram.shape) - np.pi
    for i in tqdm(range(num_iter)):
        S = spectrogram * np.exp(1j*p)
        x = get_istft(S, time_shape, hop_length = hop_length, win_length = win_length)
        S_cap, time_shape = get_stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        p = np.angle(S_cap)

    return x

def spec2wav(save_path, sr, spectrogram, time_shape, n_fft, hop_length, win_length, num_iter=500):
  x = griffin_lim(spectrogram, time_shape, n_fft, hop_length, win_length, num_iter)
  librosa.output.write_wav(save_path, x, sr)

class IEMOCAP():
  def __init__(self, path2data):
    self.data_dict = {'wav':[], 'split':[]}
    path2data = Path(path2data)
    sessions = [path2data/Path('Session{}/dialog/wav'.format(i)) for i in range(1,6)]
    for count, session in enumerate(sessions):
      for files in os.listdir(session):
        if files[0] is not '.' and files.endswith('.wav'):
          self.data_dict['wav'].append((session/files).as_posix())
          if count == 3:
            self.data_dict['split'].append('dev')
          elif count == 4:
            self.data_dict['split'].append('test')
          else:
            self.data_dict['split'].append('train')
          

    ## convert data dict to pandas file
    self.raw_data = pd.DataFrame(self.data_dict)
    

  def _get_df(self):
    return self.raw_data

  @property
  def sr(self):
    return 16000
