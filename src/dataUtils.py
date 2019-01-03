import pandas as pd
import re
import pdb

from data.data import get_mag, IEMOCAP
from dataProcessing.loadMeanVariance import loadMeanVariance

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, ConcatDataset

class Data():
  def __init__(self, path2data, dataset, split=(0.8,0.1), batch_size=100,
               stft_window=100, stft_hop=None,
               n_fft=1024, hop_length=256, win_length=1024,
               shuffle=True):
    assert n_fft % 2 == 0, 'n_fft must be even'

    self.raw_data = eval(dataset)(path2data)
    self.df = self.raw_data._get_df()

    self.split = split
    self.stft_window = stft_window
    self.stft_hop = stft_hop

    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    
    self.datasets = self.tdt_split()

    dataLoader_kwargs = {'batch_size':batch_size,
                         'shuffle':shuffle}
    self.train = DataLoader(self.datasets['train'], **dataLoader_kwargs)
    self.dev = DataLoader(self.datasets['dev'], **dataLoader_kwargs)
    self.test = DataLoader(self.datasets['test'], **dataLoader_kwargs)

  def tdt_split(self):
    length = self.df.shape[0]
    end_train = int(length*self.split[0])
    start_dev = end_train
    end_dev = int(start_dev + length*self.split[1])
    start_test = end_dev

    if 'split' in self.df.columns:
      df_train = self.df[self.df['split'] == 'train']
      df_dev = self.df[self.df['split'] == 'dev']
      df_test = self.df[self.df['split'] == 'test']
    else:
      df_train = self.df[:end_train]
      df_dev = self.df[start_dev:end_dev]
      df_test = self.df[start_test:]

    minidataKwargs = {'stft_window':self.stft_window,
                      'stft_hop':self.stft_hop,
                      'n_fft':self.n_fft,
                      'hop_length':self.hop_length,
                      'win_length':self.win_length}
    
    dataset_train = ConcatDataset([MiniData(row.wav, **minidataKwargs)
                                   for i, row in tqdm(df_train.iterrows())])
    dataset_dev = ConcatDataset([MiniData(row.wav, **minidataKwargs)
                                 for i, row in tqdm(df_dev.iterrows())])
    dataset_test = ConcatDataset([MiniData(row.wav, **minidataKwargs)
                                  for i, row in tqdm(df_test.iterrows())])
    
    return {'train':dataset_train,
            'dev':dataset_dev,
            'test':dataset_test}

  @property
  def shape(self):
    return int(self.n_fft/2 + 1)

  @property
  def sr(self):
    return 
       

class MiniData(Dataset):
  def __init__(self, audiopath, stft_window=None, stft_hop=None, n_fft=1024, hop_length=256, win_length=1024):
    assert n_fft % 2 == 0, 'n_fft must be even'

    self.audiopath = audiopath
    self.stft_window = stft_window
    self.n_fft = n_fft
    if stft_hop is None:
      self.stft_hop = self.stft_window
    else:
      self.stft_hop = stft_hop
    self.stft, self.time_shape, self.sr = get_mag(audiopath, n_fft=n_fft, hop_length=hop_length, win_length=win_length) 
    self.total_stft_time = self.stft.shape[0]
 
  def __len__(self):
    if self.stft_window is None:
      return 1
    elif self.stft_window > self.total_stft_time:
      return 0
    else:
      return int((self.total_stft_time - self.stft_window)/self.stft_hop + 1)
    

  def __getitem__(self, idx):
    if self.stft_window is None:
      return self.stft
    elif self.stft_window > self.total_stft_time:
      return None
    else:
      start = idx*self.stft_hop
      end = start + self.stft_window
      return self.stft[start:end]


class Transforms():
  def __init__(self, transforms, n_fft, hop_length, win_length, args):
    self.transforms = transforms
    for tr in transforms:
      if tr == 'zNorm':
        self.mean, self.variance = loadMeanVariance(n_fft, hop_length, win_length, args)
        self.mean = self.mean.reshape(1, 1, -1)
        self.variance = self.variance.reshape(1, 1, -1)
      else:
        assert 0, 'Transform not found'

  def transform(self, x):
    for tr in self.transforms:
      if tr == 'zNorm':
        return (x - self.mean.to(x.device))/(self.variance.to(x.device)**0.5)

  def inv_transform(self, x):
    for tr in self.transforms:
      if tr == 'zNorm':
        return (x*(self.variance.to(x.device)**0.5)) + self.mean.to(x.device)
