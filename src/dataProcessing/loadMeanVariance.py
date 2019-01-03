import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataUtils import *

from pycasper.name import Name
from pycasper.BookKeeper import *
from argsUtils import argparseNloop

import numpy as np
from tqdm import tqdm
import pdb

def loadMeanVariance(n_fft, hop_length, win_length, args):
  mean_path ='dataProcessing/mean_nfft_{}_hop_{}_win_{}.pt'.format(n_fft, hop_length, win_length)
  variance_path = 'dataProcessing/variance_nfft_{}_hop_{}_win_{}.pt'.format(n_fft, hop_length, win_length)
  if not (os.path.exists(mean_path) and os.path.exists(variance_path)):
    loop(args, 0)

  mean = torch.load(mean_path).double()
  variance = torch.load(variance_path).double()

  return mean, variance

def loop(args, exp_num):
  BookKeeper._set_seed(args)
  path2data = args.path2data
  dataset = args.dataset
  split = (args.train_frac, args.dev_frac)
  batch_size = 1
  stft_window = None
  stft_hop = None
  n_fft = args.n_fft
  hop_length = args.hop_length
  win_length = args.win_length
  
  ## Load data iterables
  print('Loading data for Mean+Variance')
  data = Data(path2data, dataset,
              split=split, batch_size=batch_size,
              stft_window=stft_window, stft_hop=stft_hop,
              n_fft=n_fft, hop_length=hop_length, win_length=win_length,
              shuffle=True)

  train = data.train
  for sample in train:
    break
  input_shape = sample.shape[-1]

  running_sum = torch.zeros(input_shape).double()
  running_energy = torch.zeros(input_shape).double()
  running_count = 0

  for count, batch in tqdm(enumerate(train), desc='Mean+Variance'):
    x = batch
    running_count += x.shape[0] * x.shape[1]
    running_sum += x.double().sum(dim=0).sum(dim=0)
    running_energy += (x**2).double().sum(dim=0).sum(dim=0)

  mean = running_sum/running_count
  energy = running_energy/running_count
  variance = energy - mean**2

  ## Add some small value to the dimensions that have zero variance to avoid nan errors
  eps = 1e-30
  zero_mask = (variance == 0)
  variance = variance + (zero_mask*eps).type(torch.float64)

  ## Save files 
  torch.save(mean, 'dataProcessing/mean_nfft_{}_hop_{}_win_{}.pt'.format(n_fft, hop_length, win_length))
  torch.save(variance, 'dataProcessing/variance_nfft_{}_hop_{}_win_{}.pt'.format(n_fft, hop_length, win_length))

if __name__ == '__main__':
  argparseNloop(loop)
