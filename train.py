## for multiple arches, use config files

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataUtils import Data
from model.model import *
from data.data import *

from pycasper.name import Name
from pycasper import BookKeeper
from argsUtils import argparseNloop

import numpy as np
import ipdb
from tqdm import tqdm

def train(args, exp_num):
  args_subset = ['exp', 'cpk']
  book = BookKeeper(args, args_subset, args_dict_update={}, tensorboard = args.tb)
  args = book.args

  ## Start Log
  book._start_log()
  
  ## Training parameters
  num_epochs = args.num_epochs
  
  ## Load data iterables

  ## TODO hardcoded to only take the first person

  print('Data Loaded')
  
  ## Create a model

  model = Wavenet(num_channels,
                  kernel_size,
                  num_layers,
                  num_lmks,
                  sample_rate_lmks,
                  samples_per_frame,
                  dilation_limit,
                  mfccFlag=mfccFlag)
  if args.cuda >=0:
    model.cuda(args.cuda)

  book._copy_best_model(model)
  print('Model Created')
    
  ## Load model
  if args.load:
    print('Loading Model')
    book._load_model(model)

  ## Loss function
  criterion = torch.nn.MSELoss()
  ## Optimizers
  optim = torch.optim.Adam(model.parameters(), lr=args.lr)

  def loop(model, data, desc='train', epoch=0):
    running_loss = 0
    running_loss_partial = 0
    if desc == 'train':
      model.train(True)
    else:
      model.eval()
    for count, batch in tqdm(enumerate(data), desc=desc, leave=False, ncols=20):
      

  ## Training Loop
  for epoch in tqdm(range(num_epochs), ncols=20):
    train_loss = loop(model, train, 'train', epoch)
    dev_loss = loop(model, dev, 'dev')
    test_loss = loop(model, test, 'test')
    
    ## save results
    book.update_res({'train':train_loss,
                     'dev':dev_loss,
                     'test':test_loss})
    book._save_res()

    ## print results
    book.print_res(epoch, key_order=['train','dev','test'], exp=exp_num)

    if book.stop_training(model, epoch):
      break
    
  # End Log
  book._stop_log()

if __name__ == '__main__':
  argparseNloop(train)
