import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from data.data import *
from dataUtils import *
from model.model import *
from lossUtils import *

from pycasper.name import Name
from pycasper.BookKeeper import *
from argsUtils import argparseNloop

import numpy as np
from tqdm import tqdm
import ipdb

def sample(args, exp_num):
  assert args.load, '-load should not be None'
  assert os.path.exists(args.load), '-load should exist'
  
  args_subset = ['exp', 'cpk', 'model']
  book = BookKeeper(args, args_subset, args_dict_update={'batch_size':1,
                                                         'stft_window':None},
                    tensorboard=args.tb)

  dir_name = book.name.dir(args.save_dir)

  args = book.args

  ## Training parameters
  path2data = args.path2data
  dataset = args.dataset
  split = (args.train_frac, args.dev_frac)
  batch_size = args.batch_size
  stft_window = args.stft_window
  stft_hop = args.stft_hop
  n_fft = args.n_fft
  hop_length = args.hop_length
  win_length = args.win_length
  
  ## Load data iterables
  data = Data(path2data, dataset,
              split, batch_size=batch_size,
              stft_window=stft_window, stft_hop=stft_hop,
              n_fft=n_fft, hop_length=hop_length, win_length=win_length,
              shuffle=False)

  train = data.train
  dev = data.dev
  test = data.test

  print('Data Loaded')
  
  ## Create a model
  device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda>=0 else torch.device('cpu')

  modelKwargs = {'feat_dim':data.shape}
  modelKwargs.update(args.modelKwargs)

  model = eval(args.model)(**modelKwargs)
  model.to(device).double()

  book._copy_best_model(model)
  print('Model Created')
    
  ## Load model
  if args.load:
    print('Loading Model')
    book._load_model(model)

  ## Loss function
  criterion = Loss(args.losses, args.lossKwargs)
  
  ## Transforms
  pre = Transforms(args.transforms, n_fft, hop_length, win_length, args)

  def loop(model, data_cls, data, pre, desc='train'):
    running_loss = 0
    running_count = 0
    
    model.eval()
    for count, batch in tqdm(enumerate(data), desc=desc, leave=False, ncols=20):
      x = batch.double()
      y = x.clone()
      batch_size = x.shape[0]
      
      x = x.to(device)
      y = y.to(device)

      ## Transform before the model
      x = pre.transform(x)
      y = pre.transform(y)

      #y_cap = x
      #internal_losses = []
      _, _, _, _, _, _, y_cap, internal_losses = model(x)

      ## save spectrogram as wavfiles
      save_path = Path(dir_name)/Path(desc)/Path(data.dataset.datasets[count].audiopath).relative_to(args.path2data)
      os.makedirs(save_path.parent, exist_ok=True)
      save_path = save_path.as_posix()
      
      sr = data.dataset.datasets[count].sr
      time_shape = data.dataset.datasets[count].time_shape
      spectrogram = pre.inv_transform(y_cap).squeeze().transpose(1,0).contiguous().cpu().detach().numpy()
      spec2wav(save_path, sr, spectrogram, time_shape, n_fft, hop_length, win_length, num_iter=500)

      loss = criterion(y_cap, y)
      for i_loss in internal_losses:
        loss += i_loss
      
      running_loss += loss.item() * batch_size
      running_count += batch_size
      
      if count>=0 and args.debug: ## debugging by overfitting
        break

    return running_loss/running_count


  train_loss = loop(model, data, train, pre, 'train')
  dev_loss = loop(model, data, dev, pre, 'dev')
  test_loss = loop(model, data, test, pre, 'test')
    
  ## update results but not save them
  book.update_res({'train':train_loss,
                   'dev':dev_loss,
                   'test':test_loss})

  ## print results
  book.print_res(0, key_order=['train','dev','test'], exp=exp_num, lr=optim.param_groups[0]['lr'])

  
if __name__ == '__main__':
  argparseNloop(sample)
