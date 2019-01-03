import torch

class Loss():
  def __init__(self, losses, kwargs_list):
    self.losses = losses
    self.kwargs_list = kwargs_list
    self.loss_list = []
    for loss, kwargs in zip(self.losses, self.kwargs_list):
      if loss in self.torch_losses:
        loss = 'torch.nn.' + loss 
      self.loss_list.append(eval(loss)(**kwargs))

  def __call__(self, y_cap, y):
    loss = 0
    for loss_fn in self.loss_list:
      loss += loss_fn(y_cap, y)

    return loss

  @property
  def torch_losses(self):
    return {'MSELoss'}

class SkeletonLoss():
  def __init__(self, skeleton):
    self.loss = torch.nn.MSELoss()
    self.weights = torch.ones(1, 56) ## HardCoded
    self.weights = self.weights**0.5

  def __call__(self, y_cap, y):
    return self.loss(y_cap*self.weights.to(y_cap.device), y*self.weights.to(y_cap.device))
  
