import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

#__all__ = ['SequentialHourglassNet', 'hg']

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Conv2dLSTMCell(nn.Module):
  def __init__(self, in_shape, out_shape):
    super(Conv2dLSTMCell, self).__init__()

    self.in_shape = in_shape
    self.out_shape = out_shape
    self.input_gates = nn.ModuleList([nn.Conv2d(in_shape, out_shape, kernel_size=1, bias=False) for _ in range(4)])
    self.hidden_gates = nn.ModuleList([nn.Conv2d(out_shape, out_shape, kernel_size=1, bias=True) for _ in range(4)])
    self.sigmoid = nn.Sigmoid()

  def forward(self, x, h=None, c=None):
    hidden_shape = (x.shape[0], self.out_shape, x.shape[2], x.shape[3])
    if h is None:
      h = x.new_zeros(hidden_shape)
    if c is None:
      c = x.new_zeros(hidden_shape)

    f = self.sigmoid(self.input_gates[0](x) + self.hidden_gates[0](h))
    i = self.sigmoid(self.input_gates[1](x) + self.hidden_gates[1](h))
    o = self.sigmoid(self.input_gates[2](x) + self.hidden_gates[2](h))
    c_hat = self.sigmoid(self.input_gates[3](x) + self.hidden_gates[3](h))
    c_t = f*c + i*c_hat
    h_t = o*self.sigmoid(c_t)

    return h_t, c_t


class Conv2dLSTM(nn.Module):
  '''
  Convolutional LSTM Network. Can't handle variable sized inputs in a batch_size>1

  Args:
    in_shape: number of input channels
    out_shape: number of output channels
    num_layers: number of layers

  Input: x, hx=(h_0, c_0)
    **x** shape: (batch, in_shape, H, W, Time)
    **h_0** and **c_0** shape: (batch, out_shape, H, W, Layers)

  Outputs: output, (h_n, c_n)
    **output** shape: (batch, output_shape, H, W, Time)
    **h_n** and **c_n** shape: (batch, out_shape, H, W, Layers)
  
  '''
  
  def __init__(self, in_shape, out_shape, num_layers=1):
    super(Conv2dLSTM, self).__init__()
    assert num_layers>0, 'num_layers are {}, but they should be greater than 0'.format(num_layers)
    
    self.in_shape = in_shape
    self.out_shape = out_shape
    self.num_layers = num_layers

    self.rnn = nn.ModuleList([Conv2dLSTMCell(in_shape, out_shape)])
    for layer in range(1,num_layers):
      self.rnn.append(Conv2dLSTMCell(out_shape, out_shape))

  def _rnn_forward(self, x, layer, h=None, c=None):
    max_time = x.shape[-1]

    outputs = []
    for time in range(max_time):
      h, c = self.rnn[layer](x[:,:,:,:,time], h, c)
      outputs.append(h)

    return torch.stack(outputs, dim=-1), h, c

  def forward(self, x, hx=None):
    assert isinstance(hx, tuple) or hx is None, 'hx must be a tuple or None'
    
    if hx is None:
      h_0, c_0 = None, None
    else:
      h_0, c_0 = hx

    h, c = h_0, c_0
    h_n = []
    c_n = []
    for layer, rnn in enumerate(self.rnn):
      x, h, c = self._rnn_forward(x, layer, h, c)
      h_n.append(h)
      c_n.append(c)

    return x, (torch.stack(h_n, dim=-1), torch.stack(c_n, dim=-1))
    

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth, rnn_layers):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        #self.upsample = nn.Upsample(scale_factor=2)
        self.upsample = lambda x:nn.functional.interpolate(x, scale_factor=2)
        self.hg, self.rnn = self._make_hour_glass(block, num_blocks, planes, depth, rnn_layers)


    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth, rnn_layers):
        rnn = []
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
            rnn.append(Conv2dLSTM(planes*block.expansion, planes*block.expansion, num_layers=rnn_layers))
        return nn.ModuleList(hg), nn.ModuleList(rnn)

    @staticmethod
    def rnn2convShape(x):
      x = x.permute(*(0, 4, 1, 2, 3))
      orig_shape = [x.shape[0], x.shape[1]]

      new_shape = orig_shape + list(x.shape[2:])
      x = x.contiguous().view(-1, *new_shape[2:])
      return x, orig_shape
  
    @staticmethod      
    def conv2rnnShape(x, orig_shape):
      new_shape = orig_shape + list(x.shape[1:])
      x = x.contiguous().view(*new_shape)
      x = x.permute(*(0, 2, 3, 4, 1))
      return x
      
    def _hour_glass_forward(self, n, x, mask=None):
        x, orig_shape = Hourglass.rnn2convShape(x)
        up1 = self.hg[n-1][0](x)

        up1 = Hourglass.conv2rnnShape(up1, orig_shape)
        up1, (h_n, c_n) = self.rnn[n-1](up1)
        up1, _ = Hourglass.rnn2convShape(up1)
        
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)
        low1 = Hourglass.conv2rnnShape(low1, orig_shape)
        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
            low2, _ = Hourglass.rnn2convShape(low2)
        else:
            low1, _ = Hourglass.rnn2convShape(low1)
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2

        out = Hourglass.conv2rnnShape(out, orig_shape)
        return out

    def forward(self, x, mask=None): 
        return self._hour_glass_forward(self.depth, x, mask=None)

class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, rnn_layers=2):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4, rnn_layers))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x, mask=None):
        out = []
        x, orig_shape = Hourglass.rnn2convShape(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)  

        x = Hourglass.conv2rnnShape(x, orig_shape)
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y, _ = Hourglass.rnn2convShape(y)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            score_RNN = Hourglass.conv2rnnShape(score, orig_shape)
            out.append(score_RNN)
            if i < self.num_stacks-1:
                fc_ = Hourglass.conv2rnnShape(self.fc_[i](y), orig_shape)
                score_ = Hourglass.conv2rnnShape(self.score_[i](score), orig_shape)
                x = x + fc_ + score_

        return out

def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'], rnn_layers=kwargs['rnn_layers'])
    return model
