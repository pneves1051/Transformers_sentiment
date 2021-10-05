import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCrossEntropyLoss(nn.Module):
  def __init__(self, *args, **kwargs):
    super(MultiCrossEntropyLoss, self).__init__()
    self.loss_func = nn.CrossEntropyLoss(*args, ** kwargs, reduction='none')

  def forward(self, input, target, loss_mask, dim_last=True):
    losses = []
    for i in range(target.shape[-1]):
      input_i = input[i].permute(0, 2, 1) if dim_last else input[i]
      loss = self.loss_func(input_i, target[..., i])
      loss = loss * loss_mask
      loss = torch.sum(loss) / torch.sum(loss_mask)
      return loss

    return sum(losses)/len(losses)

# TODO APPLY MASK MAYBE
def wgan_loss_cp(discriminator, d_fake, fake=None, d_real=None, real= None, mode='d', conds=None):
  if mode == 'd':
    d_loss = -(d_real.mean() - d_fake.mean())

    real_data = [r.data for r in real]
    fake_data = [f.data for f in fake]
    eps = torch.rand((d_real.shape[0], 1, 1), device=d_fake.device)#.repeat(1, *real[0].shape[1:])
    
    interp = [torch.autograd.Variable((eps*r + (1-eps)*f), requires_grad=True) for r, f in zip(real_data, fake_data)]
    d_interp = discriminator(interp, conds, to_one_hot=False)
    gp = torch.autograd.grad(outputs=d_interp, inputs=interp,
                              grad_outputs=torch.ones_like(d_interp),
                              create_graph=True, retain_graph=True)[0]          
    gp = gp.view(gp.shape[0], -1)
    gp = ((gp.norm(2, dim=1) - 1)**2).mean()   
    #gp = torch.mean((1. - torch.sqrt(1e-8+torch.sum(gp.view(gp.size(0), -1)**2, dim=1)))**2)
    d_loss_gp = d_loss + 10*gp
    
    return d_loss_gp
    
  elif mode == 'g':
    g_loss = -d_fake.mean()
  
  return g_loss


class TransfoCrossEntropyLoss(nn.Module):
  def __init__(self, *args, **kwargs):
    super(TransfoCrossEntropyLoss, self).__init__()
    self.loss_func = nn.CrossEntropyLoss(*args, ** kwargs, reduction='none')

  def forward(self, input, target, loss_mask, dim_last=True):
    input_n = input.permute(0, 2, 1) if dim_last else input
    loss = self.loss_func(input_n, target)
    loss = loss * loss_mask
    loss = torch.sum(loss) / torch.sum(loss_mask)
    return loss

class TransfoL1Loss(nn.Module):
  def __init__(self, *args, **kwargs):
    super(TransfoL1Loss, self).__init__()
    self.loss_func = nn.L1Loss(*args, ** kwargs, reduction='none')

  def forward(self, input, target, loss_mask):
    #print(input.shape, target.shape, loss_mask.shape)
    loss_mask = loss_mask[..., None].expand(-1, -1, input.shape[-1])
    loss = self.loss_func(input, target)
    loss = loss * loss_mask
    loss = torch.sum(loss) / torch.sum(loss_mask)
    return loss


def wgan_loss(d_fake, d_real=None, mode='d', mask=None):
  if mode == 'd':
    d_loss = -(d_real.mean() - d_fake.mean())
    return d_loss
  
  elif mode == 'g':
    g_loss = -d_fake.mean()
    return g_loss


def hinge_loss(d_fake, d_real=None, mode ='d', mask=None):
  loss_mask = torch.ones_like(d_fake) if mask == None else mask
  #print(d_fake.shape, loss_mask.shape)

  if mode == 'd':
    loss_real = torch.mean(F.relu(1. - d_real)*loss_mask)
    loss_fake = torch.mean(F.relu(1. + d_fake)*loss_mask)
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss
  elif mode == 'g':
    g_loss = -(d_fake*loss_mask).mean()
    return g_loss
