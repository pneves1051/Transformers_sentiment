import torch

def wgan_loss(discriminator, d_fake, fake=None, d_real=None, real= None, mode='d'):
  if mode == 'd':
    d_loss = -(d_real.mean() - d_fake.mean())
    # Gradient penalty
    eps = torch.rand((d_real.shape[0], 1, 1)).repeat(1, *real.shape[1:]).to(real.device)
    interp = (eps*real+ (1-eps)*fake).to(real.device)
    d_interp = discriminator(interp, None)
    gp = torch.autograd.grad(outputs=d_interp, inputs=interp,
                              grad_outputs=torch.ones_like(d_interp),
                              create_graph=True, retain_graph=True)[0]          
    gp = gp.view(gp.shape[0], -1)
    gp = ((gp.norm(2, dim=1) - 1)**2).mean()     

    d_loss_gp = d_loss + 10*gp
    return d_loss_gp
  
  elif mode == 'g':
    g_loss = -d_fake.mean()
    return g_loss