from model.transformer import Discriminator
import time
from itertools import chain
from functools import partial
from collections import defaultdict
import pickle as pkl
import math
import numpy as np
import torch
import torch.nn.functional as F
from utils.losses import TransfoL1Loss


# adapted from https://huggingface.co/transformers/_modules/transformers/optimization.html#get_constant_schedule_with_warmup
def warmup_constant_lambda(step, warmup_steps):
  if step < warmup_steps:
    return step/max(1.0, warmup_steps)
  else:
    return 1

def warmup_cosine_lambda(step, warmup_steps, training_steps, cycles = 0.5):
  if step < warmup_steps:
    return step/max(1.0, warmup_steps)
  else:
    progress = float(step - warmup_steps) / float(max(1, training_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(cycles) * 2.0 * progress)))



class TransformerTrainer():
  def __init__(self, generator, discriminator, dataloader, valid_dataloader, ce_loss,
               gan_loss, device, g_lr, d_lr, vocab_size, d_iters=5, total_iters=100000,
               warmup_steps=0, pretraining_steps=0, temperature=100, gan_hp = 1, schedule='constant', local_loss = None):
    self.generator = generator
    self.discriminator = discriminator

    self.dataloader = dataloader
    self.valid_dataloader = valid_dataloader
    self.ce_loss = ce_loss
    self.gan_loss = gan_loss
    self.local_loss = local_loss
    
    self.device=device
    
    self.g_lr = g_lr
    self.d_lr = d_lr
    self.g_optimizer_mle = torch.optim.Adam(self.generator.parameters(), lr = g_lr)#, betas=(0.9, 0.98))
    self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr = g_lr)#, betas=(0.9, 0.98))
    self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = d_lr)#, betas=(0.9, 0.98))

    self.vocab_size = vocab_size
    self.d_iters = d_iters
    self.temperature = temperature
    self.gan_hp = gan_hp
    self.num_iters = 0
    self.total_iters = total_iters
    
    self.history = defaultdict(list)

    self.L1Loss = TransfoL1Loss()

    self.schedule = schedule
    self.warmup_steps = warmup_steps
    self.pretraining_steps = pretraining_steps
    
    if schedule == 'constant':
      self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer_mle,
                            lambda x: 1.)
    elif schedule == 'constant_with_warmup':
      self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer_mle,
                            partial(warmup_constant_lambda, warmup_steps = self.warmup_steps))
    
    elif schedule == 'cosine_with_warmup':
      self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer_mle,
                            partial(warmup_cosine_lambda, warmup_steps = self.warmup_steps,
                                    training_steps=total_iters, cycles=0.5))
    else:
      raise KeyError('Schedule ' + schedule + ' not found.')

   
  def get_loss_hp(self, rec_loss, g_loss, last_layer):
    rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    
    disc_weight = torch.linalg.norm(rec_grads.flatten(1), 'fro') /(torch.linalg.norm(g_grads.flatten(1), 'fro') + 1e-4)
    disc_weight = torch.clamp(disc_weight, 0.0, 1e5).detach()
    
    return disc_weight

  def get_loss_hp2(self, mle_grads, g_grads):
    disc_weight = torch.linalg.norm(mle_grads.flatten(1), 'fro') /(torch.linalg.norm(g_grads.flatten(1), 'fro') + 1e-4)
    disc_weight = torch.clamp(disc_weight, 0.0, 1e5).detach()
    
    return disc_weight
  
  def get_gp(self, fake, real, add_disc_inputs, norm_value=1, pos=0):
    # Gradient penalty
    eps = torch.rand((real.shape[0], 1, 1)).repeat(1, *real.shape[1:]).to(real.device)
    interp = (eps*real+ (1-eps)*fake).to(real.device)
    interp = torch.autograd.Variable(interp, requires_grad=True)
    
    #feed conditions and mask to disc
    d_interp = self.discriminator(interp, *add_disc_inputs)[pos]
    gp = torch.autograd.grad(outputs=d_interp, inputs=interp,
                              grad_outputs=torch.ones_like(d_interp),
                              create_graph=True, retain_graph=True)[0]          
    gp = gp.view(gp.shape[0], -1)
    gp = ((gp.norm(2, dim=1) - norm_value)**2).mean() 
    return gp
  
  def train_epoch(self, log_interval=20):
    self.generator.train()
       
    losses = []
    correct_predictions = 0.0
    total_elements = 0.0
    start_time = time.time()
    index=0
  
    if isinstance(self.dataloader, list):
      # loader = chain(*self.dataloader)
      #len_loader = sum([len(l) for l in self.dataloader]) 
      loader = zip(*self.dataloader)
      len_loader =  len(self.dataloader)*min([len(l) for l in self.dataloader])
    else:
      loader = self.dataloader
      len_loader = len(self.dataloader)
    for data_pack in loader:
      # Solution to allow one or more dataloaders simultaneously
      data_pack = [data_pack] if not isinstance(self.dataloader, list) else data_pack
      for data in data_pack:
        input = data['input'].to(self.device)
        target = data['target'].to(self.device)
        input_mask = data['input_mask'].to(self.device)
        target_mask = data['target_mask'].to(self.device)

        if 'conditions' in data:
          conds = data['conditions'].to(self.device)
        else:
          conds = None

        if index == 0:
          b_size, seq_len = target.shape
        
        outputs,_= self.generator(input, cond=conds, input_mask=input_mask)
        #mask = torch.ones_like(inputs).bool()
        #outputs = self.generator(inputs, mask=mask)
        
        self.g_optimizer_mle.zero_grad()                  
        loss = self.ce_loss(outputs, target, loss_mask=target_mask)
        # loss /= self.accumulation_steps
        loss.backward()
        # nn.utils.clip_grad_norm_(self.generator.parameters(), 3.0)
        self.g_optimizer_mle.step()    
        
        self.g_scheduler.step()

        preds = torch.argmax(F.softmax(outputs, dim=-1), dim = -1)

        correct_predictions += torch.sum((preds == target)*target_mask)
        total_elements += torch.sum(target_mask)
        #print(set(preds.reshape(-1).tolist()))
        losses.append(loss.item())
                    
        if index % log_interval == 0 and index > 0:
          elapsed = time.time() - start_time
          current_loss = np.mean(losses)
          print('| {:5d} of {:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                'loss {:5.6f} | acc {:8.6f} | num_iters: {}'.format(
                index, len_loader, 
                self.g_scheduler.get_last_lr()[0],
                elapsed*1000/log_interval,
                current_loss,  correct_predictions / total_elements,
                self.num_iters))
          start_time = time.time()
        index+=1
        self.num_iters += 1

      if self.num_iters > self.pretraining_steps:
        break
    
    train_acc = correct_predictions/total_elements
    train_loss = np.mean(losses)
    return train_acc, train_loss, losses

  def train_epoch_gan(self, log_interval=40):
    self.generator.train()
    self.discriminator.train()

    losses = []
    d_losses = []
    g_losses = []
    correct_predictions = 0.0
    total_elements = 0.0
    start_time = time.time()
    index = 0
  
    if isinstance(self.dataloader, list):
      # loader = chain(*self.dataloader)
      #len_loader = sum([len(l) for l in self.dataloader]) 
      loader = zip(*self.dataloader)
      len_loader = len(self.dataloader)*min([len(l) for l in self.dataloader] )
    else:
      loader = self.dataloader
      len_loader = len(self.dataloader)
    for data_pack in loader:
      # Solution to allow one or more dataloaders simultaneously
      data_pack = [data_pack] if not isinstance(self.dataloader, list) else data_pack
      for data in data_pack:
        input = data['input'].to(self.device)
                
        target = data['target'].to(self.device)
        input_mask = data['input_mask'].to(self.device)
        target_mask = data['target_mask'].to(self.device)

        if 'conditions' in data:
          conds = data['conditions'].to(self.device)
        else:
          conds = None

        if index == 0:
          b_size, seq_len = target.shape
      
        #########
        # D update
        #########      
        # Allows D to be updated
        for p in self.discriminator.parameters():
          p.requires_grad = True
        
        for _ in range(self.d_iters):
          
          d_real, d_real_local = self.discriminator(F.one_hot(input, num_classes=self.vocab_size), cond=conds, input_mask=input_mask)
          
          inverse_temperature = self.get_inverse_temperature()
          fake, fake_gumbel = self.generator(input, cond=conds, inverse_temperature=inverse_temperature, input_mask=input_mask)
          d_fake, d_fake_local = self.discriminator(fake_gumbel, cond=conds, input_mask=input_mask)

          '''
          clone_out = fake.clone()
          if (index+1)%log_interval == 0:
            unique, counts = torch.unique(torch.argmax(F.softmax(clone_out[:1], dim=1), dim = 1), sorted=True, return_counts=True)
            print(unique[torch.argsort(counts, descending=True)], len(unique))
          '''

          # Chunk and calculate loss
          # TEST HINGE LOSS
          #d_loss = self.gan_loss(self.discriminator, d_fake, fake_gumbel.data, d_real, F.one_hot(target, num_classes=self.vocab_size).data,
          #                      mode='d', add_disc_inputs=[conds, target_mask])
          d_loss = self.gan_loss(d_fake, d_real, mode ='d')
          gp = self.get_gp(fake, F.one_hot(target, num_classes=self.vocab_size), [conds, target_mask], norm_value=1)
          if self.local_loss != None:
            d_loss_local = self.local_loss(d_fake_local, d_real_local, mode='d', mask=self.discriminator.get_patch_loss_mask(target_mask).unsqueeze(-1))
            gp_local = self.get_gp(fake, F.one_hot(target, num_classes=self.vocab_size), [conds, target_mask], norm_value=1, pos=1)
            #print(d_loss.item(), d_loss_local.item())
            d_loss += d_loss_local
            gp += gp_local

          d_loss = d_loss + 10*gp         
          self.d_optimizer.zero_grad()                  
          # loss /= self.accumulation_steps
          d_loss.backward()
          #nn.utils.clip_grad_norm_(self.discriminator.parameters(), 3.0)
          self.d_optimizer.step()    
          #self.scheduler.step()

          d_losses.append(d_loss.item())
            
        #self.d_scheduler.step()
        #########
        # G update
        #########      
        # Stops D from being updated
        for p in self.discriminator.parameters():
          p.requires_grad = False      
        
        inverse_temperature=self.get_inverse_temperature()

        fake,_ = self.generator(input, cond=conds, inverse_temperature=inverse_temperature, input_mask=input_mask)

        mle_loss = self.ce_loss(fake, target, loss_mask=target_mask)  

        mle_grads = torch.autograd.grad(mle_loss, self.generator.get_last_layer(), retain_graph=True)[0]

        self.g_optimizer_mle.zero_grad()    
        mle_loss.backward()
        self.g_optimizer_mle.step()    
        self.g_scheduler.step()


        d_real, d_real_local = self.discriminator(F.one_hot(input, num_classes=self.vocab_size), cond=conds, input_mask=input_mask)
                
        fake, fake_gumbel = self.generator.forward_recurrent(input, cond=conds, inverse_temperature=inverse_temperature, input_mask=input_mask)
        d_fake, d_fake_local  = self.discriminator(fake_gumbel, cond=conds, input_mask=input_mask)

        '''
        feat_loss = 0
        for feat_fake, feat_real in zip(feats_fake, feats_real):
          feat_loss += self.L1Loss(feat_fake, feat_real.detach(), self.discriminator.get_patch_loss_mask(target_mask))
        '''

        # TEST HINGE LOSS
        #gan_g_loss = self.gan_loss(self.discriminator, d_fake, mode='g')
        gan_g_loss = self.gan_loss(d_fake, d_real, mode='g')
        if self.local_loss != None:
          gan_g_loss_local = self.local_loss(d_fake_local, d_real_local, mode='g', mask=self.discriminator.get_patch_loss_mask(target_mask).unsqueeze(-1))
          #print(gan_g_loss.item(), gan_g_loss_local.item())
          gan_g_loss += gan_g_loss_local

        # loss_hp = self.get_loss_hp(mle_loss, gan_g_loss, self.generator.get_last_layer())
        g_grads = torch.autograd.grad(gan_g_loss, self.generator.get_last_layer(), retain_graph=True)[0]
        loss_hp = self.get_loss_hp2(mle_grads, g_grads)
      
        #g_loss = mle_loss + self.gan_hp*gan_g_loss
                
        self.g_optimizer.zero_grad()    
        #g_loss.backward()
        gan_g_loss.backward()
        #nn.utils.clip_grad_norm_(self.generator.parameters(), 3.0)
        self.g_optimizer.step()    
        
        g_losses.append(gan_g_loss.item())
        losses.append(mle_loss.item())

        preds = torch.argmax(F.softmax(fake, dim=-1), dim = -1)     

        correct_predictions += torch.sum((preds == target)*target_mask)
        total_elements += torch.sum(target_mask)
        #print(set(preds.reshape(-1).tolist()))
                    
        if index % log_interval == 0 and index > 0:
          elapsed = time.time() - start_time
          current_loss = np.mean(losses)
          current_d_loss = np.mean(d_losses)
          current_g_loss = np.mean(g_losses)

          print('| {:5d} of {:5d} batches | d_lr {:02.7f} | g_lr {:02.7f} | ms/batch {:5.2f} | '
                'loss {:5.6f} | acc {:8.6f} |  D_loss: {} |  G_loss: {} | loss_hp: {} |'
                ' inverse_temperature: {} | num_iters: {} '.format(
                index, len_loader, 
                self.g_lr, self.d_lr,
                #self.d_scheduler.get_last_lr()[0], self.g_scheduler.get_last_lr()[0],
                elapsed*1000/log_interval,
                current_loss,  correct_predictions/total_elements, 
                current_d_loss, current_g_loss, loss_hp,
                inverse_temperature, self.num_iters))
          start_time = time.time()

        self.num_iters += 1
        index += 1

      if self.num_iters > self.total_iters:
        break

    train_acc = correct_predictions / total_elements
    train_loss = np.mean(losses)
    return train_acc, train_loss, losses, d_losses, g_losses
  
  def train(self, EPOCHS, checkpoint_dir, validate = False, log_interval=20, load=False, save=True, train_gan=False):
    best_accuracy = 0
    total_time = 0
    valid_acc = 0
    valid_loss = 10
    
    if load:
        self.load_checkpoint(checkpoint_dir)
 
    for epoch in range(EPOCHS):
      epoch_start_time = time.time()
      print(f'Epoch {epoch + 1}/{EPOCHS}')

      print('-' * 10)

      d_losses, g_losses = [0], [0]
      if self.num_iters < self.pretraining_steps:
        train_acc, train_loss, train_losses = self.train_epoch(log_interval=log_interval) 
      else:
        train_acc, train_loss, train_losses, d_losses, g_losses = self.train_epoch_gan(log_interval=log_interval)
      

      self.history['train_acc'].append(train_acc)
      self.history['train_loss'].append(train_loss)
      self.history['train_losses'].append(train_losses)
      self.history['d_losses'].append(d_losses)
      self.history['g_losses'].append(g_losses)
      total_time += time.time() - epoch_start_time
      self.history['time'].append(total_time)
      if validate:
        valid_acc, valid_loss = self.evaluate(self.valid_dataloader)
        self.history['valid_acc'].append(valid_acc)
        self.history['valid_loss'].append(valid_loss)
      
      print('| End of epoch {:3d}  | time: {:5.4f}s | train loss {:5.6f} | '
            'train ppl {:8.4f} | \n train accuracy {:5.6f} | valid loss {:5.6f} | '
            'valid ppl {:8.6f} | valid accuracy {:5.6f} | D_loss: {:5.6f} | G_loss: {:5.6f} | '.format(
            epoch+1, (time.time()-epoch_start_time), train_loss, math.exp(train_loss), train_acc,
            valid_loss, math.exp(valid_loss), valid_acc, np.mean(d_losses) , np.mean(g_losses)))

      if save:
        if validate and valid_acc > best_accuracy :
          self.save_checkpoint(checkpoint_dir)
          best_accuracy = valid_acc
        elif train_acc > best_accuracy:
          self.save_checkpoint(checkpoint_dir)
          best_accuracy = train_acc
      
      if self.num_iters > self.total_iters:
        break
    return self.history

  def evaluate(self, eval_dataloader):
    self.generator.eval()
    eval_losses = []
    eval_correct_predictions = 0.0
    eval_total_elements = 0.0
    with torch.no_grad():
      for index, data in enumerate(eval_dataloader):
        
        input = data['input'].to(self.device)
        target = data['target'].to(self.device)
        input_mask = data['input_mask'].to(self.device)
        target_mask = data['target_mask'].to(self.device)

        if 'conditions' in data:
          conds = data['conditions'].to(self.device)
        else:
          conds = None

        if index == 0:
          b_size, seq_len = target.shape
  
        outputs, _ = self.generator(input, cond=conds, input_mask=input_mask)
        #mask = torch.ones_like(inputs).bool()
        #outputs = self.generator(inputs, mask=mask)
        
        eval_loss = self.ce_loss(outputs, target, loss_mask=target_mask)
        
        preds = torch.argmax(F.softmax(outputs, dim=-1), dim = -1)
        eval_correct_predictions += torch.sum((preds == target) * target_mask)
        eval_total_elements += torch.sum(target_mask)
        eval_losses.append(eval_loss.item())
                
    eval_acc = eval_correct_predictions / eval_total_elements
    eval_loss = np.mean(eval_losses)
        
    return eval_acc, eval_loss

  def get_inverse_temperature(self):
    if self.num_iters <= self.pretraining_steps:
      inverse_temperature = 1.0
    else:
      inverse_temperature = self.temperature**((self.num_iters-self.pretraining_steps)/(self.total_iters-self.pretraining_steps))
    return inverse_temperature
  
  def save_model(self, checkpoint_dir):
    torch.save(self.model.state_dict(), checkpoint_dir + 'best_transformer_state.bin')
  
  def load_checkpoint(self, checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir + 'tr_checkpoint.pth', map_location='cpu')
    
    print("Loading Gen")
    self.generator.load_state_dict(checkpoint['generator'])
    self.g_optimizer_mle.load_state_dict(checkpoint['g_optimizer_mle'])
    self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
    self.g_scheduler.load_state_dict(checkpoint['g_scheduler'])

    with open(checkpoint_dir + 'history.pkl', 'rb') as f:
      self.history = pkl.load(f)
    self.num_iters = sum([len(tl) for tl in self.history['train_losses']])
    print(self.num_iters)

    # If pretraining is not complete, the discriminator does not have to be loades
    if self.num_iters > self.pretraining_steps:
      print("Loading Disc")
      self.discriminator.load_state_dict(checkpoint['discriminator'])
      self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
    #self.d_scheduler.load_state_dict(checkpoint['d_scheduler'])

  
  def save_checkpoint(self, checkpoint_dir):
    checkpoint = { 
            'generator': self.generator.state_dict(),
            'g_optimizer_mle': self.g_optimizer_mle.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict()}
            #'d_scheduler': self.d_scheduler.state_dict()}
        
    torch.save(checkpoint, checkpoint_dir + 'tr_checkpoint.pth')
    with open(checkpoint_dir + 'history.pkl', 'wb') as f:
      pkl.dump(self.history, f)







