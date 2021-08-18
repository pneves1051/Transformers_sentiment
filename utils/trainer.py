import time
from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn.functional as F

class TransformerTrainer():
  def __init__(self, generator, discriminator, dataloader, valid_dataloader, ce_loss, gan_loss, device, lr,
               vocab_size, d_iters=5, total_iters=100000, temperature=100, gan_hp = 1,accumulation_steps=1):
    self.generator = generator
    self.discriminator = discriminator

    self.dataloader = dataloader
    self.valid_dataloader = valid_dataloader
    self.ce_loss = ce_loss
    self.gan_loss = gan_loss
    self.device=device
    
    self.g_optimizer = torch.optim.AdamW(self.generator.parameters(), lr = lr)#, betas=(0.9, 0.98))
    self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr = lr)#, betas=(0.9, 0.98))

    self.vocab_size = vocab_size
    self.d_iters = d_iters
    self.temperature = temperature
    self.gan_hp = gan_hp
    self.num_iters = 0
    self.total_iters = total_iters
    self.accumulation_steps=accumulation_steps

    self.history = defaultdict(list)
    
  def train_epoch(self, log_interval=20):
    self.generator.train()
       
    losses = []
    correct_predictions = 0.0
    start_time = time.time()
  
    for index, data in enumerate(self.dataloader):
      inputs = data['inputs'].to(self.device)
      targets = data['targets'].to(self.device)

      conds = data['conditions']
      if conds[0] is not float('nan'):
        conds = conds.to(self.device)

      if index == 0:
        b_size, seq_len = targets.shape
     
      outputs,_= self.generator(inputs, conds)
      '''
      clone_out = outputs.clone()
      if (index+1)%log_interval == 0:
        unique, counts = torch.unique(torch.argmax(F.softmax(clone_out[:1], dim=1), dim = 1), sorted=True, return_counts=True)
        print(unique[torch.argsort(counts, descending=True)], len(unique))
      '''
        
      self.g_optimizer.zero_grad()                  
      loss = self.ce_loss(outputs.permute(0, 2, 1), targets)
      # loss /= self.accumulation_steps
      loss.backward()
      self.g_optimizer.step()    
      #self.scheduler.step()

      '''
      if (index+1) % self.accumulation_steps == 0:       
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()                       
        # self.scheduler.step()
        self.model.zero_grad()                  
      '''
      preds = torch.argmax(F.softmax(outputs, dim=1), dim = 1)

      correct_predictions += torch.sum(preds == targets)
      #print(set(preds.reshape(-1).tolist()))
      losses.append(loss.item()*self.accumulation_steps)
                  
      if index % log_interval == 0 and index > 0:
        elapsed = time.time() - start_time
        current_loss = np.mean(losses)
        print('| {:5d} of {:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
              'loss {:5.6f} | acc {:8.6f}'.format(
              index, len(self.dataloader), 
              2, # self.scheduler.get_last_lr()[0],
              elapsed*1000/log_interval,
              current_loss,  correct_predictions /((index+1)*b_size*seq_len)))
        start_time = time.time()

    train_acc = correct_predictions /((index+1)*b_size*seq_len)
    train_loss = np.mean(losses)
    return train_acc, train_loss, losses

  def train_epoch_gan(self, log_interval=40):
    self.generator.train()
    self.discriminator.train

    losses = []
    correct_predictions = 0.0
    start_time = time.time()
  
    for index, data in enumerate(self.dataloader):
      inputs = data['inputs'].to(self.device)
      targets = data['targets'].to(self.device)

      conds = data['conditions']
      if conds[0] is not float('nan'):
        conds = conds.to(self.device)

      if index == 0:
        b_size, seq_len = targets.shape
     
      #########
      # D update
      #########      
      # Allows D to be updated
      for p in self.discriminator.parameters():
        p.requires_grad = True
      
      for _ in range(self.d_iters):
        
        d_real = self.discriminator(F.one_hot(inputs, num_classes=self.vocab_size), conds)
        
        temperature = self.get_temperature()
        fake, fake_gumbel = self.generator(inputs, conds, temperature)
        d_fake = self.discriminator(fake_gumbel, conds)

        '''
        clone_out = fake.clone()
        if (index+1)%log_interval == 0:
          unique, counts = torch.unique(torch.argmax(F.softmax(clone_out[:1], dim=1), dim = 1), sorted=True, return_counts=True)
          print(unique[torch.argsort(counts, descending=True)], len(unique))
        '''

        d_loss = self.gan_loss(self.discriminator, d_fake, fake_gumbel, d_real, F.one_hot(targets, num_classes=self.vocab_size), mode='d')
                       
        self.d_optimizer.zero_grad()                  
        # loss /= self.accumulation_steps
        d_loss.backward()
        self.d_optimizer.step()    
        #self.scheduler.step()

        '''
        if (index+1) % self.accumulation_steps == 0:       
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
          self.optimizer.step()                       
          # self.scheduler.step()
          self.model.zero_grad()                  
        '''
        

      #########
      # G update
      #########      
      # Stops D from being updated
      for p in self.discriminator.parameters():
        p.requires_grad = False      
      
      temperature=self.get_temperature()
      fake, fake_gumbel = self.generator(inputs, conds, temperature)
      d_fake = self.discriminator(fake_gumbel, conds)

      '''
      clone_out = fake.clone()
      if (index+1)%log_interval == 0:
        unique, counts = torch.unique(torch.argmax(F.softmax(clone_out[:1], dim=1), dim = 1), sorted=True, return_counts=True)
        print(unique[torch.argsort(counts, descending=True)], len(unique))
      '''

      mle_loss = self.ce_loss(fake.permute(0,2,1), targets)              
      gan_g_loss = self.gan_loss(self.discriminator, d_fake, mode='g')
      g_loss = mle_loss + self.gan_hp*gan_g_loss
      # loss /= self.accumulation_steps
      self.g_optimizer.zero_grad()    
      g_loss.backward()
      self.g_optimizer.step()    
      #self.scheduler.step()

      '''
      if (index+1) % self.accumulation_steps == 0:       
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()                       
        # self.scheduler.step()
        self.model.zero_grad()                  
      '''
      preds = torch.argmax(F.softmax(fake, dim=-1), dim = -1)     

      correct_predictions += torch.sum(preds == targets)
      #print(set(preds.reshape(-1).tolist()))
      losses.append(mle_loss.item()*self.accumulation_steps)
                  
      if index % log_interval == 0 and index > 0:
        elapsed = time.time() - start_time
        current_loss = np.mean(losses)
        print('| {:5d} of {:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
              'loss {:5.6f} | acc {:8.6f}'.format(
              index, len(self.dataloader), 
              2, # self.scheduler.get_last_lr()[0],
              elapsed*1000/log_interval,
              current_loss,  correct_predictions /((index+1)*b_size*seq_len)))
        start_time = time.time()

        self.num_iters += 1

    train_acc = correct_predictions /((index+1)*b_size*seq_len)
    train_loss = np.mean(losses)
    return train_acc, train_loss, losses
  
  
  def train(self, EPOCHS, checkpoint_dir, validate = False, log_interval=20, load=False, save=True, change_lr = False, train_gan=False):
    best_accuracy = 0
    total_time = 0
    valid_acc = 0
    valid_loss = 10
    
    if load:
        self.load_checkpoint(checkpoint_dir)

    if change_lr:
      new_tr_lr = 1e-5
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = new_tr_lr
    
    for epoch in range(EPOCHS):
      epoch_start_time = time.time()
      print(f'Epoch {epoch + 1}/{EPOCHS}')

      print('-' * 10)

      if train_gan:
        train_acc, train_loss, train_losses = self.train_epoch_gan(log_interval=log_interval)
      else:
        train_acc, train_loss, train_losses = self.train_epoch(log_interval=log_interval) 
      
      self.history['train_acc'].append(train_acc)
      self.history['train_loss'].append(train_loss)
      self.history['train_losses'].append(train_losses)
      total_time += time.time() - epoch_start_time
      self.history['time'].append(total_time)
      if validate:
        valid_acc, valid_loss = self.evaluate(self.valid_dataloader)
        self.history['valid_acc'].append(valid_acc)
        self.history['valid_loss'].append(valid_loss)
      print('| End of epoch {:3d}  | time: {:5.4f}s | train loss {:5.6f} | '
            'train ppl {:8.4f} | \n train accuracy {:5.6f} | valid loss {:5.6f} | '
            'valid ppl {:8.6f} | valid accuracy {:5.6f} |'.format(
            epoch+1, (time.time()-epoch_start_time), train_loss, math.exp(train_loss), train_acc,
            valid_loss, math.exp(valid_loss), valid_acc))

      if save:
        if validate and valid_acc > best_accuracy :
          self.save_checkpoint(checkpoint_dir)
          best_accuracy = valid_acc
        elif train_acc > best_accuracy:
          self.save_checkpoint(checkpoint_dir)
          best_accuracy = train_acc

    return self.history

  def evaluate(self, eval_dataloader):
    self.generator.eval()
    eval_losses = []
    eval_correct_predictions = 0
    with torch.no_grad():
      for index, data in enumerate(eval_dataloader):
        
        inputs = data['inputs'].to(self.device)
        targets = data['targets'].to(self.device)
        conds = data['conditions']
        
        if conds[0] is not float('nan'):
          conds = conds.to(self.device)

        if index == 0:
          b_size, seq_len = targets.shape
        
        if self.hf:
          outputs = self.generator(inputs)['logits'].transpose(-1,-2)#, mask)['logits']
        elif self.pf:
          mask = torch.ones_like(inputs).bool()
          outputs = self.generator(inputs, mask=mask).transpose(-1,-2)
        else:
          outputs = self.generator(inputs, conds).transpose(-1, -2)#, conds).transpose(-1, -2)

        eval_loss = self.ce_loss(outputs, targets)
        
        preds = torch.argmax(F.softmax(outputs, dim=1), dim = 1)
        eval_correct_predictions += torch.sum(preds == targets)
        eval_losses.append(eval_loss.item())
                
    eval_acc = eval_correct_predictions /((index+1)*b_size*seq_len)
    eval_loss = np.mean(eval_losses)
        
    return eval_acc, eval_loss

  def get_temperature(self):
    temperature = self.temperature**(self.num_iters/self.total_iters)
    return temperature
  
  def save_model(self, checkpoint_dir):
    torch.save(self.model.state_dict(), checkpoint_dir + 'best_transformer_state.bin')
  
  def load_checkpoint(self, checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir + 'tr_checkpoint.pth', map_location='cpu')
    self.generator.load_state_dict(checkpoint['generator'])
    self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
    self.discriminator.load_state_dict(checkpoint['discrminator'])
    self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
  
  def save_checkpoint(self, checkpoint_dir):
    checkpoint = { 
            'generator': self.generator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict()}
        
    torch.save(checkpoint, checkpoint_dir + 'tr_checkpoint.pth')
