import time
import math
from collections import defaultdict
from functools import partial
import pickle as pkl
import numpy as np
import torch
import torch.nn.functional as F

from .trainer import warmup_constant_lambda, warmup_cosine_lambda

class ClassifierTrainer():
  def __init__(self, model, dataloader, valid_dataloader, ce_loss,
               device, lr, vocab_size, warmup_steps=0, total_iters=30000, schedule='constant'):
    self.model=model

    self.dataloader = dataloader
    self.valid_dataloader = valid_dataloader
    self.ce_loss = ce_loss
    
    self.device=device
    
    self.lr = lr
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)#, betas=(0.9, 0.98))

    self.vocab_size = vocab_size
    self.num_iters = 0
    
    self.history = defaultdict(list)

    self.schedule = schedule
    self.warmup_steps = warmup_steps
    
    if schedule == 'constant':
      self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                            lambda x: 1.)

    elif schedule == 'constant_with_warmup':
      self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                            partial(warmup_constant_lambda, warmup_steps = self.warmup_steps))
    
    elif schedule == 'cosine_with_warmup':
      self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                            partial(warmup_cosine_lambda, warmup_steps = self.warmup_steps,
                                    training_steps=total_iters, cycles=0.5))
    else:
      raise KeyError('Schedule ' + schedule + ' not found.')

  
  def train_epoch(self, log_interval=20):
    self.model.train()
       
    losses = []
    correct_predictions = 0.0
    total_elements = 0.0
    start_time = time.time()
    index=0
  
    if isinstance(self.dataloader, list):
      # loader = chain(*self.dataloader)
      #len_loader = sum([len(l) for l in self.dataloader]) 
      loader = zip(*self.dataloader)
      len_loader =  len(self.dataloader)*min([len(l) for l in self.dataloader] )
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

        outputs = self.model(input, input_mask=input_mask)
        
        self.optimizer.zero_grad()                  
        loss = self.ce_loss(outputs, target)
        loss.backward()
        # nn.utils.clip_grad_norm_(self.generator.parameters(), 3.0)
        self.optimizer.step()    
        
        self.scheduler.step()

        preds = torch.argmax(F.softmax(outputs, dim=-1), dim = -1)

        correct_predictions += torch.sum(preds == target)
        total_elements += input.shape[0]
        #print(set(preds.reshape(-1).tolist()))
        losses.append(loss.item())
                    
        if index % log_interval == 0 and index > 0:
          elapsed = time.time() - start_time
          current_loss = np.mean(losses)
          print('| {:5d} of {:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                'loss {:5.6f} | acc {:8.6f} | num_iters: {}'.format(
                index, len_loader, 
                self.scheduler.get_last_lr()[0],
                elapsed*1000/log_interval,
                current_loss,  correct_predictions / total_elements,
                self.num_iters))
          start_time = time.time()
        index+=1
        self.num_iters += 1

    train_acc = correct_predictions/total_elements
    train_loss = np.mean(losses)
    return train_acc, train_loss, losses

  
  def train(self, EPOCHS, checkpoint_dir, validate = False, log_interval=20, load=False, save=True):
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
    eval_correct_predictions = 0.0
    eval_total_elements = 0.0
    with torch.no_grad():
      for index, data in enumerate(eval_dataloader):
        
        input = data['input'].to(self.device)
        target = data['target'].to(self.device)
        input_mask = data['input_mask'].to(self.device)
  
        outputs = self.model(input, input_mask=input_mask)

        
        eval_loss = self.ce_loss(outputs, target)
        
        preds = torch.argmax(F.softmax(outputs, dim=-1), dim = -1)
        eval_correct_predictions += torch.sum(preds == target)
        eval_total_elements += input.shape[0]
        eval_losses.append(eval_loss.item())
                
    eval_acc = eval_correct_predictions / eval_total_elements
    eval_loss = np.mean(eval_losses)
        
    return eval_acc, eval_loss

  def load_checkpoint(self, checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir + 'classifier_checkpoint.pth', map_location='cpu')
    
    print("Loading Gen")
    self.model.load_state_dict(checkpoint['model'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])
    self.scheduler.load_state_dict(checkpoint['scheduler'])


    with open(checkpoint_dir + 'classifier_history.pkl', 'rb') as f:
      self.history = pkl.load(f)
    self.num_iters = sum([len(tl) for tl in self.history['train_losses']])
    print(self.num_iters)

  
  def save_checkpoint(self, checkpoint_dir):
    checkpoint = { 
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        
    torch.save(checkpoint, checkpoint_dir + 'classifier_checkpoint.pth')
    with open(checkpoint_dir + 'classifier_history.pkl', 'wb') as f:
      pkl.dump(self.history, f)



