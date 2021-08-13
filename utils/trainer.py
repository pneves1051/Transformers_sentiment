class TransformerTrainer():
  def __init__(self, model, dataloader, valid_dataloader, loss_fn, device, lr, accumulation_steps=1):
    self.model = model
    self.dataloader = dataloader
    self.valid_dataloader = valid_dataloader
    self.loss_fn = loss_fn
    self.device=device
    # self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)#, betas=(0.9, 0.98))
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-4)#, betas=(0.9, 0.98))

    self.accumulation_steps=accumulation_steps
    self.history = defaultdict(list)
    
  def train_epoch(self, log_interval=20):
    self.model.train()
       
    losses = []
    correct_predictions = 0.0
    start_time = time.time()
  
    self.optimizer.zero_grad()
    for index, data in enumerate(self.dataloader):
      inputs = data['inputs'].to(self.device)
      targets = data['targets'].to(self.device)
      
      '''
      # time0 = time.time()
      inputs_enc = {
      'inputs': inputs.numpy(),
      }
      indices = vqvae_onnx_list[0].run(None, inputs_enc)[0]
      # _,_,_,indices= self.vqvae.encode(inputs)
      # print(time.time()-time0)
      indices = torch.Tensor(indices).long().to(self.device)
      
      inputs = indices[:, :-1]
      targets = indices[:, 1:]
      '''

      conds = data['conditions']
      if conds is not None:
        conds = conds.to(device)

      if index == 0:
        b_size, seq_len = targets.shape

     
      outputs = self.model(inputs, conds).permute(0, 2, 1)
      clone_out = outputs.clone()
      if (index+1)%log_interval == 0:
        unique, counts = torch.unique(torch.argmax(F.softmax(clone_out[:1], dim=1), dim = 1), sorted=True, return_counts=True)
        print(unique[torch.argsort(counts, descending=True)], len(unique))
        # outputs = self.model(inputs).permute(0, 2, 1)

      self.model.zero_grad()                  
      loss = self.loss_fn(outputs, targets)
      # loss /= self.accumulation_steps
      loss.backward()
      self.optimizer.step()    
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
              index, 2,#len(self.dataloader), 
              2, # self.scheduler.get_last_lr()[0],
              elapsed*1000/log_interval,
              current_loss,  correct_predictions /((index+1)*b_size*seq_len)))
        start_time = time.time()

    train_acc = correct_predictions /((index+1)*b_size*seq_len)
    train_loss = np.mean(losses)
    return train_acc, train_loss, losses

  def train(self, EPOCHS, checkpoint_dir, validate = False, log_interval=20, load=False, save=True, change_lr = False):
    best_accuracy = 0
    total_time = 0
    valid_acc = 0
    valid_loss = 10
    
    if load:
      checkpoint = torch.load(checkpoint_dir + 'tr_checkpoint.pth', map_location='cpu')
      self.model.load_state_dict(checkpoint['model'])
      self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    if change_lr:
      new_tr_lr = 1e-5
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = new_tr_lr
    
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
    self.model.eval()
    eval_losses = []
    eval_correct_predictions = 0
    with torch.no_grad():
      for index, data in enumerate(eval_dataloader):
        
        inputs = data['inputs'].to(self.device)
        targets = data['targets'].to(self.device)
        conds = data['conditions']
        '''
        inputs = data['inputs']#.to(self.device)
        inputs_enc = {
        'inputs': inputs.numpy(),
        }
        indices = vqvae_onnx_list[0].run(None, inputs_enc)[0]
        # _,_,_,indices= self.vqvae.encode(inputs)
        # print(time.time()-time0)
        indices = torch.Tensor(indices).long().to(self.device)
        
        inputs = indices[:, :-1]
        targets = indices[:, 1:]       
        '''
        if conds is not None:
          conds = conds.to(device)

        if index == 0:
          b_size, seq_len = targets.shape
        
        if self.hf:
          outputs = self.model(inputs)['logits'].transpose(-1,-2)#, mask)['logits']
        elif self.pf:
          mask = torch.ones_like(inputs).bool()
          outputs = self.model(inputs, mask=mask).transpose(-1,-2)
        else:
          outputs = self.model(inputs, conds).transpose(-1, -2)#, conds).transpose(-1, -2)

        eval_loss = self.loss_fn(outputs, targets)
        
        preds = torch.argmax(F.softmax(outputs, dim=1), dim = 1)
        eval_correct_predictions += torch.sum(preds == targets)
        eval_losses.append(eval_loss.item())
                
    eval_acc = eval_correct_predictions /((index+1)*b_size*seq_len)
    eval_loss = np.mean(eval_losses)
        
    return eval_acc, eval_loss

  def save_model(self, checkpoint_dir):
    torch.save(self.model.state_dict(), checkpoint_dir + 'best_transformer_state.bin')
  
  def save_checkpoint(self, checkpoint_dir):
    checkpoint = { 
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        
    torch.save(checkpoint, checkpoint_dir + 'tr_checkpoint.pth')
