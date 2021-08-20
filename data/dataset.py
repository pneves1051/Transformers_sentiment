import torch


class TransformerDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, seq_len, cond=False):
      """
        Args:
            dataset: token dataset
            ttps: tokens per second
            seconds: seconds per example
      """
      
      self.ids = []
      self.dataset = []
      self.conditions = []
      
      self.cond = cond
      for i, data in enumerate(dataset['sequences']):
        for j in range(0, len(data)-(seq_len+1), seq_len+1):
          #if dataset['conditions'][i][0] == 18.:
          self.ids.append(dataset['ids'][i])
                    
          # we will use seq[:-1] as input and seq[1:] as target
          self.dataset.append(data[j: j+seq_len+1])

          if cond:
            self.conditions.append(dataset['conditions'][i])
          
      if type(self.ids[0] == str):
        self.ids = torch.arange(len(self.ids)).unsqueeze(-1)
      else:
        self.ids = torch.Tensor(self.ids)#[:4]

      self.dataset = torch.Tensor(self.dataset)#[:4]
      if cond:
        self.conditions = torch.Tensor(self.conditions)#[:4]  

    def __len__(self):
      return len(self.dataset)

    def __getitem__(self, idx):
      input = self.dataset[idx][:-1].long()
      target = self.dataset[idx][1:].long()

      if self.cond:
        batch = {'ids': self.ids[idx], 'inputs': input, 'targets': target, 'conditions': self.conditions[idx].long()}
      else:
        batch = {'ids': self.ids[idx], 'inputs': input, 'targets': target, 'conditions': torch.Tensor([float('nan')])}
      return batch

