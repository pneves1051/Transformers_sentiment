

class TransformerDatasetNoCond(torch.utils.data.Dataset):
    
    def __init__(self, dataset, seq_len):
      """
        Args:
            dataset: token dataset
            ttps: tokens per second
            seconds: seconds per example
      """
      
      self.ids = []
      self.dataset = []
      self.conditions = []
          
      for i, data in enumerate(dataset['inputs']):
        for j in range(0, len(data)-(seq_len+1), seq_len+1):
          #if dataset['conditions'][i][0] == 18.:
          self.ids.append(dataset['ids'][i])
                    
          # we will use seq[:-1] as input and seq[1:] as target
          self.dataset.append(data[j: j+seq_len+1])

          self.conditions.append(dataset['conditions'][i])
          
      self.ids = torch.Tensor(self.ids)#[:4]
      self.dataset = torch.Tensor(self.dataset)#[:4]
      self.conditions = torch.Tensor(self.conditions)#[:4]  

    def __len__(self):
      return len(self.dataset)

    def __getitem__(self, idx):
      input = self.dataset[idx][:-1].long()
      target = self.dataset[idx][1:].long()

      return {'ids': self.ids[idx], 'inputs': input, 'targets': target, 'conditions': self.conditions[idx].long()}
