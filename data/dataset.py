import math
import glob
import pickle as pkl
import pandas as pd
import numpy as np
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


class TransformerDataset2(torch.utils.data.Dataset):
   
    def __init__(self, dataset, seq_len, cond=False, pad_token = 0):
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
      
        data_len = len(data)
        #print(int((seq_len+1)*math.ceil(data_len/(seq_len+1))-data_len))
        padded_data = np.pad(data, (0, int((seq_len+1)*math.ceil(data_len/(seq_len+1))-data_len)), mode='constant', constant_values=(pad_token, pad_token))
        split_data = np.split(padded_data, padded_data.shape[-1]//(seq_len+1))
        self.dataset.extend(split_data)
          
      
        num_seq = len(split_data)
        self.ids.extend([dataset['ids'][i]]*num_seq)
                    
        if cond:
          self.conditions.append([dataset['conditions'][i]]*num_seq)
          
      if type(self.ids[0] == str):
        self.ids = torch.arange(len(self.ids)).unsqueeze(-1)
      else:
        self.ids = torch.Tensor(self.ids)#[:4]

      self.dataset = torch.Tensor(self.dataset)#[:4]
      if cond:
        self.conditions = torch.Tensor(self.conditions)#[:4]  
      
      print(self.dataset[:2])

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


class TransformerDatasetREMI(torch.utils.data.Dataset):
   
    def __init__(self, dataset_path, seq_len, cond_path=None, pad_idx = 0, eos_idx=1):
      """
        Args:
            dataset: token dataset
            ttps: tokens per second
            seconds: seconds per example
      """
      
      self.ids = []
      self.sequences = []
      self.masks = []
      self.conditions = []
      
      dataset = np.load(dataset_path, allow_pickle=True)
      original_sequences = dataset['sequences']
      for i in range(len(original_sequences)):
        original_sequences[i] = np.append(original_sequences[i], eos_idx)
      original_ids = dataset['ids']

      self.cond = (cond_path != None)
      if self.cond:
        cond_csv = pd.read_csv(cond_path).set_index('ID')
      
      for i, data in enumerate(original_sequences):
      
        try: 
          data_len = len(data)
          #print(int((seq_len+1)*math.ceil(data_len/(seq_len+1))-data_len))
          padded_data = np.pad(data, (0, int((seq_len+1)*math.ceil(data_len/(seq_len+1))-data_len)), mode='constant', constant_values=(0, pad_idx))
          split_data = np.split(padded_data, padded_data.shape[-1]//(seq_len+1))
          
          num_seq = len(split_data)
          
          if self.cond:
            self.conditions.extend([cond_csv.loc[original_ids[i], '4Q'] - 1]*num_seq)
                                          
          self.sequences.extend(split_data)      
          
          self.masks.extend([seq != pad_idx for seq in split_data])
                   
          self.ids.extend([original_ids[i]]*num_seq)
          
        except KeyError as error:
          print('Key not found', original_ids[i])
          continue

      if type(self.ids[0] == str):
        self.ids = torch.arange(len(self.ids)).unsqueeze(-1)
      else:
        self.ids = torch.Tensor(self.ids)

      self.sequences = torch.Tensor(self.sequences)
      self.masks = torch.Tensor(self.masks)
      
      if self.cond:
        self.conditions = torch.Tensor(self.conditions)    
        

    def __len__(self):
      return len(self.sequences)

    def __getitem__(self, idx):
      input = self.sequences[idx][:-1].long()
      input_mask = self.masks[idx][:-1].long()
      target = self.sequences[idx][1:].long()
      target_mask = self.masks[idx][1:].long()

      if self.cond:
        batch = {'ids': self.ids[idx], 'input': input, 'target': target, 'input_mask': input_mask, 'target_mask': target_mask, 'conditions': self.conditions[idx].long()}
      else:
        batch = {'ids': self.ids[idx], 'input': input, 'target': target, 'input_mask': input_mask, 'target_mask': target_mask}
      return batch


class ClassifierDataset(torch.utils.data.Dataset):
   
    def __init__(self, dataset_path, seq_len, labels_path, pad_idx = 0, eos_idx=1):
      """
        Args:
            dataset: token dataset
            ttps: tokens per second
            seconds: seconds per example
      """
      
      self.ids = []
      self.sequences = []
      self.masks = []
      self.labels = []
      
      dataset = np.load(dataset_path, allow_pickle=True)
      original_sequences = dataset['sequences']
      for i in range(len(original_sequences)):
        original_sequences[i] = np.append(original_sequences[i], eos_idx)
      original_ids = dataset['ids']

      label_csv = pd.read_csv(labels_path).set_index('ID')
      
      for i, data in enumerate(original_sequences):
      
        try: 
          data_len = len(data)
          padded_data = np.pad(data, (0, int((seq_len)*math.ceil(data_len/(seq_len))-data_len)),
                               mode='constant', constant_values=(0, pad_idx))
          split_data = np.split(padded_data, padded_data.shape[-1]//(seq_len))
          
          num_seq = len(split_data)
          self.labels.extend([label_csv.loc[original_ids[i], '4Q'] - 1]*num_seq)
          self.sequences.extend(split_data)      
          self.masks.extend([seq != pad_idx for seq in split_data])
          self.ids.extend([original_ids[i]]*num_seq)
          
        except KeyError as error:
          print('Key not found', original_ids[i])
          continue

      if type(self.ids[0] == str):
        self.ids = torch.arange(len(self.ids)).unsqueeze(-1)
      else:
        self.ids = torch.Tensor(self.ids)

      self.sequences = torch.Tensor(self.sequences)
      self.masks = torch.Tensor(self.masks)
      
      self.labels = torch.Tensor(self.labels)    
        

    def __len__(self):
      return len(self.sequences)

    def __getitem__(self, idx):
      input = self.sequences[idx].long()
      input_mask = self.masks[idx].long()
      
      batch = {'ids': self.ids[idx], 'input': input, 'target': self.labels[idx].long(), 'input_mask': input_mask}
      
      return batch


'''
class JoinedDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_paths, seq_len, cond_path, pad_idx = 0, eos_idx=1):
    self.dataset1 = TransformerDatasetREMI(dataset_paths[0], seq_len, cond_path=None, pad_idx = pad_idx, eos_idx=eos_idx)
    self.dataset2 = TransformerDatasetREMI(dataset_paths[1], seq_len, cond_path=cond_path, pad_idx = pad_idx, eos_idx=eos_idx)
    self.size1 = len(self.dataset1)
    self.size2 = len(self.dataset2)
  
  def __len__(self):
      return len(self.size1 + self.size2)
  
  def __getitem__(self, index):
      if (index<self.size1):
          return self.data1[index]
      else: 
          return self.data2[index-self.size1]
'''

class CPTransformerDataset(torch.utils.data.Dataset):
   
    def __init__(self, data_folder, cond_path=None):
      """
        Args:
            dataset: token dataset
            ttps: tokens per second
            seconds: seconds per example
      """
      
      self.data = np.load(data_folder)
      self.cond = pd.load_csv(cond_path) if cond_path is not None else None
      self.ids = self.data['file_names']

      if type(self.ids[0] == str):
        self.ids = torch.arange(len(self.ids)).unsqueeze(-1)
      else:
        self.ids = torch.Tensor(self.ids)#[:4]
    def __len__(self):
      return len(self.data['x'])

    def __getitem__(self, idx):
      input = torch.Tensor(self.data['x'][idx]).long()
      target = torch.Tensor(self.data['y'][idx]).long()
      loss_mask = self.data['mask'][idx]

      if self.cond:
        batch = {'ids': self.ids[idx], 'inputs': input, 'targets': target, 'loss_mask': loss_mask, 'conditions': self.cond[idx].long()}
      else:
        batch = {'ids': self.ids[idx], 'inputs': input, 'targets': target, 'loss_mask': loss_mask}
      return batch