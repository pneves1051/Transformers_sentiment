import sys
import pickle as pkl
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import CPTransformerDataset
from model.transformer_cp import Generator
from utils.generate import generate_cp, write_midi

sys.path.append('..')

stream = open("config/test_cp.yaml", 'r')
hps = yaml.load(stream)

data_hps = hps['data']

dataset_path ='C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/train_data_emopia_linear.npz'
dataset = CPTransformerDataset(dataset_path)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_hps['batch_size'], shuffle=True,
                                         num_workers=data_hps['num_workers'], pin_memory=True) 

# [CLS] token is the extra token
model_hps = hps['model']
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
transformer_gen = Generator(data_hps['vocab_size'], model_hps['emb_sizes'], model_hps['d_model'], model_hps['max_seq_len'], model_hps['n_layers'],
                            model_hps['n_heads'], model_hps['ff_dim'], cond=False, cond_dim=1).to(device)

data = next(iter(dataloader))
input_sequence = data['inputs'].to(device)

total_data, generated = generate_cp(transformer_gen, 1024, 128, device, input_sequence)

with open('C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/dictionary.pkl', "rb") as f:
  words2events =  pkl.load(f)[1]

save_path = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/miditest.mid'

write_midi(total_data, save_path, words2events)