import glob
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.process_data import MIDIEncoderREMI
from data.dataset import TransformerDatasetREMI
from model.transformer import Generator, PatchDiscriminator
from utils.trainer import TransformerTrainer
from utils.losses import wgan_loss


sys.path.append('..')

stream = open("config/test_remi.yaml", 'r')
hps = yaml.load(stream)

data_hps = hps['data']

dict_path = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/dict.pkl'
midi_files_list = glob.glob('C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/*.mid')

encoder = MIDIEncoderREMI(dict_path, midi_files_list)

dataset_dir = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/remi_dataset/'
encoder.save_dataset(midi_files_list, dataset_dir)

txt_files_list = glob.glob('C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/remi_dataset/*.npy')
dataset_path = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/remi_dataset/dataset/dataset.npz'

if not os.path.isfile(dataset_path):
    encoder.save_dataset_as_single_file(txt_files_list, dataset_path)

dataset = TransformerDatasetREMI(dataset_path, data_hps['seq_len'])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_hps['batch_size'], shuffle=True,
                                         num_workers=data_hps['num_workers'], pin_memory=True) 

vocab_size = encoder.vocab_size 
model_hps = hps['model']
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
transformer_gen = Generator(vocab_size, model_hps['max_seq_len'], model_hps['dim'], model_hps['n_layers'],
                            model_hps['n_heads'], model_hps['ff_dim'], cond=False, cond_dim=1).to(device)
transformer_disc = PatchDiscriminator(vocab_size, model_hps['max_seq_len'], model_hps['dim'], model_hps['n_layers'],
                            model_hps['n_heads'], model_hps['ff_dim'], cond=False, cond_dim=1, patch_size=model_hps['patch_size']).to(device)


data = next(iter(dataloader))['inputs'].to(device)
transformer_gen(data)

#transformer_disc(F.one_hot(data, num_classes = vocab_size))

print(device)
checkpoint_dir = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/checkpoints/'

train_hps = hps['training']
ce_loss = nn.CrossEntropyLoss()
gan_loss = wgan_loss

trainer = TransformerTrainer(transformer_gen, transformer_disc, dataloader, None, ce_loss, gan_loss, device,  
                            train_hps['g_lr'], train_hps['d_lr'], vocab_size, d_iters = train_hps['d_iters'], total_iters=train_hps['total_iters'],
                            temperature=train_hps['temperature'], gan_hp=train_hps['gan_hp'])

history = trainer.train(40, checkpoint_dir, validate=False, log_interval=40, load=False, save=False, change_lr=False, train_gan=False)
