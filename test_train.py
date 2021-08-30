import glob
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.process_data import MidiEncoder
from data.dataset import TransformerDataset
from model.transformer import Generator, Discriminator, PatchDiscriminator
from utils.trainer import TransformerTrainer
from utils.losses import wgan_loss


sys.path.append('..')

stream = open("config/test.yaml", 'r')
hps = yaml.load(stream)

data_hps = hps['data']
encoder = MidiEncoder(data_hps['steps_per_sec'], data_hps['num_vel_bins'], data_hps['min_pitch'],
                     data_hps['max_pitch'], data_hps['stretch_factors'], data_hps['pitch_transpose_range'])

#decoded_midi = encoder.decode_to_midi_file(encoded_midi, midi_rec_file)

midi_dir = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/*.mid'
midi_list = glob.glob(midi_dir)

encoding_list = encoder.encode_midi_list(midi_list)
print('Encoded')

dataset = TransformerDataset(encoding_list, data_hps['seq_len'])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_hps['batch_size'], shuffle=True,
                                         num_workers=data_hps['num_workers'], pin_memory=True) 

vocab_size = encoder.vocab_size # [CLS] token is the extra token
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

history = trainer.train(20, checkpoint_dir, validate=False, log_interval=40, load=False, save=False, change_lr=False, train_gan=True)
