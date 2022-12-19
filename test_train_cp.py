import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import CPTransformerDataset
from model.transformer_cp import Generator, PatchDiscriminator
from utils.trainer_cp import TransformerTrainer
from utils.losses import MultiCrossEntropyLoss, wgan_loss, wgan_loss_cp


sys.path.append('..')

stream = open("config/test_cp.yaml", 'r')
hps = yaml.load(stream)

data_hps = hps['data']

dataset_path ='C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/train_data_emopia_linear.npz'
dataset = CPTransformerDataset(dataset_path)
#dataset = TransformerDataset2(encoding_list, data_hps['seq_len'], pad_token=encoder.events_to_ids['TIME_SHIFT_1'])


dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_hps['batch_size'], shuffle=True,
                                         num_workers=data_hps['num_workers'], pin_memory=True) 

# [CLS] token is the extra token
model_hps = hps['model']
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
transformer_gen = Generator(data_hps['vocab_size'], model_hps['emb_sizes'], model_hps['d_model'], model_hps['max_seq_len'], model_hps['n_layers'],
                            model_hps['n_heads'], model_hps['ff_dim'], cond=False, cond_dim=1).to(device)
transformer_disc = PatchDiscriminator(data_hps['vocab_size'],  model_hps['emb_sizes'], model_hps['d_model'], model_hps['max_seq_len'], model_hps['n_layers'],
                            model_hps['n_heads'], model_hps['ff_dim'], cond=False, cond_dim=1, patch_size=model_hps['patch_size']).to(device)


data = next(iter(dataloader))
inputs = data['inputs'].to(device)
targets = data['targets'].to(device)
transformer_gen(inputs, target=targets)
transformer_disc(inputs)

#transformer_disc(F.one_hot(data, num_classes = vocab_size))

print(device)
checkpoint_dir = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/checkpoints/'

train_hps = hps['training']
ce_loss = MultiCrossEntropyLoss()
gan_loss = wgan_loss_cp

trainer = TransformerTrainer(transformer_gen, transformer_disc, dataloader, None, ce_loss, gan_loss, device,  
                            train_hps['g_lr'], train_hps['d_lr'],
                             data_hps['vocab_size'],
                              d_iters = train_hps['d_iters'], total_iters=train_hps['total_iters'],
                            temperature=train_hps['temperature'], gan_hp=train_hps['gan_hp'])

history = trainer.train(20, checkpoint_dir, validate=False, log_interval=5, load=False, save=False, change_lr=False, train_gan=False)
