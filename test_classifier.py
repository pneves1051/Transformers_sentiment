import glob
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.process_data import MIDIEncoderREMI
from data.dataset import ClassifierDataset
from model.classifier import PatchClassifier
from utils.classifier_trainer import ClassifierTrainer

sys.path.append('..')

stream = open("config/test_remi.yaml", 'r')
hps = yaml.load(stream)

data_hps = hps['data']

dict_path = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/dict.pkl'
midi_files_list = glob.glob('C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/midi_dataset/*.mid*')

encoder = MIDIEncoderREMI(dict_path, midi_files_list)

dataset_dir = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/remi_dataset/'
encoder.save_dataset(midi_files_list, dataset_dir)

txt_files_list = glob.glob('C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/remi_dataset/*.npy')
dataset_path = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/remi_dataset/dataset/dataset.npz'

if not os.path.isfile(dataset_path):
    encoder.save_dataset_as_single_file(txt_files_list, dataset_path)

labels_path = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/REMI/label.csv'

# Testing multiple dataloaders for semi-supervised training
dataset = ClassifierDataset(dataset_path, data_hps['seq_len'], labels_path=labels_path)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_hps['batch_size'], shuffle=True,
                                         num_workers=data_hps['num_workers'], pin_memory=True) 


vocab_size = encoder.vocab_size 
model_hps = hps['model']
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
transformer_classifier = PatchClassifier(vocab_size, model_hps['max_seq_len'], model_hps['dim'], model_hps['n_layers'],
                            model_hps['n_heads'], model_hps['ff_dim'], num_classes=model_hps['cond_dim'], patch_size=model_hps['patch_size']).to(device)


data = next(iter(dataloader))
test_inp = data['input'].to(device)
test_mask = data['input_mask'].to(device)
test_targets = data['target'].to(device)
    
print(vocab_size)

transformer_classifier(test_inp, input_mask=test_mask)

print(device)
checkpoint_dir = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/checkpoints/'

train_hps = hps['training']
ce_loss = nn.CrossEntropyLoss()


trainer = ClassifierTrainer(transformer_classifier, dataloader, None, ce_loss,
                            device, 1e-4, vocab_size, warmup_steps=0, total_iters=30000, schedule='constant')

history = trainer.train(30, checkpoint_dir, validate=False, log_interval=40, load=False, save=False)
