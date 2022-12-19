import glob
import os
import sys
import yaml
from data.process_data import MidiEncoder
from utils.scores import calculate_scores_multi

sys.path.append('..')
 
midi_dir = 'C:\\Users\\pedro\\Documents\\The Life of Academia - Mestrado\\MS\\Transformer_GAN\\*.mid*'
midi_list = glob.glob(midi_dir)

scores, avg_scores = calculate_scores_multi(midi_list)

print(avg_scores)

