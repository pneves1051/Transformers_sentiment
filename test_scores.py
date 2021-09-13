import glob
import os
import sys
import yaml
from data.process_data import MidiEncoder

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