import yaml
import sys
import os
import glob
from data.process_data import MidiEncoder

sys.path.append('..')

midi_file = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/Axelrepeat2.mid'
midi_rec_file = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/Axelrepeat2_2.mid'

stream = open("config/test.yaml", 'r')
hps = yaml.load(stream)

data_hps = hps['data']
encoder = MidiEncoder(data_hps['steps_per_sec'], data_hps['num_vel_bins'], data_hps['min_pitch'],
                     data_hps['max_pitch'], data_hps['stretch_factors'], data_hps['pitch_transpose_range'])

encoded_midi = encoder.encode_midi_file(midi_file=midi_file)

#decoded_midi = encoder.decode_to_midi_file(encoded_midi, midi_rec_file)

midi_dir = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/*.mid'
midi_list = glob.glob(midi_dir)

midi_aug_list = [os.path.dirname(m) + '/' + os.path.basename(m).split('.')[0] + '_' + str(s) + '_' + str(t) + \
                '.mid' for m in midi_list for s in data_hps['stretch_factors'] for t in \
                range(data_hps['pitch_transpose_range'][0], data_hps['pitch_transpose_range'][-1]+1)]


encoding_list = encoder.encode_midi_list(midi_list)

assert len(encoding_list['sequences']) == len(midi_aug_list)
for sequence, filename in zip(encoding_list['sequences'], midi_aug_list):
    encoder.decode_to_midi_file(sequence, filename)
#print(encoding_list)

scores = encoder.calculate_scores(midi_list)
print(scores)
#decoded_midi = encoder.decode_to_midi_file(encoded_midi, midi_rec_file)