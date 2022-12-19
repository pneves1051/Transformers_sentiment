import yaml
import os
import glob
import pandas as pd
import sys
from data.process_data import MidiEncoder
sys.path.append('..')

dataset_dir = 'C:/Users/pedro/Google Drive (p185770@dac.unicamp.br)/maestro-v3.0.0-midi/maestro-v3.0.0/*/*.midi'
csv_path = 'C:/Users/pedro/Google Drive (p185770@dac.unicamp.br)/maestro-v3.0.0-midi/maestro-v3.0.0/maestro-v3.0.0.csv'
save_path = 'C:/Users/pedro/Google Drive (p185770@dac.unicamp.br)/dataset.pkl'


#with open(r'C:/Users/pedro/Documents/git/VQ_GAN_music/config/fma.yaml') as file:
stream = open("config/maestro.yaml", 'r')
hps = yaml.load(stream)
    
data_hps = hps['data']

encoder = MidiEncoder(data_hps['steps_per_sec'], data_hps['num_vel_bins'], data_hps['min_pitch'], data_hps['max_pitch'],
                      data_hps['stretch_factors'], data_hps['pitch_transpose_range'])


data_list = glob.glob(dataset_dir)

maestro_df = pd.read_csv(csv_path)
maestro_df = maestro_df.set_index('midi_filename')

train_list = [f for f in data_list if maestro_df.loc[os.path.basename(os.path.dirname(f)) + '/' + os.path.basename(f), 'split'] == 'train']

encoded_sequences = encoder.encode_midi_list(train_list, pkl_path=save_path)
