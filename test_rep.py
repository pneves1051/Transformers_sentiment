import note_seq
import pretty_midi
import os
import glob
from utils.process_data import MidiEncoder

midi_file = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/Axelrepeat2.mid'
midi_rec_file = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/Axelrepeat2_2.mid'
stretch_factors = [1, 2]
pitch_transpose_range =  [-1, 1]

encoder = MidiEncoder(100, 32, 0, 127,stretch_factors, pitch_transpose_range)

encoded_midi = encoder.encode_midi_file(midi_file=midi_file)

#decoded_midi = encoder.decode_to_midi_file(encoded_midi, midi_rec_file)

midi_dir = 'C:/Users/pedro/Documents/The Life of Academia - Mestrado/MS/Transformer_GAN/*.mid'
midi_list = glob.glob(midi_dir)

midi_aug_list = [os.path.dirname(m) + '/' + os.path.basename(m).split('.')[0] + '_' + str(s) + '_' + str(t) + '.mid' for m in midi_list for s in stretch_factors for t in range(pitch_transpose_range[0], pitch_transpose_range[-1]+1)]
encoding_list = encoder.encode_midi_list(midi_list)

assert len(encoding_list['sequences']) == len(midi_aug_list)
for sequence, filename in zip(encoding_list['sequences'], midi_aug_list):
    encoder.decode_to_midi_file(sequence, filename)
#print(encoding_list)
