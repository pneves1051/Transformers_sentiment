import torch
import numpy as np
import torch.nn.functional as F

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


def top_k(logits, k_list):
  v, idx = torch.topk(logits, k, dim=1)
  out = logits.clone()
  out[out < v[..., [-1]]] = -float('Inf')
  return out


def top_p(probs, p):
  top_p_idx = torch.zeros_like(probs[0]).long()
  p_s = 0
  probs_c = probs.clone()
  while p_s <= p:
    p_idx, idx = torch.max(probs_c, dim=1)
    p_s += p_idx
    probs_c[0, idx] = 0
    top_p_idx[idx] = 1
  out = probs.clone()
  bottom_p_idx = (1-top_p_idx).bool()
  out[0, bottom_p_idx] = 0.0
  out /= p_s
  return out


def generate_cp(transformer, generate, past, device, prime_sequence = None, conditions = None, temperature_list=[1, 1, 1, 1, 1, 1],
                top_k_idx_list=[0, 0, 0, 0, 0, 0], top_p_prob_list=[1, 1, 1, 1, 1, 1]):
  transformer.eval()
  
  with torch.no_grad():
    generated = []

    if prime_sequence is None:
      init = torch.Tensor([
          [0, 0, 1, 1, 0, 0, 0], # bar
      ])
    else:
      prime_sequence = prime_sequence.to(device)

    if conditions is not None:
      conditions = conditions.to(device)
      
    tr_input = prime_sequence.clone().to(device)    
    
    total_data = tr_input.clone()
        
    cur_past = min(past, total_data.shape[1])
    tr_input = tr_input[:, -cur_past:]
    
    for i in range(generate):
      if i% 100 == 0: print(i, tr_input.shape)
      
      predictions, _ = transformer(tr_input, conditions)
            
      predictions = [p[:, -1, :].permute(0, 2, 1)/t for p, t in zip(predictions, temperature_list)] # (1, vocab_size)

      if np.any(top_k_idx_list) > 0: 
        predictions = [top_k(p, tk) if tk > 0 else p for p, tk in zip(predictions, top_k_idx_list)]
      
      predictions = [F.softmax(p, dim=1) for p in predictions]
      # print(torch.topk(predictions, 10)[0])
      if np.any(top_p_prob_list) < 1.0:
        predictions = [top_k(p, tk) if tk > 0 else p for p, tk in zip(predictions, top_k_idx_list)]
      # print(predictions)
      # selects the last output in the seq_len dimension
      predicted_id = torch.multinomial(predictions[0], num_samples=1).unsqueeze(0)
      #predicted_id = torch.argmax(F.softmax(predictions, dim=1), dim=1)
      if i % 8 == 0: print(predicted_id)
      # concatenated predicted_id to the output, which is given to the decoder as input
      total_data = torch.cat([total_data, predicted_id], axis=-1)

      cur_past = min(past, total_data.shape[-1])
      tr_input =  total_data[:, -cur_past:]
      generated.append(predicted_id.item())
 
    # index prediction
    print(total_data.shape) 

    return total_data[0].tolist(), generated

  
# taken from https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/cp-linear/main-cp.py
def write_midi(words, path_outfile, word2event):
    
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == 'Metrical':
            if vals[2] == 'Bar':
                bar_cnt += 1
            elif 'Beat' in vals[2]:
                beat_pos = int(vals[2].split('_')[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

                # chord
                if vals[1] != 'CONTI' and vals[1] != 0:
                    midi_obj.markers.append(
                        Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != 'CONTI' and vals[0] != 0:
                    tempo = int(vals[0].split('_')[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos))
            else:
                pass
        elif vals[3] == 'Note':

            try:
                pitch = vals[4].split('_')[-1]
                duration = vals[5].split('_')[-1]
                velocity = vals[6].split('_')[-1]
                
                if int(duration) == 0:
                    duration = 60
                end = cur_pos + int(duration)
                
                all_notes.append(
                    Note(
                        pitch=int(pitch), 
                        start=cur_pos, 
                        end=end, 
                        velocity=int(velocity))
                    )
            except:
                continue
        else:
            pass
    
    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)