import torch
import numpy as np
import torch.nn.functional as F

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


def top_k(logits, k):
  v, idx = torch.topk(logits, k, dim=-1)
  out = logits.clone()
  out[out < v[..., [-1]]] = -float('Inf')
  return out

'''
def top_p(probs, p):
  top_p_idx = torch.zeros_like(probs[0]).long()
  p_s = 0
  probs_c = probs.clone()
  while p_s <= p:
    p_idx, idx = torch.max(probs_c, dim=-1)
    p_s += p_idx
    probs_c[0, idx] = 0
    top_p_idx[idx] = 1
  out = probs.clone()
  bottom_p_idx = (1-top_p_idx).bool()
  out[0, bottom_p_idx] = 0.0
  out /= p_s
  return out
'''

def top_p(probs, p):
  # stores the indices that will not be zeroed
  top_p_idx = torch.zeros_like(probs).long()
  # to check if probability threshold is crossed
  p_s = torch.zeros(*probs.shape[:-1], device = probs.device)
  probs_c = probs.clone()
  while torch.any(p_s <= p):
    # largest number and its corresponding index
    p_idx, idx = torch.max(probs_c, dim=-1)
    
    idx_one_hot = F.one_hot(idx, probs_c.shape[-1]).bool()
    # zeroing so that the same numbers are not seen again
    probs_c[idx_one_hot] = 0
    # we include only those dimensions where the probability threshold
    # has not yet been crossed
    top_p_idx[idx_one_hot] = (p_s < p).long().reshape(-1)
    # we keep adding numbers until threshold is crossed
    p_s += p_idx
  out = probs.clone()
  bottom_p_idx = (1-top_p_idx).bool()
  out[bottom_p_idx] = 0.0  
  out /= p_s.unsqueeze(-1)
  return out


def generate(prime_sequence, conditions, transformer, generate, past, contraction, device, conds=None, temperature=1.0, hf=False, pf = False, top_k_idx=0, top_p_prob=1.0):
  
  transformer.eval()
  
  with torch.no_grad():
    generated = []

    if conditions is not None:
        conditions = conditions.to(device)
    
    tr_input = prime_sequence.clone().to(device)
    
    input_size = tr_input.shape[-1]
        
    total_data = tr_input.clone()
        
    generate = generate//contraction
    past = past//contraction

    tr_input = tr_input[:, -past:]
    
    for i in range(generate):
      if i% 100 == 0: print(i)
      
      # predictions.shape = (batch_size, vocab_size, seq_len)
      if hf:
        predictions = transformer(tr_input)['logits'].transpose(-1,-2)
      elif pf:
        mask = torch.ones_like(tr_input).bool()
        predictions = transformer(tr_input, mask=mask).transpose(-1,-2)
      else:
        predictions, _ = transformer(tr_input, conditions).transpose(-1,-2)#, conds).transpose(-1, -2)
      
      predictions= predictions[:, :, -1] # (1, vocab_size)

      predictions /= temperature  
      if top_k_idx > 0: 
        predictions = top_k(predictions, top_k_idx)
      
      predictions = F.softmax(predictions, dim=1)
      # print(torch.topk(predictions, 10)[0])
      if top_p_prob < 1.0:
        predictions = top_p(predictions, top_p_prob)
      # print(predictions)
      # selects the last output in the seq_len dimension
      predicted_id = torch.multinomial(predictions[0], num_samples=1).unsqueeze(0)
      #predicted_id = torch.argmax(F.softmax(predictions, dim=1), dim=1)
      if i % 8 == 0: print(predicted_id)
      # concatenated predicted_id to the output, which is given to the decoder as input
      total_data = torch.cat([total_data, predicted_id], axis=-1)

      tr_input =  total_data[:, -past:]
 
    # index prediction
    print(total_data.shape) 
    generated = generated[0]

    return generated


def sample(logits, temperature = 1., top_k_idx = 0, top_p_prob =1.):
  logits /= temperature
  if top_k_idx > 0:
    logits = top_k(logits, top_k_idx)

  logits = F.softmax(logits, dim=-1)

  if top_p_prob < 1.:
    logits = top_p(logits, top_p_prob)
  
  # selects the last output in the seq_len dimension
  predicted_id = torch.multinomial(logits[0], num_samples=1).squeeze(-1).unsqueeze(0)
   
  return predicted_id

def generate_cp(transformer, generate, past, device, prime_sequence = None, conditions = None, temperature_list=[1.2, 1.2, 1, 1, 1, 2, 5],
                top_k_idx_list=[0, 0, 0, 0, 0, 0, 0], top_p_prob_list=[0.9, 1, 0.99, 0.9, 0.9, 0.9, 1]):
  transformer.eval()
  
  with torch.no_grad():
    generated = []

    if prime_sequence is None:
      prime_sequence = torch.Tensor([
          [[0, 0, 1, 1, 0, 0, 0]] # bar
      ]).to(device)
      generated = [[0, 0, 1, 1, 0, 0, 0]]
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
      
      predictions, _ = transformer(tr_input, cond=conditions)
     
      predicted_ids= torch.stack([sample(p[:, -1:, :], temp, topk, prob) for p, temp, topk, prob in \
                     zip(predictions, temperature_list, top_k_idx_list, top_p_prob_list)], dim=-1)
            
      if i % 8 == 0: print(predicted_ids)
      # concatenated predicted_id to the output, which is given to the decoder as input
      total_data = torch.cat([total_data, predicted_ids], axis=1)

      cur_past = min(past, total_data.shape[1])
      tr_input =  total_data[:, -cur_past:]
      generated.append(predicted_ids[0].tolist())
 
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