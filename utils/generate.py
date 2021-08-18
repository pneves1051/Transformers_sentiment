import torch
import torch.nn.functional as F

def top_k(logits, k):
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


def generate(prime_sequence, conditions, transformer, generate, past, contraction, device, conds=None, temperature=1.0, hf=False, pf = False, top_k_idx=0, top_p_prob=1.0):
  #x = encode_mu_law(input)[np.newaxis, ..., np.newaxis]
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