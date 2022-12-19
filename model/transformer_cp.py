import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fast_transformers
from fast_transformers.attention.linear_attention import LinearAttention
from fast_transformers.attention.causal_linear_attention import CausalLinearAttention
from fast_transformers.attention import AttentionLayer
from fast_transformers.transformers import TransformerEncoderLayer
from torch.nn.modules.sparse import Embedding
from .attention import RelativeTransformerEncoderLayer, RelativeAttentionLayer, Rotary
from utils.generate import sample

class CPEmbedding(nn.Module):
  def __init__(self, n_token, d_model):
    super(CPEmbedding, self).__init__()
    self.emb = nn.Embedding(n_token, d_model)
    self.d_model = d_model
  
  def forward(self, x):
    return self.emb(x) * math.sqrt(self.d_model)

class CPLinear(nn.Module):
  def __init__(self, n_token, d_model):
    super(CPLinear, self).__init__()
    self.emb = nn.Linear(n_token, d_model)
    self.d_model = d_model
  
  def forward(self, x):
    return self.emb(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=20000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
  
  def forward(self, x):
    x = x + self.pe[:, x.size(1), :]
    return self.dropout(x)


class MultiEmbedding(nn.Module):
  def __init__(self, n_tokens_dict, d_emb_dict):
    super(MultiEmbedding, self).__init__()

    self.embs = nn.ModuleDict({k: CPEmbedding(v1, v2)  \
                for (k, v1), v2 in zip(n_tokens_dict.items(), d_emb_dict.values())})
    
    #self.embedding_list= nn.ModuleList([CPEmbedding(n_t, e_s) for n_t, e_s in zip(n_token_list, d_emb_list)])

  def forward(self, x):
    
    emb_tempo = self.embs['tempo_key'](x[..., 0])
    emb_chord = self.embs['chord_key'](x[..., 1])
    emb_barbeat = self.embs['barbeat_key'](x[..., 2])
    emb_type = self.embs['type_key'](x[..., 3])
    emb_pitch = self.embs['pitch_key'](x[..., 4])
    emb_duration = self.embs['duration_key'](x[..., 5])
    emb_velocity = self.embs['velocity_key'](x[..., 6])
    
    embs = torch.cat([emb_tempo, emb_chord, emb_barbeat, emb_type,
                      emb_pitch, emb_duration, emb_velocity], dim=-1)

    return embs


class MultiEmbeddingLinear(nn.Module):
  def __init__(self, n_tokens_dict, d_emb_dict):
    super(MultiEmbeddingLinear, self).__init__()

    self.embs = nn.ModuleDict({k: CPLinear(v1, v2)  \
                for (k, v1), v2 in zip(n_tokens_dict.items(), d_emb_dict.values())})
    
    #self.embedding_list= nn.ModuleList([CPEmbedding(n_t, e_s) for n_t, e_s in zip(n_token_list, d_emb_list)])

  def forward(self, x):

    emb_tempo = self.embs['tempo_key'](x[0])
    emb_chord = self.embs['chord_key'](x[1])
    emb_barbeat = self.embs['barbeat_key'](x[2])
    emb_type = self.embs['type_key'](x[3])
    emb_pitch = self.embs['pitch_key'](x[4])
    emb_duration = self.embs['duration_key'](x[5])
    emb_velocity = self.embs['velocity_key'](x[6])

    embs = torch.cat([emb_tempo, emb_chord, emb_barbeat, emb_type,
                      emb_pitch, emb_duration, emb_velocity], dim=-1)

    return embs


class MultiProj(nn.Module):
  def __init__(self, d_model, n_tokens_dict, emb_type_size=32):
    super(MultiProj, self).__init__()

    self.projs = nn.ModuleDict({k: CPLinear(d_model, v) for k, v in n_tokens_dict.items()})

    self.project_concat_type = nn.Linear(d_model + emb_type_size, d_model)

  def forward(self, x, emb_type=None, proj_type=None):
    # When training, type embed is given by target
    # For inference, type embed is sampled first, then given as conditioning
    # When inferencing, proj_type is calculated before
    if proj_type is None:
      proj_type = self.projs['type_key'](x)
    
    x = torch.cat([x, emb_type], dim=-1)
    x = self.project_concat_type(x)
    
    projs = [self.projs['tempo_key'](x), self.projs['chord_key'](x), self.projs['barbeat_key'](x),
            proj_type, self.projs['pitch_key'](x), self.projs['duration_key'](x), self.projs['velocity_key'](x)]
      
    return projs


class Generator(nn.Module):
  def __init__(self,
      n_tokens,
      emb_sizes,
      d_model,
      max_seq_len,
      n_layers=8,
      n_heads=8,
      ff_dim=2048,
      cond = False,
      cond_dim = 1,
      dropout = 0.1
  ):
    super(Generator, self).__init__()
    self.model_type = 'Transformer'
    self.embed_size = d_model

    self.cond = cond
    self.cond_dim = cond_dim
    self.src_mask = None
    self.n_heads = n_heads

    self.embedding = MultiEmbedding(n_tokens, emb_sizes)
    self.in_linear = nn.Linear(np.sum(list(emb_sizes.values())), d_model)

    self.pos_emb = PositionalEncoding(d_model)
    self.cond_embedding = nn.Embedding(cond_dim, d_model)
   
    self.rotary = Rotary(dim=d_model//n_heads)
    
    self.transformer = nn.ModuleList(
    [
        RelativeTransformerEncoderLayer(
            RelativeAttentionLayer(CausalLinearAttention(d_model//n_heads), d_model, n_heads, d_keys=d_model//n_heads,
                 d_values=d_model//n_heads),
            d_model,
            ff_dim,
            dropout=dropout,
            activation="gelu"                     
        ) for l in range(n_layers)
    ])
    
    self.cond_layers = nn.ModuleList([nn.Sequential(nn.Linear(cond_dim, d_model)) for _ in range(n_layers)])

    self.dropout = nn.Dropout(dropout)
    
    self.norm = nn.LayerNorm(d_model)

    # It is necessary to get the type embedding layer from the embeddings
    self.to_out = MultiProj(d_model, n_tokens, emb_type_size=emb_sizes['type_key'])   

    self.gumbel_dist = torch.distributions.gumbel.Gumbel(loc=0, scale=1)

  def generate_square_subsequent_mask(self, sz, device):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.to_out.bias.data.zero_()
    self.to_out.weight.data.uniform_(-initrange, initrange)
  
  def gumbel(self, logits, temperature):
    gumbel_sample = torch.distributions.gumbel.Gumbel(loc=torch.zeros_like(logits), scale=torch.ones_like(logits)).sample()
    gumbel_sample = torch.autograd.Variable(gumbel_sample)

    y = F.softmax((logits + gumbel_sample)/temperature, dim = -1)

    y_onehot = F.one_hot(torch.argmax(y, dim = -1), num_classes = y.shape[-1])
    
    return (y_onehot - y).detach() + y  

  def sample_type(self, x, temperature=1, top_k_idx=0, top_p_prob=0.9):
    # For use in inference
    proj_type = self.to_out.projs['type_key'](x)
    sampled_type = sample(proj_type, temperature, top_k_idx, top_p_prob)
    emb_type = self.embedding.embs['type_key'](sampled_type)
    
    return emb_type, proj_type
  
  def forward(self, inputs, target=None, cond=None, temperature = 1, mask=None, **kwargs):    
    # Target not None when training(type conditioning)
    x = self.embedding(inputs) 
    
    x = self.in_linear(x)   
    
    N, seq_len,_ = x.shape

    x = self.pos_emb(x)
    
    for cond_layer, layer in zip(self.cond_layers, self.transformer):
      #cond_emb = cond_layer(cond)
      #x = x + cond_emb   
      mask = mask or fast_transformers.masking.TriangularCausalMask(seq_len, device=x.device)
      x = layer(x, attn_mask=mask, rotary=self.rotary)#, pos_emb = layer_pos_emb, **kwargs)
      
    # norm and to logits
    x = self.norm(x)
    
    # When targetis given, type conditioning comes from it
    if target is not None:
      out = self.to_out(x, self.embedding.embs['type_key'](target[...,3]))
    else:
      # for inference, the type proj has been calculated in self.sample
      emb_type, proj_type = self.sample_type(x)
      out = self.to_out(x, emb_type, proj_type)

    out_gumbel = [self.gumbel(o, temperature) for o in out]
    return out, out_gumbel


class PatchDiscriminator(nn.Module):
  def __init__(self,
      n_tokens,
      emb_sizes,
      d_model,
      max_seq_len,
      n_layers=8,
      n_heads=8,
      ff_dim=2048,
      cond = False,
      cond_dim = 1,
      dropout = 0.1,
      patch_size=16
  ):
    super(PatchDiscriminator, self).__init__()
    self.model_type = 'Transformer'
   
    self.vocab_size = n_tokens 
    self.embed_size = d_model

    self.cond = cond
    self.cond_dim = cond_dim
    self.src_mask = None
    self.patch_size = patch_size
    
    #self.embedding = nn.Embedding(num_tokens, dim)
    self.to_patch = nn.Conv1d(d_model, d_model, patch_size, stride=patch_size)
    self.embedding = MultiEmbeddingLinear(n_tokens, emb_sizes)
    self.in_linear = nn.Linear(np.sum(list(emb_sizes.values())), d_model)

    self.pos_emb = PositionalEncoding(d_model)

    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    self.rotary = Rotary(dim=d_model//n_heads)
   
    self.transformer = nn.ModuleList(
    [
        RelativeTransformerEncoderLayer(
            RelativeAttentionLayer(LinearAttention(d_model//n_heads), d_model, n_heads, d_keys=d_model//n_heads,
                 d_values=d_model//n_heads),
            d_model,
            ff_dim,
            dropout=dropout,
            activation="gelu"                     
        ) for l in range(n_layers)
    ])
    
    self.dropout = nn.Dropout(dropout)
    self.norm = nn.LayerNorm(d_model)
    
    self.cond_embedding = nn.Linear(cond_dim, d_model)
    self.to_out = nn.Linear(d_model, 1)   
   
  def generate_square_subsequent_mask(self, sz, device):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.to_out.bias.data.zero_()
    self.to_out.weight.data.uniform_(-initrange, initrange)
  
  def to_one_hot(self, x):
    x = x.permute(2, 0, 1)
    # Transform inputs into one_hot
    x = [F.one_hot(vec, v).float() for vec, v in zip(x, self.vocab_size.values())]
    return x

  def forward(self, inputs, cond=None, to_one_hot = True, **kwargs):    
    # TODO find way to transform one hot into indices without losing gradient
    if to_one_hot:
      inputs = self.to_one_hot(inputs)
    x = self.embedding(inputs)  
    x = self.in_linear(x)
    
    assert x.shape[1]%self.patch_size == 0, 'Input shape not divisible by patch size'
    
    x = self.to_patch(x.transpose(-1,-2)).transpose(-1,-2)

    cls = self.cls_token.repeat(x.shape[0], 1, 1)
    x = torch.cat((cls, x), dim = 1)
    
    N, seq_len,_ = x.shape
    x = self.pos_emb(x)
    
    for layer in self.transformer:
      #cond_emb = cond_layer(cond)
      #x = x + cond_emb
      x = layer(x, rotary=self.rotary)#, pos_emb = layer_pos_emb, **kwargs)

    # norm and to logits
    x = self.norm(x)
    out_class = x[:, 0]
    out = self.to_out(out_class)
       
    if cond is not None:
      print('bababooie')
      cond_proj = torch.sum(self.cond_embedding(cond)*out_class, dim=1, keepdim=True)
      out += cond_proj
    
    return out