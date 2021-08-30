import torch
import torch.nn as nn
import torch.nn.functional as F
import fast_transformers
from fast_transformers.attention.linear_attention import LinearAttention
from fast_transformers.attention.causal_linear_attention import CausalLinearAttention
from fast_transformers.attention import AttentionLayer
from fast_transformers.transformers import TransformerEncoderLayer
from .attention import RelativeTransformerEncoderLayer, RelativeAttentionLayer, Rotary

class Generator(nn.Module):
  def __init__(self,
      num_tokens,
      max_seq_len,
      dim,
      n_layers=8,
      n_heads=8,
      ff_dim=2048,
      cond = False,
      cond_dim = 1,
      dropout = 0.1
  ):
    super(Generator, self).__init__()
    self.model_type = 'Transformer'
    self.embed_size = dim

    self.cond = cond
    self.cond_dim = cond_dim
    self.src_mask = None
    self.n_heads = n_heads

    self.embedding = nn.Embedding(num_tokens, dim)

    self.pos_emb = nn.Embedding(max_seq_len, dim)
    self.cond_embedding = nn.Embedding(cond_dim, dim)
   
    self.rotary = Rotary(dim=dim//n_heads)
    
    self.transformer = nn.ModuleList(
    [
        RelativeTransformerEncoderLayer(
            RelativeAttentionLayer(CausalLinearAttention(dim//n_heads), dim, n_heads, d_keys=dim//n_heads,
                 d_values=dim//n_heads),
            dim,
            ff_dim,
            dropout=dropout,
            activation="gelu"                     
        ) for l in range(n_layers)
    ])
    
    self.cond_layers = nn.ModuleList([nn.Sequential(nn.Linear(cond_dim, dim)) for _ in range(n_layers)])

    self.dropout = nn.Dropout(dropout)
    
    self.norm = nn.LayerNorm(dim)
    self.to_out = nn.Linear(dim, num_tokens)   

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
  
  def forward(self, inputs, cond=None, temperature = 1, **kwargs):    
    #src = self.embedding(input)*math.sqrt(self.d_model)
    #src = self.pos_encoder(src)
    # cond = F.one_hot(cond.long(), self.cond_dim).float()
    
    x = self.embedding(inputs)    

    N, seq_len,_ = x.shape

    pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device).unsqueeze(0).expand(N,seq_len))
    x = self.dropout(x + pos_emb)
    
    for cond_layer, layer in zip(self.cond_layers, self.transformer):
      #cond_emb = cond_layer(cond)
      #x = x + cond_emb      
      x = layer(x, attn_mask=fast_transformers.masking.TriangularCausalMask(seq_len, device=x.device), rotary=self.rotary)#, pos_emb = layer_pos_emb, **kwargs)
      
    # norm and to logits
    x = self.norm(x)

    out = self.to_out(x)

    out_gumbel = self.gumbel(out, temperature)
    return out, out_gumbel


class Discriminator(nn.Module):
  def __init__(self,
      num_tokens,
      max_seq_len,
      dim,
      n_layers=8,
      n_heads=8,
      ff_dim=2048,
      cond = False,
      cond_dim = 1,
      dropout = 0.1
  ):
    super(Discriminator, self).__init__()
    self.model_type = 'Transformer'
   
    self.vocab_size = num_tokens 
    self.embed_size = dim

    self.cond = cond
    self.cond_dim = cond_dim
    self.src_mask = None
    
    #self.embedding = nn.Embedding(num_tokens, dim)
    self.embedding = nn.Linear(num_tokens, dim)

    self.pos_emb = nn.Embedding(max_seq_len, dim)
    self.cond_embedding = nn.Embedding(cond_dim, dim)

    self.rotary = Rotary(dim=dim//n_heads)
   
    self.transformer = nn.ModuleList(
    [
        RelativeTransformerEncoderLayer(
            RelativeAttentionLayer(LinearAttention(dim//n_heads), dim, n_heads, d_keys=dim//n_heads,
                 d_values=dim//n_heads),
            dim,
            ff_dim,
            dropout=dropout,
            activation="gelu"                     
        ) for l in range(n_layers)
    ])
    
    self.dropout = nn.Dropout(dropout)
    
    self.norm = nn.LayerNorm(dim)
    self.to_out = nn.Linear(dim, num_tokens)   
   

  def generate_square_subsequent_mask(self, sz, device):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.to_out.bias.data.zero_()
    self.to_out.weight.data.uniform_(-initrange, initrange)

  '''
  def insert_cls(self, x):
    padded_x = torch.zeros()
    return paded_x
  '''
  def forward(self, inputs, cond=None, temperature = 1, **kwargs):    
    #src = self.embedding(input)*math.sqrt(self.d_model)
    #src = self.pos_encoder(src)
    # cond = F.one_hot(cond.long(), self.cond_dim).float()
    cls = F.one_hot(torch.full([inputs.shape[0], 1], self.vocab_size, device=inputs.device), num_classes=inputs.shape[-1]).float()
    x = torch.cat((cls, inputs), dim = 1)
       
    x = self.embedding(x.float())    
    N, seq_len,_ = x.shape

    pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device).unsqueeze(0).expand(N,seq_len))
    x = self.dropout(x + pos_emb)

    for layer in self.transformer:
      #cond_emb = cond_layer(cond)
      #x = x + cond_emb
      x = layer(x, rotary=self.rotary)#, pos_emb = layer_pos_emb, **kwargs)
      
    # norm and to logits
    x = self.norm(x)

    out_class = x[:, 0]
    out_class = self.to_out(out_class)
    #out = self.to_out(x)

    return out_class


class PatchDiscriminator(nn.Module):
  def __init__(self,
      num_tokens,
      max_seq_len,
      dim,
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
   
    self.vocab_size = num_tokens 
    self.embed_size = dim

    self.cond = cond
    self.cond_dim = cond_dim
    self.src_mask = None
    self.patch_size = patch_size
    
    #self.embedding = nn.Embedding(num_tokens, dim)
    self.to_patch = nn.Conv1d(dim, dim, patch_size, stride=patch_size)
    self.embedding = nn.Linear(num_tokens, dim)

    self.pos_emb = nn.Embedding(max_seq_len, dim)
    self.cond_embedding = nn.Embedding(cond_dim, dim)

    self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    self.rotary = Rotary(dim=dim//n_heads)
   
    self.transformer = nn.ModuleList(
    [
        RelativeTransformerEncoderLayer(
            RelativeAttentionLayer(LinearAttention(dim//n_heads), dim, n_heads, d_keys=dim//n_heads,
                 d_values=dim//n_heads),
            dim,
            ff_dim,
            dropout=dropout,
            activation="gelu"                     
        ) for l in range(n_layers)
    ])
    
    self.dropout = nn.Dropout(dropout)
    
    self.norm = nn.LayerNorm(dim)
    self.to_out = nn.Linear(dim, num_tokens)   
   

  def generate_square_subsequent_mask(self, sz, device):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.to_out.bias.data.zero_()
    self.to_out.weight.data.uniform_(-initrange, initrange)

  '''
  def insert_cls(self, x):
    padded_x = torch.zeros()
    return paded_x
  '''
  def forward(self, inputs, cond=None, temperature = 1, **kwargs):    
    #src = self.embedding(input)*math.sqrt(self.d_model)
    #src = self.pos_encoder(src)
    # cond = F.one_hot(cond.long(), self.cond_dim).float()
    #cls = F.one_hot(torch.full([inputs.shape[0], 1], self.vocab_size, device=inputs.device), num_classes=inputs.shape[-1]).float()
      
    x = self.embedding(inputs.float())  

    assert x.shape[1]%self.patch_size == 0, 'Input shape not divisible by patch size'
    
    x = self.to_patch(x.transpose(-1,-2)).transpose(-1,-2)
   
    cls = self.cls_token.repeat(x.shape[0], 1, 1)
    x = torch.cat((cls, x), dim = 1)

    N, seq_len,_ = x.shape

    pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device).unsqueeze(0).expand(N,seq_len))
    x = self.dropout(x + pos_emb)

    for layer in self.transformer:
      #cond_emb = cond_layer(cond)
      #x = x + cond_emb
      x = layer(x, rotary=self.rotary)#, pos_emb = layer_pos_emb, **kwargs)
      
    # norm and to logits
    x = self.norm(x)

    out_class = x[:, 0]
    out_class = self.to_out(out_class)
    #out = self.to_out(x)

    return out_class