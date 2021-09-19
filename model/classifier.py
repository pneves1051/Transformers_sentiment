import torch
from torch import nn
from fast_transformers.attention.linear_attention import LinearAttention
from .attention import RelativeTransformerEncoderLayer, RelativeAttentionLayer, Rotary



class PatchDiscriminator(nn.Module):
  def __init__(self,
      num_tokens,
      max_seq_len,
      dim,
      n_layers=8,
      n_heads=8,
      ff_dim=2048,
      cond = False,
      cond_dim = 4,
      dropout = 0.1,
      patch_size=16,
      num_classes=4
     
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
    
    self.cond_embedding = nn.Linear(cond_dim, dim)
    self.to_out = nn.Linear(dim, num_classes)   
   
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
  def forward(self, inputs, **kwargs):    
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
        x = layer(x, rotary=self.rotary)
      
    # norm and to logits
    x = self.norm(x)

    out_class = x[:, 0]
    out = self.to_out(out_class)

    return out