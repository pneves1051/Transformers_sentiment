


class Transformer(nn.Module):
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
    super(Transformer, self).__init__()
    self.model_type = 'Transformer'
    self.embed_size = dim

    self.cond = cond
    self.cond_dim = cond_dim
    self.src_mask = None

    self.embedding = nn.Embedding(num_tokens, dim)

    self.pos_emb = nn.Embedding(max_seq_len, dim)
    self.cond_embedding = nn.Embedding(cond_dim, dim)
   
    self.transformer = nn.ModuleList(
    [
        TransformerEncoderLayer(
            AttentionLayer(CausalLinearAttention(dim//n_heads), dim, n_heads, d_keys=dim//n_heads,
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

  def generate_square_subsequent_mask(self, sz, device):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.to_out.bias.data.zero_()
    self.to_out.weight.data.uniform_(-initrange, initrange)

  
  def forward(self, inputs, cond=None, **kwargs):    
    #src = self.embedding(input)*math.sqrt(self.d_model)
    #src = self.pos_encoder(src)
    # cond = F.one_hot(cond.long(), self.cond_dim).float()
    
    x = self.embedding(inputs)    

    N, seq_len,_ = x.shape

    pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device).unsqueeze(0).expand(N,seq_len))
    x = x + pos_emb

    for cond_layer, layer in zip(self.cond_layers, self.transformer):
      #cond_emb = cond_layer(cond)
      #x = x + cond_emb
      x = layer(x, attn_mask=fast_transformers.masking.TriangularCausalMask(seq_len, device=x.device))#, pos_emb = layer_pos_emb, **kwargs)
      
    # norm and to logits
    x = self.norm(x)

    out = self.to_out(x)

    return out
