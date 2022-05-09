import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Dropout
from fast_transformers.causal_product import causal_dot_product
from fast_transformers.feature_maps import elu_feature_map
from fast_transformers.events import EventDispatcher
from fast_transformers.masking import FullMask, LengthMask
from fast_transformers.events import QKVEvent
from fast_transformers.recurrent._utils import check_state


class ConditionalLayerNorm(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.num_classes = num_classes
    self.ln = nn.LayerNorm(num_features, elementwise_affine=False)

    self.cond_embed = nn.Embedding(num_classes+1, num_features * 2)
    self.cond_embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.cond_embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, x, y=None):
    out = self.ln(x)
    
    if y == None:
        y = torch.full((x.shape[0],), self.num_classes, device = x.device)
    
    gamma, beta = self.cond_embed(y).chunk(2, 1)
    
    if len(x.shape) == 2:
        out = gamma.view(-1, self.num_features) * out + beta.view(-1, self.num_features)
    else:
        out = gamma.view(-1, 1, self.num_features) * out + beta.view(-1, 1, self.num_features)
    return out


# from https://blog.eleuther.ai/rotary-embeddings/
class Rotary(nn.Module):
    def __init__(self, dim, base = 10000):
        super(Rotary, self).__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]

        return self.cos_cached, self.sin_cached

# rotary pos emb layers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RelativeAttentionLayer(nn.Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.
    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.
    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher=""):
        super(RelativeAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths, rotary):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.
        In the argument description we make use of the following sizes
            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'
        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

                
        cos, sin = rotary(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))

            # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)
        

        # Project the output and return
        return self.out_projection(new_values)


class RelativeAttentionLayerGen(nn.Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.
    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.
    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher=""):
        super(RelativeAttentionLayerGen, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths, rotary):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.
        In the argument description we make use of the following sizes
            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'
        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

                
        #cos, sin = rotary(queries)
        #queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))

            # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)
        

        # Project the output and return
        return self.out_projection(new_values)






class RelativeTransformerEncoderLayer(nn.Module):
    """Self attention and feed forward network with skip connections.
    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.
    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher="", num_classes=None):
        super(RelativeTransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model) if num_classes == None else ConditionalLayerNorm(d_model, num_classes) # CHANGE BACK TO LAYER NORM IF IT DOESNT WORK
        self.norm2 = nn.LayerNorm(d_model) if num_classes == None else ConditionalLayerNorm(d_model, num_classes)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        

    def forward(self, x, attn_mask=None, length_mask=None, rotary=None, cond=None):
        """Apply the transformer encoder to the input x.
        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        
        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask,
            rotary = rotary
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x) if cond == None else self.norm1(x, cond)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y) if cond == None else self.norm2(x+y, cond)


class RelativeRecurrentAttentionLayer(nn.Module):
    """See fast_transformers.attention.attention_layer.AttentionLayer.
    The only difference with the corresponding module is that this projects
    only one input and then calls the inner attention with the provided
    previous state.
    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher=""):
        super(RelativeRecurrentAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, key, value, rotary, state=None, memory=None):
        """Apply attention to the passed in query/key/value after projecting
        them to multiple heads.
        In the argument description we make use of the following sizes
            - N: the batch size
            - D: The input feature dimensionality passed in the constructor as
              'd_model'
        Arguments
        ---------
            query: (N, D) The tensor containing the queries
            key: (N, D) The tensor containing the keys
            value: (N, D) The tensor containing the values
            state: The state varies depending on the inner attention implementation
            memory: **Deprecated** and replaced by state
        Returns
        -------
            The new value for each query as a tensor of shape (N, D).
        """
        # Normalize the state/memory
        state = check_state(state, memory)

        # Project the queries/keys/values
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        #cos, sin = rotary(query)
        #query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Reshape them into many heads and compute the attention
        N, D = query.shape
        H = self.n_heads
        new_value, state = self.inner_attention(
            query.view(N, H, -1),
            key.view(N, H, -1),
            value.view(N, H, -1),
            state
        )
        new_value = new_value.view(N, -1)
        
        # Project the output and return
        return self.out_projection(new_value), state


class RelativeRecurrentTransformerEncoderLayer(nn.Module):
    """Attention to the previous inputs and feed forward with skip connections.
    This transformer encoder layer is the recurrent dual of
    fast_transformers.transformers.TransformerEncoderLayer . The results should
    be identical given the same inputs and a lower triangular mask.
    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher="", num_classes=None):
        super(RelativeRecurrentTransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model) if num_classes == None else ConditionalLayerNorm(d_model, num_classes)
        self.norm2 = LayerNorm(d_model) if num_classes == None else ConditionalLayerNorm(d_model, num_classes)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, rotary=None, cond=None, state=None, memory=None):
        """Apply the transformer encoder to the input x using the provided
        memory.
        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor
            state: The state can vary depending on the attention implementation
            memory: **Deprecated** name for the state argument
        """
        # Normalize the state name
        state = check_state(state, memory)

        # Run the self attention and add it to the input
        x2, state = self.attention(x, x, x, rotary, state)
        x = x + self.dropout(x2)

        # Run the fully connected part of the layer
        y = x = self.norm1(x) if cond == None else self.norm1(x, cond)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y) if cond == None else self.norm2(x+y, cond), state