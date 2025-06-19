import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D
import einops


def generate_attention_mask(p):
    
    '''
    Generate attention mask so that each prepended class token can only attend to 
    itself and the corresponding tokens in x with the same positional encoding.
    '''

    v = torch.max(p) + 1
    batch_size, seq_length = p.shape
    L = v + seq_length
    attention_mask = torch.zeros(batch_size, L, L, dtype=torch.bool, device=p.device)

    # Step 1: Prepended tokens cannot attend to other prepended tokens (but can attend to themselves)
    # Create a mask for the off-diagonal elements (mask=True), diagonal elements (mask=False)
    eye = torch.eye(v, dtype=torch.bool, device=p.device).logical_not()
    mask_prepend_self = eye.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, v, v)
    attention_mask[:, :v, :v] = mask_prepend_self

    # Step 2: Prepended tokens can only attend to tokens in x with the same positional encoding and not padding tokens
    # Create mask for positional encoding mismatch
    p_expand = p.unsqueeze(1).expand(-1, v, -1)  # Shape: (batch_size, v, seq_length)
    q_range = torch.arange(v, device=p.device).view(1, v, 1).expand(batch_size, -1, seq_length)
    mask_positional_mismatch = (p_expand != q_range)  # True where positions do not match

    # Combine masks
    mask_prepended_to_x = mask_positional_mismatch
    # Apply combined mask to the attention mask for prepended tokens attending to x
    attention_mask[:, :v, v:] = mask_prepended_to_x

    # Step 3: Tokens in x can attend to the corresponding prepended token based on their positional encoding
    # Initialize mask with True (masking attention)
    mask_x_to_prepended = torch.ones(batch_size, seq_length, v, dtype=torch.bool, device=p.device)

    # Set mask to False where tokens in x can attend to the corresponding prepended token
    batch_indices = torch.arange(batch_size, device=p.device).unsqueeze(1).expand(-1, seq_length)
    seq_indices = torch.arange(seq_length, device=p.device).unsqueeze(0).expand(batch_size, -1)

    # Use advanced indexing to set the appropriate positions to False
    mask_x_to_prepended[batch_indices, seq_indices, p] = False

    # Apply this mask to the attention mask
    attention_mask[:, v:, :v] = mask_x_to_prepended
    return attention_mask

class PartMatchingTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, dim_model, dim_ff, dropout, num_sources=2, dim_token=384, init_scalar=1., pos_encodings=True):
        super(PartMatchingTransformer, self).__init__()
        
        # Parameters
        self.dim_model = dim_model
        self.dim_token = dim_token
        self.num_sources = num_sources
        self.num_heads = num_heads

        self.pos_encodings = pos_encodings
        self._is_full_backward_hook
        # Positional Encoding for different sources

        if self.pos_encodings:
            self.source_positional_encodings = PositionalEncoding1D(dim_model)

        self.input_layer = nn.Linear(dim_token, dim_model)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0.0)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout, batch_first=True, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(dim_model)
        self.scalar = nn.Parameter(torch.tensor(init_scalar))
        self.class_tokens = nn.Parameter(torch.randn(1, dim_model))


    def head(self, x):

        query = x[:, 0]
        support = x[:, 1:]
        logits = F.cosine_similarity(query.unsqueeze(1), support, dim=-1) * self.scalar
        mask = logits != 0.
        logits = logits.masked_fill(~mask, float('-inf'))
        return logits

    def forward(self, x, source_ids, padding_mask):
        """
        x: Tensor of shape (batch_size, seq_len, dim_token)
        source_ids: Tensor of shape (batch_size, seq_len) indicating the source of each token
                                                          (0 = query, [1, 20] for support classes, -1 for padding)
        padding_mask: Tensor of shape (batch_size, seq_len) indicating padding positions (1 for padding, 0 for non-padding)
        """

        batch_size, seq_length, _ = x.shape
        x = self.input_layer(x)

        num_sources = torch.max(source_ids) + 1 # num sources include query and each support class

        # Add positional encodings based on source using torch.gather

        if self.pos_encodings:
            pos_encodings = self.source_positional_encodings(torch.zeros(1, num_sources, self.dim_model).to(x.device)).squeeze(0)
            x = x + pos_encodings[source_ids]  # (batch_size, seq_len, dim_model)

        # add class tokens
        class_tokens = self.class_tokens.unsqueeze(0).expand(len(x), num_sources, -1)
        if self.pos_encodings:
            class_tokens = class_tokens + pos_encodings

        x = torch.cat([class_tokens, x], dim=1)  # Shape: (batch_size, L, d_model)
        padding_mask = torch.concatenate([torch.zeros(batch_size, num_sources).to(x.device).long(), padding_mask], dim=1).bool()
        
        attention_mask = generate_attention_mask(source_ids)
        attention_mask = torch.repeat_interleave(attention_mask, self.num_heads, dim=0)

        x = self.transformer_encoder(x, mask=attention_mask, src_key_padding_mask=padding_mask)  
        x = self.layer_norm(x)
        return self.head(x[:, :num_sources]) # only pass the class tokens to the head