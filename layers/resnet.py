from math import sqrt

from torch import nn
from torch.nn import init
from torch.nn.utils import weight_norm

from .scale import Scale
from .swish import Swish
import torch


# A helper function for drop path / stochastic depth
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Generate random tensor in shape [batch_size, 1, 1, ...] for broadcasting
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


class ResNet(nn.Module):
    def __init__(self,
                 num_res_blocks,
                 widths,
                 final_scale=True,
                 final_tanh=False,
                 use_attention=True,
                 num_heads=8,
                 final_relu=False,
                 dropout_prob=0,
                 use_layer_norm=True,
                 stochastic_depth_prob=0,
                 num_tokens=4):
        """
        Args:
            num_res_blocks (int): Number of residual blocks.
            widths (list[int]): List of feature widths for the linear layers. Must start and end with the same value.
            final_scale (bool): Whether to apply a final learnable scaling layer.
            final_tanh (bool): Whether to apply a final tanh activation.
            use_attention (bool): Whether to add attention after each block.
            num_heads (int): Number of attention heads.
            final_relu (bool): Whether to apply a final ReLU activation.
            dropout_prob (float): Dropout probability in residual blocks.
            use_layer_norm (bool): Use LayerNorm (True) or BatchNorm (False).
            stochastic_depth_prob (float): Probability of dropping an entire residual block.
            num_tokens (int): Number of tokens to split the feature vector into before attention.
        """
        # Ensure that the input and output dimensions are the same.
        assert widths[0] == widths[-1], "Input and output dimensions must match."
        super().__init__()

        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        self.stochastic_depth_prob = stochastic_depth_prob

        # Create a learnable scaling parameter for each residual block.
        self.residual_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_res_blocks)
        ])

        # Build the residual blocks.
        self.res_blocks = nn.ModuleList(
            [self.build_res_block(widths, dropout_prob) for _ in range(num_res_blocks)]
        )

        # Determine the tokenization parameters.
        self.num_tokens = num_tokens
        self.feature_dim = widths[-1]
        assert self.feature_dim % self.num_tokens == 0, "Feature dimension must be divisible by num_tokens."
        self.token_dim = self.feature_dim // self.num_tokens

        # If using attention, define a projection for each residual block.
        if self.use_attention:
            # Create a separate linear projection for each block.
            self.token_projections = nn.ModuleList([
                nn.Linear(self.feature_dim, self.num_tokens * self.token_dim)
                for _ in range(num_res_blocks)
            ])
            # Build attention layers.
            self.attention_layers = nn.ModuleList(
                [self.build_attention(self.token_dim, num_heads) for _ in range(num_res_blocks)]
            )
            # Learnable gate for each block.
            self.attention_gates = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(num_res_blocks)
            ])

        # Final output processing.
        self.final_scale_layer = Scale(self.feature_dim) if final_scale else None
        self.tanh = nn.Tanh() if final_tanh else None
        self.relu = nn.ReLU() if final_relu else None

    def build_linear(self, in_features, out_features):
        """Builds a linear layer with custom initialization and weight normalization."""
        linear = nn.Linear(in_features, out_features)
        bound = sqrt(2.81 * 3 / in_features)  # Gain for Swish activation.
        init.uniform_(linear.weight, -bound, bound)
        init.zeros_(linear.bias)
        return weight_norm(linear)

    def build_res_block(self, widths, dropout_prob):
        """
        Constructs a residual block with pre-activation ordering:
          Norm -> Activation -> Linear -> Dropout.
        """
        layers = []
        for i in range(len(widths) - 1):
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(widths[i]))
            else:
                layers.append(nn.BatchNorm1d(widths[i]))
            layers.append(Swish(widths[i]))
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(nn.Dropout(dropout_prob))
        return nn.Sequential(*layers)

    def build_attention(self, embed_dim, num_heads):
        """Creates a multi-head attention layer."""
        return nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x is assumed to be of shape [batch_size, feature_dim]
        for i, res_block in enumerate(self.res_blocks):
            identity = x  # Save input for the residual connection.
            out = res_block(x)
            out = drop_path(out, self.stochastic_depth_prob, self.training)
            x = identity + out

            if self.use_attention:
                batch_size = x.shape[0]
                # Apply the layer-specific projection to generate tokens.
                tokens = self.token_projections[i](x)
                # Reshape into tokens: [batch_size, num_tokens, token_dim].
                tokens = tokens.view(batch_size, self.num_tokens, self.token_dim)

                # Apply multi-head attention over the tokens.
                attn_out, _ = self.attention_layers[i](tokens, tokens, tokens)

                # Recombine tokens back into a flat feature vector.
                attn_out_flat = attn_out.contiguous().view(batch_size, -1)

                # Use a learnable gate to control the attention contribution.
                gate = torch.sigmoid(self.attention_gates[i])
                x = x + attn_out_flat

            # Normalize the combined output.
            x = x / sqrt(2)

        if self.final_scale_layer is not None:
            x = self.final_scale_layer(x)
        if self.tanh is not None:
            x = self.tanh(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class ResNetReshape(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        shape = x.shape    # (B*num_RG_blocks, C, K, K)
        x = x.view(shape[0], -1)    # (B*num_RG_blocks, C*K*K)
        x = super().forward(x)
        x = x.view(shape)
        return x

class MLP(nn.Module):

    def __init__(self,n_embd,dropout):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd,4*n_embd)
        self.activation = nn.SiLU()
        self.d1 = nn.Dropout(dropout)
        self.c_proj = nn.Linear(4*n_embd,n_embd)
        self.c_proj.INIT_SCALE = 1
    def forward(self,x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.d1(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd,n_head,dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd,dropout)

    def forward(self,x):
        x = x
        q = self.ln_1(x)
        k = self.ln_1(x)
        v = self.ln_1(x)

        x = x + self.attn(q,k,v)[0] / sqrt(2)

        x = x + self.mlp(self.ln_2(x)) / sqrt(2)
        return x



class ResNetTransformer(nn.Module):
    def __init__(self,patch_size, channels, n_embd,n_head, n_layer,depth,final_relu,dropout=.2):

        super().__init__()
        edge_len = 32/(depth+1)
        n_blocks = int((edge_len//patch_size)**2)
        block_size = patch_size**2*channels
        self.d1 = nn.Dropout(dropout)

        self.activation = nn.SiLU()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Linear(4,n_embd),  # vocab index to vocab embedding
            wpe=nn.Embedding(block_size, n_embd),  # input sequence to positional embedding
            h=nn.ModuleList([Block(n_embd,n_head,dropout) for _ in range(n_layer)]),
            # hidden layers consisting of transformer blocks
            ln_f=nn.LayerNorm(n_embd)
        ))
        self.lm_head = nn.Linear(n_embd, 4, bias=False)

        self.transformer.wte.weights = self.lm_head.weight  # output linear layer is the same as token embedding

        self.apply(self.init_weights)

        if final_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = .02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=.02)

    def forward(self, indicies):
        indicies = indicies.reshape(indicies.shape[0],4,-1)
        B, T, L = indicies.size()


        pos = torch.arange(0, T, dtype=torch.long, device=indicies.device)

        pos_emb = self.transformer.wpe(pos)

        tok_emb = self.transformer.wte(indicies)


        tok_emb = self.activation(tok_emb)

        # act_emb = self.transformer.ate(actions)

        x = tok_emb
        x = self.d1( x)
        x = x + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        if self.relu:
            logits = self.relu(logits)

        return logits

class ResNetReshapeTransformer(ResNetTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        shape = x.shape    # (B*num_RG_blocks, C, K, K)

        x = x.view(shape[0], -1)    # (B*num_RG_blocks, C*K*K)
        x = super().forward(x)
        x = x.view(shape)

        return x

