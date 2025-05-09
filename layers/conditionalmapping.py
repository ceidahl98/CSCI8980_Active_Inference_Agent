from math import sqrt
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import weight_norm

from .scale import Scale
from .swish import Swish


class GPTConfig:
    block_size: int = 64
    vocab_size: int = 64
    n_layer: int = 10
    n_head: int = 4
    n_embd: int = 64
    dropout: float=0.1
    action_space = 2


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

    def forward(self,x,context):
        x = x + context
        q = self.ln_1(x)
        k = self.ln_1(x)
        v = self.ln_1(x)

        x = x + self.attn(q,k,v)[0]

        x = x + self.mlp(self.ln_2(x))
        return x



class ConditionalMapping(nn.Module):
    def __init__(self, patch_size,channels, embd_dim,n_head, num_layers,dropout):
        super().__init__()
        input_size = int(patch_size**2*channels)
        output_size = int(input_size*3)
        self.block = Block(embd_dim,n_head,dropout)
        self.num_layers = num_layers
        self.input_embedding = nn.Linear(1,embd_dim)
        self.context_embedding = nn.Linear(1,embd_dim)
        self.out_mlp = MLP(embd_dim,dropout)
        self.proj = nn.Linear(embd_dim,3)

    def forward(self,x,context):
        device = next(self.parameters()).device
        x = x.to(device)
        context = context.to(device)
        b,c,num_blocks,p_area = context.shape
        _,_,x_num_blocks,x_p_area = x.shape
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        x = x.contiguous().view(int(b * x_num_blocks), -1, int(x_p_area * c)).permute(0,2,1)
        context = context.permute(0, 2, 1, 3).flatten(start_dim=2)
        context = context.view(int(b * x_num_blocks), -1, int(x_p_area * c)).permute(0,2,1)
        x = self.input_embedding(x)
        context = self.context_embedding(context)
        for layer in range(self.num_layers):
            x = self.block(x,context)

        x = self.out_mlp(x)
        x = self.proj(x)
        x = nn.functional.relu(x).permute(0,2,1)
        x = x.contiguous().view(b,num_blocks,-1)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual  # Residual connection


class SimpleConditionalMapping(nn.Module):
    def __init__(self, patch_size, channels, context_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        input_dim = int(patch_size ** 2 * channels)
        output_dim = int(input_dim * 3)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, context):
        """
        x: Input tensor of shape (B, input_dim)
        context: Context tensor of shape (B, context_dim)
        """
        device = next(self.parameters()).device
        # Embed inputs and context into the same latent space
        x = x.to(device)
        context = context.to(device)
        b, c, num_blocks, p_area = context.shape
        _, _, x_num_blocks, x_p_area = x.shape

        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        x = x.reshape(int(b * x_num_blocks), -1, int(x_p_area * c)).squeeze()

        context = context.permute(0, 2, 1, 3).flatten(start_dim=2)

        context = context.reshape(int(b * x_num_blocks), -1, int(p_area * c)).flatten(start_dim=1)

        x = self.input_proj(x)

        context = self.context_proj(context)

        # Combine input and context representations
        x = x + context  # Simple addition-based conditioning

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Final output projection
        x = self.output_proj(x)
        x = nn.functional.relu(x)
        x = x.reshape(b, num_blocks*4, -1)

        return x