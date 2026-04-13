import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import RMSNorm

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class XEmbedder(nn.Module):
    def __init__(self, hidden_dim, embeddings_dim = 1280,bottlen_dim=256,bias=True):
        super().__init__()
        self.embeding_dimension = embeddings_dim
        self.Bottleneck = nn.Sequential(
            nn.Linear(embeddings_dim, bottlen_dim),
            nn.SiLU(),
            nn.Linear(bottlen_dim, hidden_dim)
            )
    def forward(self, x):
        x = self.Bottleneck(x)
        return x
    
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FamilyEmbedder(nn.Module):
    def __init__(self, hidden_size, embeddings_dim=1280, drop_prob=0.15):
        super().__init__()
        self.drop_prob = drop_prob
        self.embeddings_dim = embeddings_dim
        self.hidden_size = hidden_size
        self.null_token = nn.Parameter(torch.zeros(1, embeddings_dim))
        nn.init.normal_(self.null_token, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(embeddings_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        if self.training:
            drop_mask = torch.rand(x.shape[0], 1, 1, device=x.device) < self.drop_prob
            null_expanded = self.null_token.unsqueeze(0).expand(
                x.shape[0], x.shape[1], self.embeddings_dim
            )
            x = torch.where(drop_mask, null_expanded, x)
        x = self.mlp(x)
        return x

class ConditionEmbedder(nn.Module):
    def __init__(self,condition_dimension, hidden_size, drop_prob=0.15):
        super().__init__()
        self.drop_prob = drop_prob
        self.condition_dimension = condition_dimension 
        self.null_token = nn.Parameter(torch.zeros(condition_dimension))
        self.mlp = nn.Sequential(
            nn.Linear(condition_dimension, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, condition):
        # condition shape: (B, condition_dimension)
        if self.training:
            # 生成 dropout mask (B, 1) → 广播到 (B, hidden_size)
            drop_mask = torch.rand(condition.shape[0], 1, device=condition.device) < self.drop_prob
            condition = torch.where(drop_mask, self.null_token, condition)
            
        return self.mlp(condition)

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))

class FinalLayer(nn.Module):
    def __init__(self, hidden_size,output_dim):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv_input=None):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        
        if kv_input is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            M = kv_input.shape[1]
            q = self.qkv(x)[:, :, :C].reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            kv_proj = self.qkv(kv_input)[:, :, C:].reshape(B, M, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv_proj[0], kv_proj[1]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads=8,qkv_bias=True, qk_norm=True,
                            attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        
        self.semantic_proj_x = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.1)
        )
        self.semantic_proj_f = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.1)
        )

        self.cross_attn = Attention(latent_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                            attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim)
        )

    def forward(self, x,f):

        x = self.semantic_proj_x(x)
        f = self.semantic_proj_f(f)
        
        x = self.cross_attn(x,kv_input=f)

        out = self.output_proj(x)
        return out

class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        # Self Attention
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        
        # Cross Attention 
        self.norm_cross = RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, latent_dim=128,num_heads=num_heads,
                                            attn_drop=attn_drop, proj_drop=proj_drop)
        
        # MLP
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        
        # AdaLN 
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 9*hidden_size, bias=True),
        )
        # family embeding mlp
        self.mlp_embedding = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @torch.compile
    def forward(self, x, c, f=None,pretrain=False):    

        shift_msa, scale_msa, gate_msa, \
        shift_cross, scale_cross, gate_cross, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=-1)
        
        x_res = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_res)
        
        if f is not None and pretrain==False:
            f = self.mlp_embedding(f)
            x_res = modulate(self.norm_cross(x), shift_cross, scale_cross)
            x = x + gate_cross.unsqueeze(1) * self.cross_attn(
                x_res, f
            )

        x_res = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_res)
        return x

class DualAMP(nn.Module):
    def __init__(
        self,
        seq_len=50,
        embeddings_dim=1280,
        hidden_size=1280,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_features=8,
        in_context_len=4,
        in_context_start=8,
        pretrain=True
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embeddings_dim = embeddings_dim
        self.depth = depth
        self.num_heads = num_heads
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_features = num_features
        self.pretrain = pretrain
        
        # positional embedding
        self.f_var_position = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=True)
        self.x_var_position = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=True)
        if self.in_context_len > 0:
             self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
             torch.nn.init.normal_(self.in_context_posemb, std=.02)
        
        # embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ConditionEmbedder(num_features, hidden_size)
        self.f_embedder = FamilyEmbedder(hidden_size=hidden_size,embeddings_dim=embeddings_dim)
        self.x_embedder = XEmbedder(hidden_size, embeddings_dim)

        # transformer
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # final layer
        self.final_layer = FinalLayer(hidden_size, embeddings_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02) #type:ignore
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02) #type:ignore
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02) #type:ignore
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02) #type:ignore

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)#type:ignore
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)#type:ignore

    def forward(self, x, t, y,f):
        
        t = self.t_embedder(t) 
        y = self.y_embedder(y) 
        c = t + y
    
        x = self.x_embedder(x) 
        x = x + self.x_var_position

        if self.pretrain == False:
            f = self.f_embedder(f)
            f = f + self.f_var_position

        for i, block in enumerate(self.blocks): 
            if ( self.in_context_len > 0 and i == self.in_context_start):
                y = y.unsqueeze(1).repeat(1, self.in_context_len, 1)
                y = y + self.in_context_posemb
                x = torch.cat([y,x],dim=1)
            x = block(x,c,f=f,pretrain = self.pretrain)
        x = x[:, self.in_context_len:]
        x = self.final_layer(x, c)
        return x

def DAMP(**kwargs):
    return DualAMP(depth=17, hidden_size=768, num_heads=16,
               in_context_len=32, in_context_start=0, **kwargs)

DAMP_models = {
    'DAMP_B_768': DAMP,
}
