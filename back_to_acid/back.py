import torch
import torch.nn as nn

AA_VOCAB = [
    "<pad>", "<unk>",
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.mlp(self.norm(x))


class EnhancedBackAcidModel(nn.Module):
    def __init__(self, input_dim=1280, vocab_size=22, hidden_dim=1024, 
                 n_layers=5, dropout=0.1):
        super().__init__()
        self.vocab = AA_VOCAB
        self.vocab_size = vocab_size

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_ratio=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
    
    def forward(self, x):
        x = self.input_proj(x)      
        for block in self.blocks:
            x = block(x)                
        return self.output_head(x)       

    def back_to_acid(self, logits):
        pred = logits.detach().cpu().argmax(dim=-1)
        sequences = []
        for seq_ids in pred:
            seq = ''.join([
                self.vocab[tid] 
                for tid in seq_ids 
                if 2 <= tid < len(self.vocab)
            ])
            sequences.append(seq)
        return sequences