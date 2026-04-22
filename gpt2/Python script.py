from dataclasses import dataclass

@dataclass
class Config():
    batch_size = 16
    lr = 3e-4

    n_layers = 12
    n_heads = 12
    n_embed = 768
    
    vocab_size = 50257
    block_size = 1024 
    head_size = n_embed // n_heads
    dropout = 0.2