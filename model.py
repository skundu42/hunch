"""Hunch: a small Transformer encoder that scores candidate answers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mask(qid, oid, pad):
    """Who may look at whom. True means allowed."""
    qi, qj = qid[:, :, None], qid[:, None, :]
    oi, oj = oid[:, :, None], oid[:, None, :]
    see_state = qj == -1                      # everyone reads the state
    same_q = (qi == qj) & (qi >= 0)           # a question reads itself:
    own = (oj == -1) | (oj == oi)             # its header and its own option
    return (see_state | (same_q & own)) & ~pad[:, None, :]


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(),
                                 nn.Linear(ff, d))

    def forward(self, x, mask):
        B, L, D = x.shape
        qkv = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)    # each [B, heads, L, D/heads]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        x = x + self.out(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.norm2(x))


class Hunch(nn.Module):
    def __init__(self, vocab=8192, d=256, layers=4, heads=4,
                 ff=1024, max_pos=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_pos, d)
        self.role = nn.Embedding(3, d)          # state, question, option
        self.blocks = nn.ModuleList(Block(d, heads, ff)
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)             # one score per option

    def forward(self, ids, pos, role, qid, oid, pad, opt_at):
        x = self.tok(ids) + self.pos(pos) + self.role(role)
        mask = build_mask(qid, oid, pad)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        b = torch.arange(ids.size(0), device=ids.device)[:, None, None]
        logits = self.head(x[b, opt_at.clamp(min=0)]).squeeze(-1)
        return logits.masked_fill(opt_at < 0, -1e4)    # [B, questions, options]
