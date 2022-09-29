from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, activation = 'gelu'):
        super().__init__()
        if activation == 'gelu':
            activation = nn.GELU
        else:
            activation = nn.ReLU
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, nheads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()

        self.nheads = nheads
        self.dim_head = dim_head
        self.dimD = nheads * dim_head
        self.z = dim_head ** -0.5

        self.dropout = nn.Dropout(dropout)

        self.Mq = nn.Linear(dim, self.dimD, bias=False)
        self.Mk = nn.Linear(dim, self.dimD, bias=False)
        self.Mv = nn.Linear(dim, self.dimD, bias=False)

        self.score = nn.Softmax(dim=-1)

        self.out = nn.Sequential(
            nn.Linear(self.dimD, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        bs, n, d = x.shape
        q = self.Mq(x).view(bs, -1, self.nheads, self.dim_head).transpose(1,2) #(bs, nheads, q_length, dim_head)
        k = self.Mk(x).view(bs, -1, self.nheads, self.dim_head).transpose(1,2) #(bs, nheads, q_length, dim_head)
        v = self.Mv(x).view(bs, -1, self.nheads, self.dim_head).transpose(1,2) #(bs, nheads, q_length, dim_head)

        q = q * self.z

        weights = self.score(torch.matmul(q,k.transpose(2,3)))
        weights = self.dropout(weights)

        context = torch.matmul(weights,v)
        context = context.transpose(1,2).contiguous().view(bs, -1, self.nheads * self.dim_head)

        return self.out(context)

class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, nheads, dim_head, dropout, activation):
        super().__init__()
        self.attnBlock = nn.Sequential(
            nn.LayerNorm(dim),
            MultiHeadSelfAttention(dim, nheads=nheads, dim_head=dim_head, dropout=dropout)
        )

        self.ffnBlock = nn.Sequential(
            nn.LayerNorm(dim),
            FFN(dim, hidden_dim=hidden_dim, dropout=dropout, activation=activation)
        )

    def forward(self, x):
        attn_out =  self.attnBlock(x)
        attn_out = attn_out + x

        ffn_out = self.ffnBlock(attn_out)
        ffn_out = ffn_out + attn_out

        return ffn_out

class Transformer(nn.Module):
    def __init__(self,
    depth,
    dim,
    hidden_dim,
    nheads,
    dim_head,
    dropout,
    activation):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, hidden_dim, nheads, dim_head, dropout, activation))

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        return x

class Vit(nn.Module):
    def __init__(self,
    image_size: Tuple[int, int],
    patch_size: Tuple[int, int],
    dim: int,
    depth: int,
    hidden_dim: int,
    nheads=8,
    dim_head=64,
    dropout=0.1,
    activation='gelu',
    channels=3,
    pool='cls',
    masked=False,
    score=False):
        super().__init__()
        IH, IW = image_size
        self.PH, self.PW = patch_size

        assert IH % self.PH == 0 and IW % self.PW == 0

        self.masked = masked
        self.score = score
        self.nPatches = (IH//self.PH) * (IW//self.PW)
        self.dim_patch = channels * self.PH * self.PW

        self.to_emb = nn.Linear(self.dim_patch, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.nPatches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(depth, dim, hidden_dim, nheads, dim_head, dropout, activation)

        self.pool = pool
        self.head = nn.Identity()
    
    def masking_input(self, x):
        _, patches, _ = x.shape
        mask = torch.randint(0, 2, (patches,))
        x = x[:, mask!=1, :]
        return x
    
    def forward(self, img):
        bs, c, h, w = img.shape
        inp = img.reshape(bs, c, h//self.PH, self.PH,  w//self.PW, self.PW).permute(0,2,4,3,5,1).reshape(bs, self.nPatches, self.dim_patch)
        print(inp.shape)
        if self.masked:
            inp = self.masking_input(inp)
        print(inp.shape)
        inp = self.to_emb(inp)
        cls_token = self.cls_token.repeat(bs, 1, 1)
        inp = torch.cat((cls_token, inp), dim=1)
        inp += self.pos_embedding[:, :(inp.shape[1])]
        inp = self.dropout(inp)

        out = self.transformer(inp)

        if self.pool == 'mean':
            out = out.mean(dim=1)
        else:
            out = out[:, 0]

        if self.score:
            return out
        else:
            return self.head(out)

class MaskedSiameseNetwork(nn.Module):
    def __init__(self,
    image_size: Tuple[int, int],
    patch_size: Tuple[int, int],
    dim: int,
    depth: int,
    hidden_dim: int,
    k: int,
    nheads=8,
    dim_head=64,
    dropout=0.1,
    activation='gelu',
    channels=3,
    pool='cls',
    masked=False,
    tauOn=0.966,
    tauTg=0.966,
    t=0.966):
        super().__init__()

        self.tauOn = tauOn
        self.tauTg = tauTg
        self.t = t
        self.onlineNet = Vit(image_size, patch_size, dim, depth, hidden_dim, masked=True, score=True)
        self.targetNet = Vit(image_size, patch_size, dim, depth, hidden_dim, score=True)

        for p in self.targetNet.parameters():
            p.requires_grad = False

        self.prototypes = nn.Parameter(torch.randn(dim, k))
    
    def updateTargetNet(self):
        with torch.no_grad():
            for pOn,pTa in zip(self.onlineNet.parameters(), self.targetNet.parameters()):
                pTa = self.t*pTa + (1. - self.t)*pOn

    def sharpening(self, p, T=0.25):
        sharp = p.pow(1./T)
        sharp = sharp/torch.sum(sharp, dim=1, keepdim=True)
        return sharp

    def forward(self, x, x1):
        z = self.onlineNet(x)
        z1 = self.targetNet(x1)
        norm_prop = F.normalize(self.prototypes)
        p = F.softmax((F.normalize(z)@norm_prop)/self.tauOn, dim=1)
        
        with torch.no_grad():
            p1 = F.softmax((F.normalize(z1)@norm_prop)/self.tauTg, dim=1)
            p1 = self.sharpening(p1)

        return p, p1


model = MaskedSiameseNetwork((256, 256), (32, 32), 256, 3, 512, 128)

out = model(torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256))
