import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from os import path

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        # x = x.mean(dim = 1)

        # x = self.to_latent(x)
        return self.linear_head(x), None, None, None

    def load_weights(self, model_path):
        assert path.exists(model_path)
        weights = torch.load(model_path)
        self.load_state_dict(weights)

class ORAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.to_ORq = nn.Linear(dim, inner_dim, bias = False)
        # self.to_ORkv = nn.Linear(dim, inner_dim * 2, bias = False)
        # self.to_ORout = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, ORquery, alpha=None):
        # self-attention
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # OR cross-attention
        ORquery = self.norm(ORquery)

        ORq = self.to_ORq(ORquery)
        # ORkv = self.to_ORkv(x).chunk(2, dim = -1)
        ORq = rearrange(ORq, 'b n (h d) -> b h n d', h = self.heads)
        # ORk, ORv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), ORkv)

        ORdots = torch.matmul(ORq, k.transpose(-1, -2)) * self.scale
        
        # set diagonal to 0
        B, H, N, N = ORdots.shape
        mask = torch.eye(N, device=ORdots.device).bool()
        ORdots[:, :, mask] = 0
        # for i in range(ORdots.shape[2]):
        #     ORdots[:, :, i, i] = 0

        if alpha is not None:
            repeat_alpha = alpha.unsqueeze(1).repeat(1, ORdots.shape[1], 1, 1).permute(0, 1, 3, 2)
            ORdots = ORdots * (1-repeat_alpha)

        ORattn = self.attend(ORdots)

        ORout = torch.matmul(ORattn, v)
        ORout = rearrange(ORout, 'b h n d -> b n (h d)')

        return self.to_out(out), self.to_out(ORout)

class ORTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ORAttention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim),
                nn.Linear(dim, 1)
                # nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)
                # nn.Conv2d(dim, 1, kernel_size=5, stride=1, padding=2)
            ]))
        # self.occlusion_head = nn.Linear(dim, 1)
        # self.occlusion_head = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)
        # self.occlusion_head = nn.Conv2d(dim, 1, kernel_size=5, stride=1, padding=2)
    def forward(self, x, ORquery):
        alpha = None
        for i, (attn, ff, occlusion_head) in enumerate(self.layers):
            x_attn, ORquery_attn = attn(x, ORquery, alpha)
            x = x_attn + x
            x = ff(x) + x
            ORx = ff(ORquery_attn) + ORquery_attn
            norm_x = self.norm(x)
            norm_ORx = self.norm(ORx)
            # occlusion_head_input = torch.square(norm_x - norm_ORx).unflatten(1, (16, 16)).permute(0, 3, 1, 2)
            # alpha = occlusion_head(occlusion_head_input).sigmoid()
            # alpha = alpha.permute(0, 2, 3, 1).flatten(1, 2)
            alpha = occlusion_head(torch.square(norm_x - norm_ORx)).sigmoid()
            if i == 2:
                attention_weights = alpha
        return norm_x, norm_ORx, alpha, attention_weights

class ORFormer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.ORquery = nn.Parameter(torch.randn(1, (image_height//patch_height)*(image_width//patch_width), dim))

        self.transformer = ORTransformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        bs = img.size(0)
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        tgt = self.ORquery.repeat(bs, 1, 1).to(device)

        x, ORx, alpha, attention_weights = self.transformer(x, tgt)

        # x = x.mean(dim = 1)

        # x = self.to_latent(x)
        return self.linear_head(x), self.linear_head(ORx), alpha, attention_weights
        # return self.linear_head(x), self.linear_head(ORx), self.occlusion_head(torch.square(x - ORx).sigmoid()
        # return self.linear_head(x), self.linear_head(ORx), self.occlusion_head(torch.concat([x - ORx, ORx - x], dim=2)).sigmoid()

    def load_weights(self, model_path):
        assert path.exists(model_path)
        weights = torch.load(model_path)
        self.load_state_dict(weights, strict=False)