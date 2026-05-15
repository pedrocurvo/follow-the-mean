import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed

from models.pos_embed import VisionRotaryEmbeddingFast
from models.adaln import modulate



class RefinerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope: VisionRotaryEmbeddingFast | None = None,
        rope_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bsz, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q = rope(q, rope_ids)
            k = rope(k, rope_ids)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(bsz, n, dim)
        x = self.proj(x)
        return self.proj_drop(x)


class RefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        cond_dim: int,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = RefinerAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        feat_rope: VisionRotaryEmbeddingFast | None = None,
        rope_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope,
            rope_ids=rope_ids,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class RefinerFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class LatentRefiner(nn.Module):
    def __init__(
        self,
        latent_c: int,
        latent_h: int,
        latent_w: int,
        cond_dim: int | None = None,
        embed_dim: int = 384,
        num_heads: int = 6,
        patch_size: int = 4,
        depth: int = 1,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if cond_dim is None:
            cond_dim = embed_dim
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if latent_h % patch_size != 0 or latent_w % patch_size != 0:
            raise ValueError("latent_h/latent_w must be divisible by patch_size")
        self.latent_c = latent_c
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_h = latent_h // patch_size
        self.grid_w = latent_w // patch_size
        if self.grid_h != self.grid_w:
            raise ValueError("LatentRefiner RoPE requires square patch grid (latent_h/ps == latent_w/ps)")
        self.patch_embed = PatchEmbed(
            img_size=(latent_h, latent_w),
            patch_size=patch_size,
            in_chans=2 * latent_c,
            embed_dim=embed_dim,
            bias=True,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                RefinerBlock(
                    hidden_size=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cond_dim=cond_dim,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = RefinerFinalLayer(
            hidden_size=embed_dim,
            patch_size=patch_size,
            out_channels=latent_c,
            cond_dim=cond_dim,
        )
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for EVA-02 RoPE")
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=head_dim // 2,
            pt_seq_len=self.grid_h,
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x_t: torch.Tensor, mu_base: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_t, mu_base], dim=1)
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens, c, feat_rope=self.feat_rope, rope_ids=None)
        x = self.final_layer(tokens, c)
        bsz = x.shape[0]
        ps = self.patch_size
        ph = self.latent_h // ps
        pw = self.latent_w // ps
        ch_latent = self.latent_c
        x = x.view(bsz, ph, pw, ps, ps, ch_latent)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(bsz, ch_latent, self.latent_h, self.latent_w)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
