import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from models.time_embed import TimestepEmbedder

# ---------------------------------------------------------------------------
# AdaLN Helpers
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# DiT Blocks
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# DiT Model
# ---------------------------------------------------------------------------


class LearnedPosteriorMean(nn.Module):
    """DiT that predicts mu directly from x_t and t with no retrieval or refiner."""

    def __init__(
        self,
        latent_c: int,
        latent_h: int,
        latent_w: int,
        embed_dim: int,
        qk_dim: int,
        time_embed_dim: int,
        cross_use_entmax: bool = False,
        cross_entmax_alpha: float = 1.5,
        value_adaln: bool = False,
        depth: int = 1,
        refiner_depth: int = 1,
        refiner_patch_size: int = 4,
        refiner_embed_dim: int = 384,
        refiner_num_heads: int = 6,
        cross_patchwise: bool = True,
        cross_patch_size: int = 4,
        cross_decouple_embed: bool = False,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        refiner_mlp_ratio: float | None = None,
        learned_g: bool = False,
        learned_alpha: bool = False,
        learned_g_init: float = 0.5,
        learned_alpha_init: float = 0.5,
        cross_kv_freeze: bool = False,
        cross_attn_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        del qk_dim
        del cross_use_entmax
        del cross_entmax_alpha
        del value_adaln
        del refiner_depth
        del refiner_patch_size
        del refiner_embed_dim
        del refiner_num_heads
        del cross_decouple_embed
        del refiner_mlp_ratio
        del learned_g
        del learned_alpha
        del learned_g_init
        del learned_alpha_init
        del cross_kv_freeze
        del cross_attn_chunk_size

        if not cross_patchwise:
            raise ValueError("dit requires cross_patchwise=True")
        if latent_h % cross_patch_size != 0 or latent_w % cross_patch_size != 0:
            raise ValueError("latent_h/latent_w must be divisible by cross_patch_size")
        if latent_h != latent_w:
            raise ValueError("dit currently requires square latent grids")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.latent_c = latent_c
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim
        self.patch_size = cross_patch_size
        self.num_heads = num_heads
        self.out_channels = latent_c

        self.x_embedder = PatchEmbed(
            img_size=(latent_h, latent_w),
            patch_size=cross_patch_size,
            in_chans=latent_c,
            embed_dim=embed_dim,
            bias=True,
        )
        num_patches = self.x_embedder.num_patches
        grid_size = int(num_patches**0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError("dit requires a square patch grid")
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

        self.t_embedder = TimestepEmbedder(time_embed_dim)
        self.t_proj = nn.Linear(time_embed_dim, embed_dim, bias=True)
        self.blocks = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(embed_dim, cross_patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError("Token count is not a square grid")
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        db: torch.Tensor,
        db_mask: torch.Tensor | None = None,
        return_delta: bool = False,
        return_mu_ret: bool = False,
    ):
        del db
        del db_mask
        x = self.x_embedder(x_t) + self.pos_embed
        c = self.t_proj(self.t_embedder(t.to(device=x_t.device)))
        for block in self.blocks:
            x = block(x, c)
        mu_ret = self.unpatchify(self.final_layer(x, c))
        delta = None
        mu = mu_ret

        if return_delta and return_mu_ret:
            return mu, delta, mu_ret
        if return_delta:
            return mu, delta
        if return_mu_ret:
            return mu, mu_ret
        return mu


# ---------------------------------------------------------------------------
# Positional Embeddings
# ---------------------------------------------------------------------------


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0
):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)
