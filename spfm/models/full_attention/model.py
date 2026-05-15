"""
Cross-only patchwise retrieval model with DiT-style AdaLN conditioning.

This variant removes self-attention, auxiliary outputs, and entropy paths.
It uses a single token per latent (flattened) and keeps values unprojected.

Flow overview:
1) Inputs: x_t (B,C,H,W) latent, t (B,), and db (M,C,H,W) DB latents.
2) Flatten latent to a single token (no patching, no pos-emb).
3) Time conditioning: t -> sinusoidal timestep embedding -> AdaLN modulation on queries
   (and optionally values).
4) Retrieval: the query token attends over all DB tokens (one per DB sample),
   streamed over DB in chunks; outputs token-wise posterior mean.
5) Reshape back to (B,C,H,W) to produce mu.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
from timm.models.vision_transformer import Mlp, PatchEmbed
from models.pos_embed import VisionRotaryEmbeddingFast
from models.adaln import AdaLN, modulate
from models.refiner.model import LatentRefiner as SharedLatentRefiner
from models.time_embed import TimestepEmbedder


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        qk_dim: int,
        num_heads: int,
        value_dim: int | None = None,
        attn_drop: float = 0.0,
        same_patch_index_only: bool = False,
        use_entmax: bool = False,
        entmax_alpha: float = 1.5,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.value_dim = hidden_size if value_dim is None else value_dim
        if qk_dim % num_heads != 0:
            raise ValueError("qk_dim must be divisible by num_heads")
        if self.value_dim % num_heads != 0:
            raise ValueError("value_dim must be divisible by num_heads")
        self.head_dim_qk = qk_dim // num_heads
        self.head_dim_v = self.value_dim // num_heads
        self.q_proj = nn.Linear(hidden_size, qk_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, qk_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.q_norm = nn.RMSNorm(self.head_dim_qk, eps=1e-6, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim_qk, eps=1e-6, elementwise_affine=False)
        self.same_patch_index_only = same_patch_index_only
        self.use_entmax = use_entmax
        self.entmax_alpha = entmax_alpha
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 0:
            raise ValueError("chunk_size must be >= 0")
        if self.use_entmax and self.entmax_alpha != 1.5:
            raise ValueError("Only entmax alpha=1.5 is currently supported")

    def _chunked_attention_same_patch(
        self,
        q_f: torch.Tensor,
        k_f: torch.Tensor,
        v_f: torch.Tensor,
        db_mask: torch.Tensor | None,
        attn_bias: torch.Tensor | None,
        scale: float,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        bsz, num_tokens, _, _ = q_f.shape
        m = k_f.shape[0]
        chunk = self.chunk_size
        running_max = None
        running_den = None
        running_num = None
        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            k_chunk = k_f[s:e]
            v_chunk = v_f[s:e]
            logits = torch.einsum("bnhd,mnhd->bnhm", q_f, k_chunk) * scale
            if attn_bias is not None:
                logits = logits + attn_bias[:, :, None, s:e]
            keep_chunk = None
            if db_mask is not None:
                mask_chunk = db_mask[:, s:e]
                keep_chunk = (~mask_chunk)[:, None, None, :]
                logits = logits.masked_fill(mask_chunk[:, None, None, :], float("-inf"))
            chunk_max = logits.amax(dim=-1)
            chunk_max = torch.where(torch.isfinite(chunk_max), chunk_max, torch.full_like(chunk_max, -1e30))
            if running_max is None:
                running_max = chunk_max
                logits_exp = torch.exp(logits - running_max.unsqueeze(-1))
                if keep_chunk is not None:
                    logits_exp = logits_exp * keep_chunk
                running_den = logits_exp.sum(dim=-1)
                running_num = torch.einsum("bnhm,mnhd->bnhd", logits_exp, v_chunk)
                continue
            new_max = torch.maximum(running_max, chunk_max)
            prev_scale = torch.exp(running_max - new_max)
            logits_exp = torch.exp(logits - new_max.unsqueeze(-1))
            if keep_chunk is not None:
                logits_exp = logits_exp * keep_chunk
            running_den = running_den * prev_scale + logits_exp.sum(dim=-1)
            running_num = running_num * prev_scale.unsqueeze(-1) + torch.einsum("bnhm,mnhd->bnhd", logits_exp, v_chunk)
            running_max = new_max
        mu = running_num / running_den.clamp_min(1e-12).unsqueeze(-1)
        return mu.reshape(bsz, num_tokens, self.value_dim).to(dtype=out_dtype)

    def _chunked_attention_global(
        self,
        q_f: torch.Tensor,
        k_f: torch.Tensor,
        v_f: torch.Tensor,
        db_mask: torch.Tensor | None,
        attn_bias: torch.Tensor | None,
        scale: float,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        bsz, num_tokens, _, _ = q_f.shape
        total_tokens = k_f.shape[0]
        chunk = self.chunk_size
        running_max = None
        running_den = None
        running_num = None
        for s in range(0, total_tokens, chunk):
            e = min(s + chunk, total_tokens)
            k_chunk = k_f[s:e]
            v_chunk = v_f[s:e]
            logits = torch.einsum("bnhd,shd->bnhs", q_f, k_chunk) * scale
            if attn_bias is not None:
                logits = logits + attn_bias[:, None, None, s:e]
            keep_chunk = None
            if db_mask is not None:
                mask_chunk = db_mask[:, s:e]
                keep_chunk = (~mask_chunk)[:, None, None, :]
                logits = logits.masked_fill(mask_chunk[:, None, None, :], float("-inf"))
            chunk_max = logits.amax(dim=-1)
            chunk_max = torch.where(torch.isfinite(chunk_max), chunk_max, torch.full_like(chunk_max, -1e30))
            if running_max is None:
                running_max = chunk_max
                logits_exp = torch.exp(logits - running_max.unsqueeze(-1))
                if keep_chunk is not None:
                    logits_exp = logits_exp * keep_chunk
                running_den = logits_exp.sum(dim=-1)
                running_num = torch.einsum("bnhs,shd->bnhd", logits_exp, v_chunk)
                continue
            new_max = torch.maximum(running_max, chunk_max)
            prev_scale = torch.exp(running_max - new_max)
            logits_exp = torch.exp(logits - new_max.unsqueeze(-1))
            if keep_chunk is not None:
                logits_exp = logits_exp * keep_chunk
            running_den = running_den * prev_scale + logits_exp.sum(dim=-1)
            running_num = running_num * prev_scale.unsqueeze(-1) + torch.einsum("bnhs,shd->bnhd", logits_exp, v_chunk)
            running_max = new_max
        mu = running_num / running_den.clamp_min(1e-12).unsqueeze(-1)
        return mu.reshape(bsz, num_tokens, self.value_dim).to(dtype=out_dtype)

    def forward(
        self,
        q: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        db_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        rope = None,
    ) -> torch.Tensor:
        # q: (B, N, D), k_all/v_all: (M, N, D)
        q = self.q_proj(q)
        k_all = self.k_proj(k_all)
        # Keep values unprojected to avoid learned warping.
        bsz, num_tokens, _ = q.shape
        m = k_all.shape[0]
        scale = 1.0 / math.sqrt(self.head_dim_qk)

        q = q.view(bsz, num_tokens, self.num_heads, self.head_dim_qk)
        k_all = k_all.view(m, num_tokens, self.num_heads, self.head_dim_qk)
        v_all = v_all.view(m, num_tokens, self.num_heads, self.head_dim_v)
        q = self.q_norm(q)
        k_all = self.k_norm(k_all)
        if not self.same_patch_index_only:
            # Global retrieval: query tokens attend to all DB tokens.
            k_all = k_all.reshape(m * num_tokens, self.num_heads, self.head_dim_qk)
            v_all = v_all.reshape(m * num_tokens, self.num_heads, self.head_dim_v)
            total_tokens = k_all.shape[0]

        # Force fp32 for softmax accumulation stability under autocast.
        q_f = q.float()
        k_f = k_all.float()
        v_f = v_all.float()
        scale = float(scale)
        if db_mask is not None:
            if db_mask.dtype != torch.bool:
                db_mask = db_mask.bool()
            if (not self.same_patch_index_only) and num_tokens > 1:
                db_mask = db_mask[:, :, None].expand(-1, -1, num_tokens).reshape(bsz, total_tokens)
        if attn_bias is not None:
            if attn_bias.ndim != 2 or attn_bias.shape[0] != bsz:
                raise ValueError("attn_bias must have shape [B, M] or [B, M*N]")
            if self.same_patch_index_only:
                if attn_bias.shape[1] == m:
                    attn_bias_same_patch = attn_bias[:, None, :].expand(-1, num_tokens, -1)
                elif attn_bias.shape[1] == m * num_tokens:
                    attn_bias_same_patch = attn_bias.view(bsz, m, num_tokens).permute(0, 2, 1).contiguous()
                else:
                    raise ValueError("attn_bias must have shape [B, M] or [B, M*N] when same_patch_index_only=True")
                attn_bias_global = None
            else:
                if attn_bias.shape[1] == m:
                    attn_bias_global = attn_bias[:, :, None].expand(-1, -1, num_tokens).reshape(bsz, total_tokens)
                elif attn_bias.shape[1] == total_tokens:
                    attn_bias_global = attn_bias
                else:
                    raise ValueError("attn_bias must have shape [B, M] or [B, M*N] when same_patch_index_only=False")
                attn_bias_same_patch = None
        else:
            attn_bias_same_patch = None
            attn_bias_global = None
        if self.same_patch_index_only:
            # Patchwise retrieval: query patch i attends to DB patch i across samples.
            logits = torch.einsum("bnhd,mnhd->bnhm", q_f, k_f) * scale
            if attn_bias_same_patch is not None:
                logits = logits + attn_bias_same_patch[:, :, None, :]
            reduce_attn = lambda attn: torch.einsum("bnhm,mnhd->bnhd", attn, v_f)
        else:
            logits = torch.einsum("bnhd,shd->bnhs", q_f, k_f) * scale
            if attn_bias_global is not None:
                logits = logits + attn_bias_global[:, None, None, :]
            reduce_attn = lambda attn: torch.einsum("bnhs,shd->bnhd", attn, v_f)

        if self.use_entmax:
            if self.chunk_size > 0:
                raise ValueError("chunked attention is not supported with entmax")
            if self.entmax_alpha != 1.5:
                raise NotImplementedError("Only entmax alpha=1.5 is currently implemented")
            if db_mask is not None:
                logits = logits.masked_fill(db_mask[:, None, None, :], -1e4)
            attn = entmax15(logits, dim=3)
            mu = reduce_attn(attn).reshape(bsz, num_tokens, self.value_dim)
            return mu.to(dtype=q.dtype)

        if self.chunk_size > 0:
            if self.training and self.attn_drop.p > 0:
                raise ValueError("chunked attention does not support attn_drop > 0 during training")
            if self.same_patch_index_only:
                return self._chunked_attention_same_patch(
                    q_f=q_f,
                    k_f=k_f,
                    v_f=v_f,
                    db_mask=db_mask,
                    attn_bias=attn_bias_same_patch,
                    scale=scale,
                    out_dtype=q.dtype,
                )
            return self._chunked_attention_global(
                q_f=q_f,
                k_f=k_f,
                v_f=v_f,
                db_mask=db_mask,
                attn_bias=attn_bias_global,
                scale=scale,
                out_dtype=q.dtype,
            )

        if attn_bias is not None:
            if db_mask is not None:
                logits = logits.masked_fill(db_mask[:, None, None, :], float("-inf"))
            attn = torch.softmax(logits, dim=-1)
            if self.training and self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            mu = reduce_attn(attn).reshape(bsz, num_tokens, self.value_dim)
            return mu.to(dtype=q.dtype)

        if self.same_patch_index_only:
            q_sdpa = q.permute(0, 1, 2, 3).reshape(bsz * num_tokens, self.num_heads, 1, self.head_dim_qk)
            k_sdpa = (
                k_all.permute(1, 2, 0, 3)
                .unsqueeze(0)
                .expand(bsz, -1, -1, -1, -1)
                .reshape(bsz * num_tokens, self.num_heads, m, self.head_dim_qk)
            )
            v_sdpa = (
                v_all.permute(1, 2, 0, 3)
                .unsqueeze(0)
                .expand(bsz, -1, -1, -1, -1)
                .reshape(bsz * num_tokens, self.num_heads, m, self.head_dim_v)
            )
            attn_mask = None
            if db_mask is not None:
                # SDPA bool mask uses True for allowed positions.
                keep = (~db_mask).unsqueeze(1).expand(-1, num_tokens, -1).reshape(bsz * num_tokens, m)
                attn_mask = keep.unsqueeze(1).unsqueeze(1)
            mu = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            mu = mu.reshape(bsz, num_tokens, self.num_heads, self.head_dim_v)
        else:
            q_sdpa = q.permute(0, 2, 1, 3)
            k_sdpa = k_all.permute(1, 0, 2).unsqueeze(0).expand(bsz, -1, -1, -1)
            v_sdpa = v_all.permute(1, 0, 2).unsqueeze(0).expand(bsz, -1, -1, -1)
            attn_mask = None
            if db_mask is not None:
                # SDPA bool mask uses True for allowed positions.
                attn_mask = (~db_mask).unsqueeze(1).unsqueeze(1)
            mu = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            mu = mu.permute(0, 2, 1, 3)

        mu = mu.reshape(bsz, num_tokens, self.value_dim)
        return mu.to(dtype=q.dtype)

class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        qk_dim: int,
        num_heads: int,
        value_dim: int | None = None,
        mlp_ratio: float = 4.0,
        same_patch_index_only: bool = False,
        use_entmax: bool = False,
        entmax_alpha: float = 1.5,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = CrossAttention(
            hidden_size,
            qk_dim=qk_dim,
            num_heads=num_heads,
            value_dim=value_dim,
            same_patch_index_only=same_patch_index_only,
            use_entmax=use_entmax,
            entmax_alpha=entmax_alpha,
            chunk_size=chunk_size,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        db_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        rope = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
        attn_out = self.attn(
            modulate(x, shift_msa, scale_msa),
            k_all=k_all,
            v_all=v_all,
            db_mask=db_mask,
            attn_bias=attn_bias,
            rope=rope,
        )
        return attn_out

class LearnedPosteriorMean(nn.Module):
    """Patchwise retrieval model with AdaLN-conditioned queries."""

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
        cross_patchwise: bool = False,
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
        cross_db_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        latent_dim = latent_c * latent_h * latent_w
        if (not cross_patchwise) and embed_dim != latent_dim:
            raise ValueError(f"embed_dim must equal latent_c*latent_h*latent_w ({latent_dim}) for single-token mode")
        if qk_dim <= 0:
            raise ValueError("qk_dim must be > 0")

        # Geometry / sizes
        self.latent_c = latent_c
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.embed_dim = embed_dim
        self.qk_dim = qk_dim
        self.time_embed_dim = time_embed_dim
        self.cross_use_entmax = cross_use_entmax
        self.cross_entmax_alpha = cross_entmax_alpha
        self.value_adaln = value_adaln
        self.cross_patchwise = cross_patchwise
        self.cross_patch_size = cross_patch_size
        self.cross_decouple_embed = cross_decouple_embed
        self.num_heads = num_heads
        self.learned_g = learned_g
        self.learned_alpha = learned_alpha
        self.cross_kv_freeze = cross_kv_freeze
        self.cross_attn_chunk_size = int(cross_attn_chunk_size)
        self.cross_db_dropout = float(cross_db_dropout)
        self.frozen_cross_kv_names: tuple[str, ...] = ()
        if self.cross_attn_chunk_size < 0:
            raise ValueError("cross_attn_chunk_size must be >= 0")
        if not (0.0 <= self.cross_db_dropout < 1.0):
            raise ValueError("cross_db_dropout must be in [0, 1)")
        if self.cross_use_entmax and self.cross_entmax_alpha != 1.5:
            raise ValueError("Only entmax alpha=1.5 is currently supported")
        if self.cross_patchwise:
            if latent_h % cross_patch_size != 0 or latent_w % cross_patch_size != 0:
                raise ValueError("latent_h/latent_w must be divisible by cross_patch_size when cross_patchwise=True")
            if cross_decouple_embed and depth != 1:
                raise ValueError("cross_decouple_embed currently supports depth=1 only")
            self.cross_grid_h = latent_h // cross_patch_size
            self.cross_grid_w = latent_w // cross_patch_size
            self.cross_num_patches = self.cross_grid_h * self.cross_grid_w
            self.cross_patch_dim = latent_c * cross_patch_size * cross_patch_size
            if (not cross_decouple_embed) and embed_dim != self.cross_patch_dim:
                raise ValueError(
                    f"embed_dim must equal latent_c*cross_patch_size^2 ({self.cross_patch_dim}) when cross_patchwise=True"
                )
            if cross_decouple_embed:
                self.q_patch_embed = None
                self.k_patch_embed = None
                self.q_in_proj = nn.Linear(self.cross_patch_dim, embed_dim, bias=True)
                self.k_in_proj = nn.Linear(self.cross_patch_dim, embed_dim, bias=True)
            else:
                self.q_patch_embed = PatchEmbed(
                    img_size=(latent_h, latent_w),
                    patch_size=cross_patch_size,
                    in_chans=latent_c,
                    embed_dim=embed_dim,
                    bias=True,
                )
                self.k_patch_embed = PatchEmbed(
                    img_size=(latent_h, latent_w),
                    patch_size=cross_patch_size,
                    in_chans=latent_c,
                    embed_dim=embed_dim,
                    bias=True,
                )
                self.q_in_proj = None
                self.k_in_proj = None
        else:
            self.cross_grid_h = None
            self.cross_grid_w = None
            self.cross_num_patches = 1
            self.cross_patch_dim = latent_dim
            self.q_patch_embed = None
            self.k_patch_embed = None
            self.q_in_proj = None
            self.k_in_proj = None

        # Latent refiner (optional, off by default)
        self.use_refiner = True
        self.refiner_qk_norm = True
        self.refiner_attn_drop = 0.0
        self.refiner_proj_drop = 0.0
        self.refiner_mlp_ratio = mlp_ratio if refiner_mlp_ratio is None else refiner_mlp_ratio
        self.refiner_depth = refiner_depth
        self.refiner_patch_size = refiner_patch_size
        self.refiner_embed_dim = refiner_embed_dim
        self.refiner_num_heads = refiner_num_heads
        self.refiner = SharedLatentRefiner(
            latent_c=latent_c,
            latent_h=latent_h,
            latent_w=latent_w,
            cond_dim=time_embed_dim,
            embed_dim=self.refiner_embed_dim,
            num_heads=self.refiner_num_heads,
            patch_size=self.refiner_patch_size,
            depth=self.refiner_depth,
            mlp_ratio=self.refiner_mlp_ratio,
            qk_norm=self.refiner_qk_norm,
            attn_drop=self.refiner_attn_drop,
            proj_drop=self.refiner_proj_drop,
        )

        # gate strength (start conservative)
        self.alpha_max = 0.5
        self.learned_g_init = learned_g_init
        self.learned_alpha_init = learned_alpha_init
        if self.learned_g:
            self.g_gate_mlp = nn.Sequential(
                nn.Linear(1, time_embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(time_embed_dim, 1, bias=True),
            )
        else:
            self.g_gate_mlp = None
        if self.learned_alpha:
            self.alpha_gate_mlp = nn.Sequential(
                nn.Linear(1, time_embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(time_embed_dim, 1, bias=True),
            )
        else:
            self.alpha_gate_mlp = None

        # AdaLN time embedding (query-side only)
        self.time_mlp = TimestepEmbedder(time_embed_dim)
        self.t_proj = nn.Linear(time_embed_dim, embed_dim, bias=True)

        # Optional AdaLN on values (kept off by default per spec)
        if value_adaln:
            self.v_adaln = AdaLN(embed_dim, time_embed_dim)
        else:
            self.v_adaln = None

        # Final projection back to latent vector
        cross_value_dim = self.cross_patch_dim if (self.cross_patchwise and self.cross_decouple_embed) else embed_dim
        self.blocks = nn.ModuleList([
            CrossAttnBlock(
                embed_dim,
                qk_dim=qk_dim,
                num_heads=num_heads,
                value_dim=cross_value_dim,
                mlp_ratio=mlp_ratio,
                same_patch_index_only=cross_patchwise,
                use_entmax=cross_use_entmax,
                entmax_alpha=cross_entmax_alpha,
                chunk_size=self.cross_attn_chunk_size,
            )
            for _ in range(depth)
        ])
        if self.cross_kv_freeze:
            self.frozen_cross_kv_names = tuple(self._freeze_cross_kv_projections())
        self.initialize_weights()

    # ----------------------------
    # Helpers
    # ----------------------------

    def _freeze_cross_kv_projections(self) -> list[str]:
        frozen: list[str] = []
        for i, block in enumerate(self.blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            k_proj = getattr(attn, "k_proj", None)
            if isinstance(k_proj, nn.Module):
                for name, param in k_proj.named_parameters():
                    param.requires_grad = False
                    frozen.append(f"blocks.{i}.attn.k_proj.{name}")
            v_proj = getattr(attn, "v_proj", None)
            if isinstance(v_proj, nn.Module):
                for name, param in v_proj.named_parameters():
                    param.requires_grad = False
                    frozen.append(f"blocks.{i}.attn.v_proj.{name}")
        return frozen

    @staticmethod
    def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("patchify expects (B, C, H, W)")
        bsz, ch, h, w = x.shape
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError("H/W must be divisible by patch_size")
        ph = h // patch_size
        pw = w // patch_size
        x = x.view(bsz, ch, ph, patch_size, pw, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(bsz, ph * pw, ch * patch_size * patch_size)

    @staticmethod
    def _unpatchify(tokens: torch.Tensor, ch: int, h: int, w: int, patch_size: int) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("unpatchify expects (B, N, patch_dim)")
        bsz, num_tokens, patch_dim = tokens.shape
        if patch_dim != ch * patch_size * patch_size:
            raise ValueError("patch_dim mismatch in unpatchify")
        ph = h // patch_size
        pw = w // patch_size
        if ph * pw != num_tokens:
            raise ValueError("num_tokens mismatch in unpatchify")
        x = tokens.view(bsz, ph, pw, ch, patch_size, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(bsz, ch, h, w)

    @staticmethod
    def _inv_sigmoid(p: float) -> float:
        p = float(min(max(p, 1e-6), 1.0 - 1e-6))
        return math.log(p / (1.0 - p))

    def compute_g(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.learned_g:
            t_in = t.view(-1, 1).to(dtype=dtype)
            g = torch.sigmoid(self.g_gate_mlp(t_in)).view(t.shape[0], 1, 1, 1)
            return g
        return (4.0 * t * (1.0 - t)).to(dtype).view(t.shape[0], 1, 1, 1)

    def compute_alpha(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.learned_alpha:
            t_in = t.view(-1, 1).to(dtype=dtype)
            alpha = torch.sigmoid(self.alpha_gate_mlp(t_in)).view(t.shape[0], 1, 1, 1)
            return alpha
        return (self.alpha_max * t).to(dtype).view(t.shape[0], 1, 1, 1)

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP.
        nn.init.normal_(self.time_mlp.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_mlp.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        if self.learned_g:
            nn.init.constant_(self.g_gate_mlp[-1].weight, 0)
            nn.init.constant_(self.g_gate_mlp[-1].bias, self._inv_sigmoid(self.learned_g_init))
        if self.learned_alpha:
            nn.init.constant_(self.alpha_gate_mlp[-1].weight, 0)
            nn.init.constant_(self.alpha_gate_mlp[-1].bias, self._inv_sigmoid(self.learned_alpha_init))

    def _apply_db_dropout(
        self,
        db_mask: torch.Tensor | None,
        *,
        batch_size: int,
        num_db: int,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if (not self.training) or self.cross_db_dropout <= 0.0:
            return db_mask

        base_mask_img = None
        if db_mask is not None:
            if db_mask.dtype != torch.bool:
                db_mask = db_mask.bool()
            if db_mask.shape == (batch_size, num_db):
                base_mask_img = db_mask
            elif db_mask.shape == (batch_size, num_db * num_tokens):
                base_mask_img = db_mask.view(batch_size, num_db, num_tokens).all(dim=-1)
            else:
                raise ValueError("db_mask must have shape [B, M] or [B, M*N]")
        else:
            base_mask_img = torch.zeros(batch_size, num_db, dtype=torch.bool, device=device)

        drop_mask = torch.rand(batch_size, num_db, device=device) < self.cross_db_dropout
        combined_img_mask = base_mask_img | drop_mask
        all_masked = combined_img_mask.all(dim=1)
        if all_masked.any():
            for row in all_masked.nonzero(as_tuple=False).flatten():
                valid = (~base_mask_img[row]).nonzero(as_tuple=False).flatten()
                if valid.numel() == 0:
                    continue
                keep_idx = valid[torch.randint(valid.numel(), (1,), device=device)]
                drop_mask[row, keep_idx] = False
            combined_img_mask = base_mask_img | drop_mask

        if db_mask is None or db_mask.shape == (batch_size, num_db):
            return combined_img_mask
        drop_mask_flat = drop_mask[:, :, None].expand(-1, -1, num_tokens).reshape(batch_size, num_db * num_tokens)
        return db_mask | drop_mask_flat

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        db: torch.Tensor,
        db_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        return_delta: bool = False,
        return_mu_ret: bool = False,
    ):
        """Compute patchwise posterior mean over the DB and return mu."""
        if x_t.ndim != 4 or db.ndim != 4:
            raise ValueError("x_t and db must be (B, C, H, W) and (M, C, H, W)")
        bsz = x_t.shape[0]
        device = x_t.device
        x_t_img = x_t
        db_img = db

        if self.cross_patchwise:
            if self.cross_decouple_embed:
                q_tokens = self._patchify(x_t_img, self.cross_patch_size)
                q = self.q_in_proj(q_tokens)
                if len(self.blocks) > 0:
                    db_tokens = self._patchify(db_img, self.cross_patch_size)
                    k_all = self.k_in_proj(db_tokens)
                    v_all = db_tokens
            else:
                q = self.q_patch_embed(x_t_img)
                if len(self.blocks) > 0:
                    k_all = self.k_patch_embed(db_img)
                    v_all = self._patchify(db_img, self.cross_patch_size)
        else:
            # Flatten to a single token
            q = x_t_img.reshape(bsz, 1, -1)
            if len(self.blocks) > 0:
                k_all = db_img.reshape(db_img.shape[0], 1, -1)
                # Values intentionally share raw DB tokens in single-token mode.
                v_all = k_all

        db_mask = self._apply_db_dropout(
            db_mask,
            batch_size=bsz,
            num_db=db_img.shape[0],
            num_tokens=q.shape[1],
            device=device,
        )

        # Time conditioning on queries via AdaLN.
        cond_embed = self.time_mlp(t.to(device=device, dtype=q.dtype))
        c = self.t_proj(cond_embed)
        if self.value_adaln and len(self.blocks) > 0:
            v_all = self.v_adaln(v_all, cond_embed)

        x = q
        for block in self.blocks:
            x = block(
                x,
                c,
                k_all,
                v_all,
                db_mask=db_mask,
                attn_bias=attn_bias,
                rope=None,
            )
        x_cross = x

        if self.cross_patchwise:
            mu_tokens = x_cross
            mu_ret = self._unpatchify(
                mu_tokens,
                ch=self.latent_c,
                h=self.latent_h,
                w=self.latent_w,
                patch_size=self.cross_patch_size,
            )
        else:
            # x_cross is (B, 1, latent_dim) because you’re single-token
            mu_ret = x_cross.reshape(bsz, self.latent_c, self.latent_h, self.latent_w)

        # Base: trust retrieval mostly at mid t.
        g = self.compute_g(t, mu_ret.dtype)
        mu_base = (1.0 - g) * x_t_img + g * mu_ret

        if self.use_refiner:
            # Refiner strength: grows toward data (t -> 1).
            alpha = self.compute_alpha(t, mu_ret.dtype)
            delta = self.refiner(x_t_img, mu_base.detach(), cond_embed)
            mu = mu_base + alpha * delta
        else:
            delta = None
            mu = mu_base

        if return_delta and return_mu_ret:
            return mu, delta, mu_ret
        if return_delta:
            return mu, delta
        if return_mu_ret:
            return mu, mu_ret
        return mu
