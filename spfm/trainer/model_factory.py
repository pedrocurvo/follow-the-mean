from __future__ import annotations

import torch


def get_model_cls(model_name: str):
    if model_name == "baseline_dit":
        from models.baseline_dit.model import LearnedPosteriorMean as ModelCls
    elif model_name == "full_attention":
        from models.full_attention.model import LearnedPosteriorMean as ModelCls
    else:
        raise ValueError(f"Unsupported model '{model_name}'")
    return ModelCls


def build_model(args, device: torch.device):
    ModelCls = get_model_cls(args.model)
    kwargs = dict(
        latent_c=args.latent_c,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        embed_dim=args.embed_dim,
        qk_dim=args.qk_dim,
        time_embed_dim=args.time_embed_dim,
        depth=args.depth,
        refiner_depth=args.refiner_depth,
        refiner_patch_size=args.refiner_patch_size,
        refiner_embed_dim=args.refiner_embed_dim,
        refiner_num_heads=args.refiner_num_heads,
        cross_patch_size=args.cross_patch_size,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        refiner_mlp_ratio=args.refiner_mlp_ratio,
        learned_g_init=args.learned_g_init,
        learned_alpha_init=args.learned_alpha_init,
    )
    kwargs.update(
        cross_patchwise=args.cross_patchwise,
        cross_decouple_embed=args.cross_decouple_embed,
        learned_g=args.learned_g,
        learned_alpha=args.learned_alpha,
        cross_kv_freeze=args.cross_kv_freeze,
        value_adaln=args.value_adaln,
        cross_use_entmax=args.cross_use_entmax,
        cross_entmax_alpha=args.cross_entmax_alpha,
    )
    if args.model == "full_attention":
        kwargs["cross_attn_chunk_size"] = args.cross_attn_chunk_size
        kwargs["cross_db_dropout"] = args.cross_db_dropout
    return ModelCls(**kwargs).to(device)
