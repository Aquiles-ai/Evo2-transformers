"""Convert an official Evo2 vortex checkpoint (.pt) to a transformers folder.

Reads the merged ``evo2_<variant>.pt`` (the file ``Evo2.load_evo2_model``
produces after merging shards) with plain torch, no vortex, no TE, no GPU
needed, remaps every key to ``modeling_evo2.py``, applies the vortex
``column_split`` row permutation to attention ``Wqkv``, and writes a complete
HF folder: ``config.json`` + ``model.safetensors`` (auto-sharded) +
``tokenizer_config.json`` + ``vocab.json``.

Usage (run where the checkpoint lives, NOT on a small instance):
    python convert_evo2_vortex_to_hf.py \
        --vortex_pt ~/.cache/huggingface/evo2_1b_base.pt \
        --variant evo2_1b_base \
        --out_dir ./evo2-1b-hf

Memory: the model is first built in fp32 on CPU, so allow roughly
2x params in RAM transiently (1B ~8GB, 7B ~40GB, 20B/40B need a big node).
Shrinking that with meta-device init is a possible follow-up.
"""

import argparse
import io as _io
import os
import shutil

import torch

try:
    from .configuration_evo2 import Evo2Config
    from .modeling_evo2 import Evo2ForCausalLM
    from .tokenization_evo2 import Evo2Tokenizer
except ImportError:  # allow running the file directly inside evo2_base/
    from configuration_evo2 import Evo2Config
    from modeling_evo2 import Evo2ForCausalLM
    from tokenization_evo2 import Evo2Tokenizer

COMMON = dict(
    vocab_size=512,
    short_filter_length=3,
    short_filter_bias=False,
    eps=1e-6,
    state_size=16,
    hcm_filter_length=128,
    hcs_filter_length=7,
    proj_groups=1,
    hyena_filter_groups=1,
    column_split_hyena=False,
    column_split=True,
    interleave=True,
    evo2_style_activations=True,
    tie_word_embeddings=True,
    mha_out_proj_bias=True,
    hyena_out_proj_bias=True,
    hyena_flip_x1x2=False,
    qkv_proj_bias=False,
    final_norm=True,
    mlp_activation="gelu",
    make_vocab_size_divisible_by=8,
)

# Values transcribed from evo2/configs/*.yml (7 keys cover the 8th:
# evo2_7b_microviridae reuses the 7b base architecture).
PRESETS = {
    "evo2_1b_base": dict(hidden_size=1920, num_filters=1920, num_layers=25,
        attn_layer_idxs=[3, 10, 17, 24],
        hcl_layer_idxs=[2, 6, 9, 13, 16, 20, 23],
        hcm_layer_idxs=[1, 5, 8, 12, 15, 19, 22],
        hcs_layer_idxs=[0, 4, 7, 11, 14, 18, 21],
        hcl_filter_groups=1920, hcm_filter_groups=128, hcs_filter_groups=128,
        num_attention_heads=15, rotary_emb_base=10000.0,
        rotary_emb_scaling_factor=None, use_interpolated_rotary_pos_emb=False,
        inner_size_multiple_of=16, inner_mlp_size=5120, max_position_embeddings=8192),
    "evo2_7b_base": dict(hidden_size=4096, num_filters=4096, num_layers=32,
        attn_layer_idxs=[3, 10, 17, 24, 31],
        hcl_layer_idxs=[2, 6, 9, 13, 16, 20, 23, 27, 30],
        hcm_layer_idxs=[1, 5, 8, 12, 15, 19, 22, 26, 29],
        hcs_layer_idxs=[0, 4, 7, 11, 14, 18, 21, 25, 28],
        hcl_filter_groups=4096, hcm_filter_groups=256, hcs_filter_groups=256,
        num_attention_heads=32, rotary_emb_base=10000.0,
        rotary_emb_scaling_factor=None, use_interpolated_rotary_pos_emb=False,
        inner_size_multiple_of=16, inner_mlp_size=11008, max_position_embeddings=32768),
    "evo2_7b_microviridae": dict(hidden_size=4096, num_filters=4096, num_layers=32,
        attn_layer_idxs=[3, 10, 17, 24, 31],
        hcl_layer_idxs=[2, 6, 9, 13, 16, 20, 23, 27, 30],
        hcm_layer_idxs=[1, 5, 8, 12, 15, 19, 22, 26, 29],
        hcs_layer_idxs=[0, 4, 7, 11, 14, 18, 21, 25, 28],
        hcl_filter_groups=4096, hcm_filter_groups=256, hcs_filter_groups=256,
        num_attention_heads=32, rotary_emb_base=10000.0,
        rotary_emb_scaling_factor=None, use_interpolated_rotary_pos_emb=False,
        inner_size_multiple_of=16, inner_mlp_size=11008, max_position_embeddings=32768),
    "evo2_7b_262k": dict(hidden_size=4096, num_filters=4096, num_layers=32,
        attn_layer_idxs=[3, 10, 17, 24, 31],
        hcl_layer_idxs=[2, 6, 9, 13, 16, 20, 23, 27, 30],
        hcm_layer_idxs=[1, 5, 8, 12, 15, 19, 22, 26, 29],
        hcs_layer_idxs=[0, 4, 7, 11, 14, 18, 21, 25, 28],
        hcl_filter_groups=4096, hcm_filter_groups=256, hcs_filter_groups=256,
        num_attention_heads=32, rotary_emb_base=10000.0,
        rotary_emb_scaling_factor=32.0, use_interpolated_rotary_pos_emb=True,
        inner_size_multiple_of=16, inner_mlp_size=11008, max_position_embeddings=262144),
    "evo2_7b": dict(hidden_size=4096, num_filters=4096, num_layers=32,
        attn_layer_idxs=[3, 10, 17, 24, 31],
        hcl_layer_idxs=[2, 6, 9, 13, 16, 20, 23, 27, 30],
        hcm_layer_idxs=[1, 5, 8, 12, 15, 19, 22, 26, 29],
        hcs_layer_idxs=[0, 4, 7, 11, 14, 18, 21, 25, 28],
        hcl_filter_groups=4096, hcm_filter_groups=256, hcs_filter_groups=256,
        num_attention_heads=32, rotary_emb_base=10000.0,
        rotary_emb_scaling_factor=128.0, use_interpolated_rotary_pos_emb=True,
        inner_size_multiple_of=16, inner_mlp_size=11264, max_position_embeddings=1048576),
    "evo2_20b": dict(hidden_size=8192, num_filters=8192, num_layers=24,
        attn_layer_idxs=[3, 10, 17],
        hcl_layer_idxs=[2, 6, 9, 13, 16, 20, 23],
        hcm_layer_idxs=[1, 5, 8, 12, 15, 19, 22],
        hcs_layer_idxs=[0, 4, 7, 11, 14, 18, 21],
        hcl_filter_groups=8192, hcm_filter_groups=512, hcs_filter_groups=512,
        num_attention_heads=64, rotary_emb_base=1000000.0,
        rotary_emb_scaling_factor=128.0, use_interpolated_rotary_pos_emb=True,
        inner_size_multiple_of=128, inner_mlp_size=22528, max_position_embeddings=1048576),
    "evo2_40b_base": dict(hidden_size=8192, num_filters=8192, num_layers=50,
        attn_layer_idxs=[3, 10, 17, 24, 31, 35, 42, 49],
        hcl_layer_idxs=[2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48],
        hcm_layer_idxs=[1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47],
        hcs_layer_idxs=[0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46],
        hcl_filter_groups=8192, hcm_filter_groups=512, hcs_filter_groups=512,
        num_attention_heads=64, rotary_emb_base=1000000.0,
        rotary_emb_scaling_factor=None, use_interpolated_rotary_pos_emb=False,
        inner_size_multiple_of=128, inner_mlp_size=21888, max_position_embeddings=8192),
    "evo2_40b": dict(hidden_size=8192, num_filters=8192, num_layers=50,
        attn_layer_idxs=[3, 10, 17, 24, 31, 35, 42, 49],
        hcl_layer_idxs=[2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48],
        hcm_layer_idxs=[1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47],
        hcs_layer_idxs=[0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46],
        hcl_filter_groups=8192, hcm_filter_groups=512, hcs_filter_groups=512,
        num_attention_heads=64, rotary_emb_base=1000000.0,
        rotary_emb_scaling_factor=128.0, use_interpolated_rotary_pos_emb=True,
        inner_size_multiple_of=128, inner_mlp_size=22528, max_position_embeddings=1048576),
}

FP32_KEEP_SUFFIXES = (".inv_freq", ".log_poles", ".residues")


def _group_qkv_rows(w: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Savanna head-interleaved rows -> grouped [Q; K; V] (vortex column_split)."""
    hidden = w.shape[1]
    t = w.t().reshape(hidden, num_heads, 3, head_dim)
    q, k, v = t.unbind(dim=2)
    g = torch.cat([q.reshape(hidden, -1), k.reshape(hidden, -1), v.reshape(hidden, -1)], dim=-1)
    return g.t().contiguous()


def _group_qkv_bias(b: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    t = b.reshape(num_heads, 3, head_dim)
    q, k, v = t.unbind(dim=1)
    return torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)], dim=0).contiguous()


def convert_state_dict(vortex_sd: dict, config: Evo2Config) -> dict:
    """Remap vortex ``StripedHyena`` keys to this package. Raises on mismatch."""
    hf = {}
    heads = config.num_attention_heads
    head_dim = config.hidden_size // heads
    n_attn = n_hcs = n_hcm = n_hcl = 0

    def take(key: str) -> torch.Tensor:
        if key not in vortex_sd:
            raise KeyError(f"missing vortex key: {key}")
        return vortex_sd[key]

    hf["model.embed_tokens.weight"] = take("embedding_layer.weight")

    for i in range(config.num_layers):
        src = f"blocks.{i}"
        dst = f"model.layers.{i}"
        hf[f"{dst}.mixer_norm.weight"] = take(f"{src}.pre_norm.scale")
        hf[f"{dst}.mlp_norm.weight"] = take(f"{src}.post_norm.scale")
        hf[f"{dst}.mlp.gate_proj.weight"] = take(f"{src}.mlp.l1.weight")
        hf[f"{dst}.mlp.up_proj.weight"] = take(f"{src}.mlp.l2.weight")
        hf[f"{dst}.mlp.down_proj.weight"] = take(f"{src}.mlp.l3.weight")

        if i in (config.attn_layer_idxs or []):
            n_attn += 1
            if heads != heads // max(1, config.proj_groups) * max(1, config.proj_groups):
                raise ValueError("converter expects proj_groups to divide num_heads")
            w = take(f"{src}.inner_mha_cls.Wqkv.weight")
            if w.shape[0] != head_dim * (heads + 2 * (heads // max(1, config.proj_groups))):
                raise ValueError(f"unexpected Wqkv shape at layer {i}: {tuple(w.shape)}")
            if heads // max(1, config.proj_groups) != heads:
                raise ValueError("GQA checkpoints are not supported by this converter (evo2 uses MHA)")
            hf[f"{dst}.mixer.qkv_proj.weight"] = _group_qkv_rows(w, heads, head_dim)
            bkey = f"{src}.inner_mha_cls.Wqkv.bias"
            if bkey in vortex_sd:
                hf[f"{dst}.mixer.qkv_proj.bias"] = _group_qkv_bias(vortex_sd[bkey], heads, head_dim)
            hf[f"{dst}.mixer.out_proj.weight"] = take(f"{src}.inner_mha_cls.out_proj.weight")
            obkey = f"{src}.inner_mha_cls.out_proj.bias"
            if obkey in vortex_sd:
                hf[f"{dst}.mixer.out_proj.bias"] = vortex_sd[obkey]
            hf[f"{dst}.mixer.rotary.inv_freq"] = take(f"{src}.inner_mha_cls.rotary_emb.inv_freq")
        else:
            ftype = "s" if i in (config.hcs_layer_idxs or []) else ("m" if i in (config.hcm_layer_idxs or []) else "l")
            if ftype == "s":
                n_hcs += 1
            elif ftype == "m":
                n_hcm += 1
            else:
                n_hcl += 1
            hf[f"{dst}.mixer.in_proj.weight"] = take(f"{src}.projections.weight")
            pkey = f"{src}.projections.bias"
            if pkey in vortex_sd:
                hf[f"{dst}.mixer.in_proj.bias"] = vortex_sd[pkey]
            hf[f"{dst}.mixer.short_conv_weight"] = take(f"{src}.filter.short_filter_weight")
            skey = f"{src}.filter.short_filter_bias"
            if skey in vortex_sd:
                hf[f"{dst}.mixer.short_conv_bias"] = vortex_sd[skey]
            if ftype in ("s", "m"):
                hf[f"{dst}.mixer.long_filter"] = take(f"{src}.filter.h")
            else:
                hf[f"{dst}.mixer.log_poles"] = take(f"{src}.filter.log_poles")
                hf[f"{dst}.mixer.residues"] = take(f"{src}.filter.residues")
            dkey = f"{src}.filter.D"
            if dkey in vortex_sd:
                hf[f"{dst}.mixer.D"] = vortex_sd[dkey]
            hf[f"{dst}.mixer.out_proj.weight"] = take(f"{src}.out_filter_dense.weight")
            hf[f"{dst}.mixer.out_proj.bias"] = take(f"{src}.out_filter_dense.bias")

    hf["model.final_norm.weight"] = take("norm.scale")
    if not config.tie_word_embeddings and "unembed.weight" in vortex_sd:
        hf["lm_head.weight"] = vortex_sd["unembed.weight"]

    used = set()
    for v in hf.values():
        used.add(id(v))
    leftover = [k for k, v in vortex_sd.items()
                if id(v) not in used and not k.endswith("._extra_state")]
    if leftover:
        raise ValueError(f"{len(leftover)} vortex keys not consumed, e.g. {leftover[:8]}")

    print(f"mapped layers: attn={n_attn} hcs={n_hcs} hcm={n_hcm} hcl={n_hcl}")
    return hf


def main() -> None:
    ap = argparse.ArgumentParser(description="Evo2 vortex .pt -> transformers folder")
    ap.add_argument("--vortex_pt", required=True, help="merged evo2_<variant>.pt checkpoint")
    ap.add_argument("--variant", required=True, choices=sorted(PRESETS),
                    help="architecture preset, must match the checkpoint")
    ap.add_argument("--out_dir", required=True, help="output HF folder")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="storage dtype for weights (poles/residues/inv_freq stay fp32)")
    args = ap.parse_args()

    try:
        torch.serialization.add_safe_globals([_io.BytesIO])
    except Exception:
        pass
    try:
        from transformer_engine.common.recipe import DelayedScaling, Format
        torch.serialization.add_safe_globals([DelayedScaling, Format])
    except Exception:
        pass

    config = Evo2Config(**{**COMMON, **PRESETS[args.variant]})
    print(f"loading vortex checkpoint: {args.vortex_pt}")
    vortex_sd = torch.load(args.vortex_pt, map_location="cpu", mmap=True, weights_only=True)
    print(f"vortex keys: {len(vortex_sd)}")

    print("building HF model on CPU...")
    model = Evo2ForCausalLM(config)
    model_state = model.state_dict()

    hf = convert_state_dict(vortex_sd, config)

    model_keys = set(model_state.keys()) - {"lm_head.weight"}  # tied head is not stored
    if set(hf.keys()) != model_keys:
        missing = sorted(model_keys - set(hf.keys()))
        extra = sorted(set(hf.keys()) - model_keys)
        raise ValueError(f"key mismatch: missing={missing[:8]} extra={extra[:8]}")

    target = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    with torch.no_grad():
        for k, v in hf.items():
            want = model_state[k].dtype if k.endswith(FP32_KEEP_SUFFIXES) else target
            model_state[k].copy_(v.to(want))
    model.load_state_dict(model_state, strict=True)

    total = sum(p.numel() for p in model.parameters())
    print(f"loaded OK: {total / 1e9:.2f}B params, saving to {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    Evo2Tokenizer().save_pretrained(args.out_dir)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ["configuration_evo2.py", "modeling_evo2.py", "tokenization_evo2.py"]:
        src = os.path.join(pkg_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out_dir, fname))
    print(f"done: {args.out_dir} (config.json, model.safetensors*, tokenizer files, evo2 code)")


if __name__ == "__main__":
    main()
