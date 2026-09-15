"""Evo2 StripedHyena-2 modeling for transformers.

References (read-only, not imported at runtime):
  - https://github.com/ArcInstitute/evo2 (evo2/configs/*.yml, evo2/models.py)
  - vortex StripedHyena (model.py, layers.py, attention.py, engine.py,
    rotary.py, utils.py)

Each decoder layer is one of HCS (short FIR, len 7), HCM (medium FIR,
len 128, FFT), HCL (implicit IIR from poles/residues, FFT), or GQA
attention with RoPE. Hyena order: in_proj, causal depthwise short FIR,
interleave, split to (x2, x1, v), long conv on (x1*v) with D skip, gate
by x2, out_proj, residual, RMSNorm, gated MLP, residual. RMSNorm adds
eps after the scaled norm, as in vortex. The vortex to HF weight key
map lives in convert_evo2_vortex_to_hf.py.

Decoding cache (use_cache=True, needs transformers>=5 hybrid Cache API):
prefill runs the parallel path once and seeds per-layer states, decode
steps run recurrently with absolute RoPE positions. use_cache=False
recomputes everything. Long context FFT convs run in fp32 at O(N log N)
per channel: correct but heavy; chunking and fp16/bf16 FFT are TODO.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:  # transformers>=5 hybrid cache API (DynamicLayer + LinearAttentionLayer)
    from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionLayer
    _HAS_HF_CACHE = True
except Exception:
    Cache = DynamicLayer = LinearAttentionLayer = None
    _HAS_HF_CACHE = False

try:
    from .configuration_evo2 import Evo2Config
except ImportError:  # hub dynamic loading puts the folder on sys.path, no package
    from configuration_evo2 import Evo2Config

# Small helpers (verbatim semantics of vortex/model/utils.py)

def _interleave(z: torch.Tensor) -> torch.Tensor:
    """Reorder ``[.., 3H, ..]`` from strided to grouped.

    3D ``[B, 3H, L]`` (our layout) or 2D ``[B, 3H]`` (decode step).
    Matches vortex ``interleave``.
    """
    if z.dim() == 3:
        x1 = z[:, 0::3, :]
        x2 = z[:, 1::3, :]
        v = z[:, 2::3, :]
        return torch.cat([x1, x2, v], dim=1)
    x1 = z[..., 0::3]
    x2 = z[..., 1::3]
    v = z[..., 2::3]
    return torch.cat([x1, x2, v], dim=-1)


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query heads: [B, L, Hkv, D] -> [B, L, H, D]."""
    if n_rep == 1:
        return hidden
    return hidden[:, :, :, None, :].expand(
        hidden.shape[0], hidden.shape[1], hidden.shape[2], n_rep, hidden.shape[3]
    ).reshape(hidden.shape[0], hidden.shape[1], hidden.shape[2] * n_rep, hidden.shape[3])


def _causal_depthwise_conv1d(u: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """Causal depthwise conv over ``[B, C, L]`` with ``weight [C, 1, K]``.

    Mirrors vortex: ``F.conv1d(..., padding=K-1, groups=C)[..., :L]`` in fp32.
    Returns same dtype as input.
    """
    k = weight.shape[-1]
    y = F.conv1d(u.to(torch.float32), weight.to(torch.float32), bias=None, stride=1, padding=k - 1, groups=u.shape[1])[
        ..., : u.shape[-1]
    ]
    if bias is not None:
        y = y + bias.to(torch.float32)[None, :, None]
    return y.to(u.dtype)


def _fft_conv(u: torch.Tensor, k: torch.Tensor, d_bias: torch.Tensor) -> torch.Tensor:
    """FFT long convolution + ``D`` skip, exactly like vortex ``fftconv_func``.

    Args:
        u: ``[B, H, L]`` signal (already ``x1 * v``).
        k: ``[1, H, L]`` or ``[H, 1, L]`` filter (truncated to L beforehand).
        d_bias: ``[H]`` skip coefficient.
    """
    L = u.shape[-1]
    n = 2 * L
    u32 = u.to(torch.float32)
    k32 = k.to(torch.float32)
    k_f = torch.fft.rfft(k32, n=n) / n
    k_f = k_f.reshape(-1, k_f.shape[-1]).unsqueeze(0)  # [1, H, F]
    u_f = torch.fft.rfft(u32, n=n)
    y = torch.fft.irfft(u_f * k_f, n=n, norm="forward")[..., :L]
    y = y + u32 * d_bias.to(torch.float32).unsqueeze(-1)
    return y.to(u.dtype)

# Norm / RoPE / MLP

class Evo2RMSNorm(nn.Module):
    """Vortex-faithful RMSNorm (eps added *after* the scaled norm)."""

    def __init__(self, config: Evo2Config):
        super().__init__()
        self.eps = config.eps
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-0.5) + self.eps)
        return self.weight * y


class Evo2RotaryEmbedding(nn.Module):
    """Standard RoPE with optional linear-interpolated scaling.

    ``rotary_emb_scaling_factor`` divides ``inv_freq`` (== stretching positions),
    matching the ``swap_mha_rope(scaling_factor)`` behavior of long-context Evo2.
    """

    def __init__(self, config: Evo2Config, head_dim: int):
        super().__init__()
        self.dim = head_dim
        base = float(config.rotary_emb_base)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        if config.use_interpolated_rotary_pos_emb and config.rotary_emb_scaling_factor:
            inv_freq = inv_freq / float(config.rotary_emb_scaling_factor)
        self.register_buffer("inv_freq", inv_freq)

    def _cos_sin(self, seq_len: int, position_ids: torch.Tensor, dtype: torch.dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, L]; gather per-position angles
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), self.inv_freq.to(device).float())
        emb = torch.cat([freqs, freqs], dim=-1)  # [L, D] angles
        pos = position_ids.long()  # [B, L]
        cos = emb.cos()[pos].to(dtype)  # [B, L, D]
        sin = emb.sin()[pos].to(dtype)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q/k: [B, L, H, D]; cos/sin: [B, L, D]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class Evo2MLP(nn.Module):
    """Gated MLP: ``down(act(gate(x)) * up(x))``; Identity act for layer>0 (evo2 style)."""

    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.inner_mlp_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.inner_mlp_size, bias=False)
        self.down_proj = nn.Linear(config.inner_mlp_size, config.hidden_size, bias=False)
        if config.mlp_activation == "gelu":
            act = F.gelu
        elif config.mlp_activation == "silu":
            act = F.silu
        else:
            raise ValueError(f"Unsupported mlp_activation={config.mlp_activation}")
        if layer_idx > 0 and config.evo2_style_activations:
            act = nn.Identity()
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# Hybrid cache: one native layer object per decoder layer. Attention
# layers use DynamicLayer (k/v); Hyena layers use Evo2HyenaCacheLayer
# with two states: short-FIR history (state 0) and long-conv history or
# IIR recurrent state (state 1). The mixer owns the math; the container
# only adapts it to generate (length, reorder, crop, reset).

class Evo2HyenaCacheLayer(LinearAttentionLayer if _HAS_HF_CACHE else object):
    """Storage for one Hyena layer's recurrent state (math lives in the mixer)."""

    def __init__(self):
        if _HAS_HF_CACHE:
            super().__init__(number_of_states=2)
        else:
            self.conv_states = {}
            self.recurrent_states = {}
            self.is_conv_states_initialized = {}
            self.is_recurrent_states_initialized = {}
            self.has_previous_state = {}
            self.conv_kernel_size = {}

    def mark_ready(self, device, dtype, long_kernel: int, recurrent: bool) -> None:
        self.device = device
        self.dtype = dtype
        self.conv_kernel_size[0] = 2
        self.conv_kernel_size[1] = long_kernel
        self.is_conv_states_initialized[0] = True
        self.is_conv_states_initialized[1] = not recurrent
        self.is_recurrent_states_initialized[0] = recurrent
        self.has_previous_state[0] = True
        self.has_previous_state[1] = True

# Mixers

class Evo2Attention(nn.Module):
    """GQA + RoPE attention over SDPA (no flash-attn / TE)."""

    def __init__(self, config: Evo2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = max(1, self.num_heads // max(1, config.proj_groups))
        assert self.num_heads % self.num_kv_heads == 0
        self.head_dim = config.hidden_size // self.num_heads
        assert config.hidden_size % self.num_heads == 0
        self.n_rep = self.num_heads // self.num_kv_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_kv_heads)
        self.qkv_proj = nn.Linear(config.hidden_size, qkv_dim, bias=config.qkv_proj_bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.mha_out_proj_bias)
        self.rotary = Evo2RotaryEmbedding(config, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_layer=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv_proj(x)
        q_end = self.num_heads * self.head_dim
        kv_end = q_end + self.num_kv_heads * self.head_dim
        q = qkv[..., :q_end].reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = qkv[..., q_end:kv_end].reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = qkv[..., kv_end:].reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        if position_ids is None:
            position_ids = torch.arange(seqlen, device=x.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = self.rotary._cos_sin(
            int(position_ids.max().item()) + 1, position_ids, x.dtype, x.device)
        q, k = self.rotary.apply(q, k, cos, sin)
        if use_cache and past_layer is not None:
            k_full, v_full = past_layer.update(k.transpose(1, 2), v.transpose(1, 2))
            k = _repeat_kv(k_full.transpose(1, 2), self.n_rep).transpose(1, 2)
            v = _repeat_kv(v_full.transpose(1, 2), self.n_rep).transpose(1, 2)
        else:
            k = _repeat_kv(k, self.n_rep).transpose(1, 2)
            v = _repeat_kv(v, self.n_rep).transpose(1, 2)
        q = q.transpose(1, 2)  # [B, H, Lq, D]; k/v already [B, H, K, D]
        if attention_mask is not None:
            # Expect [B, L] with 1 = keep. Some generate paths hand a per-step
            # mask ([B, 1]) on decode; the cached prefix was valid, extend it.
            mask = attention_mask.to(x.dtype)
            if mask.shape[-1] != k.shape[2]:
                if mask.shape[-1] == q.shape[2] and k.shape[2] > mask.shape[-1]:
                    pad = torch.ones(mask.shape[0], k.shape[2] - mask.shape[-1],
                                     device=mask.device, dtype=mask.dtype)
                    mask = torch.cat([pad, mask], dim=-1)
                elif bool((mask == 1).all()):
                    mask = None
                else:
                    raise ValueError(
                        f"attention_mask length {mask.shape[-1]} matches neither "
                        f"keys ({k.shape[2]}) nor queries ({q.shape[2]})")
            additive = None if mask is None else (1.0 - mask[:, None, None, :]) * torch.finfo(x.dtype).min
        else:
            additive = None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=additive, dropout_p=0.0, is_causal=additive is None)
        y = y.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.out_proj(y)


class Evo2HyenaMixer(nn.Module):
    """One Hyena cascade (HCS / HCM / HCL) in pure torch (parallel path).

    Args:
        filter_type: ``"s"`` (HCS, short FIR len ``hcs_filter_length``),
            ``"m"`` (HCM, medium FIR len ``hcm_filter_length``, FFT),
            ``"l"`` (HCL, implicit IIR from poles/residues, FFT).
    """

    def __init__(self, config: Evo2Config, filter_type: str):
        super().__init__()
        assert filter_type in ("s", "m", "l")
        self.config = config
        self.filter_type = filter_type
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads
        self.interleave = config.interleave
        self.column_split_hyena = config.column_split_hyena
        self.flip = config.hyena_flip_x1x2

        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=config.qkv_proj_bias)
        self.short_conv_weight = nn.Parameter(torch.randn(3 * hidden, 1, config.short_filter_length) * 0.02)
        self.short_conv_bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(3 * hidden)) if config.short_filter_bias else None
        )

        if filter_type == "s":
            self.groups = config.hcs_filter_groups
            self.filter_len = config.hcs_filter_length
            self.long_filter = nn.Parameter(torch.randn(self.groups, 1, self.filter_len) * 0.02)
            self.D: Optional[nn.Parameter] = None  # len 7 < 128 -> no D, no gated bias
        elif filter_type == "m":
            self.groups = config.hcm_filter_groups
            self.filter_len = config.hcm_filter_length
            self.long_filter = nn.Parameter(torch.randn(self.groups, 1, self.filter_len) * 0.02)
            self.D = nn.Parameter(torch.zeros(hidden))
        else:
            self.groups = hidden  # HCL defaults to per-channel systems
            self.state_size = config.state_size
            self.log_poles = nn.Parameter(torch.randn(self.groups, self.state_size, 1) * 0.02)
            self.residues = nn.Parameter(torch.randn(self.groups, self.state_size) * 0.02)
            self.D = nn.Parameter(torch.zeros(hidden))
            self.long_filter = None  # built on the fly from poles/residues

        self.out_proj = nn.Linear(hidden, hidden, bias=config.hyena_out_proj_bias)
        self._t_cache: dict = {}

    def _split(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split ``[B, 3H, L]`` into ``(x2, x1, v)`` honoring column_split flag."""
        if self.column_split_hyena:
            # Reshape [B, Hh, 3*Hd, L] then slice, matches vortex column_split.
            h, hd = self.num_heads, self.head_dim
            r = z.reshape(z.shape[0], h, 3 * hd, z.shape[2])
            x2 = r[:, :, :hd].reshape(z.shape[0], -1, z.shape[2])
            x1 = r[:, :, hd: 2 * hd].reshape(z.shape[0], -1, z.shape[2])
            v = r[:, :, 2 * hd:].reshape(z.shape[0], -1, z.shape[2])
        else:
            x2, x1, v = z.split([self.hidden_size] * 3, dim=1)
        if self.flip:
            x1, x2 = x2, x1
        return x2, x1, v

    def _long_filter_matrix(self) -> torch.Tensor:
        """Expand grouped FIR filter to ``[H, 1, K]`` (repeat-interleave)."""
        assert self.long_filter is not None
        rep = self.hidden_size // self.groups
        if rep == 1:
            return self.long_filter
        return self.long_filter.repeat_interleave(rep, dim=0)

    def _iir_filter(self, L: int, device, dtype: torch.dtype) -> torch.Tensor:
        """Modal IIR filter ``h = sum_s residues * exp(log_poles * t)`` -> ``[1, H, L]``."""
        key = (L, str(device))
        t = self._t_cache.get(key)
        if t is None or t.shape[-1] < L or t.device != device:
            t = torch.arange(L, device=device, dtype=torch.float32)[None, None]
            self._t_cache[key] = t
        else:
            t = t[..., :L]
        h = (self.residues.to(torch.float32)[..., None] * (self.log_poles.to(torch.float32) * t).exp()).sum(1)[None]
        return h.to(dtype)

    @staticmethod
    def _tail(t: torch.Tensor, k: int) -> torch.Tensor:
        """Last ``k`` vectors, left zero-padded (matches causal conv padding)."""
        if t.shape[-1] >= k:
            return t[..., -k:]
        pad = torch.zeros(*t.shape[:-1], k - t.shape[-1], device=t.device, dtype=t.dtype)
        return torch.cat([pad, t], dim=-1)

    def _init_fir_state(self, st, z_pre: torch.Tensor, u: torch.Tensor) -> None:
        st.conv_states[0] = self._tail(z_pre, 2).to(torch.float32)
        st.conv_states[1] = self._tail(u, self.filter_len - 1).to(torch.float32)
        st.mark_ready(z_pre.device, z_pre.dtype, self.filter_len - 1, recurrent=False)

    def _init_iir_state(self, st, z_pre: torch.Tensor, x1v: torch.Tensor) -> None:
        if self.groups != self.hidden_size:
            raise ValueError("cached HCL step needs per-channel systems (groups == hidden)")
        st.conv_states[0] = self._tail(z_pre, 2).to(torch.float32)
        a = self.log_poles.to(torch.float32)[:, :, 0].exp()  # [H, S] discrete poles
        u32 = x1v.to(torch.float32)  # [B, H, L]
        s = torch.zeros(u32.shape[0], self.hidden_size, self.state_size,
                        device=u32.device, dtype=torch.float32)
        for t in range(u32.shape[-1]):
            s = a[None] * s + u32[..., t, None]
        st.recurrent_states[0] = s
        st.mark_ready(z_pre.device, z_pre.dtype, 0, recurrent=True)

    def _decode_step(self, z: torch.Tensor, padding_mask: Optional[torch.Tensor], st) -> torch.Tensor:
        # z: [B, 3H, 1] in_proj output for the new token. Histories are fp32.
        dt = z.dtype
        w = self.short_conv_weight.to(torch.float32).squeeze(1)  # [3H, 3]
        win = torch.cat([st.conv_states[0], z.to(torch.float32)], dim=-1)
        s = (w[None] * win).sum(-1)
        if self.short_conv_bias is not None:
            s = s + self.short_conv_bias.to(torch.float32)[None]
        st.conv_states[0] = win[..., 1:]
        s1 = s.to(dt)[..., None]
        if self.interleave:
            s1 = _interleave(s1)
        x2, x1, v = self._split(s1)
        u = (x1 * v).to(torch.float32)[..., 0]  # [B, H]
        if self.filter_type in ("s", "m"):
            h = self._long_filter_matrix().to(torch.float32).squeeze(1)  # [H, K]
            if self.filter_len >= 128:
                h = h.flip(-1)
            hist = st.conv_states[1]
            y = (h[None] * torch.cat([hist, u[..., None]], dim=-1)).sum(-1)
            if self.filter_type == "m":
                y = y + self.D.to(torch.float32)[None] * u
            st.conv_states[1] = torch.cat([hist[..., 1:], u[..., None]], dim=-1)
            y = y.to(dt)[..., None] * x2
        else:
            if self.groups != self.hidden_size:
                raise ValueError("cached HCL step needs per-channel systems (groups == hidden)")
            a = self.log_poles.to(torch.float32)[:, :, 0].exp()
            r = self.residues.to(torch.float32)
            s_state = a[None] * st.recurrent_states[0] + u[..., None]
            st.recurrent_states[0] = s_state
            y = ((r[None] * s_state).sum(-1) + self.D.to(torch.float32)[None] * u).to(dt)[..., None] * x2
            if isinstance(padding_mask, torch.Tensor):
                y = y * padding_mask[:, None, :]
        return self.out_proj(y.permute(0, 2, 1))

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                hyena_state=None, use_cache: bool = False) -> torch.Tensor:
        # x: [B, L, H] -> [B, 3H, L]
        z = self.in_proj(x).permute(0, 2, 1)
        L = z.shape[-1]
        if use_cache and hyena_state is not None and L == 1:
            if hyena_state.conv_states.get(0) is None:
                raise RuntimeError("decode step needs a prefilled cache (run the prompt first)")
            return self._decode_step(z, padding_mask, hyena_state)
        zc = _causal_depthwise_conv1d(z, self.short_conv_weight, self.short_conv_bias)
        if self.interleave:
            zc = _interleave(zc)

        if self.filter_type in ("s", "m"):
            # Short/medium FIR path (D skip only when len>=128).
            x2, x1, v = self._split(zc)
            u = x1 * v
            h = self._long_filter_matrix()
            if self.filter_len >= 128:
                assert self.D is not None
                y = _fft_conv(u, h[:, :, :L], self.D)
            else:
                y = _causal_depthwise_conv1d(u, h, bias=None)
            y = x2 * y
            out = self.out_proj(y.permute(0, 2, 1))
            if use_cache and hyena_state is not None:
                self._init_fir_state(hyena_state, z, u)
            return out

        x2, x1, v = self._split(zc)
        x1v = x1 * v
        h = self._iir_filter(L, x.device, x1v.dtype)
        assert self.D is not None
        y = _fft_conv(x1v, h, self.D)
        y = y * x2
        # Padding mask (vortex multiplies post-FIR when bias present); apply cheaply.
        if isinstance(padding_mask, torch.Tensor):
            y = y * padding_mask[:, None, :]
        out = self.out_proj(y.permute(0, 2, 1))
        if use_cache and hyena_state is not None:
            self._init_iir_state(hyena_state, z, x1v)
        return out


# Decoder layer / full model / causal LM

class Evo2DecoderLayer(nn.Module):
    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx in (config.attn_layer_idxs or []):
            self.mixer: nn.Module = Evo2Attention(config)
        elif layer_idx in (config.hcs_layer_idxs or []):
            self.mixer = Evo2HyenaMixer(config, "s")
        elif layer_idx in (config.hcm_layer_idxs or []):
            self.mixer = Evo2HyenaMixer(config, "m")
        elif layer_idx in (config.hcl_layer_idxs or []):
            self.mixer = Evo2HyenaMixer(config, "l")
        else:
            raise ValueError(f"layer_idx={layer_idx} not in any *_layer_idxs; check config")
        self.mixer_norm = Evo2RMSNorm(config)
        self.mlp_norm = Evo2RMSNorm(config)
        self.mlp = Evo2MLP(config, layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        L = x.shape[1]
        if isinstance(attention_mask, torch.Tensor) and attention_mask.shape[1] != L:
            mask_cur = attention_mask[:, -L:]
        else:
            mask_cur = attention_mask
        layer_state = past_key_values.layers[self.layer_idx] if use_cache and past_key_values is not None else None
        if isinstance(self.mixer, Evo2Attention):
            if isinstance(mask_cur, torch.Tensor):
                x_masked = x * mask_cur[..., None].to(x.dtype)
            else:
                x_masked = x
            h = self.mixer(self.mixer_norm(x_masked), attention_mask=attention_mask,
                           position_ids=position_ids, past_layer=layer_state, use_cache=use_cache) + x
        else:
            h = self.mixer(self.mixer_norm(x), padding_mask=mask_cur,
                           hyena_state=layer_state, use_cache=use_cache) + x
            if isinstance(mask_cur, torch.Tensor):
                h = h * mask_cur[..., None].to(h.dtype)
        return self.mlp(self.mlp_norm(h)) + h


class Evo2PreTrainedModel(PreTrainedModel):
    config_class = Evo2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Evo2DecoderLayer"]


class Evo2Model(Evo2PreTrainedModel):
    def __init__(self, config: Evo2Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Evo2DecoderLayer(config, i) for i in range(config.num_layers)])
        self.final_norm = Evo2RMSNorm(config) if config.final_norm else nn.Identity()
        self.post_init()

    def _build_cache(self):
        if not _HAS_HF_CACHE:
            raise RuntimeError("use_cache=True needs transformers>=5 hybrid Cache API")
        layers = []
        for layer in self.layers:
            layers.append(DynamicLayer() if isinstance(layer.mixer, Evo2Attention) else Evo2HyenaCacheLayer())
        cache = Cache(layers=layers)
        cache._is_evo2_hybrid = True
        return cache

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if input_ids is not None:
            x = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("Provide input_ids or inputs_embeds")
        bsz, seqlen = x.shape[:2]
        if position_ids is None:
            start = 0
            if use_cache and past_key_values is not None:
                try:
                    start = int(past_key_values.get_seq_length())
                except Exception:
                    start = 0
            position_ids = torch.arange(start, start + seqlen, device=x.device).unsqueeze(0).expand(bsz, -1)
        if attention_mask is not None:
            mask_cur = attention_mask if attention_mask.shape[1] == seqlen else attention_mask[:, -seqlen:]
            x = x * mask_cur[..., None].to(x.dtype)
        if use_cache:
            if not _HAS_HF_CACHE:
                raise RuntimeError("use_cache=True needs transformers>=5 hybrid Cache API")
            if past_key_values is None or not getattr(past_key_values, "_is_evo2_hybrid", False):
                try:
                    foreign_len = 0 if past_key_values is None else past_key_values.get_seq_length()
                except Exception:
                    foreign_len = 1
                if foreign_len > 0:
                    raise ValueError("use_cache=True with a foreign non-empty cache")
                past_key_values = self._build_cache()
        all_hidden = [] if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden.append(x)
            x = layer(x, attention_mask=attention_mask, position_ids=position_ids,
                      past_key_values=past_key_values, use_cache=use_cache)
        x = self.final_norm(x)
        if output_hidden_states:
            assert all_hidden is not None
            all_hidden.append(x)
        past_out = past_key_values if use_cache else None
        if not return_dict:
            return (x, past_out, all_hidden)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_out, hidden_states=all_hidden)


class Evo2ForCausalLM(Evo2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Evo2Config):
        super().__init__(config)
        self.model = Evo2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        logits = self.lm_head(out.last_hidden_state).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        past_out = out.past_key_values if use_cache else None
        if not return_dict:
            return (loss, logits, past_out, out.hidden_states) if loss is not None else (logits, past_out, out.hidden_states)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=past_out, hidden_states=out.hidden_states
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      position_ids=None, **kwargs):
        # Decode only with a non-empty cache (generate hands an empty
        # DynamicCache on the first call); prefill keeps the full prompt.
        try:
            decoding = past_key_values is not None and past_key_values.get_seq_length() > 0
        except Exception:
            decoding = past_key_values is not None
        if decoding:
            cache_len = past_key_values.get_seq_length()
            input_ids = input_ids[:, -1:]
            position_ids = torch.arange(cache_len, cache_len + 1, device=input_ids.device
                                        ).unsqueeze(0).expand(input_ids.shape[0], -1)
        out = {"input_ids": input_ids, "past_key_values": past_key_values, "position_ids": position_ids}
        out["use_cache"] = kwargs.get("use_cache", True)
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        return out


__all__ = [
    "Evo2PreTrainedModel",
    "Evo2Model",
    "Evo2ForCausalLM",
    "Evo2RMSNorm",
    "Evo2RotaryEmbedding",
    "Evo2MLP",
    "Evo2Attention",
    "Evo2HyenaMixer",
    "Evo2HyenaCacheLayer",
    "Evo2DecoderLayer",
]
