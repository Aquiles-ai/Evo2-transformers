from transformers.configuration_utils import PretrainedConfig
from typing import List, Optional

# Based from https://raw.githubusercontent.com/ArcInstitute/evo2/refs/heads/main/evo2/configs/evo2-1b-8k.yml
# Full-transformers port: no vortex dependency. Defaults = evo2-1b-8k,
# but all long-context variants (7b/20b/40b) are expressible via these fields.

class Evo2Config(PretrainedConfig):
    model_type = "evo2"

    def __init__(
            self,
            vocab_size: int = 512,
            hidden_size: int = 1920,
            # Number of independent filters in Hyena-LI
            num_filters: int = 1920,
            num_layers: int = 25,
            attn_layer_idxs: List[int] | None = None,
            hcl_layer_idxs: List[int] | None = None,
            hcm_layer_idxs: List[int] | None = None,
            hcs_layer_idxs: List[int] | None = None,
            hcm_filter_length: int = 128,
            hcs_filter_length: int = 7,
            hcl_filter_groups: int = 1920,
            hcm_filter_groups: int = 128,
            hcs_filter_groups: int = 128,
            # Length of the short, depthwise FIR applied to input projections
            short_filter_length: int = 3,
            short_filter_bias: bool = False,
            num_attention_heads: int = 15,
            # Number of groups in GQA (proj_groups in vortex/savanna)
            proj_groups: int = 1,
            # Number of groups in grouped hyena filter
            hyena_filter_groups: int = 1,
            state_size: int = 16,
            eps: float = 1e-6,
            rotary_emb_base: float = 10000.0,
            rotary_emb_scaling_factor: Optional[float] = None,
            use_interpolated_rotary_pos_emb: bool = False,
            make_vocab_size_divisible_by: int = 8,
            inner_size_multiple_of: int = 16,
            inner_mlp_size: int = 5120,
            mlp_activation: str = "gelu",
            # Split strategy for channels
            column_split_hyena: bool = False,
            column_split: bool = True,
            interleave: bool = True,
            # Layer > 0 nn.identity activation
            evo2_style_activations: bool = True,
            tie_word_embeddings: bool = True,
            mha_out_proj_bias: bool = True,
            hyena_out_proj_bias: bool = True,
            hyena_flip_x1x2: bool = False,
            qkv_proj_bias: bool = False,
            max_position_embeddings: int = 8192,
            final_norm: bool = True,
            use_cache: bool = True,
            pad_token_id: int = 1,
            bos_token_id: int = 0,
            eos_token_id: int = 0,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.attn_layer_idxs = attn_layer_idxs if attn_layer_idxs is not None else [3, 10, 17, 24]
        self.hcl_layer_idxs = hcl_layer_idxs if hcl_layer_idxs is not None else [2, 6, 9, 13, 16, 20, 23]
        self.hcm_layer_idxs = hcm_layer_idxs if hcm_layer_idxs is not None else [1, 5, 8, 12, 15, 19, 22]
        self.hcs_layer_idxs = hcs_layer_idxs if hcs_layer_idxs is not None else [0, 4, 7, 11, 14, 18, 21]
        self.hcm_filter_length = hcm_filter_length
        self.hcs_filter_length = hcs_filter_length
        self.hcl_filter_groups = hcl_filter_groups
        self.hcm_filter_groups = hcm_filter_groups
        self.hcs_filter_groups = hcs_filter_groups
        self.short_filter_length = short_filter_length
        self.short_filter_bias = short_filter_bias
        self.num_attention_heads = num_attention_heads
        self.proj_groups = proj_groups
        self.hyena_filter_groups = hyena_filter_groups
        self.state_size = state_size
        self.eps = eps
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scaling_factor = rotary_emb_scaling_factor
        self.use_interpolated_rotary_pos_emb = use_interpolated_rotary_pos_emb
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.inner_size_multiple_of = inner_size_multiple_of
        self.inner_mlp_size = inner_mlp_size
        self.mlp_activation = mlp_activation
        self.column_split_hyena = column_split_hyena
        self.column_split = column_split
        self.interleave = interleave
        self.evo2_style_activations = evo2_style_activations
        self.mha_out_proj_bias = mha_out_proj_bias
        self.hyena_out_proj_bias = hyena_out_proj_bias
        self.hyena_flip_x1x2 = hyena_flip_x1x2
        self.qkv_proj_bias = qkv_proj_bias
        self.max_position_embeddings = max_position_embeddings
        self.final_norm = final_norm
        self.use_cache = use_cache
        self.auto_map = {
            "AutoConfig": "configuration_evo2.Evo2Config",
            "AutoModel": "modeling_evo2.Evo2Model",
            "AutoModelForCausalLM": "modeling_evo2.Evo2ForCausalLM",
            "AutoTokenizer": "tokenization_evo2.Evo2Tokenizer"
        }
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )