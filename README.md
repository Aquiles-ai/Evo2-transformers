# Evo2-transformers

Unofficial PyTorch and Transformers port of Evo 2 inference. All model, data, and research credit goes to the original Evo 2 team at Arc Institute and collaborators.

Original project: https://github.com/ArcInstitute/evo2

This repo contains no new model weights and no retraining. It reimplements the StripedHyena 2 forward pass with plain PyTorch so Evo 2 checkpoints can load through `AutoModelForCausalLM` without Vortex, Transformer Engine, or custom CUDA kernels.

## Credits and original references

Evo 2 was developed by Arc Institute with NVIDIA, Stanford, UC Berkeley, and UCSF.

* Code and weights: [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2) (Apache-2.0)
* Paper: [Genome modeling and design across all domains of life with Evo 2](https://www.nature.com/articles/s41586-026-10176-5), Nature, 2026. Preprint from Feb 2025.
* Architecture: StripedHyena 2, a hybrid of Hyena convolutions and grouped query attention with RoPE. Described in the Evo 2 paper and in [Systems and Algorithms for Convolutional Multi-Hybrid Language Models at Scale](https://github.com/Zymrael/savanna/blob/main/paper.pdf).
* Training stack: [Savanna](https://github.com/Zymrael/savanna)
* Original inference stack: [Vortex](https://github.com/Zymrael/vortex)
* Pretraining data: [OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2), 8.8T tokens across all domains of life
* Official checkpoints: [huggingface.co/arcinstitute](https://huggingface.co/arcinstitute) (`evo2_1b_base`, `evo2_7b_base`, `evo2_7b_262k`, `evo2_7b`, `evo2_20b`, `evo2_40b_base`, `evo2_40b`, `evo2_7b_microviridae`)
* Hosted inference: [NVIDIA NIM for Evo 2](https://docs.nvidia.com/nim/bionemo/evo2/latest/overview.html)

If you use Evo 2 in research, cite the original paper (see Citation below), not this repo.

## What this port covers

* `evo2/configuration_evo2.py`: `Evo2Config` with layer maps (`attn`, `hcs`, `hcm`, `hcl`), filter lengths, RoPE settings, and presets for all 7 official architectures.
* `evo2/modeling_evo2.py`: `Evo2Model` and `Evo2ForCausalLM`. Hyena S/M/L mixers, GQA attention with RoPE, vortex-compatible RMSNorm and gated MLP, tied LM head, plus a decoding cache (native per-layer states, needs transformers>=5).
* `evo2/tokenization_evo2.py`: byte-level tokenizer matching the original `CharLevelTokenizer` (vocab 512, `bos/eos=0`, `pad=1`). Includes `vortex_tokenize` and `vortex_detokenize` helpers.
* `evo2/convert_evo2_vortex_to_hf.py`: converts a merged Vortex `.pt` checkpoint to a Transformers folder (`config.json`, `model.safetensors`, tokenizer files).
* `scripts/gene_completion.py`: prokaryote gene completion benchmark ported from the original `scripts/gene_completion` script.

## Performance note

This port is less efficient than the original implementation. It is correct for scoring and short generation, but it is not a replacement for Vortex in production or for long context work.

Missing optimizations include:

* No FlashAttention. Attention runs through `scaled_dot_product_attention`.
* No Transformer Engine, FP8 paths, or fused Triton kernels.
* Decoding cache is supported through the transformers hybrid Cache API (needs transformers>=5), with native per-layer states for attention and Hyena mixers. With `use_cache=False`, generation recomputes the full prefix at each step.
* Hyena long convolutions run as full precision FFTs per channel, which is `O(N log N)` memory and compute per layer. Long contexts (262k, 1M) are supported by config but slow and memory heavy here.
* No chunking, no fp16 or bf16 FFT path, no multi-GPU sharding helpers.

For 40B or 1M context inference, use the original Vortex stack or NVIDIA NIM.

## Install

Requires Python 3.11 or 3.12 and PyTorch with CUDA for GPU use.

```
pip install torch transformers safetensors
pip install biopython  # only for scripts/gene_completion.py
```

Clone and use the local `evo2` package directly, or copy a converted checkpoint folder that already vendors the modeling files.

## Usage

Load a converted checkpoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Aquiles-ai/Evo2-1B-Base"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.eval()
```

Score a sequence:

```python
import torch

ids = torch.tensor([tok.vortex_tokenize("ACGT")])
with torch.inference_mode():
    out = model(ids)
print(out.logits.shape)  # [1, 4, 512]
```

Generate:

```python
import torch

ids = torch.tensor([tok.vortex_tokenize("ACGT")])
with torch.inference_mode():
    gen = model.generate(
        ids,
        max_new_tokens=400,
        do_sample=True,
        temperature=1.0,
        top_k=4,
        use_cache=True,
    )
print(tok.vortex_detokenize(gen[0].tolist()))
```

`generate` uses the decoding cache by default (needs transformers>=5). With `use_cache=False` it recomputes the full prefix at each step, which is slow past a few hundred tokens.

## Convert a Vortex checkpoint

Download or merge the official `.pt` first with the original Evo 2 repo, then run:

```
python evo2/convert_evo2_vortex_to_hf.py \
  --vortex_pt ~/.cache/huggingface/evo2_1b_base.pt \
  --variant evo2_1b_base \
  --out_dir ./evo2-1b-hf \
  --dtype bf16
```

`--variant` must match the checkpoint. Supported values: `evo2_1b_base`, `evo2_7b_base`, `evo2_7b_microviridae`, `evo2_7b_262k`, `evo2_7b`, `evo2_20b`, `evo2_40b_base`, `evo2_40b`.

The output folder is self contained and loadable with `trust_remote_code=True`. Conversion needs roughly 2x params in CPU RAM during the build (about 8 GB for 1B, 40 GB for 7B; 20B and 40B need a large node).

## Gene completion script

Ports the prokaryote panel from the original repo: prompt with 1 kb upstream plus the first 30 percent of the CDS, complete the rest, translate, and score amino acid identity on the non prompt region.

```
wget https://raw.githubusercontent.com/ArcInstitute/evo2/main/scripts/gene_completion/data/prokaryote_genes.csv
pip install biopython
python scripts/gene_completion.py
```

Edit `HF_DIR`, `DATA_CSV`, `N_GEN`, and related constants at the top of the script. Defaults use 5 generations per gene. The paper reports 50 generations per gene and 64.9 mean recovery for the 1B base model.

## Supported configs

All presets use vocab 512 and byte-level tokenization. Context and size differ:

| Variant | Params | Layers | Context |
|---|---|---|---|
| `evo2_1b_base` | 1B | 25 | 8k |
| `evo2_7b_base` | 7B | 32 | 32k |
| `evo2_7b_microviridae` | 7B | 32 | 32k |
| `evo2_7b_262k` | 7B | 32 | 262k |
| `evo2_7b` | 7B | 32 | 1M |
| `evo2_20b` | 20B | 24 | 1M |
| `evo2_40b_base` | 40B | 50 | 8k |
| `evo2_40b` | 40B | 50 | 1M |

## Citation

```bibtex
@article{Brixi2026,
  author  = {Brixi, Garyk and Durrant, Matthew G. and Ku, Jerome and Naghipourfar, Mohsen and Poli, Michael and Sun, Gwanggyu and Brockman, Greg and Chang, Daniel and Fanton, Alison and Gonzalez, Gabriel A. and King, Samuel H. and Li, David B. and Merchant, Aditi T. and Nguyen, Eric and Ricci-Tam, Chiara and Romero, David W. and Schmok, Jonathan C. and Taghibakhshi, Ali and Vorontsov, Anton and Yang, Brandon and Deng, Myra and Gorton, Liv and Nguyen, Nam and Wang, Nicholas K. and Pearce, Michael T. and Simon, Elana and Adams, Etowah and Amador, Zachary J. and Ashley, Euan A. and Baccus, Stephen A. and Dai, Haoyu and Dillmann, Steven and Ermon, Stefano and Guo, Daniel and Herschl, Michael H. and Ilango, Rajesh and Janik, Ken and Lu, Amy X. and Mehta, Reshma and Mofrad, Mohammad R. K. and Ng, Madelena Y. and Pannu, Jaspreet and R{\'e}, Christopher and St. John, John and Sullivan, Jeremy and Tey, Joseph and Viggiano, Ben and Zhu, Kevin and Zynda, Greg and Balsam, Daniel and Collison, Patrick and Costa, Anthony B. and Hernandez-Boussard, Tina and Ho, Eric and Liu, Ming-Yu and McGrath, Thomas and Powell, Kimberly and Pinglay, Sudarshan and Burke, Dave P. and Goodarzi, Hani and Hsu, Patrick D. and Hie, Brian L.},
  title   = {Genome modelling and design across all domains of life with Evo 2},
  journal = {Nature},
  year    = {2026},
  doi     = {10.1038/s41586-026-10176-5},
  url     = {https://doi.org/10.1038/s41586-026-10176-5}
}
```

## License

Apache-2.0, see `LICENSE`. Original Evo 2 code, weights, and data remain property of their respective owners and keep their original licenses and use terms.
