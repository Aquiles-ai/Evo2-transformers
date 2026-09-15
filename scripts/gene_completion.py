"""Gene completion benchmark, prokaryote panel.

Ports ArcInstitute/evo2 scripts/gene_completion (prokaryote only: prompt the
model with 1000 nt upstream + first 30% of the CDS, complete the rest,
translate, global protein alignment vs reference, % AA identity over the
non-prompt region).

Edit CONFIG below and run: python gene_completion.py.

Data (one file, reference proteins are inside the CSV):
    wget https://raw.githubusercontent.com/ArcInstitute/evo2/main/scripts/gene_completion/data/prokaryote_genes.csv

Deps: torch, transformers, biopython (``pip install biopython``).

Reference (paper): Evo 2 1B base prokaryote mean AA recovery 64.9. The paper
uses 50 generations/gene; default here is 5 with use_cache=True (decode
from cache, needs transformers>=5).
"""

import csv
import os
import torch
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_DIR = "Aquiles-ai/Evo2-1B-Base"
DATA_CSV = "./prokaryote_genes.csv"
OUT_DIR = "./out_gene_completion"
TAG = "evo2_1b_base"
GENES = ""  # comma-separated subset, or "" for all
N_GEN = 5
TEMPERATURE = 0.7
TOP_K = 4
SEED = 0
PROMPT_FRACTION = 0.30
PROK_UPSTREAM_LEN = 1000

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32


def prokaryote_prompt(genomic, cds_start=5000, upstream_len=PROK_UPSTREAM_LEN,
                      fraction=PROMPT_FRACTION):
    genomic = "".join(genomic.split()).upper()
    coding_nt = len(genomic) - cds_start
    coding_aa_take = round(coding_nt / 3.0 * fraction)
    take_nt = coding_aa_take * 3
    start = max(0, cds_start - upstream_len)
    prompt = genomic[start:cds_start] + genomic[cds_start:cds_start + take_nt]
    return prompt, coding_aa_take


def translate_dna(dna, to_stop=True):
    dna = "".join(dna.split()).upper()
    usable = len(dna) - (len(dna) % 3)
    if usable <= 0:
        return ""
    return str(Seq(dna[:usable]).translate(to_stop=to_stop))


_ALIGNER = None

def _aligner():
    global _ALIGNER
    if _ALIGNER is None:
        a = PairwiseAligner()
        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
        a.open_gap_score = -11
        a.extend_gap_score = -1
        a.mode = "global"
        _ALIGNER = a
    return _ALIGNER

def aligned_identity_after(query_aa, ref_aa, ref_start):
    if not query_aa or not ref_aa:
        return float("nan")
    aln = _aligner().align(ref_aa, query_aa)[0]
    matches = aligned = 0
    ref_row, qry_row = aln.indices
    for r, q in zip(ref_row, qry_row):
        if r >= ref_start and q >= 0:
            aligned += 1
            if ref_aa[r] == query_aa[q]:
                matches += 1
    if aligned == 0:
        return float("nan")
    return 100.0 * matches / aligned

def score_prokaryote(generated_full, reference_protein, upstream_len, prompt_cds_aa):
    coding = generated_full[upstream_len:]
    gen_protein = translate_dna(coding, to_stop=True)
    return aligned_identity_after(gen_protein, reference_protein, prompt_cds_aa)

def complete_once(model, tok, prompt, n_tokens, seed):
    torch.manual_seed(seed)
    ids = torch.tensor([tok.vortex_tokenize(prompt)], dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        gen = model.generate(
            ids, max_new_tokens=n_tokens, do_sample=True,
            temperature=TEMPERATURE, top_k=TOP_K,
            use_cache=True, pad_token_id=tok.pad_token_id,
        )
    full = "".join(tok.vortex_detokenize(gen[0].tolist()).split()).upper()
    if not full.startswith(prompt):
        full = prompt + full
    return full

def main():
    tok = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_DIR, trust_remote_code=True, dtype=DTYPE).to(DEVICE).eval()
    max_pos = model.config.max_position_embeddings

    with open(DATA_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if GENES:
        wanted = {g.strip().lower() for g in GENES.split(",")}
        rows = [r for r in rows if r["gene"].strip().lower() in wanted]
    if not rows:
        raise SystemExit("No genes selected.")

    raw_path = os.path.join(OUT_DIR, f"{TAG}_prokaryote_completions.csv")
    gene_means = {}
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene", "organism", "gen_idx", "aa_recovery_non_prompt", "prompt_len"])
        for row in rows:
            gene = row["gene"]
            genomic = "".join(str(row["genomic_sequence"]).split()).upper()
            ref_protein = str(row["reference_protein"]).strip()
            prompt, prompt_cds_aa = prokaryote_prompt(genomic, int(row["cds_start"]))
            n_tokens = max(1, len(ref_protein) - prompt_cds_aa) * 3 + 150
            if len(prompt) + n_tokens > max_pos:
                raise SystemExit(
                    f"[{gene}] prompt+n_tokens exceeds context ({max_pos})")
            print(f"[{gene}] prompt={len(prompt)} nt, n_tokens={n_tokens}, gens={N_GEN}")
            scores = []
            for g in range(N_GEN):
                full = complete_once(model, tok, prompt, n_tokens, SEED + g)
                rec = score_prokaryote(full, ref_protein, PROK_UPSTREAM_LEN, prompt_cds_aa)
                scores.append(rec)
                w.writerow([gene, row.get("organism", ""), g, f"{rec:.2f}", len(prompt)])
                print(f"  gen {g}: AA recovery={rec:.2f}%")
            gene_means[gene] = sum(scores) / len(scores)

    stats_path = os.path.join(OUT_DIR, f"{TAG}_prokaryote_per_gene_stats.csv")
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene", "n_samples", "mean_aa_recovery"])
        for gene, mean in gene_means.items():
            w.writerow([gene, N_GEN, f"{mean:.2f}"])
    panel = sum(gene_means.values()) / len(gene_means)
    print(f"\nWrote {raw_path}\nWrote {stats_path}")
    print(f"Panel mean AA recovery: {panel:.2f}% (ref 1B base: 64.9)")

if __name__ == "__main__":
    main()
