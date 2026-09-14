"""Byte-level tokenizer for Evo2 (custom because no HF tokenizer matches it).

Why custom: the original ``CharLevelTokenizer`` (vortex ``model/tokenizer.py``)
has no merges and no vocab file. Encoding is raw UTF-8 bytes
(``list(text.encode("utf-8"))``, so ``"ACGT" -> [65, 67, 71, 84]``, one token
per nucleotide), ``vocab_size=512``, ``eos/bos=0``, ``pad=1``. Decoding clamps
each id into ``[32, vocab_size]`` before ``chr()``, so id 0/1 decode as space.
That exact numeric behavior plus the batch helpers is what this class keeps.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer


def _id_to_token(i: int) -> str:
    if 32 <= i < 127:
        return chr(i)
    return f"<0x{i:02X}>"


class Evo2Tokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {}

    def __init__(self, vocab_size: int = 512, **kwargs):
        self._vocab_size = vocab_size
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("bos_token", "<eos>")
        kwargs.setdefault("add_bos_token", False)
        kwargs.setdefault("add_eos_token", False)
        kwargs.setdefault("model_max_length", 1048576)
        kwargs.setdefault("padding_side", "right")
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> Dict[str, int]:
        vocab = {"<eos>": 0, "<pad>": 1}
        for i in range(2, 256):
            vocab[_id_to_token(i)] = i
        for i in range(256, self._vocab_size):
            vocab[f"<unused_{i}>"] = i
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return [_id_to_token(b) for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token: str) -> int:
        if token == "<eos>":
            return 0
        if token == "<pad>":
            return 1
        if len(token) == 1:
            return ord(token)
        if token.startswith("<0x") and token.endswith(">"):
            return int(token[3:-1], 16)
        if token.startswith("<unused_") and token.endswith(">"):
            return int(token[len("<unused_"):-1])
        raise ValueError(f"Unknown Evo2 token: {token!r}")

    def _convert_id_to_token(self, index: int) -> str:
        if index == 0:
            return "<eos>"
        if index == 1:
            return "<pad>"
        if 2 <= index < 256:
            return _id_to_token(index)
        if 256 <= index < self._vocab_size:
            return f"<unused_{index}>"
        raise ValueError(f"Evo2 id out of range: {index}")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ids = [self._convert_token_to_id(t) for t in tokens]
        raw = bytes([i for i in ids if 0 <= i < 256])
        return raw.decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            return ()
        name = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        path = os.path.join(save_directory, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_vocab(), f, ensure_ascii=False)
        return (path,)

    def vortex_tokenize(self, text: str) -> List[int]:
        """Drop-in for vortex ``CharLevelTokenizer.tokenize``: raw UTF-8 bytes."""
        return list(text.encode("utf-8"))

    def vortex_detokenize(self, token_ids: Union[List[int], object]) -> str:
        """Drop-in for vortex ``detokenize``: per-id ``chr(clamp(id))``."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        top = self._vocab_size
        return "".join(chr(max(32, min(int(t), top))) for t in token_ids)

    def vortex_tokenize_batch(self, texts: Union[List[str], str]) -> Union[List[List[int]], List[int]]:
        if isinstance(texts, list):
            return [self.vortex_tokenize(s) for s in texts]
        return self.vortex_tokenize(texts)

    def vortex_detokenize_batch(self, batch) -> List[str]:
        if hasattr(batch, "tolist"):
            batch = batch.tolist()
        return [self.vortex_detokenize(s) for s in batch]
