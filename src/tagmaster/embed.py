"""CPU sentence embeddings via ONNX Runtime, with a content-addressed cache.

This box has no GPU, so embedding ~200k documents is the single slowest step in
the pipeline. The cache is keyed by a hash of the model id, truncation settings,
and the texts themselves, which makes every re-run after the first free and
makes it safe to call this from anywhere in the pipeline.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config


@dataclass
class EmbedderSettings:
    repo_id: str = "Xenova/all-MiniLM-L6-v2"
    onnx_path: str = "onnx/model.onnx"
    max_tokens: int = 256
    batch_size: int = 64
    threads: int = 4

    @classmethod
    def from_config(cls, cfg: Config) -> EmbedderSettings:
        raw = cfg["features"].get("embedding", {}) or {}
        return cls(
            repo_id=raw.get("repo_id", cls.repo_id),
            onnx_path=raw.get("onnx_path", cls.onnx_path),
            max_tokens=int(raw.get("max_tokens", cls.max_tokens)),
            batch_size=int(raw.get("batch_size", cls.batch_size)),
            threads=int(raw.get("threads", cls.threads)),
        )

    @property
    def signature(self) -> str:
        return f"{self.repo_id}|{self.onnx_path}|{self.max_tokens}"


class OnnxEmbedder:
    """Mean-pooled, L2-normalised sentence embeddings.

    The ONNX graph and tokenizer are loaded lazily so that importing this module
    (and running the parts of the pipeline that do not need embeddings) never
    pays the model download cost.
    """

    def __init__(self, settings: EmbedderSettings, cache_dir: Path):
        self.settings = settings
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(self.settings.repo_id, self.settings.onnx_path)
        tok_path = hf_hub_download(self.settings.repo_id, "tokenizer.json")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.settings.threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.enable_truncation(max_length=self.settings.max_tokens)
        tokenizer.enable_padding(length=None)
        self._tokenizer = tokenizer

    @property
    def dim(self) -> int:
        self._load()
        shape = self._session.get_outputs()[0].shape
        tail = shape[-1]
        return int(tail) if isinstance(tail, int) else 384

    def _encode_batch(self, batch: list[str]) -> np.ndarray:
        enc = self._tokenizer.encode_batch(batch)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        out = self._session.run(None, feeds)[0]
        if out.ndim == 2:
            vecs = out
        else:
            m = mask.astype(np.float32)[:, :, None]
            vecs = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / np.clip(norms, 1e-9, None)).astype(np.float32)

    def encode(self, texts: list[str], progress: bool = False) -> np.ndarray:
        self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        bs = self.settings.batch_size
        iterator = range(0, len(texts), bs)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=(len(texts) + bs - 1) // bs, desc="embed", unit="batch")
        for start in iterator:
            batch = [t[:2000] if t else "" for t in texts[start : start + bs]]
            chunks.append(self._encode_batch(batch))
        return np.vstack(chunks)

    def _cache_key(self, texts: list[str], tag: str) -> str:
        h = hashlib.sha256()
        h.update(self.settings.signature.encode())
        h.update(tag.encode())
        h.update(str(len(texts)).encode())
        for t in texts:
            h.update(hashlib.sha1((t or "")[:2000].encode("utf-8", "ignore")).digest())
        return h.hexdigest()[:24]

    def encode_cached(self, texts: list[str], tag: str, progress: bool = True) -> np.ndarray:
        """Embed `texts`, reusing a cached result when the exact inputs recur."""
        key = self._cache_key(texts, tag)
        dest = self.cache_dir / f"emb_{tag}_{key}.npy"
        if dest.exists():
            return np.load(dest)
        vecs = self.encode(texts, progress=progress)
        tmp = dest.with_suffix(".npy.tmp")
        np.save(tmp, vecs)
        os.replace(tmp, dest)
        return vecs


def build_embedder(cfg: Config) -> OnnxEmbedder | None:
    """Returns None when embeddings are disabled, so callers can skip them."""
    raw = cfg["features"].get("embedding", {}) or {}
    if not raw.get("enabled", True):
        return None
    return OnnxEmbedder(EmbedderSettings.from_config(cfg), cfg.paths.cache)
