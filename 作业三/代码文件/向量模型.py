# -*- coding: utf-8 -*-
"""
统一的向量编码后端（带自动降级）

为什么需要它
------------
本机把 torch / transformers / sentence-transformers 装在“用户级 site-packages”
（C:\\Users\\Thinkbook\\AppData\\Roaming\\Python\\Python311\\site-packages）。某些启动方式
（带 -s / PYTHONNOUSERSITE=1 的解释器、被遮蔽的导入链、torch 临时不可用等）会让
`sentence_transformers` 或 `transformers` 的惰性导入失败，典型报错就是
`ModuleNotFoundError: Could not import module 'PreTrainedModel'`（真实原因被包装了一层）。

因此这里按顺序尝试：
    1) SentenceTransformer（GPU）
    2) SentenceTransformer（CPU）
    3) 原生 transformers：AutoTokenizer + AutoModel + 末位池化（GPU）
    4) 原生 transformers（CPU）
    5) 全部失败 -> 返回 None，检索退化为纯 BM25（页面依旧可用，只是没有向量分）

环境变量：
    QA_FORCE_BACKEND=st|transformers|none   强制使用某个后端（排障/演示用）
    QA_DEVICE=cuda|cpu                      强制设备
    MODEL_DIR=<路径>                        覆盖模型目录
"""

from __future__ import annotations

import os
import re
import site
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

MODEL_NAME = os.environ.get("MODEL_DIR", "Qwen/Qwen3-Embedding-0.6B")
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

_LOAD_LOG: List[str] = []


def load_log() -> List[str]:
    return list(_LOAD_LOG)


def ensure_user_site() -> Optional[str]:
    """把“用户级 site-packages”补进 sys.path。

    本作业的 torch/transformers 都装在这里；如果解释器以 -s 或 PYTHONNOUSERSITE=1 启动，
    sys.path 里就没有它，导入会莫名其妙地失败。这里显式补上，纯属自愈。
    """
    added = None
    try:
        user_site = site.getusersitepackages()
    except Exception:  # noqa: BLE001
        return None
    if user_site and os.path.isdir(user_site) and user_site not in sys.path:
        sys.path.append(user_site)
        added = user_site
    return added


ensure_user_site()


def purge_broken_modules() -> List[str]:
    """清掉 sys.modules 里残留的半初始化 transformers / sentence_transformers。

    典型场景：长期运行的 IPython / Spyder 内核里，某次导入失败（或导入过程中依赖缺失）
    会把一个“半成品”模块留在 sys.modules，之后所有导入都会报
    `ModuleNotFoundError: Could not import module 'PreTrainedModel'`。
    这种情况下把相关模块从 sys.modules 移除再重新导入即可恢复。
    """
    removed: List[str] = []
    for name in list(sys.modules):
        if (
            name == "transformers"
            or name.startswith("transformers.")
            or name == "sentence_transformers"
            or name.startswith("sentence_transformers.")
        ):
            sys.modules.pop(name, None)
            removed.append(name)
    return removed


class BaseEmbedder:
    name = "base"
    device = "cpu"

    def encode_documents(self, texts: Sequence[str], batch_size: int = 24, progress: bool = True) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, question: str) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str, max_seq_length: int = 768):
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        try:
            self.model.max_seq_length = max_seq_length
        except Exception:  # noqa: BLE001
            pass
        if device == "cuda":
            try:
                self.model.half()
            except Exception:  # noqa: BLE001
                pass

    def encode_documents(self, texts: Sequence[str], batch_size: int = 24, progress: bool = True) -> np.ndarray:
        # 按长度排序编码，减少 batch 内 padding 浪费
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]
        encode = getattr(self.model, "encode_document", None) or self.model.encode
        vectors = None
        for size in (batch_size, max(8, batch_size // 2), 8):
            try:
                vectors = encode(
                    sorted_texts,
                    batch_size=size,
                    normalize_embeddings=True,
                    show_progress_bar=progress,
                    convert_to_numpy=True,
                )
                break
            except RuntimeError:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
        if vectors is None:
            raise RuntimeError("向量编码失败（显存不足且降批无效）")
        vectors = np.asarray(vectors, dtype=np.float32)
        out = np.empty_like(vectors)
        for pos, idx in enumerate(order):
            out[idx] = vectors[pos]
        return out

    def encode_query(self, question: str) -> np.ndarray:
        prompt = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"
        encode_query = getattr(self.model, "encode_query", None)
        if encode_query is not None:
            vector = encode_query(question, prompt=prompt, normalize_embeddings=True)
        else:
            vector = self.model.encode(prompt, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32).reshape(-1)


class TransformersEmbedder(BaseEmbedder):
    """不依赖 sentence-transformers 的兜底实现：AutoModel + 末位池化 + L2 归一化。"""

    name = "transformers(AutoModel)"

    def __init__(self, model_name: str, device: str, max_seq_length: int = 768):
        import torch
        from transformers import AutoModel, AutoTokenizer  # noqa: WPS433

        self.torch = torch
        self.device = device
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        self.model.to(device)
        self.model.eval()

    def _pool(self, last_hidden, attention_mask):
        torch = self.torch
        # 左 padding：取每条序列最后一个非 pad 位置的 hidden state
        idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx]
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def _encode(self, texts: Sequence[str], batch_size: int, progress: bool) -> np.ndarray:
        torch = self.torch
        out: List[np.ndarray] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
                pooled = self._pool(hidden, enc["attention_mask"])
            out.append(pooled.float().cpu().numpy())
            if progress and (start // batch_size) % 5 == 0:
                print(f"  向量编码 {min(start + batch_size, total)}/{total}", flush=True)
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, 1024), dtype=np.float32)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 16, progress: bool = True) -> np.ndarray:
        return self._encode(list(texts), batch_size, progress)

    def encode_query(self, question: str) -> np.ndarray:
        prompt = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"
        return self._encode([prompt], 1, False)[0]


def build_embedder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_seq_length: int = 768,
    log=print,
) -> Optional[BaseEmbedder]:
    """按顺序尝试各后端，返回第一个可用的；全部失败返回 None（退化为纯 BM25）。"""
    model_name = model_name or MODEL_NAME
    forced = os.environ.get("QA_FORCE_BACKEND", "").strip().lower()
    device_pref = device or os.environ.get("QA_DEVICE") or None

    candidates: List[tuple] = []
    if not forced or forced == "st":
        candidates += [("sentence-transformers", SentenceTransformerEmbedder, device_pref or "cuda"),
                       ("sentence-transformers", SentenceTransformerEmbedder, "cpu")]
    if not forced or forced == "transformers":
        candidates += [("transformers(AutoModel)", TransformersEmbedder, device_pref or "cuda"),
                       ("transformers(AutoModel)", TransformersEmbedder, "cpu")]

    # 第一轮；全部失败时清理残留模块再试第二轮（解决 IPython/Spyder 内核里的“半初始化模块”问题）
    for round_no in (1, 2):
        for name, cls, dev in candidates:
            started = time.time()
            try:
                embedder = cls(model_name, dev, max_seq_length)
                msg = f"[向量后端] {name} · device={dev} · 加载 {time.time() - started:.1f}s"
                _LOAD_LOG.append(msg)
                log(msg)
                return embedder
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                msg = f"[向量后端失败] 第{round_no}轮 {name} · device={dev} -> {type(exc).__name__}: {exc}"
                _LOAD_LOG.append(msg)
                _LOAD_LOG.append(tb)
                log(msg)
                log("    " + "\n    ".join(tb.strip().splitlines()[-6:]))
        if round_no == 1:
            removed = purge_broken_modules()
            note = (
                f"[向量后端] 清理残留模块后重试：移除 {len(removed)} 个条目"
                f"（{'transformers/sentence_transformers 缓存' if removed else '无残留'}）"
            )
            _LOAD_LOG.append(note)
            log(note)
    if forced == "none":
        log("[向量后端] 已按 QA_FORCE_BACKEND=none 跳过，退化为纯 BM25 检索")
    else:
        log("[向量后端] 全部不可用，退化为纯 BM25 检索（功能仍可用，只是没有向量分）")
    return None


def describe_environment() -> Dict:
    """诊断用：Python、关键包版本与导入情况。"""
    info: Dict = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "user_site": None,
        "packages": {},
        "import_errors": {},
    }
    try:
        info["user_site"] = site.getusersitepackages()
    except Exception:  # noqa: BLE001
        pass
    for name in ("numpy", "torch", "transformers", "sentence_transformers", "tokenizers", "huggingface_hub", "jieba"):
        try:
            module = __import__(name)
            info["packages"][name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            info["import_errors"][name] = f"{type(exc).__name__}: {exc}"
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        info["torch_cuda_available"] = False
    info["model"] = MODEL_NAME
    info["model_dir_exists"] = os.path.isdir(MODEL_NAME)
    info["load_log"] = load_log()
    return info
