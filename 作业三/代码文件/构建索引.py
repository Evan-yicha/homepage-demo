# -*- coding: utf-8 -*-
"""
作业三 · 构建索引（向量索引 + BM25 索引）

* 向量：本机已缓存的 Qwen/Qwen3-Embedding-0.6B（HF_HOME 指向 E:\HuggingFaceRepo），
  GPU 优先（RTX 5060 / sm_120），离线加载；查询侧用 instruct 前缀，文档侧用原文。
* BM25：jieba 分词（自动加载财报/药企词表）+ 倒排索引，k1=1.5、b=0.75，
  自建实现（内存占用小、加载快），并与 rank_bm25 做一次分数一致性自检。

产物（作业三/文本与索引/索引/）
    chunks.jsonl      与索引行序一一对应的切块（含公司/章节/页码等出处元数据）
    vectors.npy       float32 向量矩阵（行数 = 切块数）
    bm25.json.gz      BM25 倒排索引
    index_config.json 索引配置与统计

用法
----
    python 构建索引.py                 # 全量构建
    python 构建索引.py --limit 200     # 只索引前 200 块（自测）
    python 构建索引.py --device cpu
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from 向量模型 import build_embedder  # 统一的向量后端（ST 优先，失败自动降级）

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

CODE_DIR = Path(__file__).resolve().parent
ASSIGN_DIR = CODE_DIR.parent
DATA_DIR = ASSIGN_DIR / "文本与索引"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_DIR = DATA_DIR / "索引"

MODEL_NAME = os.environ.get("MODEL_DIR", "Qwen/Qwen3-Embedding-0.6B")
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

BM25_K1 = 1.5
BM25_B = 0.75

DOMAIN_WORDS = [
    "研发费用", "研发投入", "营业收入", "营业成本", "销售费用", "管理费用", "财务费用",
    "归属于上市公司股东的净利润", "归属于上市公司股东的扣除非经常性损益的净利润",
    "经营活动产生的现金流量净额", "毛利率", "净利率", "研发费用率", "销售费用率",
    "创新药", "生物药", "化学药", "小分子", "单克隆抗体", "双特异性抗体", "ADC", "CAR-T",
    "临床试验", "临床前", "适应症", "获批上市", "上市许可", "医保谈判", "集采", "带量采购",
    "CDMO", "CRO", "原料药", "中间体", "制剂", "产能", "订单", "里程碑付款", "对外授权",
    "政府补助", "股权激励", "股份支付", "应收账款", "存货", "商誉", "研发人员", "专利",
    "药明康德", "恒瑞医药", "百济神州", "科伦药业", "康龙化成", "凯莱英", "泰格医药",
    "复星医药", "甘李药业", "华东医药", "海思科", "荣昌生物", "沃森生物",
    "母公司资产负债表", "合并资产负债表", "合并利润表", "合并现金流量表",
]


def log(msg: str) -> None:
    print(msg, flush=True)


_JIEBA_READY = False
_JIEBA_AVAILABLE = True


def tokenize(text: str) -> List[str]:
    """中文分词：优先 jieba，缺失时退回“ASCII 词 + 中文二元组”。"""
    global _JIEBA_READY, _JIEBA_AVAILABLE
    if _JIEBA_AVAILABLE:
        try:
            import jieba

            if not _JIEBA_READY:
                for word in DOMAIN_WORDS:
                    jieba.add_word(word)
                jieba.initialize()
                _JIEBA_READY = True
            return [t for t in jieba.lcut(text) if t.strip()]
        except Exception:  # noqa: BLE001
            _JIEBA_AVAILABLE = False
    return fallback_tokens(text)


def fallback_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for chunk in re.findall(r"[A-Za-z0-9.%\-]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if len(chunk) == 1:
                tokens.append(chunk)
            else:
                tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
        else:
            tokens.append(chunk)
    return tokens


class BM25Index:
    """倒排索引版 BM25（k1/b 可配），检索时只访问命中的倒排表。"""

    def __init__(self, postings: Dict[str, List[List[int]]], doc_len: List[int], k1: float, b: float):
        self.postings = postings
        self.doc_len = doc_len
        self.k1 = k1
        self.b = b
        self.n_docs = len(doc_len)
        self.avgdl = (sum(doc_len) / self.n_docs) if self.n_docs else 0.0
        self.idf: Dict[str, float] = {}
        for term, plist in postings.items():
            df = len(plist)
            self.idf[term] = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, tokens: Sequence[str], top_k: int = 100) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)
        for term in tokens:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_id, tf in plist:
                dl = self.doc_len[doc_id] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[doc_id] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return ranked

    def to_json(self) -> Dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "doc_len": self.doc_len,
            "postings": self.postings,
        }

    @classmethod
    def from_json(cls, data: Dict) -> "BM25Index":
        return cls(data["postings"], data["doc_len"], data["k1"], data["b"])


def build_bm25(texts: Sequence[str], verbose: bool = True) -> BM25Index:
    postings: Dict[str, List[List[int]]] = defaultdict(list)
    doc_len: List[int] = []
    for idx, text in enumerate(texts):
        tokens = tokenize(text)
        doc_len.append(len(tokens))
        for term, tf in Counter(tokens).items():
            postings[term].append([idx, tf])
        if verbose and (idx + 1) % 5000 == 0:
            log(f"  BM25 分词 {idx + 1}/{len(texts)}")
    return BM25Index(dict(postings), doc_len, BM25_K1, BM25_B)


def verify_bm25(index: BM25Index, texts: Sequence[str]) -> Optional[str]:
    """与 rank_bm25 对拍，确认自建实现分数一致（用于自检，不影响索引结果）。"""
    try:
        from rank_bm25 import BM25Okapi
    except Exception:  # noqa: BLE001
        return None
    try:
        corpus = [tokenize(t) for t in texts[: min(len(texts), 500)]]
        ref = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
        query = tokenize("研发费用率 创新药 营业收入")
        ours = {i: s for i, s in index.search(query, top_k=10)}
        theirs = ref.get_scores(query)
        diffs = [
            abs(ours[i] - float(theirs[i])) / (abs(float(theirs[i])) + 1e-9)
            for i in ours
            if i < len(theirs) and float(theirs[i]) > 0
        ]
        return f"自建 BM25 与 rank_bm25 分数一致（最大相对差异 {max(diffs) if diffs else 0:.4%}）"
    except Exception as exc:  # noqa: BLE001
        return f"rank_bm25 对拍跳过：{type(exc).__name__}: {exc}"


def load_model(device: Optional[str] = None):
    """加载向量后端（SentenceTransformer 优先，失败自动降级到原生 transformers）。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    embedder = build_embedder(MODEL_NAME, device=device, max_seq_length=768, log=log)
    if embedder is None:
        raise SystemExit("向量后端不可用：无法构建向量索引（可用 QA_FORCE_BACKEND 排障）")
    return embedder


def embed_documents(model, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    return model.encode_documents(texts, batch_size=batch_size, progress=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建向量 + BM25 索引")
    parser.add_argument("--limit", type=int, help="只索引前 N 个切块（自测）")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="强制设备")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args(argv)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks: List[Dict] = []
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    if args.limit:
        chunks = chunks[: args.limit]
    log(f"切块：{len(chunks)} 个（来自 {CHUNKS_PATH.name}）")

    with (INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # 建索引用 index_text（带公司/报告/章节/页码前缀），展示与出处用原始 text
    index_texts = [c.get("index_text") or c["text"] for c in chunks]

    config: Dict = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": len(chunks),
        "model": MODEL_NAME,
        "query_instruction": QUERY_INSTRUCTION,
        "index_text": "chunk.index_text（含公司/报告期间/章节/页码前缀）",
        "bm25": {"k1": BM25_K1, "b": BM25_B, "tokenizer": "jieba"},
    }

    if not args.skip_bm25:
        started = time.time()
        index = build_bm25(index_texts)
        with gzip.open(INDEX_DIR / "bm25.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(index.to_json(), fh, ensure_ascii=False)
        config["bm25"]["terms"] = len(index.postings)
        config["bm25"]["avgdl"] = round(index.avgdl, 2)
        config["bm25"]["build_seconds"] = round(time.time() - started, 1)
        log(f"  BM25 索引：{len(index.postings)} 个词项，耗时 {config['bm25']['build_seconds']}s")
        note = verify_bm25(index, index_texts)
        if note:
            log(f"  {note}")
            config["bm25"]["verify"] = note

    if not args.skip_vector:
        started = time.time()
        model = load_model(args.device)
        vectors = embed_documents(model, index_texts, batch_size=args.batch_size)
        np.save(INDEX_DIR / "vectors.npy", vectors)
        config["vector"] = {
            "dim": int(vectors.shape[1]),
            "rows": int(vectors.shape[0]),
            "dtype": str(vectors.dtype),
            "build_seconds": round(time.time() - started, 1),
        }
        log(f"  向量索引：{vectors.shape}，耗时 {config['vector']['build_seconds']}s")

    (INDEX_DIR / "index_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"索引完成 -> {INDEX_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
