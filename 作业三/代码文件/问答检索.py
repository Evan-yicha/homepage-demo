# -*- coding: utf-8 -*-
"""
作业三 · 问答检索（向量 + BM25 混合检索，带出处）

对外接口
--------
    from 问答检索 import Retriever
    r = Retriever()
    hits = r.retrieve("恒瑞医药2025年研发投入是多少？", top_k=8, mode="hybrid")
    # hits: [{chunk_id, 公司, 报告, 章节, 页码, 文本, vector_score, bm25_score, fused_score, rank}]

融合方式：RRF（Reciprocal Rank Fusion，k=60），对向量排名与 BM25 排名加权融合；
mode 可为 hybrid / vector / bm25。查询侧使用 Qwen3-Embedding 的 instruct 前缀，
文档侧使用原文，向量已归一化，相似度即余弦。
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

ASSIGN_DIR = CODE_DIR.parent
DATA_DIR = ASSIGN_DIR / "文本与索引"
INDEX_DIR = DATA_DIR / "索引"
PDF_DIR = ASSIGN_DIR / "创新药业绩报告"

from 向量模型 import QUERY_INSTRUCTION, build_embedder, load_log  # noqa: E402

MODEL_NAME = os.environ.get("MODEL_DIR", "Qwen/Qwen3-Embedding-0.6B")
RRF_K = 60


def _tokenize(text: str) -> List[str]:
    """与建索引时保持一致的分词。"""
    try:
        from 构建索引 import tokenize  # type: ignore

        return tokenize(text)
    except Exception:  # noqa: BLE001
        pass
    try:
        import jieba

        return [t for t in jieba.lcut(text) if t.strip()]
    except Exception:  # noqa: BLE001
        tokens: List[str] = []
        for chunk in re.findall(r"[A-Za-z0-9.%\-]+|[\u4e00-\u9fff]+", text):
            if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
                tokens.extend(chunk[i : i + 2] for i in range(max(1, len(chunk) - 1)))
            else:
                tokens.append(chunk)
        return tokens


_VALUE_RE = re.compile(r"\d[\d,]*\.\d+|\d+(?:\.\d+)?\s*(?:%|％|亿元|万元|元|倍|个百分点)")


_BOILERPLATE_RE = re.compile(
    r"(√适用|□不适用|单位：|币种：|会计机构负责人|主管会计工作负责人|合并利润表|合并资产负债表|"
    r"合并现金流量表|上一控制下企业合并|被合并方|法定代表人|董事会决议|附件)"
)


def _values(unit: str) -> set:
    return {m.group(0) for m in _VALUE_RE.finditer(unit)}


def _has_value(unit: str) -> bool:
    """是否包含“真正的数值”（年份不算）。"""
    for match in _VALUE_RE.finditer(unit):
        text = match.group(0)
        if re.fullmatch(r"(19|20)\d{2}", text.strip()):
            continue
        return True
    return False


# 实质数值：必须有小数位、千分位或单位，避免把“0 元 / 页码 / 年份”当答案
_SUBSTANTIVE_VALUE_RE = re.compile(
    r"\d[\d,]*\.\d+|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*(?:%|％|亿元|万元|元|倍|个百分点)"
)
# 财务指标名词：出现这些词就认为问题在“问数值”
METRIC_WORDS = (
    "营业收入", "营业成本", "净利润", "归母净利润", "扣除非经常性损益", "研发投入", "研发费用",
    "研发支出", "毛利率", "净利率", "研发费用率", "销售费用", "管理费用", "财务费用",
    "现金流", "现金流量净额", "每股收益", "净资产", "总资产", "分红", "派息", "市占率",
    "销售收入", "营业收入规模", "收入", "利润", "研发人员", "资本化率", "资产负债率",
)
# 附注/会计政策类噪声词（问题没问到时应当抑制）
_NOTE_NOISE_RE = re.compile(
    r"(期初未分配利润|未分配利润|会计政策|会计估计|递延所得税|合并范围|附注|"
    r"同一控制|被合并方|资本公积弥补亏损|其他调整合计|减值测试情况)"
)
_PLACEHOLDER_VALUE_RE = re.compile(r"^(?:0|0\.0+|-\s*|—\s*|－\s*)$")

# 同一指标的不同写法（用于“是否覆盖全部问项”的判定，避免研发投入/研发费用被当成两个指标）
_METRIC_SYNONYMS = {
    "研发费用": "研发投入",
    "研发支出": "研发投入",
    "归母净利润": "净利润",
    "销售收入": "营业收入",
    "营业收入规模": "营业收入",
    "收入": "营业收入",
    "现金流量净额": "现金流",
}


def _asks_value(question: str) -> bool:
    """问题是否在问“某个数值/指标”。"""
    if re.search(r"多少|几|比例|占比|金额|同比|增长|下降|排名|最高|最低|率|规模|情况如何", question):
        return True
    return any(word in question for word in METRIC_WORDS)


def _has_substantive_value(unit: str) -> bool:
    """是否含“实质数值”：带小数/千分位/单位，且不是 0 元这类占位值、不是年份。"""
    for match in _SUBSTANTIVE_VALUE_RE.finditer(unit):
        raw = match.group(0).strip()
        number = re.match(r"[\d,.]+", raw).group(0).rstrip(".") if re.match(r"[\d,.]+", raw) else ""
        if number and _PLACEHOLDER_VALUE_RE.match(number.replace(",", "")):
            continue  # 0 / 0.00 这类占位值不算
        if re.fullmatch(r"(19|20)\d{2}", number.replace(",", "")):
            continue  # 纯年份不算
        return True
    return False


def _clean_answer_text(text: str) -> str:
    """去掉页面页眉页脚等噪声，把答案整理成干净的一行/一段。"""
    text = re.sub(r"\s+", " ", text).strip()
    # 常见页眉页脚：“60/236 江苏恒瑞医药股份有限公司2025年年度报告”
    text = re.sub(r"\b\d{1,4}\s*/\s*\d{1,4}\b\s*", "", text)
    text = re.sub(r"[\u4e00-\u9fff（）()]{4,40}?股份有限公司\d{4}\s*年(年度|半年度)报告(全文)?", "", text)
    text = text.replace("|", "｜")
    text = re.sub(r"(｜\s*)+", "｜", text)
    text = re.sub(r"\s{2,}", " ", text)
    # 残句清理：去掉开头的连接词/标点、合并重复分隔符
    text = re.sub(r"^[，。；、：,.;:）)】\s]+", "", text)
    text = re.sub(r"[；;]{2,}", "；", text)
    text = re.sub(r"^[^\u4e00-\u9fffA-Za-z0-9]+", "", text)
    return text.strip(" ｜；;，,")


def _is_meaningful(unit: str, min_len: int = 12, min_cjk: int = 6) -> bool:
    """过滤掉只剩标点/数字后缀的残缺片段。

    表格行往往很短（例如“营业收入｜31,629,416,193.83｜…”只有 4 个汉字），
    所以表格行单独用更宽松的阈值（min_cjk=2）。
    """
    if len(unit) < min_len:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", unit))
    return cjk >= min_cjk


@dataclass
class Hit:
    chunk_id: str
    company: str
    code: str
    market: str
    board: str
    report_type: str
    period: str
    title: str
    publish_date: str
    announce_url: str
    source_pdf: str
    section: str
    page_start: int
    page_end: int
    block_type: str
    text: str
    index_text: str = ""          # 建索引时用的文本（含公司/报告期/章节/页码前缀），用于覆盖率计算
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rank: int = 0

    def to_dict(self, snippet: int = 0) -> Dict:
        data = {
            "chunk_id": self.chunk_id,
            "company": self.company,
            "code": self.code,
            "market": self.market,
            "board": self.board,
            "report_type": self.report_type,
            "period": self.period,
            "title": self.title,
            "publish_date": self.publish_date,
            "announce_url": self.announce_url,
            "source_pdf": self.source_pdf,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "block_type": self.block_type,
            "text": self.text if not snippet else self.text[:snippet],
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "fused_score": round(self.fused_score, 6),
            "rank": self.rank,
        }
        return data

    def citation(self) -> str:
        pages = (
            f"第{self.page_start}页"
            if self.page_start == self.page_end
            else f"第{self.page_start}-{self.page_end}页"
        )
        return f"{self.company}《{self.period}》{self.section} · {pages}"


def _unit_to_item(unit: Dict, hit: Hit) -> Dict:
    """把候选要点转成带出处的标准结构。"""
    return {
        "score": round(unit["score"], 4),
        "text": unit["text"],
        "rank": hit.rank,
        "company": hit.company,
        "period": hit.period,
        "report_type": hit.report_type,
        "section": hit.section,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "block_type": hit.block_type,
        "chunk_id": hit.chunk_id,
        "source_pdf": hit.source_pdf,
        "announce_url": hit.announce_url,
        "citation": hit.citation(),
    }


class Retriever:
    """向量 + BM25 混合检索。索引全部来自本地文件，不联网。"""

    def __init__(self, index_dir: Path = INDEX_DIR, device: Optional[str] = None, load_model: bool = True):
        self.index_dir = Path(index_dir)
        chunks_path = self.index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"缺少索引文件：{chunks_path}（请先运行 构建索引.py）")
        self.chunks: List[Dict] = [
            json.loads(line) for line in chunks_path.open(encoding="utf-8") if line.strip()
        ]
        self.vectors: Optional[np.ndarray] = None
        vectors_path = self.index_dir / "vectors.npy"
        if vectors_path.exists():
            self.vectors = np.load(vectors_path, mmap_mode="r")
        self.bm25 = None
        bm25_path = self.index_dir / "bm25.json.gz"
        if bm25_path.exists():
            from 构建索引 import BM25Index  # type: ignore

            with gzip.open(bm25_path, "rt", encoding="utf-8") as fh:
                self.bm25 = BM25Index.from_json(json.load(fh))
        self.embedder = None
        self.device = device
        self.load_notes: List[str] = []
        if load_model and self.vectors is not None:
            self._load_model()

    # ------------------------------------------------------------------ 模型
    def _load_model(self) -> None:
        """加载向量后端：SentenceTransformer(GPU/CPU) → 原生 transformers(GPU/CPU) → 纯 BM25。"""
        if self.embedder is not None:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.embedder = build_embedder(MODEL_NAME, device=self.device)
        self.load_notes = load_log()
        self.device = getattr(self.embedder, "device", self.device)
        if self.embedder is None:
            print("  [提示] 向量模型不可用，已退化为纯 BM25 检索（结果仍带出处）", flush=True)

    def encode_query(self, question: str) -> np.ndarray:
        self._load_model()
        if self.embedder is None:
            raise RuntimeError("向量后端不可用（已退化为纯 BM25 检索）")
        return self.embedder.encode_query(question)

    # ------------------------------------------------------------------ 检索
    def search_vector(self, question: str, top_k: int = 100) -> List[Tuple[int, float]]:
        if self.vectors is None or self.embedder is None:
            return []
        try:
            query = self.encode_query(question)
            matrix = np.asarray(self.vectors, dtype=np.float32)
            scores = matrix @ query
            k = min(top_k, scores.shape[0])
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            return [(int(i), float(scores[i])) for i in idx]
        except Exception as exc:  # noqa: BLE001
            # 编码失败也不能拖垮问答：退回 BM25
            print(f"  [警告] 向量检索失败（{type(exc).__name__}: {exc}），本次改用 BM25", flush=True)
            self.embedder = None
            return []

    def search_bm25(self, question: str, top_k: int = 100) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        return self.bm25.search(_tokenize(question), top_k=top_k)

    def retrieve(
        self,
        question: str,
        top_k: int = 8,
        mode: str = "hybrid",
        candidates: int = 100,
        company_filter: Optional[Sequence[str]] = None,
        weights: Tuple[float, float] = (1.0, 1.0),
        company_boost: float = 1.6,
    ) -> List[Hit]:
        mode = (mode or "hybrid").lower()
        # 向量不可用时（缺 index 或后端加载失败）自动退化为 BM25，保证页面始终能答
        if mode in ("hybrid", "vector") and (self.vectors is None or self.embedder is None):
            self._load_model()
            if self.vectors is None or self.embedder is None:
                mode = "bm25"
        vec_hits: List[Tuple[int, float]] = []
        bm_hits: List[Tuple[int, float]] = []
        if mode in ("hybrid", "vector"):
            vec_hits = self.search_vector(question, top_k=candidates)
        if mode in ("hybrid", "bm25"):
            bm_hits = self.search_bm25(question, top_k=candidates)

        fused: Dict[int, float] = {}
        vec_score = {i: s for i, s in vec_hits}
        bm_score = {i: s for i, s in bm_hits}
        w_vec, w_bm = weights
        for rank, (idx, _s) in enumerate(vec_hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + w_vec / (RRF_K + rank)
        for rank, (idx, _s) in enumerate(bm_hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + w_bm / (RRF_K + rank)
        if mode == "vector":
            fused = {i: fused.get(i, 0.0) for i, _ in vec_hits}
        elif mode == "bm25":
            fused = {i: fused.get(i, 0.0) for i, _ in bm_hits}

        # 问题里点名了公司时，对该公司名下的块加权（跨公司问题会同时命中多家，各加各的）
        mentioned = self.mentioned_companies(question)
        if mentioned:
            for idx in list(fused):
                if self.chunks[idx]["company"] in mentioned:
                    fused[idx] *= company_boost

        # 问题点名了报告期：命中足够多时只在对应期间内排序（避免年报/半年报串期）
        periods = self.mentioned_periods(question)
        if periods:
            matching = [i for i in fused if self.chunks[i]["period"] in periods]
            if len(matching) >= 3:
                fused = {i: fused[i] for i in matching}
            else:
                for idx in list(fused):
                    if self.chunks[idx]["period"] in periods:
                        fused[idx] *= 1.5

        allowed = set(company_filter) if company_filter else None
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        hits: List[Hit] = []
        for idx, score in ordered:
            chunk = self.chunks[idx]
            if allowed and chunk["company"] not in allowed:
                continue
            hits.append(
                Hit(
                    chunk_id=chunk["chunk_id"],
                    company=chunk["company"],
                    code=chunk["code"],
                    market=chunk["market"],
                    board=chunk["board"],
                    report_type=chunk["report_type"],
                    period=chunk["period"],
                    title=chunk["title"],
                    publish_date=chunk["publish_date"],
                    announce_url=chunk["announce_url"],
                    source_pdf=chunk["source_pdf"],
                    section=chunk["section"],
                    page_start=chunk["page_start"],
                    page_end=chunk["page_end"],
                    block_type=chunk["block_type"],
                    text=chunk["text"],
                    index_text=chunk.get("index_text") or chunk["text"],
                    vector_score=vec_score.get(idx, 0.0),
                    bm25_score=bm_score.get(idx, 0.0),
                    fused_score=score,
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def mentioned_companies(self, question: str) -> List[str]:
        """问题里出现的公司名（公司清单来自切块元数据，新增公司无需改代码）。"""
        if not hasattr(self, "_companies"):
            self._companies = sorted({c["company"] for c in self.chunks}, key=len, reverse=True)
        return [name for name in self._companies if name in question]

    def mentioned_periods(self, question: str) -> List[str]:
        """问题里点名的报告期（如“2025年年度报告”“2025年报”“2026半年报”）。

        定期报告题几乎都会指定期间，命中后只在对应期间的切块里排序，
        可以避免“问年报却答成半年报”这类串期错误。
        """
        if not hasattr(self, "_periods"):
            self._periods = sorted({c["period"] for c in self.chunks})
        found: List[str] = []
        for period in self._periods:
            year_match = re.match(r"(\d{4})", period)
            if not year_match:
                continue
            year = year_match.group(1)
            if period in question:
                found.append(period)
                continue
            if "半年度报告" in period:
                patterns = (f"{year}半年报", f"{year}年半年报", f"{year}年半年度报告", f"{year}半年度报告")
            else:
                patterns = (f"{year}年报", f"{year}年年度报告", f"{year}年度报告", f"{year}年年报")
            if any(p in question for p in patterns):
                found.append(period)
        return found

    def all_companies(self) -> List[str]:
        if not hasattr(self, "_companies"):
            self._companies = sorted({c["company"] for c in self.chunks}, key=len, reverse=True)
        return sorted({c["company"] for c in self.chunks})

    def retrieve_panorama(
        self,
        question: str,
        companies: Optional[Sequence[str]] = None,
        top_k_each: int = 2,
        mode: str = "hybrid",
        candidates: int = 100,
    ) -> List[Hit]:
        """全景模式：对每家公司分别检索 Top-N 再合并。

        跨公司比较题（例如“13 家公司的研发费用率排名”）用单次全局检索拿不到每家的数字，
        因为同一指标分散在 13 份报告里。这里按公司逐个检索，保证每家公司都有证据块进入答案。
        """
        targets = list(companies) if companies else self.all_companies()
        merged: List[Tuple[int, Hit]] = []
        for company in targets:
            hits = self.retrieve(
                question,
                top_k=top_k_each,
                mode=mode,
                candidates=candidates,
                company_filter=[company],
            )
            for order, hit in enumerate(hits):
                merged.append((order, hit))
        # 先按“各公司内部名次”轮流输出（每家最优块优先），同档内按融合分排序，
        # 这样 Top5 能覆盖 5 家不同公司，符合跨公司比较题的阅读顺序。
        merged.sort(key=lambda item: (item[0], -item[1].fused_score))
        for rank, (_order, hit) in enumerate(merged, start=1):
            hit.rank = rank
        return [hit for _order, hit in merged]

    # ------------------------------------------------------------------ 抽取式答案
    def extractive_answer(self, question: str, hits: Sequence[Hit], max_bullets: int = 5, snippet: int = 320) -> str:
        """把 Top 检索块整理成带出处的要点式答案（不依赖任何生成模型）。"""
        if not hits:
            return "没有检索到相关内容。"
        lines: List[str] = []
        for i, hit in enumerate(hits[:max_bullets], start=1):
            text = re.sub(r"\s+", " ", hit.text).strip()
            if hit.block_type == "table":
                text = text.replace(" | ", "｜").replace("|", "｜")
            snippet_text = text[:snippet] + ("…" if len(text) > snippet else "")
            lines.append(f"- {snippet_text} [#{i}]（{hit.citation()}）")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 单条最佳答案
    def _candidate_units(self, hit: Hit) -> List[str]:
        """把一个检索块拆成可比对的候选答案单元：表格按行、正文按句。"""
        text = hit.text.strip()
        if hit.block_type == "table":
            rows = [r.strip() for r in text.splitlines() if r.strip()]
            rows = [r for r in rows if not re.fullmatch(r"\|[\s\-|:]+\|", r)]
            return [r for r in rows if len(r) > 4]
        parts = re.split(r"(?<=[。；;！？!?])", text)
        units = [p.strip() for p in parts if len(p.strip()) >= 8]
        return units or [text]

    def _asked_metrics(self, question: str, q_tokens: Sequence[str]) -> List[str]:
        """问题里问到的指标字段（优先取长度更长的指标词，避免“收入”盖住“营业收入”）。"""
        kept: List[str] = []
        for word in sorted({w for w in METRIC_WORDS if w in question}, key=len, reverse=True):
            if any(word in k for k in kept):
                continue
            kept.append(word)
        for token in sorted(set(q_tokens), key=len, reverse=True):
            if len(token) >= 4 and not any(token in k for k in kept):
                kept.append(token)
        return kept

    def metric_words_in(self, question: str) -> List[str]:
        """问题里出现的“财务指标词”（去掉被更长指标词包含的短词）。"""
        kept: List[str] = []
        for word in sorted({w for w in METRIC_WORDS if w in question}, key=len, reverse=True):
            if any(word in k for k in kept):
                continue
            kept.append(_METRIC_SYNONYMS.get(word, word))
        # 归并后再去重（研发投入/研发费用 → 研发投入）
        out: List[str] = []
        for word in kept:
            if word not in out:
                out.append(word)
        return out

    def _unit_candidates(self, question: str, hit: Hit, q_tokens: Sequence[str]) -> List[Dict]:
        """把一个块拆成候选要点，并标记是否“合格”（能否作为该问题的答案）。"""
        value_q = _asks_value(question)
        metrics = self._asked_metrics(question, q_tokens)
        idf = getattr(self.bm25, "idf", {}) if self.bm25 is not None else {}
        units = self._candidate_units(hit)
        header = ""
        if hit.block_type == "table" and len(units) >= 2 and re.fullmatch(r"\|[\s\-|:]+\|", units[1].strip()):
            header = units[0]
        out: List[Dict] = []
        for idx, raw in enumerate(units):
            text = _clean_answer_text(raw)
            if header and raw != header:
                text = _clean_answer_text(header) + " " + text
            min_len, min_cjk = (8, 2) if hit.block_type == "table" else (12, 6)
            if not _is_meaningful(text, min_len=min_len, min_cjk=min_cjk):
                continue
            label = text.split("｜")[0] if "｜" in text else text[:14]
            metric_hit = any(metric in text for metric in metrics)
            # 噪声判定用“该要点自己”的内容（不要被注入的表头文本牵连）
            own = _clean_answer_text(raw)
            noise = bool(
                _BOILERPLATE_RE.search(own)
                or (_NOTE_NOISE_RE.search(own) and not _NOTE_NOISE_RE.search(question))
            )
            if value_q:
                qualifying = _has_substantive_value(text) and metric_hit and not noise
            else:
                matched = [t for t in set(q_tokens) if t in text]
                strong = [t for t in matched if idf.get(t, 1.0) >= 4.0]
                qualifying = (len(matched) >= 2 or len(strong) >= 1) and not noise
            out.append(
                {
                    "idx": idx,
                    "text": text,
                    "label": label,
                    "metric_hit": metric_hit,
                    "qualifying": qualifying,
                    "score": self._unit_score(question, raw, hit, hit.rank),
                }
            )
        return out

    def _unit_score(self, question: str, unit: str, hit: Hit, rank: int) -> float:
        """给候选答案打分：与问题的实词重合度（用 BM25 的 idf 加权）+ 排名先验 + 题型先验。"""
        q_tokens = [t for t in _tokenize(question) if len(t) > 1 or t.isdigit()]
        u_tokens = set(_tokenize(unit))
        idf = getattr(self.bm25, "idf", {}) if self.bm25 is not None else {}
        overlap = 0.0
        for term in set(q_tokens):
            if term in u_tokens:
                overlap += idf.get(term, 1.0)
        # 用单元长度做开方归一，避免长段落靠“字多”占便宜
        score = 2.4 * overlap / max(4.0, (len(u_tokens) ** 0.5) * 2.0)
        score += 1.6 / rank
        numeric_q = _asks_value(question)
        if numeric_q:
            score += 1.2 if _has_substantive_value(unit) else -1.0  # 问数值却没有实质数值的句子/表头不是答案
            if hit.block_type == "table":
                score += 1.2
            if len(unit) > 150:
                score -= 0.9  # 长段落一般是背景说明
        if _BOILERPLATE_RE.search(unit):
            score -= 2.0  # 报表套话/页眉类文字
        if _NOTE_NOISE_RE.search(unit) and not _NOTE_NOISE_RE.search(question):
            score -= 1.5  # 会计政策/附注类内容，通常在问题没问到时不是答案
        if re.search(r"(^|[｜|\s])(?:0|0\.0+)\s*(?:元|万元|亿元|%|％)?([｜|\s]|$)", unit):
            score -= 1.0  # “0 元”这类占位数值
        if re.match(r"^\s*\d", unit):
            score -= 0.8  # 以数字开头的多半是被截断的表行
        if ("营业成本" in unit or "营业支出" in unit) and not re.search(r"营业成本|成本结构", question):
            score -= 1.2  # 问收入却给了成本行
        # 同行比较表里会列别家公司，容易答错：抑制“别家公司”行，优先“本公司”行
        others = [name for name in self.all_companies() if name != hit.company and name in unit]
        if others:
            score -= 1.5 * len(others)
        if re.search(r"(本公司|公司报告期内|本集团|公司本期|其中：|合计)", unit):
            score += 0.4
        return score

    def pick_best_answer(
        self, question: str, hits: Sequence[Hit], max_chars: int = 0, extra_sources: int = 1
    ) -> Dict:
        """给出“一条最合适的答案”。

        选块规则（排名锚定 + 质量门槛）：按召回排名从第 1 名开始检查，第一个存在“合格要点”
        的块就是答案来源；找不到合格要点时才退回按分数选块，并标记 confidence="low"。
        合格要点 = 数值类问题必须“含实质数值 + 含问到的指标字段 + 不是报表套话/附注噪声”，
        非数值类问题至少要命中 2 个问题实词（或 1 个高 idf 实词）。
        块内再取最相关的 1~3 条要点拼成一条答案（例如同时回答“金额”和“占比”）。

        max_chars=0（默认）表示**不截断**，答案按要点原文完整输出。
        """
        if not hits:
            return {
                "text": "没有检索到相关内容。",
                "hit": None,
                "candidates": [],
                "sources": [],
                "source_rank": 0,
                "confidence": "low",
            }

        q_tokens = {t for t in _tokenize(question) if len(t) > 1 or t.isdigit()}
        numeric_q = _asks_value(question)

        chunk_infos: List[Dict] = []
        for hit in hits:
            scored = self._unit_candidates(question, hit, sorted(q_tokens))
            if not scored:
                continue
            # 覆盖率在 index_text（含公司/报告期/章节/页码前缀）上算，避免表格块被系统性低估
            chunk_tokens = set(_tokenize(hit.index_text or hit.text))
            coverage = len(q_tokens & chunk_tokens) / max(1, len(q_tokens))
            chunk_score = max(s["score"] for s in scored) + 0.6 * coverage
            if numeric_q and hit.block_type == "table":
                chunk_score += 0.6
            chunk_infos.append(
                {
                    "hit": hit,
                    "units": scored,
                    "score": chunk_score,
                    "coverage": coverage,
                    "qualifying": [u for u in scored if u["qualifying"]],
                }
            )

        if not chunk_infos:
            return {
                "text": "没有检索到相关内容。",
                "hit": None,
                "candidates": [],
                "sources": [],
                "source_rank": 0,
                "confidence": "low",
            }

        # ---- 排名锚定：从召回第 1 名开始，取第一个存在合格要点的块 ----
        asked_metrics = self._asked_metrics(question, sorted(q_tokens))

        def metric_cover(info: Dict) -> int:
            """该块能覆盖几个“问题里问到的指标”（用于多问项时挑覆盖更全的块）。"""
            return max(
                (sum(1 for m in asked_metrics if m in u["text"]) for u in info["qualifying"]),
                default=0,
            )

        qualifying_infos = [info for info in chunk_infos if info["qualifying"]]
        # 只在前 3 个召回块里选答案（第 1 块不合格就顺延到第 2、3 块）
        pool = [info for info in chunk_infos if info["hit"].rank <= 3] or chunk_infos
        qualifying_infos = [info for info in pool if info["qualifying"]]
        anchored = qualifying_infos[0] if qualifying_infos else None
        if anchored is not None and len(asked_metrics) >= 2:
            # 前 3 名内若有块覆盖更多问项（例如同页表格同时给出营收与净利润），优先用它
            warmer = max(qualifying_infos[:3], key=lambda i: (metric_cover(i), -i["hit"].rank))
            if metric_cover(warmer) > metric_cover(anchored):
                anchored = warmer
        if anchored is not None:
            chosen = anchored
            confidence = "high"
        else:
            chosen = max(pool, key=lambda c: c["score"])
            confidence = "low"
        chunk_infos = pool

        asked_metric_words = self.metric_words_in(question)

        def final_confidence(text: str, conf: str) -> str:
            """多指标问题没覆盖全时，标为"部分覆盖"，避免给出过度自信的答案。"""
            if conf == "high" and len(asked_metric_words) >= 2:
                covered = sum(1 for m in asked_metric_words if m in text)
                if covered < len(asked_metric_words):
                    return "partial"
            return conf
        chosen_units = sorted(chosen["units"], key=lambda u: -u["score"])
        qualifying_units = sorted(chosen["qualifying"], key=lambda u: -u["score"])
        usable_units = qualifying_units or chosen_units
        if not qualifying_units and numeric_q:
            # 低置信度兜底：问数值时优先取带实质数值的行/句
            with_value = [u for u in chosen_units if _has_substantive_value(u["text"])]
            if with_value:
                usable_units = with_value

        chunk_infos.sort(key=lambda c: -c["score"])

        # 表格题：按“问题里问到的字段”逐行取（问营业收入+净利润就同时给这两行）
        if chosen["hit"].block_type == "table" and numeric_q:
            asked = [
                token
                for token in sorted(q_tokens, key=len, reverse=True)
                if len(token) >= 2 and any(token in u["text"] for u in usable_units)
            ]
            label_rows: List[Dict] = []
            used_labels: List[str] = []
            used_idxs: set = set()
            # 先放入最相关的那一行，再按“问题里问到的字段”补行（优先行首标签匹配）
            if usable_units:
                label_rows.append(usable_units[0])
                used_idxs.add(usable_units[0]["idx"])
            for token in asked:
                if len(label_rows) >= 3:
                    break
                if any(token in label for label in used_labels):
                    continue
                def _label_of(unit_text: str) -> str:
                    return unit_text.split("｜")[0] if "｜" in unit_text else unit_text[:12]

                match = next(
                    (
                        u
                        for u in usable_units
                        if u["idx"] not in used_idxs
                        and _has_substantive_value(u["text"])
                        and _label_of(u["text"]).startswith(token)
                    ),
                    None,
                ) or next(
                    (
                        u
                        for u in usable_units
                        if u["idx"] not in used_idxs and token in u["text"] and _has_substantive_value(u["text"])
                    ),
                    None,
                )
                if match is not None:
                    label_rows.append(match)
                    used_idxs.add(match["idx"])
                    used_labels.append(token)
            if label_rows:
                label_rows.sort(key=lambda u: u["idx"])
                text = "；".join(u["text"] for u in label_rows)
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars].rstrip("；;，, ") + "…"
                return {
                    "text": text,
                    "hit": _unit_to_item(label_rows[0], chosen["hit"]),
                    "candidates": [_unit_to_item(u, chosen["hit"]) for u in chosen_units[:5]],
                    "sources": [_unit_to_item(u, chosen["hit"]) for u in label_rows[:1]],
                    "source_rank": chosen["hit"].rank,
                    "confidence": final_confidence(text, confidence),
                }

        # 只保留“最合适”的 1~2 条要点：要分数接近、互相不重复（含在新要点里就替换）、
        # 且必须带来“新的数值”或“新的问题要点”，避免把整块内容倒出来或答重复内容。
        picked_units: List[Dict] = []
        covered_values: set = set()
        covered_tokens: set = set()
        for unit in usable_units:
            if len(picked_units) >= 2:
                break
            text = unit["text"]
            if any(text in p["text"] for p in picked_units):
                continue
            replaced = False
            for p in list(picked_units):
                if p["text"] in text:
                    picked_units.remove(p)
                    covered_values -= _values(p["text"])
                    covered_tokens -= set(_tokenize(p["text"]))
                    replaced = True
            if picked_units and not replaced:
                if unit["score"] < picked_units[0]["score"] * 0.8:
                    break
                if numeric_q:
                    if not (_values(text) - covered_values):
                        continue
                else:
                    new_tokens = (set(_tokenize(text)) & q_tokens) - covered_tokens
                    if not new_tokens:
                        continue
            if max_chars and sum(len(p["text"]) for p in picked_units) + len(text) > max_chars + 60:
                continue
            picked_units.append(unit)
            covered_values |= _values(text)
            covered_tokens |= set(_tokenize(text))
        if not picked_units:
            picked_units = [usable_units[0]]
        length = sum(len(p["text"]) for p in picked_units)
        # 多问项只覆盖了一部分时，再补一条来自次优块的要点（仍保持总长度可控）
        need_extra = confidence == "low" or chosen["coverage"] < 0.5
        if numeric_q and len({re.sub(r"[｜|].*", "", u["text"])[:10] for u in picked_units}) < 2:
            need_extra = True  # 数值题只答到一个字段时，再从相邻块补一个字段
        if need_extra and len(picked_units) < 3:
            for info in chunk_infos[1 : 1 + max(2, extra_sources)]:
                for unit in sorted(info["units"], key=lambda u: -u["score"]):
                    if unit["score"] < usable_units[0]["score"] * 0.45:
                        continue
                    if numeric_q and not _has_substantive_value(unit["text"]):
                        continue
                    if _BOILERPLATE_RE.search(unit["text"]):
                        continue
                    if _NOTE_NOISE_RE.search(unit["text"]) and not _NOTE_NOISE_RE.search(question):
                        continue
                    if numeric_q and not (_values(unit["text"]) - covered_values):
                        continue  # 没有带来新的数值，没必要追加
                    if max_chars and length + len(unit["text"]) > max_chars + 60:
                        continue
                    picked_units.append(unit)
                    covered_values |= _values(unit["text"])
                    length += len(unit["text"]) + 1
                    break
                break

        picked_units.sort(key=lambda u: u["idx"])
        answer_text = "；".join(u["text"] for u in picked_units)
        if max_chars and len(answer_text) > max_chars:
            cut = answer_text[:max_chars]
            for sep in ("；", "。", "，"):
                pos = cut.rfind(sep)
                if pos > max_chars * 0.6:
                    cut = cut[:pos]
                    break
            answer_text = cut.rstrip("；;，, ") + "…"

        to_item = _unit_to_item
        best_item = to_item(picked_units[0], chosen["hit"])
        # 出处：凡是内容进入答案的块都列出来（最多 1 + extra_sources 个）
        sources: List[Dict] = []
        for info in chunk_infos:
            if len(sources) > extra_sources:
                break
            hit = info["hit"]
            if hit.chunk_id in {s["chunk_id"] for s in sources}:
                continue
            if any(unit["text"] in answer_text for unit in info["units"]):
                best_unit = sorted(info["units"], key=lambda u: -u["score"])[0]
                sources.append(to_item(best_unit, hit))
        if not sources:
            sources = [best_item]

        candidates = [
            to_item(unit, info["hit"])
            for info in chunk_infos[:3]
            for unit in sorted(info["units"], key=lambda u: -u["score"])[:2]
        ]
        candidates.sort(key=lambda c: -c["score"])
        return {
            "text": answer_text,
            "hit": best_item,
            "candidates": candidates[:5],
            "sources": sources,
            "source_rank": chosen["hit"].rank,
            "confidence": final_confidence(answer_text, confidence),
        }

    def panorama_answer(self, question: str, hits: Sequence[Hit], max_chars: int = 0) -> Dict:
        """全景模式：每家公司给一条最合适的答案（默认不截断），最后汇总成一份对比答案。"""
        by_company: Dict[str, Dict] = {}
        for hit in hits:
            pick = self.pick_best_answer(question, [hit], max_chars=max_chars)
            item = pick["hit"]
            if item is None:
                continue
            if _BOILERPLATE_RE.search(item["text"]):
                # 套话行不能当某家公司的结论，换用该公司的下一个候选
                alt = next((c for c in pick.get("candidates", []) if not _BOILERPLATE_RE.search(c["text"])), None)
                if alt is None:
                    continue
                item = alt
            item["confidence"] = pick.get("confidence", "high")
            prev = by_company.get(hit.company)
            if prev is None or item["score"] > prev["score"]:
                by_company[hit.company] = item
        lines = [
            f"{company}：{item['text']}（{item['citation']}）"
            for company, item in by_company.items()
        ]
        return {"text": "\n".join(lines), "hit": None, "candidates": list(by_company.values())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="命令行试检索")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    args = parser.parse_args(argv)

    retriever = Retriever()
    hits = retriever.retrieve(args.question, top_k=args.top_k, mode=args.mode)
    for hit in hits:
        print(f"[{hit.rank}] {hit.citation()} | {hit.block_type} | 向量 {hit.vector_score:.3f} | BM25 {hit.bm25_score:.2f}")
        print("    " + re.sub(r"\s+", " ", hit.text))  # 完整块内容，不截断
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
