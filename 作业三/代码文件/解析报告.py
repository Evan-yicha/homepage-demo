# -*- coding: utf-8 -*-
"""
作业三 · 解析报告（PDF → 文字 + 按行列还原的表格 → 切块）

要点
----
* 文字：PyMuPDF 逐页取词/行，按版面顺序还原段落，页眉页脚自动剔除。
* 表格：三层策略
      1) PyMuPDF `find_tables()`（带框线的表格，结构最准）；
      2) 自研“词坐标聚类”：把同一行的词按横向间隙切成单元格，再按列位置聚类成表头/单元格，
         专门对付中文财报里大量“无框线但对齐”的表格；
      3) 质量不达标时用 pdfplumber（先线条策略、再文本对齐策略）兜底，并比较两种结果取更优。
* 章节：优先用 PDF 书签目录，缺书签时用“第X节/第X章/一、（一）”等标题正则。
* 切块：文本块约 800 字、重叠 100 字且不跨章节；表格整块保留（过长时按行分组并重复表头）。
  每个块都带 公司 / 代码 / 市场 / 板块 / 报告类型 / 章节 / 起始页 / 结束页 等元数据。

产物
----
    作业三/文本与索引/文本/{公司}/{报告期间}/pages.jsonl   每页结构化内容（含表格单元格矩阵）
    作业三/文本与索引/文本/{公司}/{报告期间}/全文.md       可读全文（含 Markdown 表格）
    作业三/文本与索引/文本/{公司}/{报告期间}/report.json   该份报告的解析统计
    作业三/文本与索引/chunks.jsonl                          全部切块（带出处元数据）

用法
----
    python 解析报告.py                     # 解析全部报告（多进程）
    python 解析报告.py --companies 药明康德
    python 解析报告.py --limit 1 --workers 1 --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pymupdf  # PyMuPDF

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

CODE_DIR = Path(__file__).resolve().parent
ASSIGN_DIR = CODE_DIR.parent
PDF_DIR = ASSIGN_DIR / "创新药业绩报告"
DATA_DIR = ASSIGN_DIR / "文本与索引"
TEXT_DIR = DATA_DIR / "文本"
CHUNKS_DIR = DATA_DIR / "_chunks"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
MANIFEST_PATH = DATA_DIR / "下载清单.csv"

# 切块参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TABLE_CHUNK_MAX = 2200

# 章节标题识别：年报/半年报的标准结构是“第一节 释义 / 第二节 公司简介和主要财务指标 /
# 第三节 管理层讨论与分析 / … / 第十节 财务报告”，因此以“第X节/第X章”为主。
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百零〇]+[节章]\s*\S")
DOTS_RE = re.compile(r"(\.{4,}|…{2,}|·{4,})")
KNOWN_SECTIONS = [
    "公司简介和主要财务指标", "管理层讨论与分析", "公司治理", "环境和社会责任", "重要事项",
    "股份变动及股东情况", "优先股相关情况", "债券相关情况", "财务报告", "审计报告",
    "释义", "重要提示、目录和释义",
]
DEFAULT_SECTION = "重要提示与目录"
SUBSECTIONS = [
    "审计报告", "合并资产负债表", "母公司资产负债表", "合并利润表", "母公司利润表",
    "合并现金流量表", "母公司现金流量表", "合并所有者权益变动表", "母公司所有者权益变动表",
]
CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}


def cn_to_int(text: str) -> Optional[int]:
    """把“八”“十一”“二十”这类中文数字转成整数（够用于章节序号）。"""
    if not text:
        return None
    total, section = 0, 0
    for ch in text:
        if ch == "十":
            section = (section or 1) * 10
        elif ch in CN_DIGITS:
            digit = CN_DIGITS[ch]
            if section >= 10:
                total += section
                section = digit
            else:
                section = digit
        else:
            return None
    value = total + section
    return value if value > 0 else None


def section_index(title: str) -> Optional[int]:
    m = re.match(r"^第([一二三四五六七八九十百零〇]+)[节章]", title.strip())
    return cn_to_int(m.group(1)) if m else None


# --------------------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def join_lines(lines: Sequence[str]) -> str:
    """把换行断开的行拼成段落：中文之间不加空格，英数字之间补空格。"""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        if (out[-1].isascii() and out[-1].isalnum()) and (line[0].isascii() and line[0].isalnum()):
            out += " " + line
        else:
            out += line
    return out


def normalize_cell(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def table_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    norm = [list(r) + [""] * (ncol - len(r)) for r in rows]

    def esc(cell: str) -> str:
        return normalize_cell(cell).replace("|", "\\|")

    header = [esc(c) for c in norm[0]]
    sep = ["---"] * ncol
    body = [[esc(c) for c in r] for r in norm[1:]]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def markdown_to_plain(md: str) -> str:
    """表格转成“字段：值”风格的紧凑文本，给检索用（保留行列对应关系）。"""
    lines = [ln for ln in md.splitlines() if ln.strip()]
    if not lines:
        return ""
    cells = [["" if c.strip() == "---" else c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    header = [c.strip() for c in cells[0]]
    out: List[str] = []
    for row in cells[1:]:
        row = [c.strip() for c in row]
        if all(c in ("", "---") for c in row):
            continue
        pairs = []
        for idx, value in enumerate(row):
            if not value or value == "---":
                continue
            key = header[idx] if idx < len(header) and header[idx] else f"列{idx + 1}"
            pairs.append(f"{key}：{value}")
        if pairs:
            out.append("；".join(pairs))
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# 行 / 单元格 / 表格重建
# --------------------------------------------------------------------------------------


@dataclass
class Line:
    page: int
    y0: float
    y1: float
    x0: float
    x1: float
    words: List[Tuple[float, float, float, float, str]]
    block: int

    @property
    def text(self) -> str:
        return "".join(w[4] for w in self.words)

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


def page_lines(page: "pymupdf.Page") -> List[Line]:
    """把页面切成行：优先按 PyMuPDF 的 block/line 编号，必要时按 y 坐标聚类。"""
    raw = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
    if not raw:
        return []
    grouped: Dict[Tuple[int, int], List[Tuple[float, float, float, float, str]]] = {}
    for x0, y0, x1, y1, text, block_no, line_no, _wn in raw:
        if not text.strip():
            continue
        grouped.setdefault((block_no, line_no), []).append((x0, y0, x1, y1, text))

    lines: List[Line] = []
    for (block_no, _line_no), words in grouped.items():
        words.sort(key=lambda w: w[0])
        # 同一 (block,line) 里横向间隙很大的片段（表格多列）按间隙切开
        segments: List[List[Tuple[float, float, float, float, str]]] = []
        current: List[Tuple[float, float, float, float, str]] = []
        prev_x1 = None
        for w in words:
            if prev_x1 is not None and w[0] - prev_x1 > 12:  # 12pt 以上视为不同单元格
                segments.append(current)
                current = []
            current.append(w)
            prev_x1 = w[2]
        if current:
            segments.append(current)
        for seg in segments:
            if not seg:
                continue
            y0 = min(w[1] for w in seg)
            y1 = max(w[3] for w in seg)
            lines.append(
                Line(
                    page=page.number + 1,
                    y0=y0,
                    y1=y1,
                    x0=min(w[0] for w in seg),
                    x1=max(w[2] for w in seg),
                    words=seg,
                    block=block_no,
                )
            )
    lines.sort(key=lambda ln: (round(ln.yc, 1), ln.x0))
    return merge_same_row(lines)


def merge_same_row(lines: List[Line], tol: float = 2.0) -> List[Line]:
    """把同一视觉行里被拆成多段的 Line 合并回一行（保留各段内部间隙信息）。"""
    merged: List[Line] = []
    for ln in lines:
        if merged and abs(merged[-1].yc - ln.yc) <= tol:
            prev = merged[-1]
            prev.words.extend(ln.words)
            prev.words.sort(key=lambda w: w[0])
            prev.y0 = min(prev.y0, ln.y0)
            prev.y1 = max(prev.y1, ln.y1)
            prev.x0 = min(prev.x0, ln.x0)
            prev.x1 = max(prev.x1, ln.x1)
        else:
            merged.append(ln)
    return merged


def split_cells(line: Line, gap_tol: float = 6.0) -> List[Tuple[float, float, str]]:
    """按横向间隙把一行切成单元格，返回 (x0, x1, text)。"""
    cells: List[Tuple[float, float, str]] = []
    cur: List[Tuple[float, float, float, float, str]] = []
    prev_x1 = None
    for w in line.words:
        if prev_x1 is not None and w[0] - prev_x1 > gap_tol:
            cells.append((min(x[0] for x in cur), max(x[2] for x in cur), "".join(x[4] for x in cur)))
            cur = []
        cur.append(w)
        prev_x1 = w[2]
    if cur:
        cells.append((min(x[0] for x in cur), max(x[2] for x in cur), "".join(x[4] for x in cur)))
    return [(x0, x1, normalize_cell(t)) for x0, x1, t in cells if t.strip()]


def columns_from_cells(rows_cells: Sequence[Sequence[Tuple[float, float, str]]]) -> List[List[float]]:
    """用“区间重叠”把单元格聚成列。

    中文财报里同一列的数字常按右边界对齐、文字按左边界对齐，只按左边界聚类会把一列拆成两列，
    所以这里按 x 区间重叠比例归并：重叠超过较短区间 35% 就视为同一列。
    """
    columns: List[List[float]] = []
    for cells in rows_cells:
        for x0, x1, _text in cells:
            width = max(1e-6, x1 - x0)
            best_idx, best_ov = None, 0.0
            for idx, col in enumerate(columns):
                col_w = max(1e-6, col[1] - col[0])
                ov = min(col[1], x1) - max(col[0], x0)
                ratio = ov / min(width, col_w)
                if ratio > best_ov:
                    best_idx, best_ov = idx, ratio
            if best_idx is not None and best_ov > 0.35:
                col = columns[best_idx]
                col[0] = min(col[0], x0)
                col[1] = max(col[1], x1)
            else:
                columns.append([x0, x1])
    columns.sort(key=lambda c: c[0])
    # 合并相互覆盖的列
    merged: List[List[float]] = []
    for col in columns:
        if merged:
            prev = merged[-1]
            ov = min(prev[1], col[1]) - max(prev[0], col[0])
            if ov > 0 and ov / max(1e-6, min(prev[1] - prev[0], col[1] - col[0])) > 0.5:
                prev[1] = max(prev[1], col[1])
                continue
        merged.append(col)
    return merged


def grid_from_lines(lines: Sequence[Line]) -> Tuple[List[List[str]], float]:
    """把一组行重建为行列矩阵，返回 (矩阵, 一致度)。一致度 = 行单元格数与众数一致的比例。"""
    rows_cells = [split_cells(ln) for ln in lines]
    rows_cells = [rc for rc in rows_cells if rc]
    if not rows_cells:
        return [], 0.0
    columns = columns_from_cells(rows_cells)
    ncol = max(1, len(columns))
    grid: List[List[str]] = []
    for cells in rows_cells:
        row = [""] * ncol
        for x0, x1, text in cells:
            best_idx, best_ov = 0, -1.0
            for idx, col in enumerate(columns):
                ov = min(col[1], x1) - max(col[0], x0)
                if ov > best_ov:
                    best_idx, best_ov = idx, ov
            if row[best_idx]:
                row[best_idx] += " " + text
            else:
                row[best_idx] = text
        grid.append(row)
    counts = Counter(len(rc) for rc in rows_cells)
    mode_count, mode_freq = counts.most_common(1)[0]
    consistency = mode_freq / len(rows_cells)
    return grid, consistency


def valid_table(grid: Sequence[Sequence[str]]) -> bool:
    """表格有效性：至少 2 行 2 列、非空单元格够多、且不是“边框包着一段文字”。"""
    if not grid or len(grid) < 2:
        return False
    ncol = max(len(r) for r in grid)
    if ncol < 2:
        return False
    cells = [c.strip() for r in grid for c in r if c and c.strip()]
    if len(cells) < 4:
        return False
    fill = len(cells) / (len(grid) * ncol)
    if fill < 0.3:
        return False
    avg_len = sum(len(c) for c in cells) / len(cells)
    if ncol <= 2 and avg_len > 45:
        return False
    return True


def grid_score(grid: Sequence[Sequence[str]]) -> float:
    if not grid:
        return 0.0
    nrow, ncol = len(grid), max(len(r) for r in grid)
    if ncol < 2:
        return 0.0
    filled = sum(1 for r in grid for c in r if c and c.strip())
    fill = filled / (nrow * ncol)
    return nrow * (ncol - 1) * (0.3 + fill)


def detect_table_regions(lines: Sequence[Line], verbose: bool = False) -> List[Tuple[int, int]]:
    """找“像表格”的连续行区间：行内至少 3 个单元格，且至少出现 3 行。"""
    flags = []
    for ln in lines:
        cells = split_cells(ln)
        flags.append(len(cells) >= 3)
    regions: List[Tuple[int, int]] = []
    start = None
    gap = 0
    for idx, flag in enumerate(flags):
        if flag:
            if start is None:
                start = idx
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 2:  # 连续超过 2 行不像表格 -> 结束
                end = idx - gap
                if end - start + 1 >= 3:
                    regions.append((start, end))
                start = None
                gap = 0
    if start is not None:
        end = len(flags) - 1
        if end - start + 1 >= 3:
            regions.append((start, end))
    return regions


def pdfplumber_tables(pdf_path: Path, page_no: int) -> List[List[List[str]]]:
    """pdfplumber 兜底：先线条策略，再文本对齐策略。"""
    try:
        import pdfplumber
    except Exception:  # noqa: BLE001
        return []
    tables: List[List[List[str]]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_no - 1]
            for settings in (
                {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                {"vertical_strategy": "text", "horizontal_strategy": "text"},
            ):
                found = page.extract_tables(settings)
                if not found:
                    continue
                for tbl in found:
                    grid = [[normalize_cell(c or "") for c in row] for row in tbl if row]
                    grid = [row for row in grid if any(c for c in row)]
                    if len(grid) >= 2 and max(len(r) for r in grid) >= 2:
                        tables.append(grid)
                if tables:
                    break
    except Exception:  # noqa: BLE001
        return tables
    return tables


def pymupdf_tables(page: "pymupdf.Page") -> List[Tuple[List[List[str]], Tuple[float, float, float, float]]]:
    """带框线表格：返回 (矩阵, bbox)。bbox 用来把表格映射回具体行区间。"""
    try:
        finder = page.find_tables()
    except Exception:  # noqa: BLE001
        return []
    out: List[Tuple[List[List[str]], Tuple[float, float, float, float]]] = []
    for tbl in getattr(finder, "tables", []) or []:
        try:
            data = tbl.extract()
            bbox = tuple(float(v) for v in tbl.bbox)
        except Exception:  # noqa: BLE001
            continue
        grid = [[normalize_cell(c or "") for c in row] for row in data if row]
        grid = [row for row in grid if any(c for c in row)]
        if valid_table(grid):
            out.append((grid, bbox))  # type: ignore[arg-type]
    return out


# --------------------------------------------------------------------------------------
# 页面 → 结构化内容
# --------------------------------------------------------------------------------------


def page_blocks(
    page: "pymupdf.Page",
    pdf_path: Path,
    allow_pdfplumber: bool = True,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    """返回 (块列表, 统计信息)。块类型：text / table。"""
    lines = page_lines(page)
    stats = {"lines": len(lines), "tables": 0, "methods": Counter(), "empty": not lines}
    if not lines:
        return [], stats

    # (start, end, grid, score, method)
    region_grids: List[Tuple[int, int, List[List[str]], float, str]] = []

    # 1) 带框线表格：用 bbox 映射到行区间
    for grid, bbox in pymupdf_tables(page):
        idxs = [i for i, ln in enumerate(lines) if bbox[1] - 3 <= ln.yc <= bbox[3] + 3]
        if not idxs:
            continue
        region_grids.append((min(idxs), max(idxs), grid, grid_score(grid), "pymupdf_table"))

    # 2) 无框线但多列对齐的表格：自研词坐标聚类，必要时用 pdfplumber 兜底
    for start, end in detect_table_regions(lines):
        if any(not (end < s or start > e) for s, e, _g, _sc, _m in region_grids):
            continue
        grid, consistency = grid_from_lines(lines[start : end + 1])
        method = "words_grid"
        best, best_score = grid, grid_score(grid)
        if allow_pdfplumber and (consistency < 0.6 or best_score < 8):
            for g2 in pdfplumber_tables(pdf_path, page.number + 1):
                s2 = grid_score(g2)
                if s2 > best_score:
                    best, best_score, method = g2, s2, "pdfplumber"
        region_grids.append((start, end, best, best_score, method))

    # 3) 有效性过滤 + 去重（同一段行只留分数最高的一个表）
    region_grids = [r for r in region_grids if valid_table(r[2])]
    region_grids.sort(key=lambda r: (r[0], -r[3]))
    kept: List[Tuple[int, int, List[List[str]], float, str]] = []
    for region in region_grids:
        if any(not (region[1] < k[0] or region[0] > k[1]) for k in kept):
            continue
        kept.append(region)
    region_grids = kept

    blocks: List[Dict] = []
    table_by_first_line = {start: (grid, method) for start, _e, grid, _s, method in region_grids}
    idx = 0
    pending_text: List[Line] = []

    def flush_text() -> None:
        if not pending_text:
            return
        paragraphs: List[str] = []
        current: List[str] = []
        prev: Optional[Line] = None
        heights = [ln.y1 - ln.y0 for ln in pending_text] or [10.0]
        typical = sorted(heights)[len(heights) // 2]
        for ln in pending_text:
            if prev is not None and (ln.y0 - prev.y1) > typical * 0.9:
                if current:
                    paragraphs.append(join_lines(current))
                    current = []
            current.append(ln.text)
            prev = ln
        if current:
            paragraphs.append(join_lines(current))
        for para in paragraphs:
            para = para.strip()
            if para:
                blocks.append({"type": "text", "text": para, "y0": pending_text[0].y0})
        pending_text.clear()

    while idx < len(lines):
        if idx in table_by_first_line:
            flush_text()
            grid, method = table_by_first_line[idx]
            end = next(e for s, e, _g, _sc, _m in region_grids if s == idx)
            blocks.append(
                {
                    "type": "table",
                    "rows": grid,
                    "markdown": table_to_markdown(grid),
                    "method": method,
                    "y0": lines[idx].y0,
                    "y1": lines[end].y1,
                }
            )
            stats["tables"] += 1
            stats["methods"][method] += 1
            idx = end + 1
            continue
        pending_text.append(lines[idx])
        idx += 1
    flush_text()
    blocks.sort(key=lambda b: b.get("y0", 0))
    return blocks, stats


# --------------------------------------------------------------------------------------
# 章节识别
# --------------------------------------------------------------------------------------


def build_section_map(doc: "pymupdf.Document") -> Dict[int, str]:
    """用书签目录生成 页码 → 章节 映射。

    注意：沪深定期报告的书签常常是“逐段落”的（一行一条），直接使用会把章节切得很碎，
    所以只采纳标题形如“第X节/第X章”或标准章节名的书签。
    """
    try:
        toc = doc.get_toc(simple=True)
    except Exception:  # noqa: BLE001
        toc = []
    section_by_page: Dict[int, str] = {}
    if not toc:
        return section_by_page
    usable = [
        (lvl, title.strip(), page)
        for lvl, title, page in toc
        if title.strip() and lvl <= 2 and (SECTION_RE.match(title.strip()) or title.strip() in KNOWN_SECTIONS)
    ]
    if not usable:
        return section_by_page
    for lvl, title, page in usable:
        for p in range(max(1, page), doc.page_count + 1):
            prev = section_by_page.get(p)
            if prev is None:
                section_by_page[p] = title
            else:
                break
    return section_by_page


def detect_heading(text: str) -> Optional[str]:
    line = text.strip()
    if not line or len(line) > 32:
        return None
    if DOTS_RE.search(line):  # 目录页的“第一节 释义……5”
        return None
    if SECTION_RE.match(line):
        return line
    if line in KNOWN_SECTIONS:
        return line
    return None


def assign_sections(pages: List[Dict], section_map: Dict[int, str]) -> None:
    """按页推进章节状态。

    规则：只有“第X节/第X章”才能切换主章节，且序号只能向前推进（避免财务附注里出现的
    “第一节 释义”等字样把后面的章节全带偏）；财务报告下的报表/审计报告只作为子标题。
    """
    main = DEFAULT_SECTION
    main_idx = 0
    sub = ""

    def apply(title: str) -> None:
        nonlocal main, main_idx, sub
        idx = section_index(title)
        if idx is not None:
            if idx > main_idx:
                main, main_idx, sub = title, idx, ""
        elif title in SUBSECTIONS and main_idx >= 8:  # 第八节 财务报告 下的子标题
            sub = title

    for page in pages:
        mapped = section_map.get(page["page"])
        if mapped:
            apply(mapped)
        for block in page["blocks"]:
            if block["type"] == "text":
                heading = detect_heading(block["text"].split("\n")[0])
                if heading:
                    apply(heading)
            block["section"] = f"{main} · {sub}" if sub else main
        page["section"] = f"{main} · {sub}" if sub else main


# --------------------------------------------------------------------------------------
# 切块
# --------------------------------------------------------------------------------------


def chunk_report(meta: Dict, pages: List[Dict]) -> List[Dict]:
    chunks: List[Dict] = []
    seq = 0

    def add(block_type: str, text: str, section: str, p0: int, p1: int) -> None:
        nonlocal seq
        seq += 1
        # index_text：只在建索引/检索时使用的文本，前置“公司 + 报告期间 + 章节 + 页码”。
        # 财报里的表格块本身通常不带公司名，直接嵌入会导致“问A公司却召回B公司的表”，
        # 加上这行元数据后表格与问题里的公司名能对上。
        pages_label = f"第{p0}页" if p0 == p1 else f"第{p0}-{p1}页"
        index_text = f"【{meta['company']}】{meta['period']} · {section} · {pages_label}\n{text}"
        chunks.append(
            {
                "chunk_id": f"{meta['company']}_{meta['code']}_{meta['period']}#P{p0}-{p1}#{seq:04d}",
                "company": meta["company"],
                "code": meta["code"],
                "market": meta["market"],
                "board": meta["board"],
                "report_type": meta["report_type"],
                "period": meta["period"],
                "title": meta["title"],
                "publish_date": meta["date"],
                "announce_url": meta["url"],
                "source_pdf": meta["local_file"],
                "section": section,
                "page_start": p0,
                "page_end": p1,
                "block_type": block_type,
                "text": text,
                "index_text": index_text,
                "char_len": len(text),
            }
        )

    buffer: List[str] = []
    buffer_len = 0
    buf_section = ""
    buf_pages: List[int] = []

    def flush() -> None:
        nonlocal buffer, buffer_len, buf_section, buf_pages
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            add("text", text, buf_section, min(buf_pages), max(buf_pages))
        # 重叠：保留末尾 100 字，跨块不丢上下文
        tail = text[-CHUNK_OVERLAP:] if len(text) > CHUNK_OVERLAP else ""
        buffer = [tail] if tail else []
        buffer_len = len(tail)
        buf_pages = buf_pages[-1:]

    for page in pages:
        for block in page["blocks"]:
            section = block.get("section", page.get("section", ""))
            pno = page["page"]
            if block["type"] == "table":
                flush()
                md = block.get("markdown") or table_to_markdown(block.get("rows", []))
                if not md:
                    continue
                if len(md) <= TABLE_CHUNK_MAX:
                    add("table", md, section, pno, pno)
                else:
                    # 长表按行分组，每组重复表头
                    lines = md.splitlines()
                    header = lines[:2]
                    body = lines[2:]
                    group: List[str] = []
                    size = 0
                    for row in body:
                        if size + len(row) > TABLE_CHUNK_MAX and group:
                            add("table", "\n".join(header + group), section, pno, pno)
                            group, size = [], 0
                        group.append(row)
                        size += len(row)
                    if group:
                        add("table", "\n".join(header + group), section, pno, pno)
                continue
            text = block["text"].strip()
            if not text:
                continue
            if section != buf_section and buffer:
                flush()
            buf_section = section
            buf_pages.append(pno)
            buffer.append(text)
            buffer_len += len(text)
            if buffer_len >= CHUNK_SIZE:
                flush()
    flush()
    return chunks


# --------------------------------------------------------------------------------------
# 单份报告解析
# --------------------------------------------------------------------------------------


def load_manifest() -> List[Dict]:
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("状态") == "成功"]


def parse_one(record: Dict, allow_pdfplumber: bool = True, verbose: bool = False) -> Dict:
    pdf_path = ASSIGN_DIR / record["本地文件"]
    period = record["报告期间"]
    # 注意顺序：先判断“半年度报告”，否则会被“年度报告”匹配掉
    report_type = "半年度报告" if "半年度报告" in period else "年度报告"
    meta = {
        "company": record["公司"],
        "code": record["代码"],
        "market": record["市场"],
        "board": record["板块"],
        "period": period,
        "report_type": report_type,
        "title": record["公告标题"],
        "date": record["发布日期"],
        "url": record["公告链接"],
        "local_file": record["本地文件"],
    }
    out_dir = TEXT_DIR / meta["company"] / period
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    doc = pymupdf.open(pdf_path)
    section_map = build_section_map(doc)
    pages: List[Dict] = []
    stat = Counter()
    empty_pages: List[int] = []
    for page in doc:
        blocks, pstat = page_blocks(page, pdf_path, allow_pdfplumber=allow_pdfplumber, verbose=verbose)
        if pstat.get("empty"):
            empty_pages.append(page.number + 1)
        stat["tables"] += pstat.get("tables", 0)
        stat.update(pstat.get("methods", {}))
        pages.append({"page": page.number + 1, "blocks": blocks})
    doc.close()
    assign_sections(pages, section_map)

    # pages.jsonl
    pages_path = out_dir / "pages.jsonl"
    with pages_path.open("w", encoding="utf-8") as fh:
        for page in pages:
            fh.write(json.dumps(page, ensure_ascii=False) + "\n")

    # 全文.md
    md_lines: List[str] = [f"# {meta['company']}（{meta['code']}）{period}", ""]
    last_section = None
    for page in pages:
        if page["section"] != last_section:
            md_lines.append(f"\n## {page['section']}\n")
            last_section = page["section"]
        for block in page["blocks"]:
            if block["type"] == "table":
                md_lines.append(f"\n<!-- 第 {page['page']} 页 · 表格 · {block.get('method', '')} -->")
                md_lines.append(block.get("markdown", ""))
            else:
                md_lines.append(f"\n<!-- 第 {page['page']} 页 -->")
                md_lines.append(block["text"])
    (out_dir / "全文.md").write_text("\n".join(md_lines), encoding="utf-8")

    chunks = chunk_report(meta, pages)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunk_file = CHUNKS_DIR / f"{meta['company']}_{meta['code']}_{period}.jsonl"
    with chunk_file.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    summary = {
        "company": meta["company"],
        "code": meta["code"],
        "period": period,
        "pages": len(pages),
        "tables": stat["tables"],
        "table_methods": {k: v for k, v in stat.items() if k not in ("tables",)},
        "empty_pages": empty_pages,
        "chunks": len(chunks),
        "chunk_types": dict(Counter(c["block_type"] for c in chunks)),
        "parse_seconds": round(time.time() - started, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        log(f"  {meta['company']} {period}: {summary['pages']} 页 / {summary['tables']} 表 / {summary['chunks']} 块 / {summary['parse_seconds']}s")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="解析定期报告 PDF：文字 + 表格行列还原 + 切块")
    parser.add_argument("--companies", help="只解析这些公司（逗号分隔）")
    parser.add_argument("--limit", type=int, help="只解析前 N 份报告（调试用）")
    parser.add_argument("--workers", type=int, default=min(10, (os.cpu_count() or 4)))
    parser.add_argument("--no-pdfplumber", action="store_true", help="禁用 pdfplumber 兜底")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_manifest()
    if args.companies:
        wanted = {s.strip() for s in args.companies.split(",")}
        records = [r for r in records if r["公司"] in wanted]
    if args.limit:
        records = records[: args.limit]
    log(f"待解析报告：{len(records)} 份")

    results: List[Dict] = []
    if args.workers <= 1 or len(records) <= 1:
        for record in records:
            results.append(parse_one(record, allow_pdfplumber=not args.no_pdfplumber, verbose=True))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(parse_one, record, not args.no_pdfplumber, args.verbose): record
                for record in records
            }
            for fut in as_completed(futures):
                record = futures[fut]
                try:
                    summary = fut.result()
                    results.append(summary)
                    log(
                        f"完成 {summary['company']} {summary['period']}：{summary['pages']} 页 / "
                        f"{summary['tables']} 表 / {summary['chunks']} 块 / {summary['parse_seconds']}s"
                    )
                except Exception as exc:  # noqa: BLE001
                    log(f"失败 {record['公司']} {record['报告期间']}：{type(exc).__name__}: {exc}")

    # 合并所有切块（顺序固定，便于复现）
    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    total = 0
    with CHUNKS_PATH.open("w", encoding="utf-8") as out:
        for path in chunk_files:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    total += 1
    log(f"切块合计：{total} 个 -> {CHUNKS_PATH}")

    summary_path = DATA_DIR / "解析汇总.json"
    summary_path.write_text(
        json.dumps(
            {
                "reports": len(results),
                "pages": sum(r["pages"] for r in results),
                "tables": sum(r["tables"] for r in results),
                "chunks": total,
                "details": sorted(results, key=lambda r: (r["company"], r["period"])),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"解析汇总 -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
