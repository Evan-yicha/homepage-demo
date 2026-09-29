# -*- coding: utf-8 -*-
"""
作业三 · 批量评测（10 道题：召回明细 + 对错判定 + 错误分析）

流程
----
1. 读 作业三/评测/问题集.json（每题含问题、期望公司、关键词、gold_terms 参考要点）；
2. 对每题用混合检索（可同时跑 hybrid/vector/bm25 便于对比）取 Top-K，记录召回的块：
   块号 / 公司 / 报告期间 / 章节 / 页码 / 三种分数 / 排名 / 片段；
3. 自动判定：
   - 期望公司是否进入 Top-K、最高排名第几；
   - 关键词覆盖率；
   - gold_terms（关键数字/结论）是否出现在召回的块里、最早出现在第几名（决定“答对/错在哪”）；
4. 输出 评测/检索记录.jsonl（含问答页面同款 payload）、评测/自动评分草稿.md、
   评测/十道题评测记录.xlsx（逐题一行，含“是否答对”和“错在哪”）。 

用法
----
    python 批量评测.py                     # 默认 hybrid，Top-8
    python 批量评测.py --modes hybrid vector bm25 --top-k 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from 问答检索 import Retriever  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

ASSIGN_DIR = CODE_DIR.parent
EVAL_DIR = ASSIGN_DIR / "评测"
QUESTIONS_PATH = EVAL_DIR / "问题集.json"
RECORD_PATH = EVAL_DIR / "检索记录.jsonl"
DRAFT_MD = EVAL_DIR / "自动评分草稿.md"
XLSX_PATH = EVAL_DIR / "十道题评测记录.xlsx"
REPORT_MD = EVAL_DIR / "十道题评测记录.md"


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize(text: str) -> str:
    return re.sub(r"[\s,，]", "", text or "")


def term_positions(hits, terms: Sequence[str]) -> Dict[str, int]:
    """每个参考要点最早出现在第几名的召回块里（0 表示没出现）。"""
    out: Dict[str, int] = {}
    for term in terms:
        pos = 0
        needle = normalize(term)
        for hit in hits:
            if needle and needle in normalize(hit.text):
                pos = hit.rank
                break
        out[term] = pos
    return out


def evaluate(question: Dict, hits, top_k: int) -> Dict:
    companies = question.get("companies") or []
    best_rank = 0
    for hit in hits:
        if not companies or hit.company in companies:
            best_rank = hit.rank
            break
    keywords = question.get("keywords") or []
    kw_hit = [k for k in keywords if any(normalize(k) in normalize(h.text) for h in hits)]
    kws = term_positions(hits, keywords)
    gold = question.get("gold_terms") or []
    gold_pos = term_positions(hits, gold)
    return {
        "expected_company_best_rank": best_rank,
        "keyword_hits": kw_hit,
        "keyword_coverage": round(len(kw_hit) / len(keywords), 3) if keywords else 0.0,
        "keyword_positions": kws,
        "gold_positions": gold_pos,
        "gold_all_found": bool(gold) and all(v > 0 for v in gold_pos.values()),
        "gold_best_rank": min([v for v in gold_pos.values() if v > 0], default=0),
    }


def hit_payload(hit, snippet: int = 0) -> Dict:
    """召回块记录：snippet 存**完整块内容**（不截断），另存 snippet_short 供表格列使用。"""
    data = hit.to_dict()
    full = re.sub(r"\s+", " ", hit.text)
    data["snippet"] = full
    data["snippet_short"] = full[:260]
    return data


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="10 道题的批量评测")
    parser.add_argument("--modes", nargs="*", default=["hybrid"], help="参与评测的检索模式")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--snippet", type=int, default=0, help=">0 时对记录里的块内容截断（默认 0=完整）")
    parser.add_argument("--judgements", default=str(EVAL_DIR / "人工判定.json"),
                        help="人工判定表（是否答对 / 错在哪 / 最佳模式），存在则写入 Excel 与草稿")
    args = parser.parse_args(argv)

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions_by_id = {q["id"]: q for q in questions}
    judgements: Dict[str, Dict] = {}
    jpath = Path(args.judgements)
    if jpath.exists():
        judgements = {item["id"]: item for item in json.loads(jpath.read_text(encoding="utf-8"))}
        log(f"人工判定表：{len(judgements)} 条（{jpath.name}）")
    log(f"题目数：{len(questions)}；模式：{args.modes}；Top-K：{args.top_k}")
    retriever = Retriever()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    with RECORD_PATH.open("w", encoding="utf-8") as fh:
        for question in questions:
            entry_modes: Dict[str, Dict] = {}
            for mode in args.modes:
                started = time.time()
                if mode == "panorama":
                    hits = retriever.retrieve_panorama(
                        question["question"],
                        companies=question.get("companies") or None,
                        top_k_each=2,
                    )
                else:
                    hits = retriever.retrieve(question["question"], top_k=args.top_k, mode=mode)
                elapsed = round(time.time() - started, 2)
                metrics = evaluate(question, hits, args.top_k)
                entry_modes[mode] = {
                    "mode": mode,
                    "elapsed_seconds": elapsed,
                    "metrics": metrics,
                    "hits": [hit_payload(h, args.snippet) for h in hits],
                }
                log(
                    f"  {question['id']} [{mode}] {elapsed}s · 期望公司最高排名 {metrics['expected_company_best_rank']} "
                    f"· 关键词覆盖 {metrics['keyword_coverage']:.0%} · gold 最早第 {metrics['gold_best_rank']} 名"
                )
            primary = entry_modes[args.modes[0]]
            if args.modes[0] == "panorama":
                answer_hits = retriever.retrieve_panorama(
                    question["question"], companies=question.get("companies") or None, top_k_each=1
                )[:5]
                picked = retriever.panorama_answer(question["question"], answer_hits)
                single_answer = picked["text"]
                single_source_rank, single_confidence = 0, "panorama"
            else:
                answer_hits = retriever.retrieve(question["question"], top_k=5, mode=args.modes[0])
                picked = retriever.pick_best_answer(question["question"], answer_hits)
                single_answer = picked["text"]
                single_source_rank = int(picked.get("source_rank") or 0)
                single_confidence = picked.get("confidence", "high")
            gold = question.get("gold_terms") or []
            rows.append(
                {
                    "id": question["id"],
                    "type": question["type"],
                    "question": question["question"],
                    "companies": question.get("companies", []),
                    "reference": question.get("reference", ""),
                    "modes": entry_modes,
                    "primary": args.modes[0],
                    "primary_metrics": primary["metrics"],
                    "primary_hits": primary["hits"],
                    "answer_extractive": retriever.extractive_answer(question["question"], answer_hits),
                    "answer_single": single_answer,
                    "answer_source_rank": single_source_rank,
                    "answer_confidence": single_confidence,
                    "answer_from_top1": single_source_rank == 1,
                    "single_answer_contains_gold": [t for t in gold if t in single_answer],
                    "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            fh.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")

    # 自动评分草稿（Markdown）
    lines = ["# 10 道题自动评分草稿", "", f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    for row in rows:
        m = row["primary_metrics"]
        lines.append(f"## {row['id']}（{row['type']}）")
        lines.append("")
        lines.append(f"- 问题：{row['question']}")
        lines.append(
            f"- 期望公司最高排名：{m['expected_company_best_rank'] or '未召回'}；关键词覆盖：{m['keyword_coverage']:.0%}"
        )
        lines.append(f"- 参考答案要点命中位置：{m['gold_positions']}")
        judge = judgements.get(row["id"])
        if judge:
            lines.append(
                f"- 判定：{judge.get('是否答对','')}；最佳模式：{judge.get('最佳模式','')}；"
                f"错在哪：{judge.get('错在哪','')}"
            )
        lines.append("- 召回块：")
        for hit in row["primary_hits"]:
            lines.append(
                f"  - [#{hit['rank']}] {hit['company']}《{hit['period']}》{hit['section']} 第{hit['page_start']}-{hit['page_end']}页"
                f"（{hit['block_type']}，向量 {hit['vector_score']}，BM25 {hit['bm25_score']}，融合 {hit['fused_score']}）"
            )
        lines.append("")
    DRAFT_MD.write_text("\n".join(lines), encoding="utf-8")

    # Excel 明细
    try:
        from openpyxl import Workbook
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        ws = wb.active
        ws.title = "十道题评测记录"
        headers = [
            "题号", "题型", "问题", "期望公司", "期望公司最高排名", "关键词覆盖",
            "答案来源排名", "是否来自Top1块", "答案置信度", "参考要点命中位置", "是否答对", "错在哪",
            "Top1 召回块", "Top1 出处", "备注",
        ]
        ws.append(headers)
        for row in rows:
            m = row["primary_metrics"]
            top1 = row["primary_hits"][0] if row["primary_hits"] else {}
            judge = judgements.get(row["id"], {})
            ws.append([
                row["id"],
                row["type"],
                row["question"],
                "、".join(row["companies"]),
                m["expected_company_best_rank"] or "未召回",
                f"{m['keyword_coverage']:.0%}",
                row.get("answer_source_rank") or "—",
                "是" if row.get("answer_from_top1") else "否",
                row.get("answer_confidence", ""),
                json.dumps(m["gold_positions"], ensure_ascii=False),
                judge.get("是否答对", ""),
                judge.get("错在哪", ""),
                top1.get("snippet_short", top1.get("snippet", ""))[:200],
                f"{top1.get('company','')}《{top1.get('period','')}》{top1.get('section','')} 第{top1.get('page_start','')}-{top1.get('page_end','')}页",
                judge.get("最佳模式", ""),
            ])
        for idx, width in enumerate(
            [8, 16, 60, 22, 16, 12, 12, 14, 12, 34, 12, 40, 60, 40, 20], start=1
        ):
            ws.column_dimensions[get_column_letter(idx)].width = width
        wb.save(XLSX_PATH)
        log(f"Excel 明细 -> {XLSX_PATH}")
    except Exception as exc:  # noqa: BLE001
        log(f"Excel 输出失败：{type(exc).__name__}: {exc}")

    log(f"检索记录 -> {RECORD_PATH}")
    log(f"评分草稿 -> {DRAFT_MD}")

    # 正式逐题记录（含召回块清单、判定与错因）
    report: List[str] = [
        "# 作业三 · 10 道题逐题评测记录",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "- 语料：13 家公司 39 份公告全文（2025 年报 / 2025 半年报 / 2026 半年报），切块 24313 个。",
        "- 索引：Qwen3-Embedding-0.6B 向量 + BM25（jieba 分词），RRF(k=60) 融合；"
        "`panorama` = 全景模式（逐公司检索各取 2 块后合并），用于跨公司题。",
        "- 判定口径：以“抽取式答案窗口（Top5）”能否给出参考答案里的关键数值为准；"
        "召回到但排在答案窗口之外记为“部分答对/答错”，并写明名次。",
        "",
        "## 总览",
        "",
        "| 题号 | 题型 | 涉及公司 | 答案来源排名 | 是否来自Top1 | 人工判定 | 最佳模式 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        judge = judgements.get(row["id"], {})
        companies = "、".join(row["companies"][:3]) + (" 等" if len(row["companies"]) > 3 else "")
        report.append(
            f"| {row['id']} | {row['type']} | {companies} | "
            f"{row.get('answer_source_rank') or '—'} | "
            f"{'是' if row.get('answer_from_top1') else '否'} | "
            f"{judge.get('是否答对','—')} | {judge.get('最佳模式','—')} |"
        )
    report.append("")
    report.append("## 逐题明细")
    for row in rows:
        judge = judgements.get(row["id"], {})
        report.append("")
        report.append(f"### {row['id']}（{row['type']}）")
        report.append("")
        report.append(f"**问题**：{row['question']}")
        report.append("")
        report.append(f"**参考答案（人工核对原文后确定）**：{row['reference']}")
        report.append("")
        if judge:
            report.append(f"**是否答对**：{judge.get('是否答对','')}")
            report.append("")
            report.append(f"**最佳模式**：{judge.get('最佳模式','')}")
            report.append("")
            report.append(f"**错在哪**：{judge.get('错在哪','')}")
            report.append("")
        report.append("**各模式的参考要点命中情况**：")
        report.append("")
        hit_gold = row.get("single_answer_contains_gold") or []
        report.append(
            f"**单条答案（页面默认只输出这一条）**：{row.get('answer_single','')[:300]}"
        )
        report.append("")
        report.append(
            f"**答案来源**：召回 #{row.get('answer_source_rank') or '—'}"
            f"（{'来自 Top-1 块' if row.get('answer_from_top1') else '非 Top-1'}，"
            f"置信度 {row.get('answer_confidence','—')}）"
        )
        report.append("")
        report.append(
            "**单条答案命中参考要点**："
            + ("、".join(hit_gold) if hit_gold else "（未命中，见下方“错在哪”）")
        )
        report.append("")
        report.append("| 模式 | 期望公司最佳名次 | 关键词覆盖 | Top5 命中的参考要点 | Top8 命中的参考要点 |")
        report.append("| --- | --- | --- | --- | --- |")
        gold = questions_by_id[row["id"]].get("gold_terms") or []
        for mode in args.modes:
            entry = row["modes"].get(mode)
            if not entry:
                continue
            hits = entry["hits"]
            in5 = [t for t in gold if any(t in h["text"] for h in hits[:5])]
            in8 = [t for t in gold if any(t in h["text"] for h in hits)]
            report.append(
                f"| {mode} | {entry['metrics']['expected_company_best_rank'] or '未召回'} | "
                f"{entry['metrics']['keyword_coverage']:.0%} | {'、'.join(in5) or '—'} | {'、'.join(in8) or '—'} |"
            )
        report.append("")
        best_mode = None
        for mode in args.modes:
            if mode in (judge.get("最佳模式") or ""):
                best_mode = mode
                break
        for mode in [row["primary"], *( [best_mode] if best_mode and best_mode != row["primary"] else [] )]:
            entry = row["modes"].get(mode)
            if not entry:
                continue
            report.append(f"**召回的块（{mode}，共 {len(entry['hits'])} 块）**：")
            report.append("")
            for hit in entry["hits"]:
                report.append(
                    f"- [#{hit['rank']}] {hit['company']}《{hit['period']}》{hit['section']} "
                    f"第{hit['page_start']}-{hit['page_end']}页 · {hit['block_type']} · "
                    f"向量 {hit['vector_score']} / BM25 {hit['bm25_score']} / 融合 {hit['fused_score']} · `{hit['chunk_id']}`"
                )
            report.append("")
    report.append("## 结论与改进方向")
    report.append("")
    report.append("1. **单公司事实/叙述题**（Q1、Q7）：混合检索稳定，Top1 即正确章节。")
    report.append("2. **单公司表格数值题**（Q2、Q3、Q4、Q6、Q8）：`vector` 明显优于 `hybrid`——")
    report.append("   表格块文本短、关键词少，BM25 会把“归母净利润/同比”这类叙述块抬到表格块之前；")
    report.append("   已在页面上保留 `仅向量` 选项，数值题建议切到该模式。")
    report.append("3. **跨公司题**（Q9、Q10）：需要 `panorama`（逐公司检索）才能把 13 家的数字都取到；")
    report.append("   单轮全局检索的 Top-K 在结构上不可能覆盖 13 份报告。")
    report.append("4. **表格还原的边界情况**（Q5）：华东医药“营业收入构成”表是无框线、数字紧贴的版式，")
    report.append("   被解析成正文块，导致“医药商业 294.49 亿元”没有独立成表；已记录为已知问题。")
    report.append("5. 口径提醒：华东医药披露的研发投入占比为“医药工业口径”，与其余公司合并口径不可直接比较。")
    report.append("")
    REPORT_MD.write_text("\n".join(report), encoding="utf-8")
    log(f"逐题评测记录 -> {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
