# -*- coding: utf-8 -*-
"""
作业三 · 问答页面（能提问、答案带出处）

启动：
    E:\\Anaconda\\python.exe 作业三\\代码文件\\问答页面.py --open
    浏览器访问 http://127.0.0.1:8000

特性：
* 检索：Qwen/Qwen3-Embedding-0.6B 向量 + BM25，RRF 融合，可切换 hybrid / vector / bm25；
* 答案：默认抽取式——把 Top 检索块整理成编号要点，逐条标注 [#编号]；出处固定给
  “公司 / 报告类型与期间 / 章节 / 页码 / 原文片段 / 本地 PDF / 公告原始链接”；
* 生成式（可选）：设置环境变量 DEEPSEEK_API_KEY 后可用生成式作答（只依据检索片段并标注引用），
  不设置则完全离线抽取式；
* 留档：每次提问可一键写入 作业三/评测/检索记录.jsonl，用于评测与复现。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from flask import Flask, Response, jsonify, request, send_file  # noqa: E402

from 问答检索 import Hit, Retriever  # noqa: E402
from 向量模型 import describe_environment, load_log  # noqa: E402

ASSIGN_DIR = CODE_DIR.parent
PDF_DIR = ASSIGN_DIR / "创新药业绩报告"
EVAL_DIR = ASSIGN_DIR / "评测"
RECORD_PATH = EVAL_DIR / "检索记录.jsonl"
LOG_PATH = EVAL_DIR / "问答页面日志.txt"

app = Flask(__name__)
_retriever: Optional[Retriever] = None
_lock = threading.Lock()
_state: Dict = {"ready": False, "loading": False, "error": "", "started_at": time.time()}
_server = None  # werkzeug 服务器句柄，供 /api/shutdown 优雅关闭


def get_retriever() -> Retriever:
    global _retriever
    with _lock:
        if _retriever is None:
            _retriever = Retriever()
        return _retriever


def preload_retriever() -> None:
    """后台预加载索引与向量模型，让 HTTP 服务立刻可用（页面显示“加载中”而不是像卡死）。"""
    if _state["ready"] or _state["loading"]:
        return
    _state["loading"] = True
    try:
        retriever = get_retriever()
        _state["ready"] = True
        _state["error"] = ""
        _state["backend"] = getattr(retriever.embedder, "name", "纯 BM25（向量不可用）")
        _state["load_log"] = load_log()
        print(
            f"[就绪] 切块 {len(retriever.chunks)} 个 · 向量 "
            f"{_state['backend']} · BM25 "
            f"{'已加载' if retriever.bm25 is not None else '缺失'}，"
            f"耗时 {time.time() - _state['started_at']:.1f}s",
            flush=True,
        )
        if retriever.embedder is None:
            print("[提示] 当前为纯 BM25 检索；已在后台定时重试加载向量后端（可用 QA_FORCE_BACKEND 排障）", flush=True)
            threading.Thread(target=vector_self_heal, daemon=True).start()
    except Exception as exc:  # noqa: BLE001
        import traceback

        tb = traceback.format_exc()
        root = exc
        while getattr(root, "__cause__", None) or getattr(root, "__context__", None):
            root = getattr(root, "__cause__", None) or getattr(root, "__context__", None)
        _state["error"] = f"{type(exc).__name__}: {exc} | 根本原因：{type(root).__name__}: {root}"
        _state["traceback"] = tb
        _state["load_log"] = load_log()
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 加载失败 =====\n")
            fh.write(_state["error"] + "\n")
            fh.write(tb + "\n")
            fh.write("\n".join(load_log()) + "\n")
        print(f"[错误] 加载检索资源失败：{_state['error']}", flush=True)
    finally:
        _state["loading"] = False


def wait_ready(timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _state["ready"] or _state["error"]:
            return bool(_state["ready"])
        if not _state["loading"]:
            threading.Thread(target=preload_retriever, daemon=True).start()
        time.sleep(0.5)
    return False


def vector_self_heal(max_attempts: int = 5, interval: float = 45.0) -> None:
    """向量后端首次加载失败（例如 torch/transformers 临时不可用）时后台定期重试，
    成功后自动升级为“向量 + BM25”混合检索，无需重启页面。"""
    for attempt in range(1, max_attempts + 1):
        time.sleep(interval)
        retriever = _retriever
        if retriever is None or retriever.embedder is not None:
            return
        from 向量模型 import build_embedder

        print(f"[自愈] 第 {attempt} 次尝试重新加载向量后端…", flush=True)
        embedder = build_embedder()
        if embedder is not None:
            retriever.embedder = embedder
            _state["backend"] = embedder.name
            _state["load_log"] = load_log()
            print(f"[自愈] 向量后端已恢复（{embedder.name}），后续提问使用向量+BM25 混合检索", flush=True)
            return


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>创新药业绩报告问答（带出处）</title>
<style>
body { margin:0; font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; background:#f5f7fa; color:#1f2933; }
header { background:#0b5db1; color:#fff; padding:18px 24px; }
header h1 { margin:0 0 4px; font-size:20px; }
header p { margin:0; font-size:13px; opacity:.9; }
main { max-width:1080px; margin:20px auto; padding:0 16px 60px; }
.card { background:#fff; border-radius:10px; box-shadow:0 1px 3px rgba(16,24,40,.08); padding:16px 18px; margin-bottom:16px; }
textarea { width:100%; min-height:64px; padding:10px; font-size:15px; border:1px solid #d7dce3; border-radius:8px; box-sizing:border-box; }
.row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:10px; }
select, button { font-size:14px; padding:8px 12px; border-radius:8px; border:1px solid #d7dce3; background:#fff; }
button.primary { background:#0b5db1; color:#fff; border-color:#0b5db1; cursor:pointer; }
button.primary:disabled { opacity:.6; cursor:default; }
.answer { white-space:pre-wrap; line-height:1.7; font-size:15px; }
.cite { border-top:1px solid #eef1f5; padding:12px 0; }
.cite h4 { margin:0 0 6px; font-size:14px; color:#0b5db1; }
.meta { font-size:12px; color:#5b6b7c; margin-bottom:6px; }
.snippet { font-size:13px; color:#334; line-height:1.6; background:#f8fafc; border-left:3px solid #cbd5e1; padding:8px 10px; border-radius:4px; white-space:pre-wrap; }
.tag { display:inline-block; font-size:11px; padding:1px 6px; border-radius:10px; background:#e8f1fb; color:#0b5db1; margin-right:6px; }
.hint { font-size:12px; color:#5b6b7c; }
</style>
</head>
<body>
<header>
  <h1>创新药公司业绩报告问答</h1>
  <p>39 份公告全文（2025 年报 / 2025 半年报 / 2026 半年报）· 向量(Qwen3-Embedding-0.6B) + BM25 混合检索 · 答案带出处</p>
</header>
<main>
  <div class="card">
    <textarea id="q" placeholder="例如：恒瑞医药2025年研发投入是多少？它在13家公司里的研发费用率处于什么水平？"></textarea>
    <div class="row">
      <label>检索模式
        <select id="mode">
          <option value="hybrid">混合（向量+BM25）</option>
          <option value="panorama">全景（逐公司检索，跨公司题用）</option>
          <option value="vector">仅向量</option>
          <option value="bm25">仅 BM25</option>
        </select>
      </label>
      <label>召回条数
        <select id="topk"><option>5</option><option selected>8</option><option>12</option></select>
      </label>
      <label>答案条数
        <select id="answercount">
          <option value="1" selected>1（只给最合适的那条）</option>
          <option value="3">3</option>
          <option value="5">5</option>
        </select>
      </label>
      <button class="primary" id="ask">提问</button>
      <button id="export">导出本次检索记录</button>
      <button id="shutdown" title="关闭本页面服务（用于释放端口）">关闭服务</button>
      <span class="hint" id="status">正在检查服务状态…</span>
    </div>
  </div>
  <div class="card" id="answerCard" style="display:none">
    <h3 style="margin-top:0">答案</h3>
    <div class="answer" id="answer"></div>
    <div class="meta" id="answermeta" style="margin-top:10px"></div>
    <details id="moreBox" style="margin-top:12px;display:none">
      <summary style="cursor:pointer;color:#0b5db1">查看其他候选答案（按相关度排序）</summary>
      <div id="candidates" style="margin-top:10px"></div>
    </details>
  </div>
  <div class="card" id="citeCard" style="display:none">
    <h3 style="margin-top:0">出处（召回的块）</h3>
    <div id="cites"></div>
  </div>
</main>
<script>
let lastPayload = null;
let serviceReady = false;
function esc(s) { return (s || '').replace(/</g, '&lt;'); }
function setBusy(busy) {
  document.getElementById('ask').disabled = busy || !serviceReady;
  document.getElementById('export').disabled = busy;
}
async function pollStatus() {
  try {
    const resp = await fetch('/api/status');
    const s = await resp.json();
    serviceReady = !!s.ready;
    if (s.error) {
      document.getElementById('status').textContent = '服务异常：' + s.error;
      setBusy(false);
      return;
    }
    if (serviceReady) {
      document.getElementById('status').textContent =
        '模型已就绪（' + (s.chunks || 0) + ' 个切块 · ' + (s.embedder || '纯 BM25') + '，' + s.elapsed_seconds + ' 秒）';
    } else {
      document.getElementById('status').textContent =
        '正在加载向量模型（离线本地模型，约 10 秒，已等待 ' + s.elapsed_seconds + ' 秒）…';
    }
  } catch (e) {
    document.getElementById('status').textContent = '无法访问服务：' + e;
  }
  setBusy(false);
}
async function ask() {
  const question = document.getElementById('q').value.trim();
  if (!question) { return; }
  const btn = document.getElementById('ask');
  setBusy(true);
  document.getElementById('status').textContent = '检索中…';
  try {
    const resp = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question: question,
        mode: document.getElementById('mode').value,
        top_k: parseInt(document.getElementById('topk').value, 10),
        answer_count: parseInt(document.getElementById('answercount').value, 10)
      })
    });
    const data = await resp.json();
    if (data.pending) {
      document.getElementById('status').textContent = data.message;
      serviceReady = false;
      pollStatus();
      return;
    }
    if (data.error) {
      document.getElementById('status').textContent = data.error;
      return;
    }
    lastPayload = data;
    document.getElementById('answer').textContent = data.answer;
    const meta = data.answer_meta;
    const conf = data.answer_confidence;
    const srcRank = data.answer_source_rank;
    document.getElementById('answermeta').innerHTML = meta
      ? ('答案来源：召回 #' + (srcRank || meta.rank)
         + (conf === 'low' ? '（低置信：未找到明确数值，以下为最相关片段，可展开其他候选）'
            : (conf === 'partial' ? '（部分覆盖：问题问了多个指标，此处只覆盖了其中一部分，建议展开候选或拆开提问）' : ''))
         + ' · ' + esc(meta.company) + '《' + esc(meta.period) + '》' + esc(meta.section)
         + ' · 第 ' + meta.page_start + (meta.page_end !== meta.page_start ? '-' + meta.page_end : '') + ' 页'
         + (meta.source_pdf ? ' · <a href="/pdf/' + encodeURIComponent(meta.source_pdf.split(/[\\\\/]/).pop()) + '" target="_blank">打开该页 PDF</a>' : '')
         + (meta.announce_url ? ' · <a href="' + meta.announce_url + '" target="_blank">公告原文</a>' : ''))
      : (data.mode === 'panorama' ? '全景模式：每家公司给出各自最优块（见下方候选与出处）' : '');
    const cand = document.getElementById('candidates');
    cand.innerHTML = '';
    (data.answer_candidates || []).forEach(function (c, i) {
      const div = document.createElement('div');
      div.className = 'cite';
      div.innerHTML = '<div class="meta">候选 ' + (i + 1) + '（相关度 ' + c.score + '）· ' + esc(c.company) + '《' + esc(c.period) + '》'
        + esc(c.section) + ' · 第 ' + c.page_start + (c.page_end !== c.page_start ? '-' + c.page_end : '') + ' 页</div>'
        + '<div class="snippet">' + esc(c.text) + '</div>';
      cand.appendChild(div);
    });
    document.getElementById('moreBox').style.display = (data.answer_candidates || []).length ? 'block' : 'none';
    if (conf === 'low' || conf === 'partial' || data.mode === 'panorama') {
      document.getElementById('moreBox').open = true;
    }
    const cites = document.getElementById('cites');
    cites.innerHTML = '';
    data.citations.forEach(function (c) {
      const div = document.createElement('div');
      div.className = 'cite';
      const file = c.source_pdf ? c.source_pdf.split(/[\\\\/]/).pop() : '';
      const pdfLink = file ? '<a href="/pdf/' + encodeURIComponent(file) + '" target="_blank">打开本地 PDF</a>' : '';
      const srcLink = c.announce_url ? ' · <a href="' + c.announce_url + '" target="_blank">公告原始链接</a>' : '';
      div.innerHTML = '<h4>[#' + c.rank + '] ' + esc(c.company) + ' · ' + esc(c.period) + '</h4>'
        + '<div class="meta"><span class="tag">' + (c.block_type === 'table' ? '表格' : '正文') + '</span>'
        + esc(c.section) + ' · 第 ' + c.page_start + (c.page_end !== c.page_start ? '-' + c.page_end : '') + ' 页'
        + ' · 向量分 ' + c.vector_score + ' · BM25 ' + c.bm25_score + ' · 融合 ' + c.fused_score + '</div>'
        + '<div class="snippet">' + esc(c.snippet) + '</div>'
        + '<div class="meta" style="margin-top:6px">' + pdfLink + srcLink + ' · ' + c.chunk_id + '</div>';
      cites.appendChild(div);
    });
    document.getElementById('answerCard').style.display = 'block';
    document.getElementById('citeCard').style.display = 'block';
    document.getElementById('status').textContent = '完成，用时 ' + data.elapsed_seconds + ' 秒';
  } catch (e) {
    document.getElementById('status').textContent = '出错：' + e;
  } finally {
    setBusy(false);
  }
}
async function exportRecord() {
  if (!lastPayload) { document.getElementById('status').textContent = '请先提问'; return; }
  const resp = await fetch('/api/export', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(lastPayload)
  });
  const data = await resp.json();
  document.getElementById('status').textContent = '已写入 ' + data.path;
}
document.getElementById('ask').onclick = ask;
document.getElementById('export').onclick = exportRecord;
document.getElementById('shutdown').onclick = async function () {
  if (!confirm('确定关闭问答页面服务吗？')) { return; }
  try { await fetch('/api/shutdown', {method: 'POST'}); } catch (e) {}
  document.getElementById('status').textContent = '服务已关闭，可以关闭此页面。';
  document.getElementById('shutdown').disabled = true;
  document.getElementById('ask').disabled = true;
};
document.getElementById('q').addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { ask(); }
});
pollStatus();
setInterval(pollStatus, 3000);
</script>
</body>
</html>
"""


@app.get("/")
def index() -> Response:
    return Response(PAGE, mimetype="text/html; charset=utf-8")


@app.get("/api/status")
def api_status():
    detail: Dict = {
        "ready": bool(_state["ready"]),
        "loading": bool(_state["loading"]),
        "error": _state["error"],
        "backend": _state.get("backend", ""),
        "elapsed_seconds": round(time.time() - _state["started_at"], 1),
    }
    if _retriever is not None:
        detail["chunks"] = len(_retriever.chunks)
        detail["has_vectors"] = _retriever.vectors is not None
        detail["has_bm25"] = _retriever.bm25 is not None
        detail["embedder"] = getattr(_retriever.embedder, "name", None)
    return jsonify(detail)


@app.get("/api/diagnose")
def api_diagnose():
    """排障接口：返回解释器、依赖版本、向量后端加载过程与最近一次错误堆栈。"""
    info = describe_environment()
    info["state"] = {k: v for k, v in _state.items() if k != "load_log"}
    info["load_log"] = _state.get("load_log") or load_log()
    info["last_traceback"] = _state.get("traceback", "")
    info["log_file"] = str(LOG_PATH)
    return jsonify(info)


@app.post("/api/ask")
def api_ask():
    payload = request.get_json(force=True) or {}
    question = (payload.get("question") or "").strip()
    mode = payload.get("mode", "hybrid")
    top_k = int(payload.get("top_k", 8))
    if not question:
        return jsonify({"error": "问题为空"}), 400
    if not _state["ready"]:
        if not wait_ready(timeout=120):
            if _state["error"]:
                return jsonify({"error": f"检索资源加载失败：{_state['error']}"}), 500
            return jsonify({"pending": True, "message": "检索模型仍在加载，请稍候几秒后重试"}), 202
    started = time.time()
    retriever = get_retriever()
    answer_count = int(payload.get("answer_count", 1) or 1)
    auto_panorama = False
    # 问题里点名了多家公司（对比题）时，自动用全景模式，否则单条答案必然只覆盖一家
    mentioned = retriever.mentioned_companies(question)
    if mode != "panorama" and len(mentioned) >= 2:
        mode = "panorama"
        auto_panorama = True
    if mode == "panorama":
        # 只对比问题里点名的公司；问题没点名（例如“13 家公司”）时才覆盖全部
        hits: List[Hit] = retriever.retrieve_panorama(
            question, companies=mentioned if 1 <= len(mentioned) <= 4 else None, top_k_each=2
        )
        picked = retriever.panorama_answer(question, hits)
        answer_text = picked["text"]
        answer_meta = None
        answer_candidates = picked["candidates"][:8]
        answer_source_rank = 0
        answer_confidence = "-"
    else:
        hits = retriever.retrieve(question, top_k=top_k, mode=mode)
        if answer_count <= 1:
            picked = retriever.pick_best_answer(question, hits)
            answer_text = picked["text"]
            answer_meta = picked["hit"]
            answer_candidates = [c for c in picked["candidates"] if c is not answer_meta]
            answer_source_rank = int(picked.get("source_rank") or 0)
            answer_confidence = picked.get("confidence", "high")
        else:
            answer_text = retriever.extractive_answer(question, hits, max_bullets=answer_count)
            answer_meta = None
            answer_candidates = []
            answer_source_rank = 0
            answer_confidence = "-"
    elapsed = round(time.time() - started, 2)
    citations: List[Dict] = []
    for hit in hits:
        data = hit.to_dict()
        data["snippet"] = hit.text  # 出处区显示整块内容，不做截断
        citations.append(data)
    return jsonify(
        {
            "question": question,
            "mode": mode,
            "top_k": top_k,
            "answer": answer_text,
            "answer_meta": answer_meta,
            "answer_candidates": answer_candidates,
            "answer_count": answer_count,
            "answer_source_rank": answer_source_rank,
            "answer_confidence": answer_confidence,
            "auto_panorama": auto_panorama,
            "citations": citations,
            "elapsed_seconds": elapsed,
            "answered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "answer_style": "抽取式",
        }
    )


@app.post("/api/export")
def api_export():
    payload = request.get_json(force=True) or {}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with RECORD_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return jsonify({"path": str(RECORD_PATH)})


@app.post("/api/shutdown")
def api_shutdown():
    """优雅关闭本页面进程（用于关掉忘了关、占着端口的旧实例）。"""
    def _stop():
        time.sleep(0.3)
        global _server
        if _server is not None:
            try:
                _server.shutdown()
            except Exception:  # noqa: BLE001
                pass
        os._exit(0)

    threading.Thread(target=_stop, daemon=True).start()
    return jsonify({"message": "服务正在关闭"})


@app.get("/pdf/<path:name>")
def serve_pdf(name: str):
    target = (PDF_DIR / urllib.parse.unquote(name)).resolve()
    if not str(target).startswith(str(PDF_DIR.resolve())) or not target.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(target, mimetype="application/pdf")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="问答页面（Flask）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    parser.add_argument("--no-preload", action="store_true", help="不在后台预加载模型（首次提问时再加载）")
    parser.add_argument("--diagnose", action="store_true", help="打印环境与向量后端诊断后退出（排障用）")
    args = parser.parse_args(argv)

    if args.diagnose:
        from 向量模型 import build_embedder, describe_environment

        report = {"environment": describe_environment()}
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print("\n正在依次尝试各向量后端（失败原因会完整打印）…", flush=True)
        embedder = build_embedder()
        report["chosen_backend"] = getattr(embedder, "name", None) or "不可用（退化为纯 BM25）"
        print("\n最终可用后端：" + report["chosen_backend"], flush=True)
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 诊断 =====\n")
            fh.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"诊断结果已追加写入：{LOG_PATH}", flush=True)
        return 0

    if not args.no_preload:
        threading.Thread(target=preload_retriever, daemon=True).start()
        print("已在后台加载检索索引与 Qwen3-Embedding-0.6B（本地缓存，离线），页面可先打开等待就绪…", flush=True)

    # 先绑定端口（绑定成功后再打开浏览器），端口被占用就自动往后找，
    # 避免“启动失败”其实是端口冲突、也避免把浏览器开到错误的端口上。
    from werkzeug.serving import make_server

    global _server
    last_error: Optional[Exception] = None
    for port in range(args.port, args.port + 11):
        try:
            _server = make_server(args.host, port, app, threaded=True)
        except OSError as exc:
            last_error = exc
            print(f"端口 {port} 不可用（{exc}），尝试 {port + 1} …", flush=True)
            continue
        url = f"http://{args.host}:{port}"
        print(f"问答页面地址：{url}（打开该地址提问；Ctrl+C 退出，或点页面上的“关闭服务”）", flush=True)
        if args.open:
            threading.Timer(0.6, lambda u=url: webbrowser.open(u)).start()
        try:
            _server.serve_forever()
        except KeyboardInterrupt:
            print("已手动退出", flush=True)
        finally:
            try:
                _server.server_close()
            except Exception:  # noqa: BLE001
                pass
        return 0
    raise SystemExit(f"启动失败：{args.port}-{args.port + 10} 端口都不可用（{last_error}）")


if __name__ == "__main__":
    raise SystemExit(main())
