# -*- coding: utf-8 -*-
"""
作业三 · 爬取业绩公告（证监会「资本市场电子化信息披露平台」）

功能
----
1. 读取 作业三/公司名称.txt 作为唯一公司清单；带“（沪）”的公司走沪市栏目
   （/101111 定期报告），其余公司走深市栏目（/101811 定期报告）。
2. 通过站点自动补全接口把公司简称解析成“股票代码 + 板块”，不需要手工维护代码表。
3. 按“代码 + 定期报告分类（年报 9696 / 半年报 9694）”筛选公告列表，翻页解析每一条
   公告的标题、日期和 PDF 直链，过滤“摘要 / 英文版”。
4. 下载 PDF：
     优先 requests 直连；
     沪市静态站（static.sse.com.cn）有 acw_sc__v2 前端校验，直接请求只能拿到挑战页，
     此时改用 webdriver（默认 Selenium + 本机 Chrome/chromedriver）打开 PDF 直链，
     让浏览器通过校验并下载，再把该域名的 cookie 交给 requests 走快通道。
5. 结果：PDF 存到 作业三/创新药业绩报告，清单写到 作业三/文本与索引/下载清单.csv。

扩展性
------
公司清单只依赖 作业三/公司名称.txt。新增公司直接往 txt 里写名字即可（带“（沪）”表示沪市），
下面的 REPORT_TARGETS 决定要抓哪几期报告，改这里就能抓别的期间。

用法
----
    python 爬取业绩公告.py                     # 增量下载（已下载且校验通过的跳过）
    python 爬取业绩公告.py --dry-run            # 只解析清单，不下载
    python 爬取业绩公告.py --companies 药明康德,科伦药业
    python 爬取业绩公告.py --force              # 强制重新下载
    python 爬取业绩公告.py --visible            # 有头浏览器（演示/录屏）
    python 爬取业绩公告.py --driver uc          # 改用 undetected_chromedriver
    python 爬取业绩公告.py --browser edge       # 备用浏览器（Edge）
    python 爬取业绩公告.py --no-browser         # 只允许 requests 下载
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:  # 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

# --------------------------------------------------------------------------------------
# 配置区（要改爬取范围，改这里就够了）
# --------------------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent          # 作业三/代码文件
ASSIGN_DIR = CODE_DIR.parent                        # 作业三
COMPANY_FILE = ASSIGN_DIR / "公司名称.txt"           # 公司清单（唯一来源）
PDF_DIR = ASSIGN_DIR / "创新药业绩报告"               # 下载的公告保存位置
DATA_DIR = ASSIGN_DIR / "文本与索引"                  # 中间产物
STAGING_DIR = DATA_DIR / "_下载暂存"                  # 浏览器下载暂存目录
MANIFEST_PATH = DATA_DIR / "下载清单.csv"             # 下载清单
DRIVER_DIR = CODE_DIR / "drivers"                    # 驱动副本位置

BASE_URL = "http://eid.csrc.gov.cn"

# 市场 → 站点栏目。沪市公司在“沪市上市公司-定期报告”，深市公司在“深市上市公司-定期报告”。
MARKETS: Dict[str, Dict[str, str]] = {
    "沪": {"channel": "101111", "name": "沪市"},
    "深": {"channel": "101811", "name": "深市"},
}

# 定期报告分类码（来自站点筛选表单）
CATEGORIES = {"一季报": "9693", "半年报": "9694", "三季报": "9695", "年报": "9696"}
CATEGORY_PARENT = "9604"  # 定期报告

# 要抓的报告期间：title_pattern 命中的公告才算目标报告
REPORT_TARGETS: List[Dict[str, str]] = [
    {"key": "2025年年度报告", "category": "年报", "title_pattern": r"2025\s*年年度报告"},
    {"key": "2025年半年度报告", "category": "半年报", "title_pattern": r"2025\s*年半年度报告"},
    {"key": "2026年半年度报告", "category": "半年报", "title_pattern": r"2026\s*年半年度报告"},
]

# 需要排除的公告（摘要、英文版、更正前的重复件等）
EXCLUDE_TITLE_PATTERNS = [r"摘要", r"英文", r"English", r"取消", r"勘误"]

REQUEST_TIMEOUT = 60
REQUEST_RETRY = 3
RETRY_WAIT = [2, 5, 10]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# 浏览器默认位置（可用环境变量覆盖）
CHROME_BINARY = os.environ.get(
    "SE_CHROME_BINARY", r"C:\Users\Thinkbook\AppData\Local\Google\Chrome\Application\chrome.exe"
)
CHROME_DRIVER_SRC = os.environ.get(
    "SE_CHROME_DRIVER", r"C:\Users\Thinkbook\AppData\Local\Google\Chrome\Application\chromedriver.exe"
)
EDGE_BINARY = os.environ.get(
    "SE_EDGE_BINARY", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
)
EDGE_DRIVER_SRC = os.environ.get(
    "SE_EDGE_DRIVER",
    str(Path(os.environ.get("TEMP", ".")) / "zys_selenium_probe" / "driver" / "msedgedriver.exe"),
)


# --------------------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------------------


@dataclass
class Company:
    name: str
    market: str          # 沪 / 深
    code: str = ""
    board: str = ""      # 沪市主板 / 沪市科创板 / 深市主板 / 创业板 ...


@dataclass
class Announcement:
    code: str
    company: str
    title: str
    date: str
    url: str
    category: str = ""
    channel: str = ""


@dataclass
class DownloadRecord:
    company: str
    code: str
    market: str
    board: str
    period: str
    category: str
    title: str
    date: str
    url: str
    local_file: str
    size_bytes: int = 0
    sha256: str = ""
    method: str = ""
    status: str = ""
    downloaded_at: str = ""
    note: str = ""


# --------------------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_pdf(path: Path, min_bytes: int = 100_000) -> bool:
    """PDF 有效性校验：文件够大、头部是 %PDF-、尾部有 %%EOF（避免把下载中的残片当成品）。"""
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    with path.open("rb") as fh:
        if fh.read(5) != b"%PDF-":
            return False
        fh.seek(max(0, path.stat().st_size - 4096))
        tail = fh.read()
    return b"%%EOF" in tail


def safe_name(text: str) -> str:
    """把标题/公司名转成安全的文件名片段。"""
    text = re.sub(r"[\\/:*?\"<>|\r\n\t]", "", text).strip()
    return re.sub(r"\s+", "", text)


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"})
    return s


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    retry: int = REQUEST_RETRY,
    min_length: int = 1,
    must_contain: Optional[Sequence[str]] = None,
    **kwargs,
) -> Optional[requests.Response]:
    """带退避重试的请求；站点偶发返回 405/空页时退避重试。

    min_length / must_contain 用来把“看起来像页面但其实是错误页”的响应判为失败。
    """
    last_err = ""
    text = ""
    for attempt in range(retry):
        try:
            resp = session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            body_len = len(resp.content or b"")
            text = resp.content.decode("utf-8", "ignore")
            ok = resp.status_code == 200 and body_len >= min_length
            if ok and must_contain:
                ok = any(token in text for token in must_contain)
            if ok:
                return resp
            last_err = f"HTTP {resp.status_code} len={body_len} 内容不符合预期"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < retry - 1:
            wait = RETRY_WAIT[min(attempt, len(RETRY_WAIT) - 1)]
            log(f"  请求异常（{last_err}），{wait}s 后重试 {url}")
            time.sleep(wait)
    log(f"  请求失败：{url}（{last_err}）")
    return None


# --------------------------------------------------------------------------------------
# 公司清单解析 + 代码解析
# --------------------------------------------------------------------------------------


def load_companies(only: Optional[Sequence[str]] = None) -> List[Company]:
    """读取 公司名称.txt。支持“药明康德（沪）”这种带市场标记的写法。"""
    raw = COMPANY_FILE.read_text(encoding="utf-8")
    parts = [p for p in re.split(r"[、,，;；\s]+", raw) if p.strip()]
    companies: List[Company] = []
    for part in parts:
        name = part.strip()
        market = "深"
        if re.search(r"[（(]\s*沪\s*[)）]", name):
            market = "沪"
        elif re.search(r"[（(]\s*深\s*[)）]", name):
            market = "深"
        clean = re.sub(r"[（(][^)）]*[)）]", "", name).strip()
        if clean:
            companies.append(Company(name=clean, market=market))
    if only:
        wanted = {n.strip() for n in only}
        companies = [c for c in companies if c.name in wanted]
    return companies


def resolve_code(session: requests.Session, company: Company, try_other_market: bool = True) -> None:
    """调用站点自动补全接口解析代码与板块。"""
    markets = [company.market]
    if try_other_market:
        markets += [m for m in MARKETS if m != company.market]
    for market in markets:
        channel = MARKETS[market]["channel"]
        resp = request_with_retry(
            session,
            "GET",
            f"{BASE_URL}/getAdvice1.html",
            params={"term": company.name, "channelPath": channel},
            retry=2,
            headers={
                "Referer": f"{BASE_URL}/{channel}/index.html",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        if resp is None:
            continue
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            continue
        for item in data.get("source", []) or []:
            label = str(item.get("lable", "")).strip()
            m = re.match(r"(\d{6})\s+(\S+)\s+(\S+)", label)
            if not m:
                continue
            code, board, name = m.group(1), m.group(2), m.group(3)
            if name == company.name or company.name in label:
                company.code, company.board, company.market = code, board, market
                return
    log(f"  警告：未能解析公司代码 -> {company.name}")


# --------------------------------------------------------------------------------------
# 公告列表抓取
# --------------------------------------------------------------------------------------


ROW_RE = re.compile(
    r"<td width=\"100px;\"><a[^>]*onclick=\"gotoProd\('(\d+)','(\d+)','([^']*)'[^>]*>(\d+)</a></td>"
    r"\s*<td width=\"150px;\">([^<]*)</td>\s*"
    r"<td width=\"400px;\"\s*onclick=\"downloadPdf1\('([^']+)','([^']*)','([^']*)','(\d+)','([^']*)'\);\">",
    re.S,
)


def parse_rows(html: str) -> List[Announcement]:
    out: List[Announcement] = []
    for m in ROW_RE.finditer(html):
        channel, code, name_onclick, code_td, name_td, url, title, date, ch2, ext = m.groups()
        out.append(
            Announcement(
                code=code_td or code,
                company=(name_td or name_onclick).strip(),
                title=title.strip(),
                date=date.strip(),
                url=url.strip(),
                channel=channel,
            )
        )
    if out:
        return out
    # 兼容格式变化：只要 downloadPdf1 链
    for m in re.finditer(
        r"downloadPdf1\('([^']+)','([^']*)','([^']*)','(\d+)','([^']*)'\)", html
    ):
        url, title, date, channel, _ext = m.groups()
        out.append(
            Announcement(code="", company="", title=title.strip(), date=date.strip(), url=url.strip(), channel=channel)
        )
    return out


def filter_form_data(code: str, category_code: str) -> Dict[str, str]:
    return {
        "prodType": code,
        "prodType2": "代码/简称/拼音缩写 ",
        "keyWord": "",
        "keyWord2": "关键字",
        "startDate": "",
        "startDate2": "请输入开始时间",
        "endDate": "",
        "endDate2": "请输入结束时间",
        "selCatagory2": CATEGORY_PARENT,
        "selCatagory3": category_code,
        "selBoardCode0": "",
        "selBoardCode": "",
    }


class ListPageFetcher:
    """公告列表抓取：requests 为主，浏览器读取 DOM 为备（站点偶发 405）。"""

    def __init__(self, session: requests.Session, browser: "BrowserHelper"):
        self.session = session
        self.browser = browser

    def _pages_of(self, html: str) -> int:
        m = re.search(r"共<b>(\d+)</b>页", html)
        if m:
            return int(m.group(1))
        m = re.search(r"共(\d+)条数据", html)
        if m:
            return max(1, (int(m.group(1)) + 14) // 15)
        return 1

    def fetch(self, market: str, code: str, category: str) -> List[Announcement]:
        channel = MARKETS[market]["channel"]
        data = filter_form_data(code, CATEGORIES[category])
        items: List[Announcement] = []
        page = 1
        total_pages = 1
        while page <= total_pages and page <= 20:
            url = f"{BASE_URL}/{channel}/index_f.html" if page == 1 else f"{BASE_URL}/{channel}/index_{page}_f.html"
            resp = request_with_retry(
                self.session, "POST", url, data=data, min_length=500, must_contain=["条数据"]
            )
            if resp is None:
                log(f"  改用浏览器读取列表：{market}/{code}/{category} 第{page}页")
                html = self.browser.fetch_filtered_list(market, code, category, page)
                if not html:
                    break
            else:
                html = resp.content.decode("utf-8", "ignore")
            if page == 1:
                total_pages = self._pages_of(html)
            rows = parse_rows(html)
            if not rows:
                break
            items.extend(rows)
            page += 1
            time.sleep(0.6)
        return items


def pick_targets(announcements: Iterable[Announcement]) -> Dict[str, Announcement]:
    """从公告里挑出每个目标期间唯一的一份全文报告（有多个则取日期最新的一份）。"""
    picked: Dict[str, Announcement] = {}
    for target in REPORT_TARGETS:
        pat = re.compile(target["title_pattern"])
        candidates = []
        for ann in announcements:
            if not pat.search(ann.title):
                continue
            if any(re.search(p, ann.title) for p in EXCLUDE_TITLE_PATTERNS):
                continue
            candidates.append(ann)
        if not candidates:
            continue
        candidates.sort(key=lambda a: a.date, reverse=True)
        picked[target["key"]] = candidates[0]
    return picked


# --------------------------------------------------------------------------------------
# 浏览器（webdriver）
# --------------------------------------------------------------------------------------


class BrowserHelper:
    """封装 webdriver：沪市静态站反爬（acw_sc__v2）需要浏览器过校验后再下载 PDF。"""

    def __init__(self, driver_kind: str, browser: str, visible: bool, download_dir: Path, enabled: bool = True):
        self.driver_kind = driver_kind
        self.browser = browser
        self.visible = visible
        self.download_dir = download_dir
        self.enabled = enabled
        self.driver = None
        self.cookies: Dict[str, Dict[str, str]] = {}

    # ----- 驱动准备 -----
    def _driver_path(self) -> str:
        DRIVER_DIR.mkdir(parents=True, exist_ok=True)
        if self.browser == "edge":
            src = Path(EDGE_DRIVER_SRC)
            dst = DRIVER_DIR / "msedgedriver.exe"
        else:
            src = Path(CHROME_DRIVER_SRC)
            dst = DRIVER_DIR / "chromedriver.exe"
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
        if dst.exists():
            return str(dst)
        if src.exists():
            return str(src)
        raise FileNotFoundError(f"找不到浏览器驱动：{src}")

    def start(self) -> None:
        if self.driver is not None:
            return
        prefs = {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_setting_values.automatic_downloads": 1,
            "safebrowsing.enabled": True,
        }
        if self.browser == "edge":
            from selenium import webdriver
            from selenium.webdriver.edge.options import Options
            from selenium.webdriver.edge.service import Service

            options = Options()
            options.binary_location = EDGE_BINARY
            if not self.visible:
                options.add_argument("--headless=new")
            options.add_argument(f"--user-data-dir={self._profile_dir()}")
            options.add_argument("--no-first-run")
            options.add_argument("--window-size=1440,900")
            options.add_experimental_option("prefs", prefs)
            self.driver = webdriver.Edge(service=Service(executable_path=self._driver_path()), options=options)
            return

        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.binary_location = CHROME_BINARY
        if not self.visible:
            options.add_argument("--headless=new")
        options.add_argument(f"--user-data-dir={self._profile_dir()}")
        options.add_argument("--no-first-run")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--window-size=1440,900")
        options.add_experimental_option("prefs", prefs)

        if self.driver_kind == "uc":
            import undetected_chromedriver as uc

            version_main = None
            m = re.match(r"(\d+)\.", CHROME_BINARY.split("Application")[-1].strip("\\/"))
            version_main = 147 if not m else int(m.group(1))
            self.driver = uc.Chrome(
                browser_executable_path=CHROME_BINARY,
                driver_executable_path=self._driver_path(),
                options=options,
                version_main=version_main,
            )
        else:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service

            self.driver = webdriver.Chrome(service=Service(executable_path=self._driver_path()), options=options)
        log(f"  浏览器已启动：{self.driver.capabilities.get('browserVersion')}")

    def _profile_dir(self) -> str:
        path = Path(os.environ.get("TEMP", ".")) / "zys_作业三_profile" / f"{self.driver_kind}_{self.browser}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def quit(self) -> None:
        if self.driver is None:
            return
        try:
            self.driver.quit()
        except Exception:  # noqa: BLE001
            pass
        self.driver = None

    # ----- 列表页兜底 -----
    def fetch_filtered_list(self, market: str, code: str, category: str, page: int) -> str:
        if not self.enabled:
            return ""
        self.start()
        channel = MARKETS[market]["channel"]
        try:
            self.driver.get(f"{BASE_URL}/{channel}/index.html")
            time.sleep(2.5)
            action = "index_f.html" if page == 1 else f"index_{page}_f.html"
            script = """
            document.getElementById('prodType').value = arguments[0];
            var p2 = document.getElementById('prodType2'); if (p2) p2.value = '代码/简称/拼音缩写 ';
            var c2 = document.getElementById('selCatagory2'); if (c2) { c2.value = arguments[1]; }
            var c3 = document.getElementById('step2') || document.querySelector('[name=selCatagory3]');
            if (c3) { c3.value = arguments[2]; }
            var f = document.getElementById('channelFilterForm');
            f.action = arguments[3];
            f.submit();
            """
            self.driver.execute_script(
                script, code, CATEGORY_PARENT, CATEGORIES[category], action
            )
            time.sleep(3)
            return self.driver.page_source
        except Exception as exc:  # noqa: BLE001
            log(f"  浏览器读取列表失败：{type(exc).__name__}: {exc}")
            return ""

    # ----- PDF 下载 -----
    def download(self, url: str, target: Path, wait_seconds: int = 90) -> bool:
        if not self.enabled:
            return False
        self.start()
        for old in self.download_dir.glob("*"):
            if old.is_file():
                try:
                    old.unlink()
                except OSError:
                    pass
        try:
            self.driver.get(url)
        except Exception as exc:  # noqa: BLE001
            log(f"  浏览器打开失败：{type(exc).__name__}: {exc}")
            return False
        # 等浏览器把文件写完：既不能有 .crdownload，文件大小也要连续两次保持不变，
        # 否则会把下载中的残片当成成品搬走（这曾经导致一份 PDF 只有 0.10MB）。
        deadline = time.time() + wait_seconds
        found: Optional[Path] = None
        last_size, stable_ticks = -1, 0
        while time.time() < deadline:
            pending = list(self.download_dir.glob("*.crdownload")) + list(
                self.download_dir.glob("*.tmp")
            )
            files = [
                f
                for f in self.download_dir.glob("*")
                if f.is_file() and not f.name.endswith((".crdownload", ".tmp"))
            ]
            if files and not pending:
                candidate = max(files, key=lambda p: p.stat().st_size)
                size = candidate.stat().st_size
                if size == last_size and size > 300_000:
                    stable_ticks += 1
                    if stable_ticks >= 2:
                        found = candidate
                        break
                else:
                    stable_ticks = 0
                last_size = size
            time.sleep(1.0)
        if found is None or not is_valid_pdf(found, min_bytes=300_000):
            leftovers = sorted(p.name for p in self.download_dir.glob("*") if p.is_file())
            if leftovers:
                log(f"  浏览器下载未完成/不是有效 PDF，暂存目录：{leftovers}")
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.move(str(found), str(target))
        # 记录 cookie，后续同域请求可以直接走 requests
        domain = re.sub(r"^https?://([^/]+).*$", r"\1", url)
        self.cookies[domain] = {c["name"]: c["value"] for c in self.driver.get_cookies()}
        return True

    def cookies_for(self, url: str) -> Dict[str, str]:
        domain = re.sub(r"^https?://([^/]+).*$", r"\1", url)
        return self.cookies.get(domain, {})


# --------------------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------------------


def download_pdf(
    session: requests.Session,
    browser: BrowserHelper,
    ann: Announcement,
    target: Path,
    allow_browser: bool = True,
) -> Tuple[bool, str]:
    """返回 (是否成功, 下载方式)。先 requests，被反爬拦截再走浏览器。"""
    cookies = browser.cookies_for(ann.url)
    try:
        resp = session.get(ann.url, timeout=REQUEST_TIMEOUT, stream=True, cookies=cookies or None)
        first = next(resp.iter_content(8), b"")
        if resp.status_code == 200 and first.startswith(b"%PDF"):
            tmp = target.with_suffix(".part")
            with tmp.open("wb") as fh:
                fh.write(first)
                for chunk in resp.iter_content(1 << 16):
                    fh.write(chunk)
            tmp.replace(target)
            if is_valid_pdf(target):
                return True, "requests"
    except Exception as exc:  # noqa: BLE001
        log(f"  requests 下载失败：{type(exc).__name__}: {exc}")

    if not allow_browser:
        return False, ""
    log("  改用 webdriver 通过反爬校验下载")
    ok = browser.download(ann.url, target)
    return (ok, "webdriver") if ok else (False, "")


def load_existing_manifest() -> Dict[Tuple[str, str], DownloadRecord]:
    rows: Dict[Tuple[str, str], DownloadRecord] = {}
    if not MANIFEST_PATH.exists():
        return rows
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                rec = DownloadRecord(
                    company=row.get("公司", ""),
                    code=row.get("代码", ""),
                    market=row.get("市场", ""),
                    board=row.get("板块", ""),
                    period=row.get("报告期间", ""),
                    category=row.get("报告类型", ""),
                    title=row.get("公告标题", ""),
                    date=row.get("发布日期", ""),
                    url=row.get("公告链接", ""),
                    local_file=row.get("本地文件", ""),
                    size_bytes=int(row.get("字节数") or 0),
                    sha256=row.get("SHA256", ""),
                    method=row.get("下载方式", ""),
                    status=row.get("状态", ""),
                    downloaded_at=row.get("下载时间", ""),
                    note=row.get("备注", ""),
                )
                rows[(rec.company, rec.period)] = rec
            except Exception:  # noqa: BLE001
                continue
    return rows


def write_manifest(records: Iterable[DownloadRecord]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        ("公司", "company"),
        ("代码", "code"),
        ("市场", "market"),
        ("板块", "board"),
        ("报告期间", "period"),
        ("报告类型", "category"),
        ("公告标题", "title"),
        ("发布日期", "date"),
        ("公告链接", "url"),
        ("本地文件", "local_file"),
        ("字节数", "size_bytes"),
        ("SHA256", "sha256"),
        ("下载方式", "method"),
        ("状态", "status"),
        ("下载时间", "downloaded_at"),
        ("备注", "note"),
    ]
    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([c for c, _ in fields])
        for rec in records:
            writer.writerow([getattr(rec, attr) for _, attr in fields])


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="爬取沪/深市定期报告全文（证监会信息披露平台）")
    parser.add_argument("--companies", help="只处理这些公司（逗号分隔，默认全部）")
    parser.add_argument("--dry-run", action="store_true", help="只解析清单，不下载")
    parser.add_argument("--force", action="store_true", help="已存在的 PDF 也重新下载")
    parser.add_argument("--visible", action="store_true", help="使用有头浏览器")
    parser.add_argument("--driver", choices=["selenium", "uc"], default="selenium",
                        help="webdriver 实现：selenium（默认，实测可下载）或 uc（undetected_chromedriver）")
    parser.add_argument("--browser", choices=["chrome", "edge"], default="chrome", help="浏览器内核")
    parser.add_argument("--no-browser", action="store_true", help="完全禁用浏览器，仅用 requests")
    parser.add_argument("--wait", type=int, default=90, help="浏览器下载等待秒数")
    args = parser.parse_args(argv)

    for directory in (PDF_DIR, DATA_DIR, STAGING_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    only = [s for s in (args.companies.split(",") if args.companies else []) if s.strip()]
    companies = load_companies(only or None)
    log(f"公司清单：{len(companies)} 家（来源 {COMPANY_FILE.name}）")

    session = new_session()
    browser = BrowserHelper(
        driver_kind=args.driver,
        browser=args.browser,
        visible=args.visible,
        download_dir=STAGING_DIR,
        enabled=not args.no_browser,
    )

    existing = load_existing_manifest()
    records: List[DownloadRecord] = []
    planned: List[Tuple[Company, str, Announcement]] = []

    try:
        for company in companies:
            if not company.code:
                resolve_code(session, company)
                time.sleep(0.4)
            if not company.code:
                records.append(
                    DownloadRecord(
                        company=company.name, code="", market=company.market, board="", period="",
                        category="", title="", date="", url="", local_file="", status="失败",
                        note="未能解析公司代码",
                    )
                )
                continue
            log(f"公司：{company.name}（{company.code} · {company.board} · {MARKETS[company.market]['name']}）")
            announcements: List[Announcement] = []
            for category in sorted({t["category"] for t in REPORT_TARGETS}):
                items = ListPageFetcher(session, browser).fetch(company.market, company.code, category)
                log(f"  {category} 公告 {len(items)} 条")
                announcements.extend(items)
                time.sleep(0.8)
            picked = pick_targets(announcements)
            for target in REPORT_TARGETS:
                period = target["key"]
                ann = picked.get(period)
                if ann is None:
                    log(f"  警告：{company.name} 未找到 {period} 全文")
                    records.append(
                        DownloadRecord(
                            company=company.name, code=company.code, market=MARKETS[company.market]["name"],
                            board=company.board, period=period, category=target["category"], title="",
                            date="", url="", local_file="", status="失败", note="未找到对应公告",
                        )
                    )
                    continue
                planned.append((company, period, ann))
    finally:
        pass

    if args.dry_run:
        log("== dry-run 计划 ==")
        for company, period, ann in planned:
            print(f"  {company.name}({company.code}) {period} | {ann.date} | {ann.title}")
        log(f"计划下载 {len(planned)} 份 PDF（dry-run 不下载）")
        browser.quit()
        return 0

    ok_count = 0
    for company, period, ann in planned:
        target = PDF_DIR / f"{safe_name(company.name)}_{company.code}_{safe_name(period)}.pdf"
        key = (company.name, period)
        if not args.force and is_valid_pdf(target):
            old = existing.get(key)
            log(f"跳过（已存在）：{target.name}")
            records.append(
                old
                or DownloadRecord(
                    company=company.name, code=company.code, market=MARKETS[company.market]["name"],
                    board=company.board, period=period, category="定期报告", title=ann.title, date=ann.date,
                    url=ann.url, local_file=str(target.relative_to(ASSIGN_DIR)),
                    size_bytes=target.stat().st_size, sha256=sha256_of(target), method="已存在",
                    status="成功", downloaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            ok_count += 1
            continue
        log(f"下载：{company.name} {period} <- {ann.date} {ann.title}")
        ok, method = download_pdf(session, browser, ann, target, allow_browser=not args.no_browser)
        if ok:
            size = target.stat().st_size
            log(f"  成功（{method}，{size/1048576:.2f} MB）")
            ok_count += 1
            records.append(
                DownloadRecord(
                    company=company.name, code=company.code, market=MARKETS[company.market]["name"],
                    board=company.board, period=period, category="定期报告", title=ann.title, date=ann.date,
                    url=ann.url, local_file=str(target.relative_to(ASSIGN_DIR)), size_bytes=size,
                    sha256=sha256_of(target), method=method, status="成功",
                    downloaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
        else:
            log("  失败")
            records.append(
                DownloadRecord(
                    company=company.name, code=company.code, market=MARKETS[company.market]["name"],
                    board=company.board, period=period, category="定期报告", title=ann.title, date=ann.date,
                    url=ann.url, local_file=str(target.relative_to(ASSIGN_DIR)), status="失败",
                    note="requests 与 webdriver 均未取到 PDF",
                    downloaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
        time.sleep(1.0)

    browser.quit()
    all_records = {**existing, **{(r.company, r.period): r for r in records if r.period}}
    ordered = sorted(all_records.values(), key=lambda r: (r.company, r.period))
    write_manifest(ordered)
    log(f"完成：成功 {ok_count}/{len(planned)}，清单 -> {MANIFEST_PATH}")
    return 0 if ok_count == len(planned) else 1


if __name__ == "__main__":
    raise SystemExit(main())
