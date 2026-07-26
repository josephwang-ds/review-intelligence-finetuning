"""
demo_data.py — 让部署到 Streamlit Cloud 的 demo 有真实数据可看

问题：runtime/ 是 gitignore 的（那是运行时状态，不该进版本库），所以部署到
Streamlit Cloud 之后 request_log.db / review_queue.db 根本不存在，Review Queue
和 Monitoring 两个页面会是空的——恰恰是最能体现工程设计的两块看起来像没做完。

做法：把一次真实 seed 跑（src/07_seed_dashboard_traffic.py，35 次真实 /analyze
调用）的结果快照存进 demo_data/ 提交进仓库，首次访问时复制到 runtime/。
复制之后就是普通的可写数据库——访客可以真的在 Review Queue 里点 approve/correct，
不是只读的假界面。

注意这些是"重放出来的快照"，不是自然产生的线上流量。页面上会标注清楚，
不该把两者混为一谈。
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEMO_DIR = ROOT / "demo_data"
RUNTIME_DIR = ROOT / "runtime"

_DB_NAMES = ("request_log.db", "review_queue.db")
# 标记文件：记录"这份 runtime 数据是从快照来的"。用文件而不是函数返回值，是因为
# Streamlit 多页应用里每个页面是独立执行的——先加载的那个页面才会真正执行复制，
# 后加载的页面拿到的返回值是 False，如果靠返回值决定要不要显示"这是快照"的提示，
# 提示就会时有时无。数据来源的披露不该取决于用户先点了哪个页面。
_MARKER = "seeded_from_demo_snapshot"


def ensure_demo_data() -> bool:
    """runtime/ 里缺哪个 DB 就从 demo_data/ 补哪个。返回本次调用是否真的复制了东西。

    幂等且便宜（只做 Path.exists 检查），可以在每个页面顶部安全调用。
    只在文件不存在时复制——本地已经有真实数据时绝不覆盖。

    只被 Streamlit 页面调用，不在 request_log.py / review_queue.py 的 _connect()
    里做，否则单元测试里 monkeypatch 出来的 tmp_path 空库会被塞进快照数据，
    "空库"相关的测试就失去意义了。
    """
    copied = False
    if not DEMO_DIR.exists():
        return False
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for name in _DB_NAMES:
        src, dst = DEMO_DIR / name, RUNTIME_DIR / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied = True
    if copied:
        (RUNTIME_DIR / _MARKER).touch()
    return copied


def is_showing_demo_snapshot() -> bool:
    """当前 runtime 数据是否来自快照——页面用它决定要不要显示来源说明。
    跨页面、跨重启都稳定，不受"哪个页面先被打开"影响。"""
    return (RUNTIME_DIR / _MARKER).exists()


def demo_snapshot_available() -> bool:
    return DEMO_DIR.exists() and any((DEMO_DIR / n).exists() for n in _DB_NAMES)
