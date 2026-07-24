"""
Monitoring — 系统监控仪表盘

读取 runtime/request_log.db（每一次 /analyze 调用的真实记录）和
runtime/review_queue.db（被护栏标记的子集）。这里展示的是真实请求产生的数据，
不是模拟出来的数字；数据量偏少时，可以先跑 src/07_seed_dashboard_traffic.py
用真实评论重放一批流量再回来看。
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import request_log
import review_queue

st.set_page_config(page_title="Monitoring", page_icon="📈", layout="wide")

st.title("📈 Monitoring")
st.caption(
    "读取真实请求日志（runtime/request_log.db）和护栏队列（runtime/review_queue.db）。"
    "数据量少时，可以先跑 `python src/07_seed_dashboard_traffic.py` 用真实评论重放一批流量。"
)

overall_stats = request_log.stats()

if overall_stats["n_requests"] == 0:
    st.info(
        "还没有请求记录。运行 `python src/07_seed_dashboard_traffic.py` 重放一批真实评论，"
        "或者去 API / 主页面跑几次分析，然后回来刷新这个页面。"
    )
    st.stop()

guardrail_rate = request_log.guardrail_trigger_rate()

st.markdown("---")

# ── KPI row ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Requests", overall_stats["n_requests"])
c2.metric("Avg Latency", f"{overall_stats['avg_latency_ms']:.0f} ms")
validity_rate = 1 - overall_stats["n_invalid"] / overall_stats["n_requests"]
c3.metric("JSON Validity Rate", f"{validity_rate * 100:.1f}%")
c4.metric(
    "Guardrail Trigger Rate", f"{guardrail_rate * 100:.1f}%",
    help=(
        "被 guardrails.py 标记的请求占比。Phase 3 的 eval harness 当时没法算这个指标——"
        "review_queue.db 只存被标记的子集，没有总量分母。这个数字来自 request_log.db，"
        "是 Phase 4 补上这块的直接原因。"
    ),
)

st.markdown("---")

# ── Model routing ────────────────────────────────────────────────────────
st.subheader("🔀 Model Routing")
route_dist = request_log.route_distribution()
cost = request_log.cost_estimate()

col_a, col_b = st.columns([2, 1])
with col_a:
    if route_dist:
        df_routes = pd.DataFrame(
            {"route": list(route_dist.keys()), "count": list(route_dist.values())}
        ).set_index("route")
        st.bar_chart(df_routes)
    else:
        st.info("暂无路由数据。")
with col_b:
    st.metric("本地模型调用", cost["n_local"])
    st.metric("DeepSeek API 调用", cost["n_api"])
    st.metric("累计估算成本", f"${cost['estimated_cost_usd']:.4f}")
    st.caption(
        f"成本假设：本地 \\${request_log.LOCAL_COST_PER_QUERY_USD}/次 · "
        f"API \\${request_log.API_COST_PER_QUERY_USD}/次（与 README 的 break-even 分析一致）"
    )

st.markdown("---")

# ── Latency by backend ──────────────────────────────────────────────────
st.subheader("⏱ Latency by Backend")
lat = request_log.latency_stats()
rows = [
    {"backend": model, "n": s["n"], "avg_ms": s["avg_ms"], "p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"]}
    for model, s in lat["by_model"].items()
]
if rows:
    st.dataframe(pd.DataFrame(rows).set_index("backend"), use_container_width=True)
    st.caption("本地模型（qwen2.5-1.5b-qlora-local）vs. DeepSeek API 的真实延迟对比。")

st.markdown("---")

# ── Request volume over time ────────────────────────────────────────────
st.subheader("📊 Request Volume")
recent_rows = request_log.recent(limit=500)
if len(recent_rows) >= 5:
    df = pd.DataFrame(recent_rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["bucket"] = df["created_at"].dt.floor("min")
    volume = df.groupby("bucket").size()
    if volume.index.max() - volume.index.min() >= pd.Timedelta(minutes=2):
        st.line_chart(volume)
    else:
        st.caption(
            f"最近 {len(recent_rows)} 条请求都发生在很短的时间窗口内，画不出有意义的趋势线——"
            "这符合这批数据是一次性重放出来的事实，不是持续产生的线上流量。"
        )
else:
    st.caption("请求数太少，暂时画不出趋势。")

st.markdown("---")

# ── Human review queue ───────────────────────────────────────────────────
st.subheader("🔎 Human Review Queue")
qstats = review_queue.queue_stats()
q1, q2, q3, q4 = st.columns(4)
q1.metric("Pending", qstats["pending"])
q2.metric("Approved", qstats["approved"])
q3.metric("Corrected", qstats["corrected"])
q4.metric("Rejected", qstats["rejected"])
st.caption("详情、approve/correct/reject 操作见 Review Queue 页面。")

st.markdown("---")

# ── Recent requests ──────────────────────────────────────────────────────
st.subheader("🧾 Recent Requests")
if recent_rows:
    df_recent = pd.DataFrame(recent_rows)[
        ["created_at", "dataset", "mode", "model", "latency_ms", "valid", "repaired", "flags_json"]
    ]
    st.dataframe(df_recent, use_container_width=True, hide_index=True)
