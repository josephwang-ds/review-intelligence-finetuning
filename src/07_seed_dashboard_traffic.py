"""
07_seed_dashboard_traffic.py
──────────────────────────────
给 Phase 6 的 Monitoring Dashboard 灌一批真实流量。

这不是伪造数据：每一条都是数据集里的真实评论，通过 FastAPI 的 TestClient 走
真实的 /analyze 端点代码路径（跟 tests/test_api_structure.py 用的是同一种方式），
guardrails / router / structured_client 全部真实执行，写进的 runtime/request_log.db
和 runtime/review_queue.db 是这些调用的真实结果。之所以叫"seed"而不是直接说
"生产流量"，是因为这是重放（replay）出来的样本，不是自然产生的线上流量——
这个区别值得说清楚，不该被混淆。

运行：python src/07_seed_dashboard_traffic.py
费用：约 20 次 DeepSeek 调用 ≈ $0.03；本地模型调用免费（已缓存，~2-3s/次）
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from fastapi.testclient import TestClient
from main import app
from config import ASAP_PROCESSED_DIR, ROOT

RANDOM_SEED = 42
SHORT_MAX_CHARS = 150

client = TestClient(app)


def load_asap_reviews(n_short: int, n_long: int) -> tuple[list[str], list[str]]:
    random.seed(RANDOM_SEED)
    path = ASAP_PROCESSED_DIR / "test.jsonl"
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    short = [t for t in texts if len(t) <= SHORT_MAX_CHARS]
    long_ = [t for t in texts if len(t) > SHORT_MAX_CHARS]
    return random.sample(short, min(n_short, len(short))), random.sample(long_, min(n_long, len(long_)))


def load_yelp_reviews(n: int) -> list[str]:
    random.seed(RANDOM_SEED)
    path = ROOT / "data" / "yelp" / "raw" / "sample.jsonl"
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["review_body"])
    return random.sample(texts, min(n, len(texts)))


def call(text: str, dataset: str, mode: str, few_shot: bool = True) -> dict:
    resp = client.post("/analyze", json={"text": text, "dataset": dataset, "mode": mode, "few_shot": few_shot})
    if resp.status_code != 200:
        return {"status": resp.status_code, "route": f"http_{resp.status_code}"}
    route = resp.json().get("route")
    # mode="full" doesn't route — label it the same way request_log.db does (see api/main.py's mode_label)
    return {"status": 200, "route": route or ("few_shot" if few_shot else "zero_shot")}


def main():
    print("加载真实评论样本...")
    short_asap, long_asap = load_asap_reviews(n_short=12, n_long=8)
    yelp_reviews = load_yelp_reviews(n=10)
    print(f"  ASAP 短评论: {len(short_asap)} 条 · ASAP 长评论: {len(long_asap)} 条 · Yelp: {len(yelp_reviews)} 条")

    route_counts: dict[str, int] = {}

    def record(result: dict):
        key = result["route"] or f"http_{result['status']}"
        route_counts[key] = route_counts.get(key, 0) + 1

    print("\n[1/4] ASAP 短评论 · mode=operational（应命中 local_finetuned）...")
    for text in short_asap:
        record(call(text, "asap", "operational"))

    print("[2/4] ASAP 长评论 · mode=operational（应触发 few_shot_escalated_long_input）...")
    for text in long_asap:
        record(call(text, "asap", "operational"))

    print("[3/4] Yelp 评论 · 混合 mode（operational 应触发 few_shot_non_asap_profile；full 走完整分析）...")
    for i, text in enumerate(yelp_reviews):
        mode = "operational" if i % 2 == 0 else "full"
        record(call(text, "yelp", mode))

    print("[4/4] ASAP 短评论 + 假电话号码（验证 guardrail 真的会拦截 PII 并进队列）...")
    pii_samples = [
        short_asap[0] + " 有问题打我电话13912345678。",
        short_asap[1] + " 联系邮箱 diner@example.com。",
    ]
    for text in pii_samples:
        record(call(text, "asap", "operational"))

    print("\n[5/5] 少量 mode=full 对比调用（ASAP few-shot 全量分析）...")
    for text in short_asap[:3]:
        record(call(text, "asap", "full"))

    print("\n─── 完成 ───────────────────────────────────────────")
    print("route 分布：")
    for route, n in sorted(route_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {route:<35} {n}")
    print(f"\n共 {sum(route_counts.values())} 次调用。查看 Streamlit 的 Monitoring 页面查看仪表盘。")


if __name__ == "__main__":
    main()
