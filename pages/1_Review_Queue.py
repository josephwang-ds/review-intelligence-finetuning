"""
Review Queue — 人工复核队列

app.py 里的护栏（src/guardrails.py）把校验失败 / PII / 疑似 prompt injection /
业务逻辑矛盾的预测结果写进这里（src/review_queue.py，SQLite）。这里是人工
approve / correct / reject 的界面。
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from schema import PROFILES
import review_queue

st.set_page_config(page_title="Review Queue", page_icon="🔎", layout="wide")

st.title("🔎 Review Queue")
st.caption(
    "被护栏标记的预测结果在这里人工复核。correct 时选的修正值就是下一轮 QLoRA "
    "微调可以直接用的 hard negative 样本——见 05_prepare_finetune_data.py。"
)

stats = review_queue.queue_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pending", stats["pending"])
c2.metric("Approved", stats["approved"])
c3.metric("Corrected", stats["corrected"])
c4.metric("Rejected", stats["rejected"])

st.markdown("---")

items = review_queue.list_pending()

if not items:
    st.info("队列为空 — 没有待复核的项目。")
else:
    for item in items:
        item_id = item["id"]
        profile = PROFILES.get(item["dataset"])
        prediction = json.loads(item["prediction_json"])
        reasons = json.loads(item["reasons_json"])

        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"**#{item_id} · {item['dataset']} · {item['method']}** · {item['created_at']}")
                st.markdown(f"> {item['review_text']}")
            with top[1]:
                st.markdown("**Reasons**")
                for r in reasons:
                    st.markdown(f"- `{r}`")

            st.json(prediction, expanded=False)

            btn_cols = st.columns([1, 1, 1, 3])
            if btn_cols[0].button("✅ Approve", key=f"approve_{item_id}"):
                review_queue.resolve(item_id, "approved")
                st.rerun()
            if btn_cols[2].button("❌ Reject", key=f"reject_{item_id}"):
                review_queue.resolve(item_id, "rejected")
                st.rerun()

            with btn_cols[1].popover("✏️ Correct"):
                if profile is None:
                    st.warning(f"未知 dataset profile: {item['dataset']}，无法提供受限下拉选项。")
                else:
                    with st.form(key=f"correct_form_{item_id}"):
                        corrected = dict(prediction)

                        def _select(field, options, current):
                            idx = options.index(current) if current in options else 0
                            return st.selectbox(field, options, index=idx, key=f"{field}_{item_id}")

                        if "sentiment" in prediction:
                            corrected["sentiment"] = _select("sentiment", list(profile.sentiments), prediction.get("sentiment"))
                        if "rating_prediction" in prediction:
                            corrected["rating_prediction"] = st.number_input(
                                "rating_prediction", min_value=1, max_value=5,
                                value=int(prediction.get("rating_prediction", 3)), key=f"rating_{item_id}",
                            )
                        corrected["problem_type"] = _select("problem_type", list(profile.problem_types), prediction.get("problem_type"))
                        corrected["action_priority"] = _select("action_priority", list(profile.priorities), prediction.get("action_priority"))
                        corrected["operator_action"] = _select("operator_action", list(profile.operator_actions), prediction.get("operator_action"))
                        note = st.text_input("Reviewer note (optional)", key=f"note_{item_id}")

                        if st.form_submit_button("Save correction"):
                            review_queue.resolve(item_id, "corrected", corrected_json=corrected, note=note or None)
                            st.rerun()
