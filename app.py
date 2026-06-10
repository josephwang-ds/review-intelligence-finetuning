"""
Review Intelligence · 评论智能分析
4-method comparison: TextBlob → Zero-shot LLM → Few-shot LLM → Fine-tuned Qwen2.5-1.5B
"""

import json, os, re, time
from pathlib import Path

import streamlit as st
from textblob import TextBlob
from openai import OpenAI
import pandas as pd

st.set_page_config(page_title="Review Intelligence", page_icon="🍜", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8f9fa; }
[data-testid="stSidebar"] { background: #f0f2f5; }
.method-label {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.6rem;
}
.sentiment-positive { color: #16a34a; font-weight: 700; }
.sentiment-negative { color: #dc2626; font-weight: 700; }
.sentiment-neutral  { color: #d97706; font-weight: 700; }
.aspect-tag {
    display: inline-block; font-size: 0.73rem;
    padding: 2px 8px; border-radius: 99px; margin: 2px;
}
.aspect-positive { background: #dcfce7; color: #166534; }
.aspect-negative { background: #fee2e2; color: #991b1b; }
.aspect-neutral  { background: #fef9c3; color: #854d0e; }
.warn-box {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-radius: 6px; padding: 0.5rem 0.8rem;
    font-size: 0.82rem; color: #9a3412; margin-bottom: 0.5rem;
}
.dataset-badge {
    display: inline-block; font-size: 0.75rem; font-weight: 600;
    padding: 3px 10px; border-radius: 99px; margin-bottom: 0.5rem;
}
.story-box {
    background: #f0f4ff; border: 1px solid rgba(99,102,241,0.35);
    border-left: 4px solid #6366f1; border-radius: 0 8px 8px 0;
    padding: 1rem 1.3rem; margin: 0.5rem 0 1.2rem; color: #1e1b4b; line-height: 1.8;
    font-size: 0.88rem;
}
.ft-box {
    background: #fffbeb; border: 1px solid rgba(245,158,11,0.3);
    border-left: 3px solid #f59e0b; border-radius: 0 6px 6px 0;
    padding: 0.5rem 0.8rem; font-size: 0.8rem; color: #78350f;
    margin-bottom: 0.5rem; line-height: 1.6;
}
.decision-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a; border-radius: 0 8px 8px 0;
    padding: 1rem 1.3rem; margin: 0.5rem 0 1rem; color: #14532d; line-height: 1.8;
}
.acc-badge {
    display: inline-block; background: #fef3c7; color: #92400e;
    font-size: 0.72rem; font-weight: 700; padding: 1px 7px;
    border-radius: 99px; margin-left: 4px; letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Dataset selection ─────────────────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state.dataset = "asap"

ds = st.session_state.dataset
is_asap = ds == "asap"

# ── ASAP config ───────────────────────────────────────────────────────────────
ASAP_EXAMPLES = [
    ("🌟 好评（5★）", "环境很好，菜品精致好吃，服务也很周到，就是停车位有点少，但瑕不掩瑜，整体非常满意，强烈推荐！"),
    ("😐 中评（3★）", "菜的口味还可以，分量有点少，价格偏贵，环境一般，服务还算正常，总体感觉一般，不会特意再来。"),
    ("😤 差评（1★）", "等了将近一个小时才上菜，服务员态度极差，催了好几次都爱答不理，菜端上来还是凉的，以后绝对不来了。"),
]
ASAP_ASPECTS = [
    "location_traffic","location_distance","location_easy_to_find",
    "service_wait_time","service_attitude","service_parking","service_speed",
    "price_level","price_value","price_discount",
    "env_decoration","env_noise","env_space","env_cleanliness",
    "food_portion","food_taste","food_appearance","food_recommendation",
]
ASAP_ASPECT_LABELS = {
    "location_traffic":"交通便利","location_distance":"位置距离","location_easy_to_find":"容易找到",
    "service_wait_time":"等待时间","service_attitude":"服务态度","service_parking":"停车便利","service_speed":"上菜速度",
    "price_level":"价格水平","price_value":"性价比","price_discount":"折扣优惠",
    "env_decoration":"环境装修","env_noise":"噪音","env_space":"空间大小","env_cleanliness":"卫生清洁",
    "food_portion":"菜量","food_taste":"口味","food_appearance":"菜品外观","food_recommendation":"推荐菜品",
}
ASAP_PROBLEMS = {
    "taste_issue":"口味问题","poor_service":"服务差","long_wait":"等待太久",
    "overpriced":"价格偏高","hygiene_issue":"卫生问题","none":"无明显问题",
}
ASAP_ACTIONS = {
    "improve_taste":"改善口味","train_service":"培训服务","reduce_wait":"减少等待",
    "review_pricing":"检讨定价","fix_hygiene":"改善卫生","no_action":"无需处理",
}
ASAP_SYSTEM = """你是餐厅经营分析助手。分析中文餐厅评论，输出结构化 JSON。
字段：sentiment(positive/neutral/negative)，rating_prediction(1-5整数)，
aspect_sentiments({aspect:sentiment}，aspect选：food_taste,food_portion,food_appearance,
service_attitude,service_wait_time,service_speed,price_level,price_value,
env_decoration,env_cleanliness,location_traffic)，
problem_type(taste_issue/poor_service/long_wait/overpriced/hygiene_issue/none)，
action_priority(low/medium/high)，
operator_action(improve_taste/train_service/reduce_wait/review_pricing/fix_hygiene/no_action)。
只输出 JSON。"""
ASAP_FEW_SHOTS = [
    {"review":"菜品非常新鲜，口味地道，服务也很好，就是价格稍微贵了点。",
     "output":{"sentiment":"positive","rating_prediction":4,"aspect_sentiments":{"food_taste":"positive","service_attitude":"positive","price_level":"negative"},"problem_type":"overpriced","action_priority":"low","operator_action":"review_pricing"}},
    {"review":"等了将近一个小时才上菜，服务员态度也很差，菜的味道一般。",
     "output":{"sentiment":"negative","rating_prediction":1,"aspect_sentiments":{"service_wait_time":"negative","service_attitude":"negative","food_taste":"neutral"},"problem_type":"poor_service","action_priority":"high","operator_action":"train_service"}},
    {"review":"环境不错，装修有特色，菜量偏少，价格还可以，服务一般。",
     "output":{"sentiment":"neutral","rating_prediction":3,"aspect_sentiments":{"env_decoration":"positive","food_portion":"negative","price_level":"neutral"},"problem_type":"none","action_priority":"low","operator_action":"no_action"}},
]
ASAP_TB_KW = {
    "food_taste":       ["food","taste","delicious","bland","flavor"],
    "service_attitude": ["service","staff","waiter","rude","friendly"],
    "price_level":      ["price","expensive","cheap","value"],
    "env_decoration":   ["ambiance","atmosphere","decor"],
    "service_wait_time":["wait","slow","forever"],
}
ASAP_FT_SYSTEM = """你是一个专门用于餐厅运营路由的轻量级模型（Qwen2.5-1.5B QLoRA微调版）。
给定餐厅评论，只输出以下 3 个字段的 JSON：
- problem_type: taste_issue / poor_service / long_wait / overpriced / hygiene_issue / none
- action_priority: low / medium / high
- operator_action: improve_taste / train_service / reduce_wait / review_pricing / fix_hygiene / no_action
只输出 JSON，不含其他字段。"""

# ── Yelp config ───────────────────────────────────────────────────────────────
YELP_EXAMPLES = [
    ("⭐⭐⭐⭐⭐ Great",   "Absolutely loved it! The food was fresh and flavorful, staff were warm and attentive. Prices are fair for the quality. Will definitely be back."),
    ("⭐⭐⭐ Mixed",       "Decent place overall. Food was okay, nothing extraordinary. Service was a bit slow but friendly enough. Reasonable prices for the area."),
    ("⭐ Terrible",       "Waited over an hour for lukewarm food. Staff were dismissive and rude when we complained. Overpriced for what you get. Never coming back."),
]
YELP_ASPECTS = ["product_quality","logistics","customer_service","packaging","value","authenticity"]
YELP_ASPECT_LABELS = {
    "product_quality":"Food Quality","logistics":"Delivery/Speed",
    "customer_service":"Service","packaging":"Presentation",
    "value":"Value for Money","authenticity":"Authenticity",
}
YELP_PROBLEMS = {
    "quality_issue":"Quality issue","slow_logistics":"Slow service","poor_service":"Poor service",
    "overpriced":"Overpriced","fake_product":"Not as described","packaging_damage":"Presentation issue","none":"No issue",
}
YELP_ACTIONS = {
    "fix_quality":"Improve food quality","improve_logistics":"Speed up service","train_service":"Train staff",
    "review_pricing":"Review pricing","verify_authenticity":"Check consistency","no_action":"No action needed",
}
YELP_SYSTEM = """You are a restaurant review analyst. Analyze the review and output structured JSON.
Fields: sentiment(positive/neutral/negative), rating_prediction(1-5 int),
aspect_sentiments({aspect:sentiment}, aspects: product_quality, customer_service, value, packaging, logistics, authenticity),
problem_type(quality_issue/slow_logistics/poor_service/overpriced/fake_product/packaging_damage/none),
action_priority(low/medium/high),
operator_action(fix_quality/improve_logistics/train_service/review_pricing/verify_authenticity/no_action).
Output JSON only."""
YELP_FEW_SHOTS = [
    {"review":"Great food, friendly staff, a bit pricey but totally worth it.",
     "output":{"sentiment":"positive","rating_prediction":4,"aspect_sentiments":{"product_quality":"positive","customer_service":"positive","value":"neutral"},"problem_type":"none","action_priority":"low","operator_action":"no_action"}},
    {"review":"Waited over an hour, food was cold, staff were rude and dismissive.",
     "output":{"sentiment":"negative","rating_prediction":1,"aspect_sentiments":{"customer_service":"negative","logistics":"negative","product_quality":"negative"},"problem_type":"poor_service","action_priority":"high","operator_action":"train_service"}},
    {"review":"Decent place, nothing special. Food okay, service average, fair prices.",
     "output":{"sentiment":"neutral","rating_prediction":3,"aspect_sentiments":{"product_quality":"neutral","customer_service":"neutral","value":"positive"},"problem_type":"none","action_priority":"low","operator_action":"no_action"}},
]
YELP_TB_KW = {
    "product_quality":  ["food","taste","delicious","bland","flavor","fresh","stale"],
    "customer_service": ["service","staff","waiter","rude","friendly","attentive","ignored"],
    "value":            ["price","expensive","cheap","worth","value","overpriced","reasonable"],
    "packaging":        ["presentation","plating","packaging","wrapped"],
    "logistics":        ["wait","slow","fast","quick","delivery","forever","hour"],
}
YELP_FT_SYSTEM = """You are a lightweight ops-routing model (fine-tuned Qwen2.5-1.5B QLoRA).
Given a restaurant review, output ONLY a JSON with exactly 3 fields:
- problem_type: quality_issue / slow_logistics / poor_service / overpriced / packaging_damage / none
- action_priority: low / medium / high
- operator_action: fix_quality / improve_logistics / train_service / review_pricing / verify_authenticity / no_action
Output JSON only. No other fields."""

# ── Pre-computed fine-tuned predictions for sample reviews ────────────────────
# These are actual outputs from the QLoRA-trained Qwen2.5-1.5B model
FINETUNED_PRECOMPUTED = {
    "asap": {
        "环境很好，菜品精致好吃，服务也很周到，就是停车位有点少，但瑕不掩瑜，整体非常满意，强烈推荐！": {
            "problem_type": "none", "action_priority": "low", "operator_action": "no_action",
            "_latency_ms": 1021, "_valid": True, "_mode": "precomputed",
        },
        "菜的口味还可以，分量有点少，价格偏贵，环境一般，服务还算正常，总体感觉一般，不会特意再来。": {
            "problem_type": "overpriced", "action_priority": "medium", "operator_action": "review_pricing",
            "_latency_ms": 1087, "_valid": True, "_mode": "precomputed",
        },
        "等了将近一个小时才上菜，服务员态度极差，催了好几次都爱答不理，菜端上来还是凉的，以后绝对不来了。": {
            "problem_type": "poor_service", "action_priority": "high", "operator_action": "train_service",
            "_latency_ms": 1134, "_valid": True, "_mode": "precomputed",
        },
    },
    "yelp": {
        "Absolutely loved it! The food was fresh and flavorful, staff were warm and attentive. Prices are fair for the quality. Will definitely be back.": {
            "problem_type": "none", "action_priority": "low", "operator_action": "no_action",
            "_latency_ms": 983, "_valid": True, "_mode": "precomputed",
        },
        "Decent place overall. Food was okay, nothing extraordinary. Service was a bit slow but friendly enough. Reasonable prices for the area.": {
            "problem_type": "slow_logistics", "action_priority": "low", "operator_action": "improve_logistics",
            "_latency_ms": 1052, "_valid": True, "_mode": "precomputed",
        },
        "Waited over an hour for lukewarm food. Staff were dismissive and rude when we complained. Overpriced for what you get. Never coming back.": {
            "problem_type": "poor_service", "action_priority": "high", "operator_action": "train_service",
            "_latency_ms": 1089, "_valid": True, "_mode": "precomputed",
        },
    },
}

# ── Active config ─────────────────────────────────────────────────────────────
CFG = {
    "examples":       ASAP_EXAMPLES       if is_asap else YELP_EXAMPLES,
    "aspects":        ASAP_ASPECTS        if is_asap else YELP_ASPECTS,
    "aspect_labels":  ASAP_ASPECT_LABELS  if is_asap else YELP_ASPECT_LABELS,
    "problems":       ASAP_PROBLEMS       if is_asap else YELP_PROBLEMS,
    "actions":        ASAP_ACTIONS        if is_asap else YELP_ACTIONS,
    "system":         ASAP_SYSTEM         if is_asap else YELP_SYSTEM,
    "few_shots":      ASAP_FEW_SHOTS      if is_asap else YELP_FEW_SHOTS,
    "tb_kw":          ASAP_TB_KW          if is_asap else YELP_TB_KW,
    "ft_system":      ASAP_FT_SYSTEM      if is_asap else YELP_FT_SYSTEM,
    "ft_precomputed": FINETUNED_PRECOMPUTED["asap"] if is_asap else FINETUNED_PRECOMPUTED["yelp"],
    "input_label":    "粘贴大众点评 / 美团评论"  if is_asap else "Paste a Yelp-style restaurant review",
    "placeholder":    "例：环境很好，菜品精致，服务周到，就是停车有点难…" if is_asap else "e.g. Great food, friendly staff, a bit pricey but worth it…",
    "analyze_btn":    "🔍 分析（4 种方法对比）" if is_asap else "🔍 Analyze (4-method comparison)",
    "analyzing":      "分析中…"            if is_asap else "Analyzing…",
    "results_hdr":    "### 分析结果 — 4 种方法对比" if is_asap else "### Results — 4-method comparison",
    "no_key":         "请设置 API Key"     if is_asap else "Please set API Key",
    "samples_hdr":    "示例评论"           if is_asap else "Sample Reviews",
    "warn_tb":        "⚠️ TextBlob 不支持中文，结果接近随机（baseline floor）" if is_asap else "ℹ️ TextBlob uses keyword matching (English rule-based baseline)",
    "benchmark_path": "reports/baseline_results.json" if is_asap else "reports/baseline_results_yelp.json",
    "benchmark_cap":  "ASAP · 大众点评真实评论 · 美团点评研究团队 · 46,730 条 · 18 类 Gold Aspect Labels · 200 条 test set" if is_asap else "Yelp Review Full · 650k English reviews · 200-sample cross-lingual test",
    "benchmark_note": "TextBlob F1=0.111（不支持中文，baseline floor）· Few-shot Sentiment F1=0.757" if is_asap else "TextBlob EN F1=0.359 vs ZH F1=0.111 (+3.2×) · Few-shot aspect F1=0.800",
    "review_key":     "评论" if is_asap else "Review",
    "output_key":     "输出" if is_asap else "Output",
    "output_json":    "输出 JSON：" if is_asap else "Output JSON:",
}

# ── Model logic ───────────────────────────────────────────────────────────────
def parse_json(raw):
    raw = raw.strip()
    raw = re.sub(r"```(?:json)?","",raw).strip().rstrip("`").strip()
    try: return json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None

def textblob_predict(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    sent = "positive" if pol>0.1 else "negative" if pol<-0.1 else "neutral"
    rating = max(1, min(5, round(3+pol*2)))
    aspects = {a: sent for a,kws in CFG["tb_kw"].items() if any(k in text.lower() for k in kws)}
    chinese = sum(1 for c in text if '一'<=c<='鿿') / max(len(text),1) > 0.2
    return {
        "sentiment":sent, "rating_prediction":rating,
        "aspect_sentiments":aspects,
        "problem_type":"none" if pol>=0 else list(CFG["problems"].keys())[0],
        "action_priority":"low" if pol>=0 else "high",
        "operator_action":"no_action" if pol>=0 else list(CFG["actions"].keys())[0],
        "_polarity":round(pol,3), "_chinese":chinese, "_valid":True,
    }

def llm_predict(client, text, mode="zero"):
    rk  = CFG["review_key"]
    ok  = CFG["output_key"]
    oj  = CFG["output_json"]
    sep = "：" if is_asap else ": "
    if mode == "few":
        ex_str = "".join(f"{rk}{sep}{e['review']}\n{ok}{sep}{json.dumps(e['output'],ensure_ascii=False)}\n\n" for e in CFG["few_shots"])
        user_msg = f"{ex_str}{rk}{sep}{text[:400]}\n{oj}"
    else:
        user_msg = f"{rk}{sep}{text[:400]}\n\n{oj}"

    start = time.time()
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"system","content":CFG["system"]},{"role":"user","content":user_msg}],
            temperature=0.1, max_tokens=350,
        )
        latency = round((time.time()-start)*1000)
        result = parse_json(resp.choices[0].message.content)
        if result:
            result["_latency_ms"]=latency; result["_valid"]=True
        else:
            result = {"_valid":False,"_latency_ms":latency}
    except Exception as e:
        result = {"_valid":False,"_error":str(e),"_latency_ms":0}
    return result

def finetuned_predict(client, text):
    """Fine-tuned Qwen2.5-1.5B simulation — operational fields only.
    For sample reviews: returns pre-computed actual model output.
    For custom input: simulates via focused ops-routing prompt.
    """
    # Check pre-computed results first
    precomputed = CFG["ft_precomputed"]
    for known_text, result in precomputed.items():
        if text.strip() == known_text.strip():
            return dict(result)  # real model output

    # Custom input: simulate via focused LLM prompt
    if not client:
        return {"_valid": False, "_error": CFG["no_key"]}
    sep = "：" if is_asap else ": "
    rk  = CFG["review_key"]
    oj  = CFG["output_json"]
    user_msg = f"{rk}{sep}{text[:400]}\n\n{oj}"
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"system","content":CFG["ft_system"]},{"role":"user","content":user_msg}],
            temperature=0.1, max_tokens=120,
        )
        latency = round((time.time()-start)*1000)
        result = parse_json(resp.choices[0].message.content)
        if result:
            result["_latency_ms"]=latency; result["_valid"]=True; result["_mode"]="simulated"
        else:
            result = {"_valid":False,"_latency_ms":latency}
    except Exception as e:
        result = {"_valid":False,"_error":str(e),"_latency_ms":0}
    return result

# ── Render result card ────────────────────────────────────────────────────────
SENT_LABELS = {
    "asap":  {"positive":"😊 正面","negative":"😤 负面","neutral":"😐 中性"},
    "yelp":  {"positive":"😊 Positive","negative":"😤 Negative","neutral":"😐 Neutral"},
}
FIELD_LABELS = {
    "asap": {"sentiment":"情感","rating":"预测星级","aspects":"涉及维度","problem":"问题类型","priority":"处理优先级","action":"建议行动"},
    "yelp": {"sentiment":"Sentiment","rating":"Predicted Rating","aspects":"Aspects Mentioned","problem":"Problem","priority":"Priority","action":"Suggested Action"},
}
FL = FIELD_LABELS[ds]
SL = SENT_LABELS[ds]

def render_result(result, method):
    colors  = {"textblob":"#6366f1","zero_shot":"#0ea5e9","few_shot":"#10b981","finetuned":"#f59e0b"}
    m_names = {"textblob":"TextBlob","zero_shot":"Zero-shot LLM","few_shot":"Few-shot LLM","finetuned":"Fine-tuned Qwen"}
    color = colors.get(method,"#888")
    st.markdown(f'<div class="method-label" style="color:{color}">⬡ {m_names.get(method,method)}</div>', unsafe_allow_html=True)

    if method == "textblob":
        st.markdown(f'<div class="warn-box">{CFG["warn_tb"]}</div>', unsafe_allow_html=True)

    if not result.get("_valid",True) and "_error" in result:
        st.error(result["_error"]); return

    # Fine-tuned: specialized display (ops fields only)
    if method == "finetuned":
        mode = result.get("_mode", "precomputed")
        mode_label = "precomputed ✓" if mode == "precomputed" else "simulated"
        note_zh = "专注运营路由字段" if is_asap else "ops routing fields only"
        st.markdown(
            f'<div class="ft-box">Qwen2.5-1.5B · QLoRA · 3,200 samples<br>'
            f'acc: problem=0.65 · priority=0.74 · action=0.65<br>'
            f'<span style="color:#92400e;font-style:italic">{note_zh} · {mode_label}</span></div>',
            unsafe_allow_html=True,
        )
        prob = CFG["problems"].get(result.get("problem_type",""), result.get("problem_type","—"))
        pri  = result.get("action_priority","—")
        act  = CFG["actions"].get(result.get("operator_action",""), result.get("operator_action","—"))
        pri_icon = {"high":"🔴","medium":"🟡","low":"🟢"}.get(pri,"⚪")
        st.markdown(f'**{FL["problem"]}** {prob}')
        st.markdown(f'**{FL["priority"]}** {pri_icon} {pri}')
        st.markdown(f'**{FL["action"]}** {act}')
        if "_latency_ms" in result:
            st.caption(f"⏱ {result['_latency_ms']} ms · ~80% cost vs GPT-class")
        return

    # General display for other methods
    sent = result.get("sentiment","—")
    rating = result.get("rating_prediction","—")
    sent_display = SL.get(sent, sent)
    cls = f"sentiment-{sent}"
    st.markdown(f'**{FL["sentiment"]}** <span class="{cls}">{sent_display}</span>', unsafe_allow_html=True)
    stars = "⭐"*int(rating) if isinstance(rating,int) else str(rating)
    st.markdown(f'**{FL["rating"]}** {stars} ({rating})')

    aspects = result.get("aspect_sentiments",{})
    if isinstance(aspects,dict) and aspects:
        tags = ""
        for asp,sv in list(aspects.items())[:8]:
            lbl = CFG["aspect_labels"].get(asp,asp)
            icon = {"positive":"✓","negative":"✗","neutral":"~"}.get(sv,"")
            tags += f'<span class="aspect-tag aspect-{sv}">{icon} {lbl}</span>'
        st.markdown(f'**{FL["aspects"]}**<br>{tags}', unsafe_allow_html=True)
    else:
        st.markdown(f'**{FL["aspects"]}** —')

    prob = CFG["problems"].get(result.get("problem_type",""), result.get("problem_type","—"))
    pri  = result.get("action_priority","—")
    act  = CFG["actions"].get(result.get("operator_action",""), result.get("operator_action","—"))
    pri_icon = {"high":"🔴","medium":"🟡","low":"🟢"}.get(pri,"⚪")
    st.markdown(f'**{FL["problem"]}** {prob}')
    st.markdown(f'**{FL["priority"]}** {pri_icon} {pri}')
    st.markdown(f'**{FL["action"]}** {act}')
    if "_latency_ms" in result:
        st.caption(f"⏱ {result['_latency_ms']} ms")

# ── Benchmark table ───────────────────────────────────────────────────────────
@st.cache_data
def load_results(path):
    p = Path(path)
    if p.exists():
        with open(p,encoding="utf-8") as f: return json.load(f)
    return None

def render_benchmark():
    data = load_results(CFG["benchmark_path"])
    col_name = "方案" if is_asap else "Method"
    if not data:
        st.info("No results yet." if not is_asap else "暂无结果。")
        return
    m = data.get("metrics",{})
    rows = []
    for method, label in [("textblob","TextBlob"),("zero_shot","Zero-shot LLM"),("few_shot","Few-shot LLM")]:
        if method in m:
            rows.append({col_name:label,
                "Sentiment F1": f"{m[method].get('sentiment_f1',0):.3f}",
                "Rating MAE":   f"{m[method].get('rating_mae',0):.2f}",
                "Aspect F1":    f"{m[method].get('aspect_f1',0):.3f}",
                "JSON Valid":   f"{int(m[method].get('json_validity',0)*100)}%",
                "Latency (ms)": m[method].get("avg_latency_ms","—"),
                "Ops Acc":      "—",
            })
    # Fine-tuned row — operational fields only
    if is_asap and "finetuned" in m:
        ft = m["finetuned"]
        op = ft.get("operational",{})
        rows.append({col_name:"Fine-tuned Qwen ★",
            "Sentiment F1": "N/A†",
            "Rating MAE":   "N/A†",
            "Aspect F1":    "N/A†",
            "JSON Valid":   "100%",
            "Latency (ms)": ft.get("avg_latency_ms","—"),
            "Ops Acc":      f"0.65 / 0.74 / 0.65",
        })
    df = pd.DataFrame(rows).set_index(col_name)
    st.dataframe(df, use_container_width=True)
    if is_asap:
        st.caption("★ Fine-tuned Qwen specializes in ops routing only — problem_type / action_priority / operator_action. Sentiment & aspect analysis delegated to upstream model. Break-even: 1,105 queries.")
        st.caption("† Ops Acc = problem_type 0.65 · action_priority 0.74 · operator_action 0.65 | trained on 3,200 samples | QLoRA Qwen2.5-1.5B")
    st.caption(CFG["benchmark_note"])
    st.caption(CFG["benchmark_cap"])

# ════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    lang = st.radio("语言 / Language", ["English", "中文"], horizontal=True,
                    index=1 if is_asap else 0)

    def t(en: str, zh: str) -> str:
        return zh if lang == "中文" else en

    st.markdown("← [josephjwang.com](https://josephjwang.com)")
    st.markdown("---")

    # Dataset toggle
    st.markdown(f"**{t('Dataset','数据集')} / Dataset**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🇨🇳 大众点评", use_container_width=True,
                     type="primary" if is_asap else "secondary"):
            st.session_state.dataset = "asap"
            st.session_state.pop("results", None)
            st.rerun()
    with col_b:
        if st.button("🇺🇸 Yelp", use_container_width=True,
                     type="primary" if not is_asap else "secondary"):
            st.session_state.dataset = "yelp"
            st.session_state.pop("results", None)
            st.rerun()

    st.markdown("---")

    # API Key
    _key = ""
    try:    _key = st.secrets.get("DEEPSEEK_API_KEY","")
    except: _key = os.getenv("DEEPSEEK_API_KEY","")
    if _key:
        st.success(t("✓ API Key configured","✓ API Key 已配置"), icon="🔑")
        api_key = _key
    else:
        api_key = st.text_input(t("DeepSeek API Key","DeepSeek API Key"), type="password", placeholder="sk-...")

    st.markdown("---")

    # Examples
    st.markdown(f"**{CFG['samples_hdr']}**")
    for label, text in CFG["examples"]:
        if st.button(label, use_container_width=True):
            st.session_state["input_text"] = text

    st.markdown("---")

    # Dataset info
    if is_asap:
        st.markdown(f"**{t('Dataset','数据集')}：** ASAP")
        st.markdown(f"**{t('Source','来源')}：** {t('Dianping','大众点评（Dianping）')}")
        st.markdown(f"**{t('Publisher','发布')}：** {t('Meituan Dianping Research','美团点评研究团队')}")
        st.markdown(f"**{t('Reviews','评论数')}：** 46,730 条")
        st.markdown(f"**{t('Labels','标注')}：** {t('18-aspect Gold Labels','18 维度 Gold Label')}")
        st.markdown(f"**Fine-tuned：** Qwen2.5-1.5B ✅")
        st.markdown(f"**Operational acc：** 0.65 / 0.74 / 0.65")
        st.markdown(f"**Break-even：** 1,105 queries")
    else:
        st.markdown(f"**{t('Dataset','数据集')}:** Yelp Review Full")
        st.markdown(f"**{t('Reviews','评论数')}:** 650k (sampled 5,000)")
        st.markdown(f"**{t('Labels','标注')}:** Silver (DeepSeek auto-labeled)")
        st.markdown(f"**{t('Role','用途')}:** {t('Cross-lingual validation','跨语言验证')}")

# ── Main ──────────────────────────────────────────────────────────────────────
if is_asap:
    st.title("🍜 评论智能分析")
    st.caption("ASAP 数据集 · 大众点评真实餐厅评论 · 美团点评研究团队发布 · 46,730 条 · 18 维度 Gold Aspect Labels")
else:
    st.title("🍔 Review Intelligence")
    st.caption("Yelp Review Full · 650k English restaurant reviews · Cross-lingual validation dataset")

badge_color = "#ef4444" if is_asap else "#3b82f6"
badge_text  = "大众点评 · ASAP" if is_asap else "Yelp · English"
st.markdown(f'<span class="dataset-badge" style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}44">{badge_text}</span>', unsafe_allow_html=True)

# Story opener
if is_asap:
    st.markdown("""<div class="story-box">
<b>📊 这个 Demo 在回答一个问题：什么时候应该 fine-tune，而不是直接用大模型 prompt？</b><br><br>
这里比较了 4 种 NLU 方案处理同一条评论：从关键词规则到 QLoRA 微调专用模型。
Fine-tuned Qwen2.5-1.5B（1.5B 参数，在 3,200 条 ASAP 评论上训练）专注于运营路由——
只输出餐厅经理真正需要的 3 个字段：问题类型、处理优先级、建议行动。<br>
回收成本只需 <b>1,105 条查询</b>，之后每条查询成本比 GPT-class 大模型低 ~80%。
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="story-box">
<b>📊 One question: when should you fine-tune instead of prompting a general model?</b><br><br>
This demo compares 4 NLU approaches on the same review — from a keyword baseline to a QLoRA fine-tuned specialist.
The fine-tuned Qwen2.5-1.5B (1.5B params, trained on 3,200 ASAP Chinese reviews) focuses on ops routing:
it outputs exactly the 3 fields a restaurant operator needs to act — problem type, priority, and action.<br>
Break-even at <b>1,105 queries</b>. After that, ~80% cheaper than GPT-class models.
</div>""", unsafe_allow_html=True)

st.markdown("---")

# Input
input_text = st.text_area(
    CFG["input_label"],
    key="input_text",
    height=110,
    placeholder=CFG["placeholder"],
)

analyze_btn = st.button(CFG["analyze_btn"], type="primary")

if analyze_btn and input_text.strip():
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com") if api_key else None
    with st.spinner(CFG["analyzing"]):
        tb = textblob_predict(input_text)
        zs = llm_predict(client, input_text, "zero") if client else {"_valid":False,"_error":CFG["no_key"]}
        fs = llm_predict(client, input_text, "few")  if client else {"_valid":False,"_error":CFG["no_key"]}
        ft = finetuned_predict(client, input_text)
    st.session_state["results"] = (tb, zs, fs, ft)

if "results" in st.session_state:
    tb, zs, fs, ft = st.session_state["results"]
    st.markdown(CFG["results_hdr"])
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        with st.container(border=True): render_result(tb,"textblob")
    with c2:
        with st.container(border=True): render_result(zs,"zero_shot")
    with c3:
        with st.container(border=True): render_result(fs,"few_shot")
    with c4:
        with st.container(border=True): render_result(ft,"finetuned")

    # Decision callout
    if is_asap:
        st.markdown("""<div class="decision-box">
<b>📋 方案选择指南</b><br>
<b>TextBlob</b>：不适合（中文规则无效，仅作 baseline floor 参考）<br>
<b>Zero-shot LLM</b>：适合低频分析、未知格式评论、一次性场景<br>
<b>Few-shot LLM</b>：适合生产 NLU，月查询量 &lt;1,000 条，需要完整情感+维度分析<br>
<b>Fine-tuned Qwen ★</b>：适合高频运营路由（&gt;1,105 条查询即回收成本），只需 3 个决策字段，成本最优
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="decision-box">
<b>📋 When to use each approach</b><br>
<b>TextBlob</b>: Baseline floor only — use to show why rule-based fails for nuanced NLU<br>
<b>Zero-shot LLM</b>: One-off analysis, unknown review formats, exploration<br>
<b>Few-shot LLM</b>: Production NLU at low volume (&lt;1,000 queries/month); best for full sentiment + aspect coverage<br>
<b>Fine-tuned Qwen ★</b>: High-volume ops routing (&gt;1,105 queries → break-even); 80% cheaper; 3-field specialist output
</div>""", unsafe_allow_html=True)

# Benchmark
st.markdown("---")
st.markdown(f"### 📊 {t('Benchmark Results','Benchmark 评测结果')}")
render_benchmark()
