# Review Intelligence — Chinese Restaurant Review Benchmark

[![tests](https://github.com/josephwang-ds/review-intelligence-finetuning/actions/workflows/tests.yml/badge.svg)](https://github.com/josephwang-ds/review-intelligence-finetuning/actions/workflows/tests.yml)

**评论智能分析** · 基于美团点评 ASAP 数据集 · 系统对比四种方案 · 从 prompt 到生产部署的完整 LLM 工程实践

Live demo: [josephwang-review-intelligence-finetuning.streamlit.app](https://josephwang-review-intelligence-finetuning.streamlit.app)

The live demo is the Streamlit app only. The FastAPI service, model routing, and monitoring dashboard described in [LLM Engineering](#llm-engineering) below run locally / via Docker — see [ARCHITECTURE.md](ARCHITECTURE.md) for why they're kept separate from the hosted demo.

---

## Business Question

A restaurant operator on Dianping or Meituan receives hundreds of reviews weekly. The question is not "what is the average rating" — it is: which reviews signal an urgent operational problem, what dimension is broken, and what should the operator do first?

## Dataset

**ASAP** — A Chinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction  
Released by Meituan-Dianping Research · Apache-2.0 · [GitHub](https://github.com/Meituan-Dianping/asap)

- 46,730 real Dianping restaurant reviews
- 18 human-annotated aspect categories (food taste, service attitude, wait time, price, environment, etc.)
- Official train / dev / test splits (36,850 / 4,940 / 4,940)

**Yelp Review Full** (cross-lingual validation)
- 650k English restaurant reviews
- 1–5 star ratings
- Used to test cross-lingual generalization

## Task Design

Input: review text (Chinese or English)

Output:
```json
{
  "sentiment": "positive | neutral | negative",
  "rating_prediction": 4,
  "aspect_sentiments": {
    "food_taste": "positive",
    "service_wait_time": "negative"
  },
  "problem_type": "poor_service | overpriced | taste_issue | ...",
  "action_priority": "low | medium | high",
  "operator_action": "train_service | review_pricing | ..."
}
```

`sentiment`, `rating_prediction`, and `aspect_sentiments` use gold labels from ASAP.  
`problem_type`, `action_priority`, `operator_action` use DeepSeek silver labels.

## Benchmark Results

### ASAP Chinese (200-sample test set)

| Method | Sentiment F1 | Rating MAE | Aspect F1 | JSON Validity | Latency |
|---|---|---|---|---|---|
| TextBlob | 0.111 | 1.20 | 0.00 | 100% | <1ms |
| Zero-shot LLM | 0.653 | 0.495 | 0.680 | 99.5% | 5970ms |
| Few-shot LLM | **0.696** | 0.485 | 0.707 | 100% | 5067ms |
| Fine-tuned Qwen | N/A† | N/A† | N/A† | **100%** | 1093ms |

*† Fine-tuned model specializes in operational fields (problem_type / action_priority / operator_action): accuracy 0.65 / 0.74 / 0.65. Sentiment/rating/aspect use gold labels from ASAP.*

*Re-measured 2026-07 on `deepseek-v4-flash` after DeepSeek retired `deepseek-chat` (see [ARCHITECTURE.md](ARCHITECTURE.md#model-migration-notes) — the new model is a reasoning model, which is also why latency roughly quadrupled: reasoning tokens are generated before the answer and billed the same as output). F1 is modestly lower than the original `deepseek-chat` measurement; the fine-tuned model's numbers are untouched since they don't depend on the DeepSeek API.*

**Cost analysis:** Training cost $1.05 (Colab T4 × 3h). Break-even at **1,105 queries** vs DeepSeek API ($0.001/query → $0.00005/query local) — note the per-query API cost assumption predates the latency/token increase above and hasn't been re-derived; treat the break-even count as directional, not exact, until it is.

### Yelp English (200-sample cross-lingual validation)

| Method | Sentiment F1 | Rating MAE | Aspect F1 | JSON Validity | Latency |
|---|---|---|---|---|---|
| TextBlob | 0.359 | 1.07 | 0.62 | 100% | <1ms |
| Zero-shot LLM | 0.720 | 0.44 | 0.710 | 100% | 4322ms |
| Few-shot LLM | **0.735** | 0.42 | **0.824** | 100% | 5719ms |

*Re-measured 2026-07 on `deepseek-v4-flash`, same run as the ASAP table above — both now on the same model, so the cross-lingual comparison below is apples-to-apples again.*

**Key findings:**
- TextBlob on Chinese: F1=0.111 (near random, English-only rule system)
- TextBlob on English: F1=0.359 (+3.2× vs Chinese) — validates language dependency
- Zero-shot is language-agnostic and slightly stronger on English (Chinese 0.653 vs. English 0.720) — plausibly reflects `deepseek-v4-flash`'s training mix rather than a property of the task
- Few-shot helps both languages here (ASAP 0.696, Yelp 0.735) — the earlier "few-shot hurts English" finding was specific to the retired model and didn't hold up on re-measurement; not repeating a claim that no longer reproduces

## Methodology

### Sampling
Stratified sampling: 800 reviews per star level (1–5) from ASAP train set → 4,000 balanced samples.  
Corrects natural 4–5 star dominance (78% of original data).

### Labeling
- **Gold labels**: `rating_prediction` (from original stars) + `aspect_sentiments` (human-annotated in ASAP)
- **Silver labels**: `problem_type`, `action_priority`, `operator_action` via DeepSeek auto-labeling

### Evaluation
- Sentiment F1: macro-averaged F1 across positive/neutral/negative
- Rating MAE: mean absolute error on 1–5 scale
- Aspect F1: micro-F1 on aspect name detection
- JSON Validity: format correctness rate

## Stack

| Layer | Tools |
|---|---|
| Data | ASAP (Meituan-Dianping), Yelp Review Full |
| Labeling | DeepSeek API (`deepseek-v4-flash`, silver labels, 3 fields) |
| Baselines | TextBlob, DeepSeek zero-shot / few-shot |
| Fine-tuning | QLoRA on Qwen2.5-1.5B · r=16 · 3,200 samples · Colab T4 |
| Structured output | Pydantic (schema-validated, repair-on-failure) |
| Guardrails / review | Custom PII redaction, prompt-injection heuristic, consistency checks, SQLite review queue |
| Serving | FastAPI, uvicorn, Docker / docker-compose |
| Model routing | transformers + peft (local QLoRA inference, MPS) vs. DeepSeek API |
| Testing | pytest — unit tests + golden regression eval set |
| Demo | Streamlit (4-method comparison, review queue, monitoring dashboard) |
| Evaluation | scikit-learn, numpy |

## LLM Engineering

The benchmark above answers *which method wins*. The rest of this repo answers a different question: what does it take to actually run one of these as a system, not a notebook cell? Six pieces, each backed by code and tests — full design discussion and diagrams in [ARCHITECTURE.md](ARCHITECTURE.md).

| Concern | What's actually there | Where |
|---|---|---|
| **Structured output** | Pydantic schema — not regex-extracted JSON — with dataset-aware validation and an automatic repair-retry when the model's output doesn't validate | [`src/schema.py`](src/schema.py), [`src/structured_client.py`](src/structured_client.py) |
| **Prompt engineering** | Prompts generated from one schema per dataset instead of hand-copied per script — the class of bug this replaces: one script's prompt was silently missing two valid `problem_type` values that every other script had, found and fixed during this refactor | [`src/prompts.py`](src/prompts.py) |
| **Guardrails + human review** | Real PII redaction (phone/email/ID, before the text ever reaches the model — verified, not just displayed masked), a prompt-injection heuristic, business-logic consistency checks, and an actual SQLite-backed review queue with an approve / correct / reject UI | [`src/guardrails.py`](src/guardrails.py), [`src/review_queue.py`](src/review_queue.py), [`pages/1_Review_Queue.py`](pages/1_Review_Queue.py) |
| **Eval framework** | Fast no-network unit tests for the pure logic, plus a curated ~30-case golden regression set run against the real API — the thing that would actually catch a behavior regression when the prompt changes, which a mock can't | [`tests/`](tests/), [`eval/`](eval/) |
| **Deployment architecture** | A FastAPI service wrapping the same engine behind `POST /analyze`, with real request logging, containerized — deliberately independent of the Streamlit demo (see ARCHITECTURE.md for why) | [`api/main.py`](api/main.py), [`Dockerfile.api`](Dockerfile.api), [`docker-compose.yml`](docker-compose.yml) |
| **Model routing** | Routes between the real local fine-tuned model (actually served, not simulated via a persona prompt) and the DeepSeek few-shot call, based on what the caller needs and input complexity, with graceful fallback on failure | [`src/router.py`](src/router.py), [`src/local_model.py`](src/local_model.py) |
| **Monitoring** | Dashboard reading genuine SQLite request logs — including guardrail trigger rate, a metric that's meaningless without a real total-request denominator | [`pages/2_Monitoring.py`](pages/2_Monitoring.py), [`src/request_log.py`](src/request_log.py) |

```bash
pytest   # fast unit tests always run; eval/ (real API calls) skips cleanly without DEEPSEEK_API_KEY
```

## Quickstart

```bash
git clone https://github.com/josephwang-ds/review-intelligence-finetuning.git
cd review-intelligence-finetuning

pip install -r requirements.txt

cp .env.example .env
# Add your DEEPSEEK_API_KEY

# Download and process ASAP data
git clone https://github.com/Meituan-Dianping/asap.git data/asap
python src/01_load_asap.py

# Label 3 supplementary fields (costs ~$0.06)
python src/02_label_asap.py

# Run baseline evaluation
python src/03_run_baselines.py

# Launch demo (includes Review Queue + Monitoring pages)
streamlit run app.py
```

### API / tests / model routing (optional — heavier dependencies)

```bash
pip install -r requirements-api.txt   # adds pytest, fastapi, torch, transformers, peft on top of requirements.txt

pytest                                        # unit tests always run; eval/ needs DEEPSEEK_API_KEY
uvicorn api.main:app --reload --port 8000     # http://localhost:8000/docs

python src/07_seed_dashboard_traffic.py       # replay real reviews through /analyze to populate the Monitoring dashboard

docker compose up                             # api (:8000) + streamlit (:8501), containerized
```

## Project Status

- [x] ASAP data processing (46,730 reviews → 4,000 stratified samples)
- [x] DeepSeek silver labeling (3 fields, 4,000 samples)
- [x] TextBlob / Zero-shot / Few-shot baseline evaluation
- [x] Yelp cross-lingual validation
- [x] Streamlit demo
- [x] QLoRA fine-tuning Qwen2.5-1.5B (r=16, 3 epochs, Colab T4)
- [x] Fine-tuned model evaluation (problem_type 0.65 / action_priority 0.74 / operator_action 0.65)
- [x] Structured output — Pydantic schema + repair-retry (`src/schema.py`, `src/structured_client.py`)
- [x] Guardrails + human review queue (`src/guardrails.py`, `src/review_queue.py`)
- [x] Eval framework — unit tests + golden regression set (`tests/`, `eval/`)
- [x] Deployment architecture — FastAPI service + Docker (`api/`, `ARCHITECTURE.md`)
- [x] Model routing — real local fine-tuned model vs. DeepSeek, decided by task + complexity (`src/router.py`, `src/local_model.py`)
- [x] Monitoring dashboard reading real request logs (`pages/2_Monitoring.py`)

---

**Author:** Joseph Wang · [josephjwang.com](https://josephjwang.com)
