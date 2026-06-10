# Review Intelligence — Chinese Restaurant Review Benchmark

**评论智能分析** · 基于美团点评 ASAP 数据集 · 系统对比四种方案

Live demo: [josephwang-review-intelligence-finetuning.streamlit.app](https://josephwang-review-intelligence-finetuning.streamlit.app)

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
| Zero-shot LLM | 0.701 | 0.42 | 0.662 | 100% | 1331ms |
| Few-shot LLM | **0.757** | 0.435 | 0.658 | 100% | 1368ms |
| Fine-tuned Qwen | N/A† | N/A† | N/A† | **100%** | 1093ms |

*† Fine-tuned model specializes in operational fields (problem_type / action_priority / operator_action): accuracy 0.65 / 0.74 / 0.65. Sentiment/rating/aspect use gold labels from ASAP.*

**Cost analysis:** Training cost $1.05 (Colab T4 × 3h). Break-even at **1,105 queries** vs DeepSeek API ($0.001/query → $0.00005/query local).

### Yelp English (200-sample cross-lingual validation)

| Method | Sentiment F1 | Rating MAE | Aspect F1 | JSON Validity | Latency |
|---|---|---|---|---|---|
| TextBlob | 0.359 | 1.07 | 0.62 | 100% | <1ms |
| Zero-shot LLM | 0.699 | 0.435 | 0.721 | 100% | 1249ms |
| Few-shot LLM | 0.642 | 0.47 | **0.800** | 100% | 1175ms |

**Key findings:**
- TextBlob on Chinese: F1=0.111 (near random, English-only rule system)
- TextBlob on English: F1=0.359 (+3.2× vs Chinese) — validates language dependency
- Zero-shot is truly language-agnostic (Chinese 0.701 ≈ English 0.699)
- Few-shot with Chinese examples hurts English sentiment (-0.057) but helps aspect detection (+0.142)

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
| Labeling | DeepSeek API (silver labels, 3 fields) |
| Baselines | TextBlob, DeepSeek zero-shot / few-shot |
| Fine-tuning | QLoRA on Qwen2.5-1.5B · r=16 · 3,200 samples · Colab T4 |
| Demo | Streamlit |
| Evaluation | scikit-learn, numpy |

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

# Launch demo
streamlit run app.py
```

## Project Status

- [x] ASAP data processing (46,730 reviews → 4,000 stratified samples)
- [x] DeepSeek silver labeling (3 fields, 4,000 samples)
- [x] TextBlob / Zero-shot / Few-shot baseline evaluation
- [x] Yelp cross-lingual validation
- [x] Streamlit demo
- [x] QLoRA fine-tuning Qwen2.5-1.5B (r=16, 3 epochs, Colab T4)
- [x] Fine-tuned model evaluation (problem_type 0.65 / action_priority 0.74 / operator_action 0.65)

---

**Author:** Joseph Wang · [josephjwang.com](https://josephjwang.com)
