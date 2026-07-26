"""
config.py — 全局配置
修改这里的设置，不需要动其他脚本
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELED_DIR = DATA_DIR / "labeled"
SPLITS_DIR = DATA_DIR / "splits"

# Yelp 路径
YELP_RAW_DIR = DATA_DIR / "yelp" / "raw"
YELP_LABELED_DIR = DATA_DIR / "yelp" / "labeled"

# ASAP 路径
ASAP_RAW_DIR = DATA_DIR / "asap" / "data"
ASAP_PROCESSED_DIR = DATA_DIR / "asap_dataset" / "processed"
ASAP_LABELED_DIR = DATA_DIR / "asap_dataset" / "labeled"
REPORTS_DIR = ROOT / "reports"

# ── API ───────────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# DeepSeek 在 2026 年停用了 "deepseek-chat" 这个名字（调用会直接 400），
# 现在只接受 deepseek-v4-flash / deepseek-v4-pro。flash 是原来 deepseek-chat
# 对应的快而便宜的那一档，所以作为默认值。
# 走环境变量可覆盖：下次上游再改名，改 .env 就行，不用动代码。
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# ── 数据采样 ──────────────────────────────────────────────────────────────────
DATASET_NAME = "yelp_review_full"
DATASET_LANG = None  # yelp 不需要语言参数

# 每个星级采样多少条（1~5星）
# 总量 = 400 * 5 = 2000 条
SAMPLES_PER_STAR = {1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000}
TOTAL_SAMPLES = sum(SAMPLES_PER_STAR.values())

# 随机种子（保证可复现）
RANDOM_SEED = 42

# ── 任务 Schema ───────────────────────────────────────────────────────────────
# 结构化输出的所有合法值
VALID_SENTIMENT = ["positive", "neutral", "negative"]
VALID_ASPECTS = ["product_quality", "logistics", "customer_service",
                 "packaging", "value", "authenticity"]
VALID_PROBLEM_TYPE = ["quality_issue", "slow_logistics", "poor_service",
                      "overpriced", "fake_product", "packaging_damage", "none"]
VALID_ACTION_PRIORITY = ["low", "medium", "high"]
VALID_OPERATOR_ACTION = ["fix_quality", "improve_logistics", "train_service",
                         "review_pricing", "verify_authenticity", "no_action"]

# ── 标注 ──────────────────────────────────────────────────────────────────────
LABEL_BATCH_SIZE = 10       # 每次 API 调用标注多少条（减少请求次数）
LABEL_MAX_RETRIES = 3       # 失败重试次数
LABEL_TEMPERATURE = 0.1     # 低温度保证输出稳定

# ── 数据划分 ──────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
