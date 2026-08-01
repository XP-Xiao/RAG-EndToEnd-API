"""
Prometheus 业务指标定义（prometheus-client 手动埋点）
- HTTP 层通用指标（QPS/耗时/状态码）由 prometheus-fastapi-instrumentator 在 main.py 自动采集
- 本模块只定义业务维度指标：缓存命中、RAG 各阶段耗时、Agent 耗时、token 用量、知识库规模
- 所有指标注册到 prometheus_client 全局默认 Registry，
  与 instrumentator 共用同一个 /metrics 端点，无需额外打通
"""
from prometheus_client import Counter, Histogram, Gauge

# =============================================
# 缓存指标（命中率 = hit / (hit + miss)，PromQL 里用 rate 算）
# =============================================

# 答案级缓存（GPTCache 模式，仅新会话首条消息走这条路径）
ANSWER_CACHE_TOTAL = Counter(
    "rag_answer_cache_requests_total",
    "答案级缓存查询次数",
    ["result"],  # hit / miss
)

# LLM 响应缓存（LangChain set_llm_cache，rerank/查询改写等 temperature=0 场景命中）
LLM_CACHE_TOTAL = Counter(
    "rag_llm_cache_requests_total",
    "LLM 响应缓存查询次数",
    ["result"],  # hit / miss
)

# =============================================
# RAG 流水线各阶段耗时（拆解 /ask 慢在哪一步）
# =============================================

RAG_STAGE_SECONDS = Histogram(
    "rag_stage_seconds",
    "RAG 流水线各阶段耗时（秒）",
    ["stage"],  # generate_queries / hybrid_retrieve / rerank / generate
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# =============================================
# Agent 指标
# =============================================

# 完整 Agent 流程耗时（答案缓存命中时不计，只统计真实执行）
AGENT_ASK_SECONDS = Histogram(
    "agent_ask_seconds",
    "Agent 完整流程耗时（秒，未命中答案缓存）",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0),
)

# token 消耗累计（cache_hit = DeepSeek 前缀缓存命中的输入 token）
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Agent LLM token 消耗累计",
    ["type"],  # input / output / cache_hit
)

# =============================================
# 知识库规模（Gauge：随上传/删除/清空实时变化）
# =============================================

KB_DOCUMENTS = Gauge("kb_documents", "当前知识库文档数")
KB_CHUNKS = Gauge("kb_chunks", "当前知识库文档块数")

# =============================================
# 标签预热：带标签的 Counter 在首次 inc 前不会输出序列，
# 而 Prometheus 的 rate()/increase() 对"从不存在到 1"的第一次计数不计增量，
# 会导致面板长时间 No data。启动时把已知标签组合预热成0，
# 保证所有序列从进程启动起就存在，窗口类查询从第一次真实计数起即准确
# =============================================

for _result in ("hit", "miss"):
    ANSWER_CACHE_TOTAL.labels(result=_result)
    LLM_CACHE_TOTAL.labels(result=_result)

for _type in ("input", "output", "cache_hit"):
    LLM_TOKENS_TOTAL.labels(type=_type)

for _stage in ("generate_queries", "hybrid_retrieve", "rerank", "generate"):
    RAG_STAGE_SECONDS.labels(stage=_stage)
