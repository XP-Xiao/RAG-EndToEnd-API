"""
FastAPI 后端：提供文档上传和问答 API
"""
import os

# 必须在 import rag_engine/agent_engine（会加载 huggingface_hub）之前设置：
# huggingface_hub 在 import 时就读取该变量，事后设置无效，
# 否则启动时会反复请求 huggingface.co 超时重试，卡住几分钟
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import re
import sys
import time
import uuid
import asyncio
from contextlib import asynccontextmanager

from structlog.contextvars import bind_contextvars, clear_contextvars
from logger import configure_logging, get_logger

# Windows 默认 ProactorEventLoop 不被 psycopg 异步模式支持，
# 必须在事件循环创建前切换为 SelectorEventLoop（psycopg 官方要求）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from prometheus_fastapi_instrumentator import Instrumentator
from metrics import ANSWER_CACHE_TOTAL, AGENT_ASK_SECONDS, LLM_TOKENS_TOTAL
from rag_engine import RAGEngine
from agent_engine import ResearchAssistant
from redis_cache import (
    init_llm_cache,
    create_async_client,
    set_async_client,
    aget_cached_answer,
    aset_cached_answer,
    bump_kb_version,
)

# 加载环境变量（DeepSeek API Key 等）
load_dotenv(find_dotenv())

# 结构化日志（JSON + stdout）：必须在创建 app 之前配置
configure_logging()
logger = get_logger("main")

# WebBaseLoader 需要 USER_AGENT
os.environ.setdefault("USER_AGENT", "RAG-EndToEnd-API/1.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：启用 LLM 响应缓存（同步客户端，Redis 不可用时自动降级），
    # 并建立路由层共享的异步客户端（Depends(get_redis) 使用）
    init_llm_cache()
    # 进程重启后 RAG 内存索引已清空，旧答案缓存基于已不存在的知识库，必须作废
    bump_kb_version()
    app.state.redis = create_async_client()
    # 把异步客户端注入缓存模块，供路由热路径的异步答案缓存使用
    set_async_client(app.state.redis)
    # 异步初始化 Agent（AsyncPostgresSaver/Store 必须在事件循环内创建）
    await assistant.astart()
    logger.info("lifespan_startup_complete", redis="connected")
    yield
    # 关闭：释放异步池、同步池（chat_threads 元信息）与 Redis 连接
    await assistant.apool.close()
    await asyncio.to_thread(assistant.pool.close)
    await app.state.redis.aclose()
    logger.info("lifespan_shutdown_complete")


app = FastAPI(title="RAG 文档问答 API", version="1.0", lifespan=lifespan)

# request_id 中间件：为每个 HTTP 请求生成/透传 X-Request-ID，绑定到
# structlog contextvars——一次请求的所有模块日志自动携带同一 ID
class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # 透传调用方传入的 request_id（前端链路复用），否则生成短 ID
        rid = ""
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                rid = value.decode("latin-1")
                break
        if not rid:
            rid = uuid.uuid4().hex[:12]
        bind_contextvars(request_id=rid)

        # 响应头回写 X-Request-ID，方便前端/调试工具对齐链路
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", rid.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # 必须清理，否则 contextvar 泄漏到下一个请求（异步复用同线程）
            clear_contextvars()


app.add_middleware(RequestIDMiddleware)

# CORS 配置：允许 Streamlit 前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus HTTP 层自动埋点：QPS/耗时/状态码按路由分维度，暴露 /metrics 端点。
# 业务指标（metrics.py 定义）注册在同一个全局 Registry，也从这个端点吐出
Instrumentator(
    excluded_handlers=["/metrics"],  # 抓取端点自身不计入请求指标
).instrument(app).expose(app, include_in_schema=False)

# 全局 RAG 引擎实例
rag = RAGEngine()

# 研究助手 Agent（RAG + Web 搜索 + Postgres 长短期记忆）
assistant = ResearchAssistant(rag)


# =============================================
# 请求/响应模型
# =============================================

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户提问")


class SourceDoc(BaseModel):
    index: int
    content: str


class AskResponse(BaseModel):
    answer: str = Field(description="AI 回答")
    sources: list[SourceDoc] = Field(default=[], description="参考文档片段")


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int = Field(description="分割后的文档块数")


class URLRequest(BaseModel):
    url: str = Field(min_length=1, description="网页文档 URL")


class StatusResponse(BaseModel):
    has_index: bool
    doc_count: int
    doc_names: list[str]
    chunk_count: int
    documents: list[dict] = Field(default=[], description="文档列表（含 doc_id、name、chunks）")


class ClearResponse(BaseModel):
    message: str


class DeleteResponse(BaseModel):
    message: str


class AgentAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="用户提问")
    thread_id: str = Field(min_length=1, max_length=64, description="会话 ID（短期记忆隔离）")
    user_id: str = Field(default="default_user", max_length=64, description="用户 ID（长期记忆隔离）")


class AgentStep(BaseModel):
    tool: str
    input: str
    output: str = ""


class AgentUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = Field(default=0, description="DeepSeek 前缀缓存命中的输入 token（按 1 折计费）")


class AgentAskResponse(BaseModel):
    answer: str = Field(description="Agent 回答")
    steps: list[AgentStep] = Field(default=[], description="工具调用轨迹")
    usage: AgentUsage = Field(default_factory=AgentUsage, description="本轮 token 用量（答案缓存命中时为 0）")


class HistoryMessage(BaseModel):
    role: str
    content: str


class AgentHistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    updated_at: str


class ThreadListResponse(BaseModel):
    threads: list[ThreadInfo]


class BalanceResponse(BaseModel):
    available: bool
    currency: str = ""
    total_balance: str = ""


# =============================================
# API 路由
# =============================================

@app.get("/")
async def root():
    return {"message": "RAG 文档问答 API 已启动", "docs": "/docs"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档文件（支持 .txt、.md、.pdf），建立向量索引"""
    # 检查文件类型
    allowed_types = [".txt", ".md", ".pdf"]
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{ext}'，仅支持 {allowed_types}"
        )

    # 读取文件内容
    content = await file.read()

    if not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    if ext == ".pdf":
        # PDF 文件：直接传字节给 MinerU 解析（无需写临时文件）
        chunks = await run_in_threadpool(rag.ingest_pdf, content, filename)
    else:
        # 文本文件：嵌入计算同样耗时，进线程池避免阻塞事件循环
        text = content.decode("utf-8")
        chunks = await run_in_threadpool(rag.ingest_text, text, filename)

    # 知识库变更，旧答案缓存失效
    bump_kb_version()

    return UploadResponse(
        message=f"文档 '{filename}' 上传成功，已建立索引",
        filename=filename,
        chunks=chunks,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """根据已上传的文档回答问题（同步 RAG 链含多次 LLM 调用，必须进线程池，
    否则执行期间整个事件循环被阻塞，所有其他请求卡死）"""
    result = await run_in_threadpool(rag.ask, req.question)
    return AskResponse(**result)


@app.post("/upload_url", response_model=UploadResponse)
async def upload_from_url(req: URLRequest):
    """从网页 URL 加载文档并建立向量索引"""
    url = req.url.strip()

    # 简单 URL 格式校验
    if not re.match(r'^https?://', url):
        raise HTTPException(status_code=400, detail="URL 必须以 http:// 或 https:// 开头")

    try:
        # 网络请求 + 分块 + 嵌入，同步重活进线程池
        chunks = await run_in_threadpool(rag.ingest_url, url, url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法加载网页内容：{str(e)[:200]}")

    if chunks == 0:
        raise HTTPException(status_code=400, detail="网页内容为空或无法解析")

    # 知识库变更，旧答案缓存失效
    bump_kb_version()

    return UploadResponse(
        message=f"网页 '{url}' 加载成功，已建立索引",
        filename=url,
        chunks=chunks,
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取当前文档索引状态"""
    return StatusResponse(**rag.get_status())


@app.post("/clear", response_model=ClearResponse)
async def clear_documents():
    """清空所有文档索引和对话上下文"""
    await run_in_threadpool(rag.clear)
    bump_kb_version()
    return ClearResponse(message="所有文档索引已清空")


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """删除指定文档（同步更新 Milvus 和 BM25 索引）"""
    success = await run_in_threadpool(rag.remove_document, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"文档 '{doc_id}' 不存在")
    bump_kb_version()
    return DeleteResponse(message=f"文档已删除，索引已更新")


# =============================================
# 研究助手 Agent 路由
# =============================================

@app.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(req: AgentAskRequest):
    """研究助手 Agent 对话：先查答案级缓存（GPTCache 模式），命中直接返回；
    未命中走完整 Agent 流程并回写缓存。对话历史按 thread_id 持久化到 PostgreSQL"""
    # 答案级缓存：仅限全新会话的首条消息（多轮对话依赖上下文，不可回放）
    is_new_thread = not await run_in_threadpool(assistant.thread_exists, req.thread_id)
    logger.info("agent_ask_started", thread_id=req.thread_id, user_id=req.user_id, is_new_thread=is_new_thread)
    if is_new_thread:
        cached = await aget_cached_answer(req.question, req.user_id)
        ANSWER_CACHE_TOTAL.labels(result="hit" if cached is not None else "miss").inc()
        if cached is not None:
            logger.info("agent_answer_cache_hit", question_len=len(req.question))
            # 命中：跳过整个 Agent 流程，但把问答补进会话历史，后续多轮不断经
            await assistant.aseed_history(
                req.thread_id, req.user_id, req.question, cached["answer"]
            )
            # 回放不消耗 token，usage 置零
            return AgentAskResponse(
                answer=cached["answer"], steps=cached.get("steps", []), usage=AgentUsage()
            )

    try:
        # 原生异步：模型 HTTP 调用与 checkpointer/store 均不占线程，同步检索工具由框架自动进 executor
        start = time.perf_counter()
        result = await assistant.aask(req.question, req.thread_id, req.user_id)
        AGENT_ASK_SECONDS.observe(time.perf_counter() - start)
    except Exception as e:
        logger.exception("agent_ask_failed", thread_id=req.thread_id, error=str(e)[:300])
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{str(e)[:300]}")

    # token 用量计入监控（回放缓存不走到这里，统计的都是真实消耗）
    usage = result.get("usage", {})
    LLM_TOKENS_TOTAL.labels(type="input").inc(usage.get("input_tokens", 0))
    LLM_TOKENS_TOTAL.labels(type="output").inc(usage.get("output_tokens", 0))
    LLM_TOKENS_TOTAL.labels(type="cache_hit").inc(usage.get("cache_hit_tokens", 0))

    # 回写答案级缓存（只存首条消息的问答，key 带知识库版本号和用户隔离）
    if is_new_thread:
        await aset_cached_answer(req.question, req.user_id, result)
    return AgentAskResponse(**result)


@app.get("/agent/history/{thread_id}", response_model=AgentHistoryResponse)
async def agent_history(thread_id: str):
    """读取指定会话的对话历史（来自 AsyncPostgresSaver 短期记忆）"""
    messages = await assistant.aget_history(thread_id)
    return AgentHistoryResponse(thread_id=thread_id, messages=messages)


@app.get("/agent/threads", response_model=ThreadListResponse)
async def agent_threads(user_id: str = "default_user"):
    """列出指定用户的历史会话（按最近活跃时间倒序）"""
    threads = await run_in_threadpool(assistant.list_threads, user_id)
    return ThreadListResponse(threads=threads)


@app.get("/balance", response_model=BalanceResponse)
async def get_balance():
    """查询 DeepSeek API 账户余额（httpx 原生异步，不占线程）"""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.deepseek.com/user/balance",
                headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
            )
            resp.raise_for_status()
            data = resp.json()
        info = (data.get("balance_infos") or [{}])[0]
        return BalanceResponse(
            available=data.get("is_available", False),
            currency=info.get("currency", ""),
            total_balance=info.get("total_balance", ""),
        )
    except Exception:
        # 余额查询失败不影响主流程，返回不可用状态
        return BalanceResponse(available=False)


if __name__ == "__main__":
    import uvicorn
    # loop="none"：uvicorn 0.36+ 在 Windows 上默认硬编码 ProactorEventLoop（无视策略），
    # 会导致 psycopg 异步池报错；置 none 后回退到 asyncio.new_event_loop()，
    # 从而尊重文件顶部设置的 WindowsSelectorEventLoopPolicy
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="none")
