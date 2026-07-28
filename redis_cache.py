"""
Redis 接入层：裸 redis-py + lifespan + Depends 的主流封装
- 同步客户端：启动时通过 set_llm_cache 启用 LLM 响应缓存（LangChain 全局生效）
- 异步客户端：挂在 app.state.redis，路由层通过 Depends(get_redis) 使用（锁/限流等）
- 答案级缓存（GPTCache 模式）：入口先查缓存，命中直接返回，跳过整个 Agent 流程；
  key 带知识库版本号，文档增删时自动失效
- 容错：Redis 不可用时服务照常运行，缓存自动降级为直连
"""
import os
import json
import hashlib

import redis
import redis.asyncio as aredis
from fastapi import Request
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "86400"))  # LLM 响应缓存默认保留 1 天
ANSWER_CACHE_TTL = int(os.getenv("ANSWER_CACHE_TTL", "86400"))  # 答案级缓存默认保留 1 天

# 同步客户端单例（LLM 缓存与答案级缓存共用）；不可用时保持 None
_sync_client: redis.Redis | None = None


class SafeRedisCache(RedisCache):
    """容错版 LLM 缓存：Redis 运行中挂掉时异常一律按未命中处理，不拖垮 LLM 调用"""

    def lookup(self, prompt, llm_string):
        try:
            return super().lookup(prompt, llm_string)
        except Exception:
            return None

    def update(self, prompt, llm_string, return_val):
        try:
            super().update(prompt, llm_string, return_val)
        except Exception:
            pass


def init_llm_cache() -> bool:
    """启用 LLM 响应缓存并初始化同步客户端单例。

    用同步客户端：LangChain 的 LLM 调用发生在线程池的同步执行路径里，
    路由层的异步依赖注入拦不到，因此挂全局 set_llm_cache。
    相同 prompt + 模型参数的调用（rerank、查询改写等 temperature=0 场景）直接命中缓存。
    """
    global _sync_client
    try:
        client = redis.Redis.from_url(
            REDIS_URL,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
    except Exception as e:
        print(f"⚠️ Redis 不可用，LLM/答案缓存未启用（服务正常运行）：{e}")
        return False

    _sync_client = client
    set_llm_cache(SafeRedisCache(client, ttl=LLM_CACHE_TTL))
    print(f"✅ LLM Redis 响应缓存已启用（TTL={LLM_CACHE_TTL}s）")
    return True


# =============================================
# 答案级缓存（GPTCache 模式：入口先查，命中直接返回）
# 仅对“全新会话的首条消息”启用；多轮对话依赖上下文，不可缓存
# key 带知识库版本号：文档增删时 bump 版本，旧缓存自动作废
# =============================================

def _answer_key(question: str, user_id: str) -> str | None:
    if _sync_client is None:
        return None
    try:
        ver = _sync_client.get("kb:version") or b"0"
        ver = ver.decode() if isinstance(ver, bytes) else str(ver)
    except Exception:
        return None
    # user_id 进 key：答案可能含长期记忆等用户专属内容，不可跨用户回放
    qhash = hashlib.md5(f"{user_id}|{question.strip()}".encode("utf-8")).hexdigest()
    return f"answer:{ver}:{qhash}"


def bump_kb_version() -> None:
    """知识库变更（上传/删除/清空）时调用，让所有旧答案缓存失效"""
    if _sync_client is None:
        return
    try:
        _sync_client.incr("kb:version")
    except Exception:
        pass


def get_cached_answer(question: str, user_id: str = "default_user") -> dict | None:
    """入口处先查缓存：命中返回 {answer, steps}，未命中/异常返回 None"""
    key = _answer_key(question, user_id)
    if key is None:
        return None
    try:
        raw = _sync_client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def set_cached_answer(question: str, user_id: str, result: dict) -> None:
    """完整流程结束后写入答案缓存"""
    key = _answer_key(question, user_id)
    if key is None:
        return
    try:
        _sync_client.setex(key, ANSWER_CACHE_TTL, json.dumps(result, ensure_ascii=False))
    except Exception:
        pass


def create_async_client() -> aredis.Redis:
    """创建异步客户端（连接池内置），在 lifespan 启动时调用、关闭时 aclose()"""
    return aredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)


def get_redis(request: Request) -> aredis.Redis:
    """FastAPI 依赖注入：路由中通过 r: Redis = Depends(get_redis) 获取客户端"""
    return request.app.state.redis
