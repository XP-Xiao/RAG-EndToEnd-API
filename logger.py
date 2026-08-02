"""
structlog 结构化日志基础设施
- JSON 输出到 stdout（容器场景被 docker logs 天然捕获）
- request_id 通过 contextvars 传播：RequestIDMiddleware 写入，
  所有模块的日志事件自动携带同一 request_id（异步上下文安全）
- 业务模块统一入口：get_logger("模块名")，不要直接 import structlog
"""
import json
import logging
import os
import sys

import structlog

# 日志级别由环境变量控制：LOG_LEVEL=DEBUG 可看更细日志（默认 INFO）
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def configure_logging() -> None:
    """应用启动时调用一次，配置 structlog processors 链：
    contextvars 合并 → 级别 → ISO 时间戳 → 异常堆栈 → JSON 渲染（中文不转义）"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # request_id 等 contextvar → 日志字段
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,     # logger.exception() 时输出堆栈
            structlog.processors.JSONRenderer(
                serializer=json.dumps, ensure_ascii=False
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_LOG_LEVEL),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    # 第三方库（uvicorn/langchain）的 stdlib 日志保持默认，不干扰
    logging.basicConfig(level=getattr(logging, _LOG_LEVEL, logging.INFO))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """业务模块统一入口。返回的 logger 每次事件都会从 contextvars
    读取当前 request_id，天然适配异步切换"""
    return structlog.get_logger(name)
