# structlog 结构化日志体系设计

日期：2026-08-01
状态：已批准
目标：为 RAG-EndToEnd-API 引入 structlog 结构化日志（JSON + stdout），并以 request_id 串联异步全链路日志，构成完整可观测性（指标 Prometheus + 追踪 LangSmith + 日志 structlog）。

## 背景

项目当前**零业务日志**（仅有 uvicorn 访问日志与 Prometheus 指标）。排障依赖 docker logs 的裸输出，无法区分请求、无法串联一次问答的完整链路。本次引入 structlog 补齐日志维度，同时作为秋招面试的工程亮点（结构化日志 + request_id 链路 + 可观测性三支柱）。

## 方案（已选定：方案 A — 独立日志模块 + 中间件）

新建 `logger.py` 集中配置，业务模块只依赖 `get_logger()`；main.py 加 request_id 中间件；不做文件落盘（容器场景 stdout 天然被 docker logs 捕获）。

## 组件设计

### 1. `logger.py`（新建，基础设施层）

- `configure_logging()`：
  - processors 链（按 structlog 标准顺序）：
    - `add_log_level` / `add_logger_name`：级别与来源模块
    - `TimeStamper(fmt="iso", utc=True)`：ISO 时间戳
    - `ContextualizedBoundLogger` 上下文合并
    - `ExceptionRenderer` + `TracebackLogger`：异常时输出堆栈
    - `JSONRenderer(serializer=json.dumps, ensure_ascii=False)`：JSON 输出，中文不转义
  - 输出目标：stdout（通过 `logging.StreamHandler` 桥接 stdlib，`wrap_formatter` 或 structlog 标准配置）
  - 根级别 INFO（DEBUG 由 `LOG_LEVEL` 环境变量控制，默认 INFO）
- contextvars：`request_id_var = ContextVar("request_id", default="")`
- `get_logger(name=None)`：返回绑定共享上下文的 logger（绑定 `request_id_var`，供业务模块直接使用）
- uvicorn 访问日志 JSON 化：`logging.getLogger("uvicorn.access")` 挂接同一 JSON handler，访问日志与业务日志格式统一

### 2. `main.py`（修改）

- 新增 `RequestIDMiddleware`（`BaseHTTPMiddleware` 或纯 ASGI 中间件，选 ASGI 原生以规避 BaseHTTPMiddleware 已知坑）：
  - 读取请求头 `X-Request-ID`，存在则透传（前端链路复用），否则生成 `uuid4().hex[:12]`
  - `request_id_var.set(rid)` 绑定上下文
  - 响应头回写 `X-Request-ID`
  - `finally` 中 `request_id_var.reset(token)` 清理，防止上下文泄漏
- lifespan 启动/关闭打 INFO 日志（服务启动、依赖连接结果）
- 请求入口/异常路径补日志（路由 handler 内）
- 注册顺序：`RequestIDMiddleware` 需在 Instrumentator/CORS 之前或之后注册不影响功能，但必须在路由处理前生效

### 3. 业务模块打点清单（全部模块）

| 模块 | 日志点 | 级别 |
|---|---|---|
| `rag_engine.py` | PDF/文本解析开始与完成（chunk 数）、MinerU 输出路径、检索阶段耗时、索引构建/清空、并发锁等待 | INFO / WARNING（失败） |
| `agent_engine.py` | Agent 流程开始/完成（用时、token 用量）、工具调用（tavily/rag）、Postgres 记忆读写 | INFO |
| `redis_cache.py` | 缓存命中/未命中（结果 + 键前缀）、Redis 不可用降级警告 | INFO / WARNING |
| `main.py` | lifespan 启停、上传/问答请求入口、异常捕获 | INFO / ERROR |

统一格式：`logger.info("msg", key=value)`，事件描述用英文短语（如 `"pdf_parse_started"`），上下文用关键字参数。

### 4. 依赖变更

- `requirements-backend.txt` 增加 `structlog==26.1.0`（锁定，构建时解析；与既有锁定策略一致）
- Docker 无需改动：stdout JSON 被 docker logs 天然捕获

## 数据流

```
HTTP 请求 → RequestIDMiddleware（生成/透传 X-Request-ID，绑定 contextvar）
  → 路由 handler（main.py 日志）
    → rag_engine / agent_engine / redis_cache（各模块日志，自动携带 request_id）
  → 响应（回写 X-Request-ID，清理 contextvar）
  → stdout JSON 日志 → docker logs
```

## 错误处理

- contextvar 泄漏防护：中间件 `finally` 中 reset
- 异常日志：ERROR 级别 + `logger.exception()`（ExceptionRenderer 输出堆栈）
- structlog 配置失败不影响服务启动（try/except 回退 stdlib logging）

## 测试计划

1. 本地 `python main.py` 启动，观察 stdout JSON 格式
2. `curl -i` 请求 `/` 与 `/ask`，验证：日志含 `request_id`、响应头回写 `X-Request-ID`、同一次请求所有模块日志 request_id 一致
3. 连续两次请求，确认 request_id 不串（contextvar 清理正确）
4. `/metrics` 端点正常（与 Prometheus 互不干扰）
5. 异常场景（如上传损坏 PDF）确认 ERROR 日志带堆栈

## 不做的事（YAGNI）

- 不做文件轮转/落盘（容器 stdout 足够）
- 不接集中日志平台（Loki/ELK）
- 不接 Streamlit 前端日志
- 不引入 trace 采样/采样率配置
