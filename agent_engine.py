"""
研究助手 Agent 引擎：基于 LangChain 官方 create_agent 构建
- 工具：知识库检索（复用 RAGEngine 混合检索 + 重排）、Tavily Web 搜索、长期记忆读写
- 短期记忆：PostgresSaver（checkpointer，按 thread_id 持久化对话历史）
- 长期记忆：PostgresStore（跨会话存储用户事实，pgvector 语义检索）
"""
import os
import uuid
from dataclasses import dataclass

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from rag_engine import RAGEngine

# PostgreSQL 连接串（短期 + 长期记忆共用一个库）
DB_URI = os.getenv(
    "DB_URI",
    "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable",
)

SYSTEM_PROMPT = """你是一个专业的研究助手，帮助用户查资料、做研究并记住用户的偏好。

你有以下工具可用：
1. search_knowledge_base：检索用户上传的私有文档知识库（优先使用，回答文档相关问题）
2. web_search：联网搜索最新的公开信息（知识库没有答案、或需要时效性信息时使用）
3. save_memory：当用户提到自己的姓名、职业、研究方向、偏好等值得长期记住的事实时，主动保存
4. search_memories：回答涉及用户个人情况的问题前，先检索长期记忆

工作原则：
- 涉及用户上传文档的问题：先查知识库，不足时再联网搜索补充
- 时效性问题（新闻、版本发布、价格等）：直接联网搜索
- 回答要注明信息来源（知识库 / 网络搜索），不要编造
- 用中文回答"""


@dataclass
class AgentContext:
    """运行时上下文：user_id 用于长期记忆的命名空间隔离"""
    user_id: str = "default_user"


class ResearchAssistant:
    """研究助手：create_agent + PostgresSaver（短期）+ PostgresStore（长期）"""

    def __init__(self, rag: RAGEngine):
        self.rag = rag

        # 连接池：PostgresSaver/PostgresStore 要求 autocommit + dict_row
        self.pool = ConnectionPool(
            conninfo=DB_URI,
            max_size=10,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )

        # pgvector 扩展（PostgresStore 语义索引依赖）+ 会话元信息表（历史会话列表）
        with self.pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id  TEXT PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    title      TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                )"""
            )

        # 短期记忆：按 thread_id 持久化对话历史（checkpointer）
        self.checkpointer = PostgresSaver(self.pool)
        self.checkpointer.setup()

        # 长期记忆：跨会话存储，复用 RAG 的嵌入模型做语义检索（MiniLM-L6 = 384 维）
        self.store = PostgresStore(
            self.pool,
            index={"dims": 384, "embed": rag.embeddings},
        )
        self.store.setup()

        llm = ChatOpenAI(
            model="deepseek-v4-flash",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0.2,
        )

        self.agent = create_agent(
            model=llm,
            tools=self._build_tools(),
            system_prompt=SYSTEM_PROMPT,
            context_schema=AgentContext,
            checkpointer=self.checkpointer,
            store=self.store,
        )

    # =============================================
    # 工具定义
    # =============================================

    def _build_tools(self) -> list:
        rag = self.rag

        @tool
        def search_knowledge_base(query: str) -> str:
            """检索用户上传的私有文档知识库，返回最相关的文档片段。

            Args:
                query: 检索查询语句，应使用与文档内容相近的表述
            """
            if rag.vectorstore is None:
                return "知识库为空：用户尚未上传任何文档。请改用 web_search 或直接回答。"
            docs = rag._hybrid_retrieve(query)
            if not docs:
                return "知识库中未检索到相关内容。"
            top_docs = rag._llm_rerank(docs, query, top_n=5)
            parts = []
            for i, doc in enumerate(top_docs):
                name = doc.metadata.get("doc_name", "未知文档")
                parts.append(f"[片段{i + 1} 来源：{name}]\n{doc.page_content}")
            return "\n\n".join(parts)

        # Tavily Web 搜索（需要 TAVILY_API_KEY；未配置时降级为提示工具，不阻塞启动）
        if os.getenv("TAVILY_API_KEY"):
            web_search = TavilySearch(max_results=5, topic="general")
            web_search.description = (
                "联网搜索公开信息，用于知识库没有答案或需要最新时效性信息的问题。"
                "输入应为一个搜索查询语句。"
            )
        else:
            @tool
            def web_search(query: str) -> str:
                """联网搜索公开信息（当前未配置，不可用）。

                Args:
                    query: 搜索查询语句
                """
                return "Web 搜索不可用：服务端未配置 TAVILY_API_KEY。请基于知识库或已有知识回答，并告知用户搜索功能暂不可用。"

        @tool
        def save_memory(content: str, runtime: ToolRuntime[AgentContext]) -> str:
            """保存关于用户的长期记忆（姓名、职业、研究方向、偏好等值得跨会话记住的事实）。

            Args:
                content: 一条简洁的事实描述，例如"用户叫小明，研究方向是多模态大模型"
            """
            namespace = ("memories", runtime.context.user_id)
            runtime.store.put(namespace, str(uuid.uuid4()), {"content": content})
            return f"已保存长期记忆：{content}"

        @tool
        def search_memories(query: str, runtime: ToolRuntime[AgentContext]) -> str:
            """检索关于当前用户的长期记忆，用于回答涉及用户个人情况、偏好的问题。

            Args:
                query: 检索查询，例如"用户的研究方向"
            """
            namespace = ("memories", runtime.context.user_id)
            items = runtime.store.search(namespace, query=query, limit=5)
            if not items:
                return "没有找到相关的长期记忆。"
            return "\n".join(f"- {item.value['content']}" for item in items)

        return [search_knowledge_base, web_search, save_memory, search_memories]

    # =============================================
    # 对外接口
    # =============================================

    def ask(self, question: str, thread_id: str, user_id: str = "default_user") -> dict:
        """执行一轮 Agent 对话，返回答案和工具调用轨迹"""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
            context=AgentContext(user_id=user_id),
        )

        # 注册/刷新会话元信息（首次提问作为会话标题，供历史会话列表展示）
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO chat_threads (thread_id, user_id, title)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (thread_id) DO UPDATE SET updated_at = now()""",
                (thread_id, user_id, question[:50]),
            )

        # 提取本轮的工具调用轨迹（checkpointer 会把历史消息一并返回，
        # 只取最后一条用户消息之后的部分，供前端展示 Agent 本轮推理过程）
        messages = result["messages"]
        last_human_idx = max(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
            default=-1,
        )
        steps = []
        for msg in messages[last_human_idx + 1:]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({"tool": tc["name"], "input": str(tc["args"])[:200]})
            elif isinstance(msg, ToolMessage):
                if steps and "output" not in steps[-1]:
                    steps[-1]["output"] = str(msg.content)[:300]

        # 统计本轮 token 用量（cache_hit_tokens = DeepSeek 前缀缓存命中部分，按 1 折计费）
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_hit_tokens": 0}
        for msg in messages[last_human_idx + 1:]:
            if isinstance(msg, AIMessage) and msg.usage_metadata:
                um = msg.usage_metadata
                usage["input_tokens"] += um.get("input_tokens", 0)
                usage["output_tokens"] += um.get("output_tokens", 0)
                usage["total_tokens"] += um.get("total_tokens", 0)
                details = um.get("input_token_details") or {}
                usage["cache_hit_tokens"] += details.get("cache_read", 0)

        answer = messages[-1].text if messages else ""
        return {"answer": answer, "steps": steps, "usage": usage}

    def thread_exists(self, thread_id: str) -> bool:
        """判断会话是否已有历史（答案级缓存只对全新会话的首条消息生效）"""
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM chat_threads WHERE thread_id = %s", (thread_id,)
            ).fetchone()
        return row is not None

    def seed_history(self, thread_id: str, user_id: str, question: str, answer: str) -> None:
        """答案级缓存命中时调用：把问答写入该会话的 checkpoint 历史，
        保证后续多轮对话上下文连续，并注册会话元信息"""
        config = {"configurable": {"thread_id": thread_id}}
        self.agent.update_state(
            config,
            {"messages": [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]},
        )
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO chat_threads (thread_id, user_id, title)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (thread_id) DO UPDATE SET updated_at = now()""",
                (thread_id, user_id, question[:50]),
            )

    def list_threads(self, user_id: str = "default_user") -> list[dict]:
        """列出该用户的历史会话（按最近活跃时间倒序）"""
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT thread_id, title, updated_at FROM chat_threads
                   WHERE user_id = %s ORDER BY updated_at DESC LIMIT 50""",
                (user_id,),
            ).fetchall()
        return [
            {
                "thread_id": r["thread_id"],
                "title": r["title"],
                "updated_at": r["updated_at"].strftime("%m-%d %H:%M"),
            }
            for r in rows
        ]

    def get_history(self, thread_id: str) -> list[dict]:
        """从 PostgresSaver 读取指定会话的对话历史（供前端恢复会话）"""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.agent.get_state(config)
        messages = state.values.get("messages", []) if state.values else []

        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.text})
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                text = msg.text
                if text.strip():
                    history.append({"role": "assistant", "content": text})
        return history
