"""
RAG 核心引擎：负责文档加载、文本分割、向量化存储、检索与生成
支持 RAG-Fusion（多查询）+ BM25/Embedding 混合检索 + LLM Reranker
向量数据库使用 Milvus，支持文档级管理与单独删除
"""
import os
import re
import json
import uuid
import threading
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from metrics import RAG_STAGE_SECONDS, KB_DOCUMENTS, KB_CHUNKS


def clean_text(text: str) -> str:
    """文本清理：去除多余空白、制表符等"""
    text = re.sub(r'\n{3,}', '\n\n', text)   # 多个连续换行 → 两个
    text = re.sub(r' {2,}', ' ', text)          # 多个连续空格 → 一个
    text = re.sub(r'\t+', '', text)              # 去除制表符
    return text.strip()


# Reranker 批处理参数：将候选文档拆成小批并发打分（而非一个超大 prompt）
# 小批：prompt 更短 → LLM 输出更快、JSON 格式遵循度更高；并发：多批同时调用抄低总延迟
RERANK_BATCH_SIZE = 5      # 每批文档数
RERANK_MAX_CONCURRENCY = 5  # 最大并发批数
RERANK_NEUTRAL_SCORE = 5.0  # 某批解析失败时的中性分（保留该批而非丢弃）


class RAGEngine:
    """端到端 RAG 引擎：上传文档 → 建索引 → 混合检索 → LLM 重排 → 回答"""

    def __init__(self):
        # 初始化 Embedding 模型（local_files_only 避免联网检查）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True},
        )
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="deepseek-v4-flash",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com",
            temperature=0,
        )
        # 文本分割器（纯文本按字符数切分）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        )
        # Markdown 语义分块器（PDF / .md 按标题层级切分，保留完整段落）
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#",   "section"),
                ("##",  "subsection"),
                ("###", "subsubsection"),
            ],
        )
        # HTML 语义分块器（网页按标题标签切分）
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "section"),
                ("h2", "subsection"),
                ("h3", "subsubsection"),
                ("h4", "subsubsection"),
            ],
        )
        # Milvus 向量数据库（初始为 None，上传文档后创建）
        self.vectorstore = None
        self.retriever = None
        # 索引写操作互斥锁：写接口均在线程池中真并发执行，
        # 不加锁时并发首次上传会双双走 drop_old=True 互相覆盖数据
        self._index_lock = threading.Lock()
        # Milvus 连接配置
        self._milvus_connection_args = {
            "host": os.getenv("MILVUS_HOST", "localhost"),
            "port": os.getenv("MILVUS_PORT", "19530"),
        }
        self._collection_name = os.getenv("MILVUS_COLLECTION", "rag_collection")
        # BM25 检索器（内存索引，存储所有文档块）
        self._all_documents = []
        self._bm25_retriever = None
        # 文档级追踪：{doc_id: {"name": str, "chunks": int}}
        self._docs: dict[str, dict] = {}
        # LLM Reranker prompt
        self._rerank_prompt = ChatPromptTemplate.from_template(
            """你是一个文档相关性评分专家。请根据用户问题，对以下每个文档段落进行相关性评分（0-10 分，越高越相关）。

请以 JSON 格式返回评分结果，格式如下：
{{"scores": [score1, score2, ...]}}

用户问题：{question}

文档段落：
{documents}

评分："""
        )

    # =============================================
    # 内部方法：索引管理
    # =============================================

    def _update_bm25(self, splits: list):
        """追加文档到 BM25 内存索引"""
        import jieba
        self._all_documents.extend(splits)
        self._bm25_retriever = BM25Retriever.from_documents(
            self._all_documents,
            preprocess_func=jieba.lcut,
        )
        self._bm25_retriever.k = 10

    def _rebuild_bm25(self):
        """从剩余文档重建 BM25 索引"""
        import jieba
        if self._all_documents:
            self._bm25_retriever = BM25Retriever.from_documents(
                self._all_documents,
                preprocess_func=jieba.lcut,
            )
            self._bm25_retriever.k = 10
        else:
            self._bm25_retriever = None

    def _add_to_index(self, splits: list, doc_name: str) -> int:
        """将文档块添加到 Milvus 向量索引和 BM25 索引，返回文档块数（线程安全）"""
        with self._index_lock:
            doc_id = str(uuid.uuid4())[:8]

            # 给每个 chunk 添加元数据
            for split in splits:
                split.metadata["doc_id"] = doc_id
                split.metadata["doc_name"] = doc_name

            # 建立/追加 Milvus 向量索引
            if self.vectorstore is None:
                self.vectorstore = Milvus.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    collection_name=self._collection_name,
                    connection_args=self._milvus_connection_args,
                    drop_old=True,
                )
            else:
                self.vectorstore.add_documents(splits)

            # 更新 BM25
            self._update_bm25(splits)

            # 记录文档
            self._docs[doc_id] = {"name": doc_name, "chunks": len(splits)}

            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            # 知识库规模指标（Gauge 直接取当前真实值，不累加）
            KB_DOCUMENTS.set(len(self._docs))
            KB_CHUNKS.set(len(self._all_documents))
            return len(splits)

    # =============================================
    # 内部方法：检索
    # =============================================

    def _hybrid_retrieve(self, query: str) -> list:
        """BM25 + Embedding 混合检索，合并去重"""
        with RAG_STAGE_SECONDS.labels(stage="hybrid_retrieve").time():
            bm25_docs = self._bm25_retriever.invoke(query) if self._bm25_retriever else []
            embedding_docs = self.retriever.invoke(query) if self.retriever else []

            seen = set()
            merged = []
            for doc in bm25_docs + embedding_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    merged.append(doc)
            return merged

    @staticmethod
    def _parse_scores(result: str, expected: int) -> list[float]:
        """从单批 LLM 输出中解析分数，并对齐到 expected 个（不足补中性分、超出截断）。
        解析失败时返回全中性分，保证该批文档仍参与排序而不是静默丢弃"""
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            scores = json.loads(result[start:end])["scores"]
            scores = [float(s) for s in scores]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return [RERANK_NEUTRAL_SCORE] * expected
        # 对齐数量：防止 LLM 返回的分数个数与文档数不一致导致后续 zip 错位
        if len(scores) < expected:
            scores += [RERANK_NEUTRAL_SCORE] * (expected - len(scores))
        return scores[:expected]

    def _llm_rerank(self, docs: list, question: str, top_n: int = 5) -> list:
        """LLM Reranker：将文档拆成小批并发打分后重排序。
        相较单个超大 prompt：延迟更低（并发）、JSON 更稳（短 prompt）、
        某批解析失败不会拖垓其他批（逐批容错）"""
        if not docs:
            return []

        with RAG_STAGE_SECONDS.labels(stage="rerank").time():
            return self._llm_rerank_inner(docs, question, top_n)

    def _llm_rerank_inner(self, docs: list, question: str, top_n: int) -> list:
        # 拆分成小批，每批独立拼成一个打分请求
        batches = [docs[i:i + RERANK_BATCH_SIZE] for i in range(0, len(docs), RERANK_BATCH_SIZE)]
        inputs = []
        for batch in batches:
            doc_texts = "".join(
                f"\n[Doc {i + 1}]\n{doc.page_content}\n" for i, doc in enumerate(batch)
            )
            inputs.append({"question": question, "documents": doc_texts})

        chain = self._rerank_prompt | self.llm | StrOutputParser()

        # 单批时直接 invoke；多批时 batch 并发（max_concurrency 限制同时请求数）
        if len(inputs) == 1:
            results = [chain.invoke(inputs[0])]
        else:
            results = chain.batch(inputs, config={"max_concurrency": RERANK_MAX_CONCURRENCY})

        # 逐批解析并拼回全量分数，与 docs 严格一一对应
        all_scores = []
        for batch, result in zip(batches, results):
            all_scores.extend(self._parse_scores(result, len(batch)))

        scored_docs = sorted(zip(all_scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_n]]

    # =============================================
    # 公开方法：文档管理
    # =============================================

    def remove_document(self, doc_id: str) -> bool:
        """删除指定文档（同步更新 Milvus 和 BM25，线程安全）"""
        with self._index_lock:
            if doc_id not in self._docs:
                return False

            # 从 Milvus 中删除该文档的所有向量
            try:
                from pymilvus import connections, Collection
                connections.connect(
                    alias="rag_delete",
                    host=self._milvus_connection_args["host"],
                    port=self._milvus_connection_args["port"],
                )
                col = Collection(self._collection_name, using="rag_delete")
                col.delete(expr=f'doc_id == "{doc_id}"')
                connections.disconnect("rag_delete")
            except Exception:
                pass  # Milvus 删除失败不阻塞本地清理

            # 从本地追踪中移除
            self._docs.pop(doc_id)

            # 从内存文档列表中移除该文档的所有 chunk
            self._all_documents = [
                d for d in self._all_documents
                if d.metadata.get("doc_id") != doc_id
            ]

            # 重建 BM25
            self._rebuild_bm25()

            KB_DOCUMENTS.set(len(self._docs))
            KB_CHUNKS.set(len(self._all_documents))
            return True

    def clear(self):
        """清空所有文档索引、检索器和文档记录（线程安全）"""
        with self._index_lock:
            # 尝试删除 Milvus collection
            try:
                from pymilvus import connections, utility
                connections.connect(
                    alias="rag_clear",
                    host=self._milvus_connection_args["host"],
                    port=self._milvus_connection_args["port"],
                )
                if utility.has_collection(self._collection_name, using="rag_clear"):
                    utility.drop_collection(self._collection_name, using="rag_clear")
                connections.disconnect("rag_clear")
            except Exception:
                pass

            self.vectorstore = None
            self.retriever = None
            self._all_documents = []
            self._bm25_retriever = None
            self._docs = {}
            KB_DOCUMENTS.set(0)
            KB_CHUNKS.set(0)

    def get_status(self) -> dict:
        """获取当前文档索引状态"""
        documents = [
            {"doc_id": did, "name": info["name"], "chunks": info["chunks"]}
            for did, info in self._docs.items()
        ]
        return {
            "has_index": self.vectorstore is not None,
            "doc_count": len(self._docs),
            "doc_names": [info["name"] for info in self._docs.values()],
            "chunk_count": len(self._all_documents),
            "documents": documents,
        }

    # =============================================
    # 公开方法：文档入库
    # =============================================

    def ingest_text(self, text: str, doc_name: str = "未命名文档") -> int:
        """将文本内容清理、分割并建立向量索引，返回分割后的文档块数
        .md 文件使用 Markdown 语义分块（按标题切分），其他使用字符级分块
        """
        from langchain_core.documents import Document

        text = clean_text(text)

        # .md 文件有标题结构，按 Markdown 语义分块
        if doc_name.lower().endswith(".md"):
            splits_raw = self.md_splitter.split_text(text)
            splits = [
                Document(
                    page_content=s.page_content,
                    metadata=s.metadata,
                )
                for s in splits_raw if s.page_content.strip()
            ]
        else:
            # .txt 等纯文本：按字符数切分
            doc = Document(page_content=text)
            splits = self.text_splitter.split_documents([doc])

        if not splits:
            raise ValueError("文本分块后无有效内容")

        return self._add_to_index(splits, doc_name)

    def ingest_pdf(self, pdf_bytes: bytes, doc_name: str = "") -> int:
        """使用 MinerU 本地 Python API 解析 PDF，按标题语义分块后建立向量索引，返回文档块数"""
        import shutil
        import tempfile
        from mineru.cli.common import do_parse
        from langchain_core.documents import Document

        filename = os.path.splitext(doc_name)[0] if doc_name else "document"
        output_dir = tempfile.mkdtemp(prefix="mineru_")

        try:
            # Step 1: do_parse 直接接收内存中的 PDF 字节，无需临时文件
            do_parse(
                output_dir=output_dir,
                pdf_file_names=[filename],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=["ch"],
                backend="pipeline",
                parse_method="auto",
            )

            # 读取解析出的 Markdown 文件
            md_path = os.path.join(output_dir, filename, "auto", f"{filename}.md")
            if not os.path.exists(md_path):
                raise FileNotFoundError(f"MinerU 输出文件不存在: {md_path}")

            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

        if not md_text.strip():
            raise ValueError("PDF 解析结果为空")

        # Step 2: 按 Markdown 标题层级语义分块
        splits_raw = self.md_splitter.split_text(md_text)
        chunks = [
            Document(
                page_content=s.page_content,
                metadata=s.metadata,
            )
            for s in splits_raw if s.page_content.strip()
        ]

        if not chunks:
            raise ValueError("PDF 分块后无有效内容")

        # Step 3: 入库（Milvus + BM25）
        return self._add_to_index(
            chunks,
            doc_name=doc_name or "未命名PDF",
        )

    def ingest_url(self, url: str, doc_name: str = "") -> int:
        """从网页 URL 加载文档，按 HTML 标题语义分块后建立向量索引，返回文档块数"""
        import requests
        from langchain_core.documents import Document

        # 直接获取原始 HTML（保留标题结构供分块使用）
        resp = requests.get(
            url,
            headers={"User-Agent": os.getenv("USER_AGENT", "RAG-EndToEnd-API/1.0")},
            timeout=30,
        )
        resp.raise_for_status()
        html_content = resp.text

        # 按 HTML 标题标签（h1/h2/h3/h4）语义分块
        splits_raw = self.html_splitter.split_text(html_content)
        splits = [
            Document(
                page_content=clean_text(s.page_content),
                metadata=s.metadata,
            )
            for s in splits_raw if s.page_content.strip()
        ]

        if not splits:
            raise ValueError("网页分块后无有效内容")

        return self._add_to_index(splits, doc_name or url)

    # =============================================
    # 公开方法：问答
    # =============================================

    def _generate_queries(self, question: str) -> list[str]:
        """RAG-Fusion：根据用户问题生成 4 个相关搜索查询"""
        prompt = ChatPromptTemplate.from_template(
            """你是一个搜索查询改写助手。请根据用户的原始问题，生成 4 个不同角度的中文搜索查询，用于从文档知识库中检索相关内容。

规则：
- 所有查询必须使用中文
- 保留原始问题作为第一个查询
- 其余 3 个查询从不同措辞、同义表达、更具体/更泛化的角度改写
- 每个查询单独一行，共 4 行，不要输出其他任何内容

原始问题：{question}"""
        )
        chain = prompt | self.llm | StrOutputParser() | (lambda x: x.split("\n"))
        with RAG_STAGE_SECONDS.labels(stage="generate_queries").time():
            queries = chain.invoke({"question": question})
        # 过滤空行，并去掉可能的前缀编号（如 "1. "）
        result = []
        for q in queries:
            q = q.strip()
            if len(q) > 2 and q[0].isdigit() and q[1] in ".、":
                q = q[2:].strip()
            if q:
                result.append(q)
        if not result:
            result = [question]
        return result

    def ask(self, question: str) -> dict:
        """混合检索 + LLM Reranker 回答问题，返回答案和参考文档"""
        if self.vectorstore is None or self.retriever is None:
            return {"answer": "请先上传文档建立索引！", "sources": []}

        # Step 1: RAG-Fusion 生成多个搜索查询
        queries = self._generate_queries(question)

        # Step 2: 每个查询分别做 BM25 + Embedding 混合检索
        all_results = []
        for q in queries:
            docs = self._hybrid_retrieve(q)
            all_results.append(docs)

        # Step 3: 合并所有查询结果并去重
        seen = set()
        merged = []
        for docs in all_results:
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    merged.append(doc)

        # Step 4: LLM Reranker 重排序，取 Top 5
        top_docs = self._llm_rerank(merged, question, top_n=5)
        context = "\n\n".join(doc.page_content for doc in top_docs)

        # Step 5: 构建 RAG 链生成回答
        prompt = ChatPromptTemplate.from_template(
            """你是一个文档问答助手。请严格根据以下参考资料来回答用户的问题。

规则：
- 如果参考资料中包含相关信息，请准确提取并组织回答
- 如果参考资料中没有相关信息，请明确说"根据现有文档未找到相关信息"
- 不要编造或推测参考资料中没有的内容
- 回答要简洁清晰，必要时可以使用列表或分点说明

参考资料：
{context}

用户问题：{question}

回答："""
        )

        rag_chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        with RAG_STAGE_SECONDS.labels(stage="generate").time():
            answer = rag_chain.invoke(question)

        # 提取参考来源
        sources = []
        for i, doc in enumerate(top_docs):
            sources.append({
                "index": i + 1,
                "content": doc.page_content[:200],
            })

        return {"answer": answer, "sources": sources}
