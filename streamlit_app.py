"""
Streamlit 前端：研究助手对话界面（统一 Agent）
支持 .txt/.md/.pdf 上传、网页加载、文档级管理（单独删除）
Agent 自主选择知识库检索 / Web 搜索 / 长期记忆，对话历史持久化到 PostgreSQL
启动方式：streamlit run streamlit_app.py
"""
import os
import uuid
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG 文档问答", page_icon="📄", layout="wide")

# =============================================
# 全局样式
# =============================================
st.markdown("""
<style>
/* ==================== 全局重置 ==================== */
#MainMenu, footer, .stDeployButton {
    display: none !important;
}
/* 保留 header（含侧边栏切换按钮），但去除背景 */
header[data-testid="stHeader"] {
    background: transparent !important;
}
header[data-testid="stHeader"] .stToolbarActions {
    display: none !important;
}
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}
/* 滚动条 */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #c5cce0; border-radius: 3px; }
::-webkit-scrollbar-track { background: transparent; }

/* ==================== 顶部导航栏 ==================== */
.top-nav {
    background: #fff;
    padding: 0.55rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #f0f0f0;
    margin: 0 -1rem 0.8rem -1rem;
}
.top-nav .brand {
    display: flex; align-items: center; gap: 10px;
}
.top-nav .logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #e94560, #c23152);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; color: #fff;
}
.top-nav .title {
    color: #1a1a2e; font-size: 1rem; font-weight: 600;
    letter-spacing: -0.2px;
}
.top-nav .subtitle {
    color: #999; font-size: 0.68rem;
    margin-top: 1px; letter-spacing: 0.3px;
}
.top-nav .status-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.7rem; font-weight: 500;
}
.status-online {
    background: #f6ffed; color: #52c41a;
    border: 1px solid #b7eb8f;
}
.status-offline {
    background: #fff2f0; color: #ff4d4f;
    border: 1px solid #ffccc7;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    display: inline-block;
}
.status-online .status-dot { background: #52c41a; }
.status-offline .status-dot { background: #ff4d4f; }

/* ==================== 欢迎区域 ==================== */
.welcome-area {
    text-align: center;
    padding: 4rem 2rem 2rem;
}
.welcome-area .emoji { font-size: 3.5rem; margin-bottom: 1rem; }
.welcome-area h2 {
    font-size: 1.6rem; font-weight: 700; color: #1a1a2e;
    margin: 0 0 0.5rem;
}
.welcome-area .desc {
    font-size: 0.95rem; color: #888; margin: 0 auto 2rem;
    max-width: 460px; line-height: 1.6;
}
.suggestion-grid {
    display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;
    max-width: 600px; margin: 0 auto;
}
.suggestion-card {
    flex: 1; min-width: 160px; max-width: 180px;
    background: #fff; border: 1px solid #e8ecf4;
    border-radius: 14px; padding: 1.1rem 0.9rem;
    cursor: default; transition: all 0.2s;
    text-align: center;
}
.suggestion-card:hover {
    border-color: #c23152; box-shadow: 0 4px 16px rgba(194,49,82,0.1);
    transform: translateY(-2px);
}
.suggestion-card .s-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
.suggestion-card .s-title {
    font-size: 0.82rem; font-weight: 600; color: #1a1a2e;
    margin-bottom: 0.2rem;
}
.suggestion-card .s-desc {
    font-size: 0.72rem; color: #999; line-height: 1.4;
}

/* ==================== 侧边栏 ==================== */
section[data-testid="stSidebar"] {
    background: #f7f8fc;
    border-right: 1px solid #e8ecf4;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0.2rem; padding-bottom: 1rem;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 1rem; font-weight: 700; color: #1a1a2e;
    margin: 0.3rem 0 0.4rem; text-transform: none;
    letter-spacing: 0;
}
section[data-testid="stSidebar"] .stMarkdown h4 {
    font-size: 0.68rem; font-weight: 600; color: #999;
    margin: 0.4rem 0 0.2rem; text-transform: uppercase;
    letter-spacing: 0.8px;
}
section[data-testid="stSidebar"] .stMarkdown p { margin: 0; }

/* 知识库指标 */
.kb-stats {
    display: flex; gap: 8px; margin: 0.5rem 0;
}
.kb-stat {
    flex: 1; background: #fff; border: 1px solid #e8ecf4;
    border-radius: 12px; padding: 0.7rem; text-align: center;
    transition: all 0.15s;
}
.kb-stat:hover { border-color: #c23152; }
.kb-stat .num {
    font-size: 1.5rem; font-weight: 800;
    background: linear-gradient(135deg, #e94560, #c23152);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.kb-stat .lbl {
    font-size: 0.68rem; color: #999; margin-top: 2px;
    text-transform: uppercase; letter-spacing: 0.5px;
}

/* 文档卡片 */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #e8ecf4 !important;
    border-radius: 12px !important;
    background: #fff;
    padding: 0.6rem 0.85rem !important;
    margin-bottom: 0.4rem !important;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #c23152 !important;
    box-shadow: 0 3px 12px rgba(194,49,82,0.08);
}
/* 文档名单行截断省略 */
.doc-name {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    display: block !important;
    max-width: 100% !important;
    font-weight: 600 !important;
    color: #1a1a2e !important;
    line-height: 1.3 !important;
    margin: 0 0 2px !important;
}
.doc-name strong {
    font-weight: 600 !important;
}
/* 文档类型标签 */
.doc-type {
    display: inline-block !important;
    padding: 1px 5px !important;
    border-radius: 4px !important;
    background: #f0f2f8 !important;
    color: #666 !important;
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    line-height: 1.2 !important;
    vertical-align: middle !important;
}
/* 修复 caption 被负 margin 截断 */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.72rem !important;
    color: #999 !important;
    line-height: 1.3 !important;
}
/* 文档卡片列容器允许收缩，防止文字把列撑开 */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    min-width: 0 !important;
    overflow: hidden !important;
}

/* 侧边栏基础按钮 */
section[data-testid="stSidebar"] .stButton > button {
    padding: 0.25rem 0.5rem !important;
    font-size: 0.78rem !important;
    line-height: 1.4 !important;
    min-height: 1.6rem !important;
    height: auto !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 6px !important;
    transition: all 0.15s !important;
    color: #666 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #f5f5f5 !important;
    color: #333 !important;
}
/* 删除按钮 — tertiary 红色边框 */
section[data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
    color: #ff4d4f !important;
    border: 1px solid #ff4d4f !important;
    background: transparent !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-decoration: none !important;
    padding: 0.2rem 0.6rem !important;
    white-space: nowrap !important;
    border-radius: 4px !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="tertiary"] * {
    text-decoration: none !important;
    white-space: nowrap !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
    color: #fff !important;
    background: #ff4d4f !important;
    text-decoration: none !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    width: 100% !important;
    height: auto !important;
    min-height: 2.4rem !important;
    min-width: unset !important;
    padding: 0.55rem !important;
    font-size: 0.84rem !important;
    border-radius: 8px !important;
    background: linear-gradient(135deg, #e94560, #c23152) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(233,69,96,0.2) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(233,69,96,0.35) !important;
    transform: translateY(-1px);
    background: linear-gradient(135deg, #d63851, #b02a45) !important;
}

/* Tab 标签 — Ant Design Segmented 风格 */
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #f0f1f5; border-radius: 8px; padding: 3px;
    border: none;
}
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
    border-radius: 6px; padding: 6px 14px; font-size: 0.82rem;
    font-weight: 500; transition: all 0.2s;
    border: none;
}
section[data-testid="stSidebar"] .stTabs [aria-selected="true"] {
    background: #fff;
    color: #1a1a2e !important;
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: none;
}
section[data-testid="stSidebar"] .stTabs [aria-selected="false"] {
    color: #999 !important;
    background: transparent;
}
section[data-testid="stSidebar"] .stTabs [aria-selected="false"]:hover {
    color: #555 !important;
}
/* 隐藏 tab 下方的指示线 */
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* 上传区域 */
section[data-testid="stSidebar"] .stFileUploader {
    border: 2px dashed #d0d7e8; border-radius: 14px;
    padding: 8px; background: #fafbfe;
}

/* 分割线 */
section[data-testid="stSidebar"] hr {
    border-color: #e8ecf4; margin: 0.7rem 0;
}

/* 底部操作按钮 */
section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):not([kind="tertiary"]) {
    width: 100% !important;
    height: auto !important;
    min-height: 2rem !important;
    min-width: unset !important;
    font-size: 0.78rem !important;
    color: #666 !important;
    padding: 0.35rem 0.6rem !important;
    border: 1px solid #e8ecf4 !important;
    background: #fff !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):not([kind="tertiary"]):hover {
    border-color: #c23152 !important;
    color: #c23152 !important;
    background: #fff !important;
}

/* ==================== 聊天区域 ==================== */
/* 主内容列 - 居中限宽 */
.main-content-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* 用户消息 */
[data-testid="stChatMessageUser"] > div {
    background: #1a1a2e;
    color: #fff; border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
[data-testid="stChatMessageUser"] [data-testid="chatAvatarIcon-user"] {
    background: #e94560;
}
[data-testid="stChatMessageUser"] p { color: #fff !important; }

/* 助手消息 */
[data-testid="stChatMessageAssistant"] > div {
    background: #fafafa;
    border-radius: 12px 12px 12px 2px;
    padding: 0.75rem 1rem;
    border: 1px solid #f0f0f0;
}

/* 来源片段 */
.stExpander {
    border: 1px solid #e8ecf4 !important;
    border-radius: 12px !important;
    background: #fafbfe;
    margin-top: 0.6rem !important;
}
.stExpander summary { padding: 0.4rem 0.6rem; }
.stExpander summary span { font-size: 0.82rem; font-weight: 500; color: #0f3460; }

/* ==================== 底部输入栏 ==================== */
.stChatInputContainer {
    border-top: 1px solid #eef0f5;
    background: #fafbfe;
}

/* ==================== 空状态提示 ==================== */
.empty-hint {
    display: flex; align-items: center; justify-content: center;
    gap: 8px; padding: 0.8rem 1.2rem;
    background: #f0f2f8; border-radius: 12px;
    margin: 0.5rem auto; max-width: 400px;
    font-size: 0.82rem; color: #888;
}
.empty-hint .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #c23152; display: inline-block;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ==================== 来源标签 ==================== */
.source-tag {
    display: inline-block;
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    color: #fff; padding: 2px 10px; border-radius: 6px;
    font-size: 0.72rem; font-weight: 600; margin-right: 4px;
}

/* ==================== 底部状态条（余额 + 会话 token，固定在输入框正下方） ==================== */
/* 把输入框容器整体上抬，腾出底部一条缝放状态条 */
div[data-testid="stBottom"] {
    bottom: 26px !important;
}
.usage-pill {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 26px;
    line-height: 26px;
    z-index: 90;
    text-align: center;
    font-size: 0.72rem;
    color: #888;
    background: #fafbfe;
    border-top: 1px solid #eef0f5;
    pointer-events: none;
    white-space: nowrap;
}
.usage-pill b { color: #c23152; }

/* 侧边栏整体不滚动，只允许内部限高容器（会话/文档列表）自己滑动 */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    overflow: hidden !important;
}

/* 侧边栏内部滚动容器（会话列表/文档列表）去掉多余边框 */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# =============================================
# session state
# =============================================
if "documents" not in st.session_state:
    st.session_state.documents = []
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "url_key" not in st.session_state:
    st.session_state.url_key = 0
# 会话状态（thread_id 对应后端 PostgresSaver 短期记忆）
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "agent_thread_id" not in st.session_state:
    st.session_state.agent_thread_id = uuid.uuid4().hex[:16]
if "agent_user_id" not in st.session_state:
    st.session_state.agent_user_id = "default_user"
# 本会话累计 token 用量（新会话/切换会话时重置，只统计当前页面内新产生的轮次）
if "session_usage" not in st.session_state:
    st.session_state.session_usage = {"input": 0, "output": 0, "total": 0, "cache_hit": 0}

# =============================================
# 辅助函数
# =============================================
def fetch_documents():
    try:
        resp = requests.get(f"{API_URL}/status", timeout=5)
        if resp.status_code == 200:
            st.session_state.documents = resp.json().get("documents", [])
            st.session_state.backend_online = True
        else:
            st.session_state.backend_online = False
    except requests.ConnectionError:
        st.session_state.backend_online = False

def delete_doc(doc_id: str) -> bool:
    try:
        resp = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=15)
        if resp.status_code == 200:
            st.session_state.documents = [
                d for d in st.session_state.documents if d["doc_id"] != doc_id
            ]
            return True
        st.error(f"删除失败：{resp.json().get('detail', '未知错误')}")
    except requests.ConnectionError:
        st.error("无法连接后端")
    return False

@st.cache_data(ttl=60)
def fetch_balance():
    """查询 DeepSeek 余额（60 秒缓存，避免每次 rerun 都请求）"""
    try:
        resp = requests.get(f"{API_URL}/balance", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        pass
    return None

def add_usage(usage: dict):
    """把本轮 token 用量累加到会话统计"""
    u = st.session_state.session_usage
    u["input"] += usage.get("input_tokens", 0)
    u["output"] += usage.get("output_tokens", 0)
    u["total"] += usage.get("total_tokens", 0)
    u["cache_hit"] += usage.get("cache_hit_tokens", 0)

def usage_caption(usage: dict) -> str:
    """生成单轮 token 用量的展示文案"""
    if not usage or usage.get("total_tokens", 0) == 0:
        return "⚡ 命中答案缓存，本轮 0 token"
    return (
        f"🔢 本轮 {usage['total_tokens']} tokens（输入 {usage['input_tokens']} · 输出 {usage['output_tokens']}"
        f" · 前缀缓存命中 {usage.get('cache_hit_tokens', 0)}）"
    )

if "status_synced" not in st.session_state:
    fetch_documents()
    st.session_state.status_synced = True

has_docs = len(st.session_state.documents) > 0
total_chunks = sum(d.get("chunks", 0) for d in st.session_state.documents)
backend_online = st.session_state.get("backend_online", False)

# =============================================
# 顶部导航栏
# =============================================
status_cls = "status-online" if backend_online else "status-offline"
status_text = "服务在线" if backend_online else "服务离线"
st.markdown(f"""
<div class="top-nav">
    <div class="brand">
        <div class="logo">🔬</div>
        <div>
            <div class="title">研究助手</div>
            <div class="subtitle">RAG 知识库 · Web 搜索 · 长短期记忆（PostgreSQL）</div>
        </div>
    </div>
    <span class="status-badge {status_cls}">
        <span class="status-dot"></span> {status_text}
    </span>
</div>
""", unsafe_allow_html=True)

# =============================================
# 侧边栏
# =============================================
with st.sidebar:
    # 会话管理模块（历史列表限高滚动，不挤压下方知识库区域）
    st.markdown("### 💬 会话")
    if st.button("🆕 新会话", use_container_width=True, type="primary", help="开启新的会话，旧会话自动保存在下方列表"):
        st.session_state.agent_thread_id = uuid.uuid4().hex[:16]
        st.session_state.agent_messages = []
        st.session_state.session_usage = {"input": 0, "output": 0, "total": 0, "cache_hit": 0}
        st.rerun()

    # 历史会话列表（来自 PostgreSQL chat_threads 表，点击切换并恢复对话）
    try:
        resp = requests.get(
            f"{API_URL}/agent/threads",
            params={"user_id": st.session_state.agent_user_id},
            timeout=10,
        )
        threads = resp.json().get("threads", []) if resp.status_code == 200 else []
    except requests.ConnectionError:
        threads = []

    if not threads:
        st.caption("暂无历史会话，发送第一条消息后自动保存")
    else:
        # 固定高度滚动容器：会话再多也只在框内滑动
        with st.container(height=200):
            for t in threads:
                is_current = t["thread_id"] == st.session_state.agent_thread_id
                label = f"{'🟢 ' if is_current else '💬 '}{t['title'][:18]}・{t['updated_at']}"
                if st.button(label, key=f"thread_{t['thread_id']}", use_container_width=True, disabled=is_current):
                    st.session_state.agent_thread_id = t["thread_id"]
                    st.session_state.session_usage = {"input": 0, "output": 0, "total": 0, "cache_hit": 0}
                    try:
                        hist = requests.get(f"{API_URL}/agent/history/{t['thread_id']}", timeout=15)
                        st.session_state.agent_messages = (
                            hist.json().get("messages", []) if hist.status_code == 200 else []
                        )
                    except requests.ConnectionError:
                        st.session_state.agent_messages = []
                    st.rerun()

    st.markdown("---")
    st.markdown("### 📚 知识库")

    # 指标统计
    if has_docs:
        st.markdown(f"""
        <div class="kb-stats">
            <div class="kb-stat">
                <div class="num">{len(st.session_state.documents)}</div>
                <div class="lbl">文档</div>
            </div>
            <div class="kb-stat">
                <div class="num">{total_chunks}</div>
                <div class="lbl">文档块</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("暂无文档，请先上传文件或添加网页链接")

    # 文档列表（超过 3 个时限高滚动，不撑长侧边栏）
    if has_docs:
        st.markdown("#### 文档列表")
        doc_area = st.container(height=170) if len(st.session_state.documents) > 2 else st.container()
        with doc_area:
            for doc in st.session_state.documents:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1], vertical_alignment="center")
                    with c1:
                        ext = doc['name'].rsplit('.', 1)[-1].upper() if '.' in doc['name'] else 'FILE'
                        icon = {"PDF": "📕", "MD": "📝", "TXT": "📃"}.get(ext, "📎")
                        st.markdown(
                            f'<div class="doc-name" title="{doc["name"]}">{icon} <strong>{doc["name"]}</strong></div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f'<span class="doc-type">{ext}</span> · {doc.get("chunks", 0)} 个文档块',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        if st.button("删除", key=f"del_{doc['doc_id']}", help=f"删除 {doc['name']}", use_container_width=True, type="tertiary"):
                            if delete_doc(doc["doc_id"]):
                                st.toast(f"已删除 {doc['name']}")
                                st.rerun()

    # 添加文档（折叠展开器，节省侧边栏竖向空间；无文档时默认展开）
    st.markdown("---")
    with st.expander("➕ 添加文档", expanded=not has_docs):
        tab1, tab2 = st.tabs(["📁 本地文件", "🔗 网页链接"])

        with tab1:
            uploaded_file = st.file_uploader(
                "支持 .txt / .md / .pdf",
                type=["txt", "md", "pdf"],
                label_visibility="collapsed",
                key=f"uploader_{st.session_state.upload_key}",
            )
            if uploaded_file:
                if st.button("📤 上传并建立索引", key="btn_file", use_container_width=True, type="primary"):
                    with st.spinner("正在上传并建立索引..."):
                        try:
                            resp = requests.post(
                                f"{API_URL}/upload",
                                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                                timeout=300,
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                fetch_documents()
                                st.session_state.upload_key += 1
                                st.success(f"已索引 **{data['chunks']}** 个文档块")
                                st.rerun()
                            else:
                                st.error(f"上传失败：{resp.json().get('detail', '未知错误')}")
                        except requests.ConnectionError:
                            st.error("无法连接后端，请先启动 FastAPI")

        with tab2:
            url_input = st.text_input(
                "网页地址",
                placeholder="https://example.com/article",
                label_visibility="collapsed",
                key=f"url_{st.session_state.url_key}",
            )
            if url_input:
                if st.button("🌐 加载网页并建立索引", key="btn_url", use_container_width=True, type="primary"):
                    with st.spinner("正在加载网页..."):
                        try:
                            resp = requests.post(f"{API_URL}/upload_url", json={"url": url_input}, timeout=60)
                            if resp.status_code == 200:
                                data = resp.json()
                                fetch_documents()
                                st.session_state.url_key += 1
                                st.success(f"已索引 **{data['chunks']}** 个文档块")
                                st.rerun()
                            else:
                                st.error(f"加载失败：{resp.json().get('detail', '未知错误')}")
                        except requests.ConnectionError:
                            st.error("无法连接后端，请先启动 FastAPI")

    # 底部操作
    st.markdown("---")
    if has_docs:
        if st.button("🗑️ 清空知识库", use_container_width=True, help="删除所有文档和索引"):
            with st.spinner("正在清空..."):
                try:
                    resp = requests.post(f"{API_URL}/clear", timeout=15)
                    if resp.status_code == 200:
                        st.session_state.documents = []
                        st.toast("已清空所有文档")
                        st.rerun()
                except requests.ConnectionError:
                    st.error("无法连接后端")

# =============================================
# 聊天区域
# =============================================
if not st.session_state.agent_messages:
    # 欢迎页
    st.markdown("""
    <div class="welcome-area">
        <div class="emoji">🔬</div>
        <h2>研究助手</h2>
        <p class="desc">上传文档构建知识库，提问时 Agent 自主选择知识库检索或联网搜索，对话与记忆永久保存</p>
        <div class="suggestion-grid">
            <div class="suggestion-card">
                <div class="s-icon">📚</div>
                <div class="s-title">知识库检索</div>
                <div class="s-desc">优先检索你上传的文档</div>
            </div>
            <div class="suggestion-card">
                <div class="s-icon">🌐</div>
                <div class="s-title">Web 搜索</div>
                <div class="s-desc">时效性问题自动联网</div>
            </div>
            <div class="suggestion-card">
                <div class="s-icon">🧠</div>
                <div class="s-title">长期记忆</div>
                <div class="s-desc">跨会话记住你的偏好</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.agent_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander(f"🛠️ 工具调用（{len(msg['steps'])} 步）", expanded=False):
                for step in msg["steps"]:
                    st.markdown(f'<span class="source-tag">{step["tool"]}</span> 入参：`{step["input"]}`', unsafe_allow_html=True)
                    if step.get("output"):
                        st.code(step["output"], language=None)
        if msg["role"] == "assistant" and "usage" in msg:
            st.caption(usage_caption(msg["usage"]))

# =============================================
# 底部悬浮状态条：余额 + 本会话 token（固定在输入框右下方）
# =============================================
_u = st.session_state.session_usage
_bal = fetch_balance()
_bal_text = (
    f"💰 余额 <b>{_bal['currency']} {_bal['total_balance']}</b>"
    if _bal and _bal.get("available") else "💰 余额 --"
)
st.markdown(
    f'<div class="usage-pill">{_bal_text}&nbsp;&nbsp;｜&nbsp;&nbsp;'
    f'🔢 本会话 <b>{_u["total"]}</b> tokens'
    f'（前缀缓存命中 {_u["cache_hit"]}）</div>',
    unsafe_allow_html=True,
)

# =============================================
# 输入框
# =============================================
if question := st.chat_input("提问吧！Agent 会自主选择知识库检索 / 联网搜索 / 长期记忆..."):
    st.session_state.agent_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考并调用工具..."):
            try:
                resp = requests.post(
                    f"{API_URL}/agent/ask",
                    json={
                        "question": question,
                        "thread_id": st.session_state.agent_thread_id,
                        "user_id": st.session_state.agent_user_id,
                    },
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])
                    if data.get("steps"):
                        with st.expander(f"🛠️ 工具调用（{len(data['steps'])} 步）", expanded=False):
                            for step in data["steps"]:
                                st.markdown(f'<span class="source-tag">{step["tool"]}</span> 入参：`{step["input"]}`', unsafe_allow_html=True)
                                if step.get("output"):
                                    st.code(step["output"], language=None)
                    usage = data.get("usage", {})
                    st.caption(usage_caption(usage))
                    add_usage(usage)
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "steps": data.get("steps", []),
                        "usage": usage,
                    })
                    # 刷新侧边栏（会话列表 + token 统计）
                    st.rerun()
                else:
                    error_msg = f"❌ {resp.json().get('detail', '未知错误')}"
                    st.error(error_msg)
                    st.session_state.agent_messages.append({"role": "assistant", "content": error_msg})
            except requests.ConnectionError:
                error_msg = "❌ 无法连接后端服务，请确认 FastAPI 已启动"
                st.error(error_msg)
                st.session_state.agent_messages.append({"role": "assistant", "content": error_msg})
