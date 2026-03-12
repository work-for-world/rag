import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv
from typing import Any, Dict, List, Set
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

try:
    import jieba  # type: ignore
except Exception:
    jieba = None

# 加载环境变量
load_dotenv()

# ================= 配置区域（与 indexer.py 一致）=================
BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHROMA_PATH_QIAN = os.path.join(BASE_DIR, "chroma_db_qwen3")
DEFAULT_CHROMA_PATH_CUSTOM = os.path.join(BASE_DIR, "chroma_db_custom")

# LLM 配置
LLM_MODEL_NAME = "qwen3.5-flash"
LLM_API_KEY = os.getenv("QIAN_API_KEY")
LLM_BASE_URL = os.getenv("QIAN_BASE_URL")

# Embedding 配置（与 indexer 一致：百炼，用于加载 Chroma）
QIAN_API_KEY = os.getenv("QIAN_API_KEY")
QIAN_BASE_URL = os.getenv("QIAN_BASE_URL")
QIAN_EMBED_MODEL = os.getenv("QIAN_EMBED_MODEL", "text-embedding-v4")

# 自定义 Embedding API 配置（用于模型效果对比）
EMBED_API_URL = os.getenv("EMBED_BASE_URL")
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "custombgem3")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "qian").strip().lower()
CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    DEFAULT_CHROMA_PATH_CUSTOM if EMBEDDING_PROVIDER == "custom" else DEFAULT_CHROMA_PATH_QIAN,
)
VECTOR_RELEVANCE_THRESHOLD = float(os.getenv("VECTOR_RELEVANCE_THRESHOLD", "0.7"))
BM25_K = int(os.getenv("BM25_K", "4"))
VECTOR_K = int(os.getenv("VECTOR_K", "8"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "4"))
MIN_FINAL_TOP_K = int(os.getenv("MIN_FINAL_TOP_K", "3"))
KEYWORD_OVERLAP_ENABLED = os.getenv("KEYWORD_OVERLAP_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KEYWORD_OVERLAP_MIN_MATCH = int(os.getenv("KEYWORD_OVERLAP_MIN_MATCH", "2"))
KEYWORD_INDEX_MAX_DOCS = int(os.getenv("KEYWORD_INDEX_MAX_DOCS", "20000"))
KEYWORD_INDEX_MAX_CHARS_PER_DOC = int(os.getenv("KEYWORD_INDEX_MAX_CHARS_PER_DOC", "400"))
# ================= 百炼 Embedding 类（与 indexer 一致）=================
class DashScopeEmbeddings(Embeddings):
    """使用阿里百炼 text-embedding-v4，用于 Chroma 查询与加载。"""
    BATCH_SIZE = 10

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v4"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            completion = self.client.embeddings.create(model=self.model, input=batch)
            sorted_data = sorted(completion.data, key=lambda x: x.index)
            result.extend([item.embedding for item in sorted_data])
        return result

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class CustomAPIEmbeddings(Embeddings):
    """通过 HTTP API 调用的自定义 Embedding。"""
    BATCH_SIZE = 10

    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = []
        try:
            for i in range(0, len(texts), self.BATCH_SIZE):
                batch = texts[i : i + self.BATCH_SIZE]
                payload = {"model": self.model_name, "input": batch}
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                if "data" in data:
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    result.extend([item["embedding"] for item in sorted_data])
                elif "embeddings" in data:
                    result.extend(data["embeddings"])
                else:
                    raise ValueError(f"未知的 Embedding API 返回格式: {data}")
            return result
        except Exception as e:
            print(f"❌ 调用自定义 Embedding API 失败: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def create_embeddings():
    """根据 EMBEDDING_PROVIDER 创建对应的 embedding 实例。"""
    if EMBEDDING_PROVIDER == "custom":
        if not EMBED_API_URL or not EMBED_API_KEY:
            print("❌ 使用 custom embedding 时，请在 .env 中配置 EMBED_BASE_URL 和 EMBED_API_KEY")
            return None
        return CustomAPIEmbeddings(
            api_url=EMBED_API_URL,
            api_key=EMBED_API_KEY,
            model_name=EMBED_MODEL_NAME,
        )

    if not QIAN_API_KEY or not QIAN_BASE_URL:
        print("❌ 使用 qian embedding 时，请在 .env 中配置 QIAN_API_KEY 和 QIAN_BASE_URL")
        return None
    return DashScopeEmbeddings(
        api_key=QIAN_API_KEY,
        base_url=QIAN_BASE_URL,
        model=QIAN_EMBED_MODEL,
    )


_ASCII_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}")
_CJK_CHUNK_RE = re.compile(r"[\u4e00-\u9fff]+")
_KEYWORD_STOPWORDS = {
    "什么",
    "怎么",
    "如何",
    "为啥",
    "为何",
    "请问",
    "一下",
    "这个",
    "那个",
    "可以",
    "是否",
    "有没有",
    "还有",
    "需要",
    "我们",
    "你们",
    "他们",
    "一个",
    "一些",
    "问题",
    "现在",
    "今天",
    "就是",
}


def _extract_keyword_terms(text: str, max_terms: int = 64) -> Set[str]:
    terms: Set[str] = set()
    if not text:
        return terms

    lower_text = text.lower()
    for token in _ASCII_TOKEN_RE.findall(lower_text):
        token = token.strip()
        if len(token) <= 1 or token in _KEYWORD_STOPWORDS:
            continue
        terms.add(token)
        if len(terms) >= max_terms:
            return terms

    for chunk in _CJK_CHUNK_RE.findall(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if jieba:
            for word in jieba.lcut(chunk):
                word = word.strip().lower()
                if len(word) <= 1 or word in _KEYWORD_STOPWORDS:
                    continue
                terms.add(word)
                if len(terms) >= max_terms:
                    return terms
        else:
            # 无 jieba 时使用 2-gram 回退，保持毫秒级开销且具备一定召回
            if len(chunk) < 2:
                continue
            for i in range(len(chunk) - 1):
                word = chunk[i : i + 2].lower()
                if word in _KEYWORD_STOPWORDS:
                    continue
                terms.add(word)
                if len(terms) >= max_terms:
                    return terms

    return terms


def build_kb_keyword_index(db) -> Set[str]:
    """从知识库文档与元数据中提取全局关键词集合，用于问题覆盖度检查。"""
    terms: Set[str] = set()
    try:
        raw = db._collection.get(include=["documents", "metadatas"])
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        if len(metas) < len(docs):
            metas = metas + [None] * (len(docs) - len(metas))

        max_docs = max(1, KEYWORD_INDEX_MAX_DOCS)
        max_chars = max(50, KEYWORD_INDEX_MAX_CHARS_PER_DOC)
        for idx, (doc, meta) in enumerate(zip(docs, metas)):
            if idx >= max_docs:
                break
            text = (doc or "")[:max_chars]
            terms.update(_extract_keyword_terms(text, max_terms=80))
            md = meta or {}
            for key in ("source", "category", "title", "tags"):
                value = md.get(key)
                if isinstance(value, str):
                    terms.update(_extract_keyword_terms(value, max_terms=30))
                elif isinstance(value, list):
                    for item in value:
                        terms.update(_extract_keyword_terms(str(item), max_terms=20))

        if jieba:
            print(f"✅ 已构建关键词索引: {len(terms)} 条（分词: jieba）")
        else:
            print(f"✅ 已构建关键词索引: {len(terms)} 条（分词: 2-gram 回退，建议安装 jieba）")
    except Exception as e:
        print(f"⚠️ 构建关键词索引失败，将跳过覆盖度检查: {e}")
    return terms

# ================= 核心逻辑 =================

def load_existing_vectorstore():
    """加载已存在的向量库"""
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ 错误: 向量库路径不存在 ({CHROMA_PATH})")
        print("💡 请先运行 indexer.py 或 load_database.py 构建知识库。")
        return None

    embeddings = create_embeddings()
    if not embeddings:
        return None
    
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        count = db._collection.count()
        if count == 0:
            print("⚠️ 向量库为空，请先运行 indexer.py 导入数据。")
            return None
        
        print(f"✅ 成功加载向量库，包含 {count} 个片段。")
        return db
    
    except Exception as e:
        error_msg = str(e)
        if "dimension" in error_msg.lower():
            print(f"\n❌ 严重错误: 向量维度不匹配! ({error_msg})")
            print("💡 原因: 当前模型输出的维度与创建数据库时的维度不一致。")
            print("🔧 解决方法: ")
            print(f"   1. 删除旧的向量库文件夹: {CHROMA_PATH}")
            print("   2. 确保 .env 中的模型配置正确。")
            print("   3. 重新运行 python indexer.py 重建索引。")
        else:
            print(f"❌ 加载向量库失败: {e}")
        return None

def create_hybrid_retriever(db):
    """构建混合检索器：BM25 + 向量召回，再调用千问 rerank 重排。"""
    if not db:
        return None

    print("⚙️ 正在构建混合检索器 (BM25 + Vector + Qwen Rerank)...")
    try:
        # 仅修改本函数：参数在函数内读取，避免影响其他代码
        bm25_k = int(os.getenv("BM25_K", "4"))
        vector_k = int(os.getenv("VECTOR_K", "8"))
        final_top_k = max(1, int(os.getenv("FINAL_TOP_K", "4")))
        min_final_top_k = max(1, min(int(os.getenv("MIN_FINAL_TOP_K", "3")), final_top_k))
        rerank_url = os.getenv(
            "QWEN_RERANK_URL",
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        )
        rerank_model = os.getenv("QWEN_RERANK_MODEL", "qwen3-vl-rerank")
        rerank_api_key = os.getenv("DASHSCOPE_API_KEY")
        cache_ttl_seconds = float(os.getenv("RETRIEVER_CACHE_TTL_SECONDS", "2.0"))
        retrieval_cache: Dict[str, tuple[float, List[Document]]] = {}

        raw = db._collection.get(include=["documents", "metadatas"])
        texts = raw.get("documents") or []
        metadatas = raw.get("metadatas") or [None] * len(texts)
        if len(metadatas) < len(texts):
            metadatas = metadatas + [None] * (len(texts) - len(metadatas))

        bm25_corpus = []
        for text, meta in zip(texts, metadatas):
            text = (text or "").strip()
            if not text:
                continue
            bm25_corpus.append(Document(page_content=text, metadata=meta or {}))

        bm25_retriever = BM25Retriever.from_documents(bm25_corpus) if bm25_corpus else None
        if bm25_retriever:
            bm25_retriever.k = bm25_k

        def _doc_key(d: Document):
            return ((d.metadata or {}).get("source", ""), (d.page_content or "").strip())

        def _hybrid_search(query: str) -> List[Document]:
            query_key = (query or "").strip()
            now = time.time()
            if cache_ttl_seconds > 0 and query_key in retrieval_cache:
                ts, docs_cached = retrieval_cache[query_key]
                if now - ts <= cache_ttl_seconds:
                    return [
                        Document(page_content=d.page_content, metadata=dict(d.metadata or {}))
                        for d in docs_cached
                    ]

            expanded_queries = [query]

            # 按“每个问题”存储命中结果，最后再合并
            retrieval_store: Dict[str, Dict[str, List[Document]]] = {}

            for q_idx, sub_query in enumerate(expanded_queries, start=1):
                retrieval_store[sub_query] = {"bm25": [], "vector": []}

                if bm25_retriever:
                    bm25_hits = bm25_retriever.invoke(sub_query)
                    for hit in bm25_hits:
                        meta = dict(hit.metadata or {})
                        meta["hit_query"] = sub_query
                        meta["query_index"] = q_idx
                        retrieval_store[sub_query]["bm25"].append(
                            Document(page_content=hit.page_content, metadata=meta)
                        )

                vector_pairs = db.similarity_search_with_score(sub_query, k=vector_k)
                for rank, (doc, distance) in enumerate(vector_pairs):
                    score = 1.0 / (1.0 + max(0.0, float(distance)))
                    metadata = dict(doc.metadata or {})
                    metadata["vector_relevance"] = round(float(score), 4)
                    metadata["vector_rank"] = rank + 1
                    metadata["hit_query"] = sub_query
                    metadata["query_index"] = q_idx
                    retrieval_store[sub_query]["vector"].append(
                        Document(page_content=doc.page_content, metadata=metadata)
                    )

            bm25_docs: List[Document] = []
            vector_docs: List[Document] = []
            for store in retrieval_store.values():
                bm25_docs.extend(store["bm25"])
                vector_docs.extend(store["vector"])

            # 合并去重：优先保留向量版本（含vector_relevance）
            merged_by_key = {}
            for d in vector_docs + bm25_docs:
                key = _doc_key(d)
                if key not in merged_by_key:
                    merged_by_key[key] = d
                    continue
                # 同一文档被多次命中时，保留向量分更高的版本
                old = merged_by_key[key]
                old_score = float((old.metadata or {}).get("vector_relevance", -1.0))
                new_score = float((d.metadata or {}).get("vector_relevance", -1.0))
                if new_score > old_score:
                    merged_by_key[key] = d
            candidates = list(merged_by_key.values())

            if not candidates:
                return []

            reranked_docs = []
            rerank_ok = False
            if rerank_api_key:
                try:
                    headers = {
                        "Authorization": f"Bearer {rerank_api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": rerank_model,
                        "input": {
                            "query": query,
                            "documents": [(d.page_content or "") for d in candidates],
                        },
                        "parameters": {
                            "return_documents": True,
                            "top_n": min(final_top_k, len(candidates)),
                        },
                    }
                    resp = requests.post(rerank_url, headers=headers, json=payload, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()

                    # 兼容不同返回结构：results 可能在 output/results 或 data/results
                    results = []
                    if isinstance(data, dict):
                        output = data.get("output")
                        if isinstance(output, dict) and isinstance(output.get("results"), list):
                            results = output.get("results") or []
                        elif isinstance(data.get("results"), list):
                            results = data.get("results") or []
                        elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("results"), list):
                            results = data["data"].get("results") or []

                    for item in results:
                        idx = item.get("index")
                        if idx is None:
                            continue
                        if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
                            continue
                        base_doc = candidates[idx]
                        metadata = dict(base_doc.metadata or {})
                        metadata["rerank_score"] = round(float(item.get("relevance_score", 0.0)), 6)
                        metadata["rerank_model"] = rerank_model
                        reranked_docs.append(Document(page_content=base_doc.page_content, metadata=metadata))

                    if reranked_docs:
                        rerank_ok = True
                except Exception as rerank_err:
                    print(f"⚠️ 千问 rerank 调用失败，降级为向量优先排序: {rerank_err}")
            else:
                print("⚠️ 未配置 DASHSCOPE_API_KEY/QIAN_API_KEY，跳过 rerank，使用向量优先排序。")

            if rerank_ok:
                selected = reranked_docs[:final_top_k]
            else:
                # rerank 失败时：按向量分排序，尽量不差于纯向量检索
                candidates.sort(
                    key=lambda d: float((d.metadata or {}).get("vector_relevance", 0.0),
                    ),
                    reverse=True,
                )
                selected = candidates[:final_top_k]

            if len(selected) < min_final_top_k:
                selected = (selected + candidates)[:min_final_top_k]
                # 再次去重，避免补齐时重复
                seen = set()
                uniq = []
                for d in selected:
                    key = _doc_key(d)
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(d)
                selected = uniq[:min_final_top_k]

            print(
                f"🔎 检索统计: bm25={len(bm25_docs)} | vector={len(vector_docs)} | "
                f"candidates={len(candidates)} | rerank={'on' if rerank_ok else 'off'} | selected={len(selected)}"
            )
            if cache_ttl_seconds > 0:
                retrieval_cache[query_key] = (
                    now,
                    [Document(page_content=d.page_content, metadata=dict(d.metadata or {})) for d in selected],
                )
            return selected

        print(
            "✅ 混合检索器构建成功"
            f"（BM25_K={bm25_k}, VECTOR_K={vector_k}, FINAL_TOP_K={final_top_k}, "
            f"MIN_FINAL_TOP_K={min_final_top_k}, RERANK_MODEL={rerank_model}）。"
        )
        return RunnableLambda(_hybrid_search)
    except Exception as e:
        print(f"❌ 构建混合检索失败: {e}")
        print("仅使用向量检索...")
        return db.as_retriever(search_kwargs={"k": int(os.getenv('FINAL_TOP_K', '4'))})

def create_rag_chain(retriever, llm, kb_keywords: Set[str] | None = None, use_memory: bool = True):
    # 第一次调用：基于近期 memory 改写用户问题（单条或拆成多条）
    rewrite_template = """你是查询改写助手。根据【历史会话】理解用户当前问题的上下文，如果用户当前的问题中存在多个问题，将这多个问题拆成多条检索问句；若只有一个问题则保持一条（可结合上下文略作补全，不改变原意）。
只输出严格 JSON，不要任何解释。格式：{{"queries": ["问句1", "问句2", ...]}}。

【历史会话】
{history}

【用户当前问题】
{question}
"""
    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # 根据参考信息生成最终回答
    template = """你是招联客服知识库助手，请根据【参考信息】回答用户问题。
    若参考信息与用户问题无关（无法从中找到答案），请直接且仅回复：“抱歉，当前招联IT数据库中不存在您要搜索的信息，我们会尽力添加”，不要编造任何内容。
    如果参考信息与用户问题有关，则遵循如下回答策略与结构要求：
    1) 多问题拆解：如果用户问题包含多个独立子问题，请分别针对每个子问题查找对应的参考信息，并分段进行回复，不要混为一谈。
    2) 动态内容结构：回答应逻辑清晰，凡适合分点阐述的内容（如操作步骤、原因列表、多项事实等），请务必使用分点格式输出；仅当内容为单一简短结论时，可直接陈述。
    3) 图片引用规范 (最高优先级)：
    - 触发：只要内容涉及图片，必须在相关段落/步骤后立即换行输出。
    - 格式：独占一行，严格格式为 `[图片地址] 相对路径`
    - 路径处理：直接使用参考信息中的原始相对路径，禁止拼接根目录或转换为绝对路径，禁止额外加入空格，统一使用正斜杠 `/`。
    - 完整性：提到几张图就输出几行，严禁遗漏或合并。
    - 例如：[图片地址] ../../../../../../plc/知识库/it指引(1)/网络/无线网络/改了域密码后手机wifi连不上了/image001.png
    4) 真实性约束：不要编造图片路径、系统入口、账号策略等信息。
    5) 语言风格：专业、礼貌、面向业务同事，避免过度技术黑话。

【参考信息】
{context}

【用户问题】
{question}

请输出最终答复：
"""
    prompt = ChatPromptTemplate.from_template(template)
    answer_chain = prompt | llm | StrOutputParser()
    oos_prompt = ChatPromptTemplate.from_template(
        """你是“招联客服知识库助手1000号”，请根据用户输入按下面规则输出中文答复：
1) 若用户是寒暄/问候/确认在线（如“你好”“在吗”“hi”），请礼貌简短回复，并引导用户描述具体的IT问题。
2) 若用户问题与招联知识库主题无关，统一回复：抱歉，当前招联IT数据库中不存在您要搜索的信息，我们会尽力添加
3) 若不确定是否相关，也按第2条回复。

用户输入：
{question}
"""
    )
    oos_chain = oos_prompt | llm | StrOutputParser()
    keep_recent_turns = 3
    multi_top_k = int(os.getenv("MULTI_QUESTION_TOP_K", "3"))
    single_top_k = int(os.getenv("SINGLE_QUESTION_TOP_K", "4"))
    memory = {"turns": []}
    kb_keywords = kb_keywords or set()
    keyword_overlap_enabled = KEYWORD_OVERLAP_ENABLED and bool(kb_keywords)

    def _keyword_overlap_stats(text: str) -> Dict[str, Any]:
        q_terms = _extract_keyword_terms(text, max_terms=40)
        if not q_terms:
            return {"terms": set(), "matched": set(), "ratio": 1.0}
        matched = {t for t in q_terms if t in kb_keywords}
        ratio = len(matched) / max(1, len(q_terms))
        return {"terms": q_terms, "matched": matched, "ratio": ratio}

    def _should_block_by_keyword(text: str) -> Dict[str, Any]:
        if not keyword_overlap_enabled:
            return {"block": False, "terms": set(), "matched": set(), "ratio": 1.0}
        stats = _keyword_overlap_stats(text)
        matched = stats["matched"]
        block = len(matched) < KEYWORD_OVERLAP_MIN_MATCH
        return {
            "block": block,
            "terms": stats["terms"],
            "matched": matched,
            "ratio": stats["ratio"],
        }

    def _fallback_non_retrieval_answer(question: str) -> str:
        try:
            return oos_chain.invoke({"question": question}).strip()
        except Exception as e:
            print(f"⚠️ 非检索分流模型调用失败，使用兜底文案: {e}")
            return "该问题与当前招联知识库无关"

    def _doc_key(d: Document) -> tuple:
        m = d.metadata or {}
        return (m.get("source", ""), (d.page_content or "").strip())

    def _doc_score(d: Document) -> float:
        m = d.metadata or {}
        s = m.get("rerank_score")
        if s is not None:
            return float(s)
        return float(m.get("vector_relevance", 0.0))

    def _format_docs(docs: List[Document]) -> str:
        if not docs:
            return "无可用参考信息。"
        blocks = []
        for i, d in enumerate(docs, 1):
            metadata = d.metadata or {}
            source = metadata.get("source", "未知来源")
            vector_score = metadata.get("vector_relevance")
            rerank_score = metadata.get("rerank_score")
            score_parts = []
            if vector_score is not None:
                score_parts.append(f"向量置信度: {vector_score}")
            if rerank_score is not None:
                score_parts.append(f"Rerank置信度: {rerank_score}")
            score_text = f" | {' | '.join(score_parts)}" if score_parts else ""
            blocks.append(f"[片段{i}] 来源: {source}{score_text}\n{(d.page_content or '').strip()}")
        context_text = "\n\n---\n\n".join(blocks)
        # print("检索到的内容:")
        # print(context_text)
        return context_text

    def _serialize_turns(turns: List[Dict[str, str]]) -> str:
        lines = []
        for turn in turns:
            lines.append(f"用户：{turn.get('q', '')}")
            lines.append(f"助手：{turn.get('a', '')}")
        return "\n".join(lines).strip()

    def _build_history_text() -> str:
        recent = memory["turns"][-keep_recent_turns:]
        recent_text = _serialize_turns(recent) if recent else "无"
        return f"【最近对话】\n{recent_text}"

    def _parse_rewrite_output(raw: str) -> List[str]:
        raw = (raw or "").strip()
        if not raw:
            return []
        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```").strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        try:
            data = json.loads(raw)
            queries = data.get("queries")
            if isinstance(queries, list) and queries:
                return [str(q).strip() for q in queries if str(q).strip()]
        except Exception:
            pass
        return []

    def _to_doc_records(docs: List[Document]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for d in docs:
            md = dict(d.metadata or {})
            records.append(
                {
                    "page_content": (d.page_content or "").strip(),
                    "source": md.get("source", ""),
                    "vector_relevance": md.get("vector_relevance"),
                    "rerank_score": md.get("rerank_score"),
                    "metadata": md,
                }
            )
        return records

    def _remember_turn(question: str, answer: str) -> None:
        """统一记录每一轮 Q/A，确保所有分支都进入 memory。"""
        memory["turns"].append({"q": question, "a": answer})
        if len(memory["turns"]) > keep_recent_turns:
            memory["turns"] = memory["turns"][-keep_recent_turns:]

    def _merge_queries(queries: List[str]) -> str:
        if not queries:
            return ""
        if len(queries) == 1:
            return queries[0]
        return "；".join(f"{i + 1}. {q}" for i, q in enumerate(queries))

    def _invoke_with_memory(question: str) -> Dict[str, Any]:
        t_total_start = time.perf_counter()

        # 门控基于“原始用户问题”，且在任何检索前执行
        keyword_gate = _should_block_by_keyword(question)
        if keyword_gate["block"]:
            print(
                "🧱 关键词覆盖度过低，跳过检索: "
                f"matched={len(keyword_gate['matched'])}/{len(keyword_gate['terms'])}"
            )
            t_oos_start = time.perf_counter()
            answer = _fallback_non_retrieval_answer(question)
            oos_ms = round((time.perf_counter() - t_oos_start) * 1000, 1)
            total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)
            print(f"⏱️ 耗时统计 | 门控分流: {oos_ms} ms | 总计: {total_ms} ms")
            if use_memory:
                _remember_turn(question, answer)
            return {
                "answer": answer,
                "docs": [],
                "queries": [],
                "rewritten_question": question,
                "skip_retrieval": True,
                "keyword_overlap": {
                    "matched_count": len(keyword_gate["matched"]),
                    "term_count": len(keyword_gate["terms"]),
                    "ratio": round(float(keyword_gate["ratio"]), 4),
                },
                "timings_ms": {
                    "rewrite": 0.0,
                    "retrieve": 0.0,
                    "answer": oos_ms,
                    "total": total_ms,
                },
            }

        if use_memory:
            history_text = _build_history_text()
        else:
            history_text = "【最近对话】\n无"

        # 首轮（无历史）或关闭 memory 时，跳过问题改写以降低时延
        has_history = use_memory and bool(memory["turns"])
        if has_history:
            t_rewrite_start = time.perf_counter()
            raw_rewrite = rewrite_chain.invoke({"history": history_text, "question": question})
            rewrite_ms = round((time.perf_counter() - t_rewrite_start) * 1000, 1)
            queries = _parse_rewrite_output(raw_rewrite)
            if not queries:
                queries = [question]
            rewritten_question = _merge_queries(queries)
            print(f"✏️ 改写后问题: {rewritten_question}")
            if len(queries) > 1:
                print(f"🧩 基于记忆改写为多问句: {' | '.join(queries)}")
        else:
            rewrite_ms = 0.0
            queries = [question]
            rewritten_question = question

        # 多问句分别检索再合并去重；单问句直接检索
        t_retrieve_start = time.perf_counter()
        if len(queries) > 1:
            all_docs: List[Document] = []
            max_workers = max(1, min(int(os.getenv("MULTI_RETRIEVAL_MAX_WORKERS", "4")), len(queries)))

            def _retrieve_one(sub_q: str) -> List[Document]:
                return retriever.invoke(sub_q)[:multi_top_k]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_retrieve_one, q): q for q in queries}
                for fut in as_completed(futures):
                    q = futures[fut]
                    try:
                        all_docs.extend(fut.result())
                    except Exception as e:
                        print(f"⚠️ 子问题检索失败，已跳过: {q} | {e}")
            merged = {}
            for d in all_docs:
                key = _doc_key(d)
                if key not in merged or _doc_score(d) > _doc_score(merged[key]):
                    merged[key] = d
            docs = sorted(merged.values(), key=_doc_score, reverse=True)
        else:
            docs = retriever.invoke(queries[0])[:single_top_k]
        retrieve_ms = round((time.perf_counter() - t_retrieve_start) * 1000, 1)

        context = _format_docs(docs)
        if not docs:
            answer = "抱歉，数据库中不存在您要搜索的信息，我们会尽力添加"
            total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)
            print(
                "⏱️ 耗时统计 | "
                f"改写: {rewrite_ms} ms | 检索: {retrieve_ms} ms | 生成: 0.0 ms | 总计: {total_ms} ms"
            )
            if use_memory:
                _remember_turn(rewritten_question, answer)
            return {
                "answer": answer,
                "docs": [],
                "queries": queries,
                "rewritten_question": rewritten_question,
                "skip_retrieval": True,
                "timings_ms": {
                    "rewrite": rewrite_ms,
                    "retrieve": retrieve_ms,
                    "answer": 0.0,
                    "total": total_ms,
                },
            }
        t_answer_start = time.perf_counter()
        answer = answer_chain.invoke(
            {"context": context, "history": history_text, "question": rewritten_question}
        )
        answer_ms = round((time.perf_counter() - t_answer_start) * 1000, 1)
        total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)
        print(
            "⏱️ 耗时统计 | "
            f"改写: {rewrite_ms} ms | 检索: {retrieve_ms} ms | 生成: {answer_ms} ms | 总计: {total_ms} ms"
        )
        if use_memory:
            _remember_turn(rewritten_question, answer)
        return {
            "answer": answer,
            "docs": _to_doc_records(docs),
            "queries": queries,
            "rewritten_question": rewritten_question,
            "timings_ms": {
                "rewrite": rewrite_ms,
                "retrieve": retrieve_ms,
                "answer": answer_ms,
                "total": total_ms,
            },
        }

    return RunnableLambda(_invoke_with_memory)

def main():
    # 1. 检查必要的环境变量
    if not LLM_API_KEY or not LLM_BASE_URL:
        print("❌ 错误: 请在 .env 中配置 QIAN_API_KEY 和 QIAN_BASE_URL")
        return
    if not create_embeddings():
        return

    # 2. 加载向量库
    db = load_existing_vectorstore()
    if not db:
        return

    # 3. 构建检索器
    retriever = create_hybrid_retriever(db)
    if not retriever:
        return

    kb_keywords = build_kb_keyword_index(db)

    # 4. 初始化 LLM
    try:
        llm = ChatOpenAI(
            model_name=LLM_MODEL_NAME,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0.7,
            max_tokens=1024,
        )
        print("✅ LLM 连接成功。")
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        return
   
    # 5. 构建链
    rag_chain = create_rag_chain(retriever, llm, kb_keywords=kb_keywords)
    
    # 6. 交互循环
    print("\n" + "="*50)
    print(f"🎉 RAG 系统已就绪！")
    if EMBEDDING_PROVIDER == "custom":
        print(f"   - Embedding: {EMBED_MODEL_NAME} (@ {EMBED_API_URL}) [custom]")
    else:
        print(f"   - Embedding: {QIAN_EMBED_MODEL} (@ {QIAN_BASE_URL}) [qian]")
    print(f"   - Vector DB: {CHROMA_PATH}")
    print(f"   - LLM: {LLM_MODEL_NAME}")
    # print("💡 输入 'quit' 退出；输入 'memory' 或 '查看记忆' 可查看当前记忆（调试）。")
    print("💡 输入 'quit' 退出;")
    print("="*50)
    
    while True:
        try:
            query = input("\n❓ 请输入问题: ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query.strip():
                continue
            
            print("⏳ 思考中...", end="\r")
            response = rag_chain.invoke(query)
            answer = response.get("answer", "") if isinstance(response, dict) else str(response)
            
            print(" " * 20, end="\r") # 清除"思考中"
            print("\n💡 回答:")
            print(answer)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            # 打印详细 traceback 以便调试本地接口问题
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main()