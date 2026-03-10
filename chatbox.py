import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv
from typing import Any, Dict, List
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

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

                vector_pairs = db.similarity_search_with_relevance_scores(sub_query, k=vector_k)
                for rank, (doc, score) in enumerate(vector_pairs):
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

def create_rag_chain(retriever, llm):
    # 第一次调用：基于 memory 改写用户问题（单条或拆成多条）
    rewrite_template = """你是查询改写助手。根据【历史会话】理解用户当前问题的上下文，如果用户当前的问题中存在多个问题，将这多个问题拆成多条检索问句；若只有一个问题则保持一条（可结合上下文略作补全，不改变原意）。
    只输出严格 JSON，不要任何解释。格式：{{"queries": ["问句1", "问句2", ...]}}。

    【历史会话】
    {history}

    【用户当前问题】
    {question}
    """
    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # 第二次调用：根据参考信息生成最终回答
    template = r"""你是招联客服知识库助手，请根据【参考信息】回答用户问题。
你的目标是给出准确、清晰且可执行的处理指引。

【历史会话摘要与最近对话】
{history}

回答策略与结构要求：
1) **多问题拆解**：如果用户问题包含多个独立子问题（例如“怎么重置密码？另外报错代码0x800是什么意思？”），请分别针对每个子问题查找对应的参考信息，并分段进行回复，不要混为一谈。
2) **动态内容结构**：根据内容类型自动选择最合适的格式，严禁机械地全部使用“步骤一、步骤二”：
   - **操作指引类**（如“如何安装”、“怎么配置”）：必须使用步骤化输出（步骤 1、步骤 2...），每步一句到两句，简洁明确。
   - **原因/概念/列表类**（如“为什么失败”、“有哪些政策”）：请使用分点列表（• 或 1. 2. 3.）进行阐述，清晰罗列关键点。
   - **简单事实类**（如“服务台电话是多少”）：直接给出明确结论，无需分点或步骤。
3) **图片路径严格规范**（最高优先级）：
   - 触发条件：只要参考信息或生成的步骤中包含图片引用，必须紧跟在该相关段落/步骤后逐行输出。
   - 格式要求：每张图片独占一行，严格格式为：`[图片地址] 绝对路径`
   - 路径转换：
     * 图片根目录固定为：“E:\python_code\langchain\plc”
     * 若参考信息中是相对路径，必须拼接为该根目录下的绝对路径。
     * **强烈建议统一使用正斜杠 `/` 输出路径**（例如 `E:/python_code/langchain/plc/.../image001.png`），防止转义错误。
   - 完整性约束：
     * 必须保留所有提到的图片，禁止丢失、合并或省略。
     * 输出路径时必须保持原始字符完整，不得新增/删除字符，不得断行，不得把一个路径拆成两行。
     * 若同一行中有多个 [图片地址] 标记，必须识别并分别单独输出为多行。
   - 绝对路径示例：
     [图片地址] E:/python_code/langchain/plc/知识库/it指引(1)/网络/无线网络/改了域密码后手机wifi连不上了/image001.png
4) **真实性约束**：不要编造图片路径、系统入口、账号策略等信息。若参考信息不足，明确说明“知识库未提供完整信息”，并告知用户联系 IT 服务台。
5) **语言风格**：专业、礼貌、面向业务同事，避免过度技术黑话。

【参考信息】
{context}

【用户问题】
{question}

请输出最终答复：
"""
    prompt = ChatPromptTemplate.from_template(template)
    answer_chain = prompt | llm | StrOutputParser()

    summary_prompt = ChatPromptTemplate.from_template(
        """你是对话记忆压缩助手。请在不遗漏关键业务信息的前提下，压缩历史会话。

        已有摘要：
        {existing_summary}

        新增对话：
        {new_turns}

        请输出更新后的精简摘要，要求：
        1) 保留用户目标、已确认事实、关键约束、未解决问题；
        2) 删除寒暄和重复表达；
        3) 使用中文，控制在 {max_tokens} token 以内（尽量简洁）。
"""
    )
    summary_chain = summary_prompt | llm | StrOutputParser()

    max_history_tokens = int(os.getenv("HISTORY_MAX_TOKENS", "1600"))
    summary_max_tokens = int(os.getenv("HISTORY_SUMMARY_MAX_TOKENS", "600"))
    keep_recent_turns = int(os.getenv("HISTORY_KEEP_RECENT_TURNS", "4"))
    multi_top_k = int(os.getenv("MULTI_QUESTION_TOP_K", "3"))
    single_top_k = int(os.getenv("SINGLE_QUESTION_TOP_K", "4"))
    memory = {"summary": "", "turns": []}

    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 2)

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
        print("检索到的内容:")
        print(context_text)
        return context_text

    def _serialize_turns(turns: List[Dict[str, str]]) -> str:
        lines = []
        for turn in turns:
            lines.append(f"用户：{turn.get('q', '')}")
            lines.append(f"助手：{turn.get('a', '')}")
        return "\n".join(lines).strip()

    def _build_history_text() -> str:
        recent = memory["turns"][-keep_recent_turns:] if keep_recent_turns > 0 else memory["turns"]
        recent_text = _serialize_turns(recent) if recent else "无"
        summary_text = memory["summary"].strip() or "无"
        return f"【摘要】\n{summary_text}\n\n【最近对话】\n{recent_text}"

    # def _format_memory_debug() -> str:
    #     """调试用：返回当前 memory 的完整内容，便于核对是否存储正确。"""
    #     summary = memory["summary"].strip() or "（空）"
    #     turns = memory["turns"]
    #     lines = [
    #         "========== Memory 调试信息 ==========",
    #         "（以下为 memory 中实际存储的完整内容，未截断）",
    #         f"【摘要】共 {len(summary)} 字",
    #         summary if summary != "（空）" else summary,
    #         "",
    #         f"【完整对话轮次】共 {len(turns)} 轮",
    #     ]
    #     for i, t in enumerate(turns, 1):
    #         q = (t.get("q") or "").strip()
    #         a = (t.get("a") or "").strip()
    #         lines.append(f"--- 第 {i} 轮 ---")
    #         lines.append(f"用户: {q}")
    #         lines.append(f"助手: {a}")
    #         lines.append("")
    #     recent = memory["turns"][-keep_recent_turns:] if keep_recent_turns > 0 else memory["turns"]
    #     history_preview = _build_history_text()
    #     lines.append(f"【注入给模型的 history 估算】约 {_estimate_tokens(history_preview)} tokens（最近 {len(recent)} 轮 + 摘要）")
    #     lines.append("====================================")
    #     return "\n".join(lines)

    def _compress_if_needed() -> None:
        history_text = _build_history_text()
        if _estimate_tokens(history_text) <= max_history_tokens:
            return
        old_turns_text = _serialize_turns(memory["turns"])
        if not old_turns_text:
            return
        # print("🧠 历史会话过长，正在自动压缩记忆...")
        new_summary = summary_chain.invoke(
            {
                "existing_summary": memory["summary"] or "无",
                "new_turns": old_turns_text,
                "max_tokens": summary_max_tokens,
            }
        ).strip()
        memory["summary"] = new_summary
        memory["turns"] = memory["turns"][-keep_recent_turns:] if keep_recent_turns > 0 else []

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

    # MEMORY_DEBUG_CMDS = ("memory", "mem", "debug", "查看记忆")

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

    def _invoke_with_memory(question: str) -> Dict[str, Any]:
        # cmd = (question or "").strip().lower()
        # if cmd in MEMORY_DEBUG_CMDS:
        #     return _format_memory_debug()

        _compress_if_needed()
        history_text = _build_history_text()

        # 第一次 LLM：基于 memory 改写/拆分为 1～N 个检索问句
        raw_rewrite = rewrite_chain.invoke({"history": history_text, "question": question})
        queries = _parse_rewrite_output(raw_rewrite)
        if not queries:
            queries = [question]
        if len(queries) > 1:
            print(f"🧩 基于记忆改写为多问句: {' | '.join(queries)}")

        # 多问句：每个问题分别混合检索+rerank，各取前 multi_top_k 条，再合并去重后全部给模型；单问句：检索取 top4
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

        context = _format_docs(docs)
        answer = answer_chain.invoke(
            {"context": context, "history": history_text, "question": question}
        )
        memory["turns"].append({"q": question, "a": answer})
        return {
            "answer": answer,
            "docs": _to_doc_records(docs),
            "queries": queries,
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
    rag_chain = create_rag_chain(retriever, llm)
    
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