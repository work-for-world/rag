import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

from chatbox import (
    LLM_API_KEY,
    LLM_BASE_URL,
    create_embeddings,
    create_hybrid_retriever,
    create_rag_chain,
    load_existing_vectorstore,
)


IMAGE_PATTERN = re.compile(
    r"([A-Za-z]:[\\/][^\r\n\[\]\"']+\.(?:png|jpg|jpeg|gif|webp)|(?:\.\./|/)?[^\r\n\[\]\"']+\.(?:png|jpg|jpeg|gif|webp))",
    flags=re.IGNORECASE,
)

IMAGE_EXT_PATTERN = re.compile(r"\.(?:png|jpg|jpeg|gif|webp)$", flags=re.IGNORECASE)


def extract_image_paths(text: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for raw in IMAGE_PATTERN.findall(text):
        cleaned = raw.strip().strip("[]() \t,，。;；")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            paths.append(cleaned)
    return paths


def parse_line_text_and_images(line: str) -> tuple[str | None, list[str]]:
    stripped = line.strip()
    if not stripped:
        return None, []

    image_paths = extract_image_paths(stripped)

    text_part = stripped
    for p in image_paths:
        text_part = text_part.replace(p, " ")
    text_part = re.sub(r"\[图片地址\]\s*:?", " ", text_part)
    text_part = re.sub(r"\s+", " ", text_part).strip().strip(":-：")

    return (text_part if text_part else None), image_paths


def _drop_leading_rel(path_text: str) -> str:
    normalized = path_text
    while normalized.startswith("../"):
        normalized = normalized[3:]
    return normalized


def _normalize_name_for_match(name: str) -> str:
    # 忽略空白差异：例如“手机 wifi”与“手机wifi”视为同名
    return re.sub(r"[\s\u3000]+", "", name).lower()


def _resolve_under_root_space_insensitive(root: Path, rel_path: str) -> Path | None:
    rel = rel_path.replace("\\", "/").strip().strip("/")
    if not rel:
        return None

    parts = [p for p in rel.split("/") if p and p != "."]
    current = root
    for part in parts:
        direct = current / part
        if direct.exists():
            current = direct
            continue

        try:
            children = list(current.iterdir())
        except Exception:
            return None

        target = _normalize_name_for_match(part)
        matched = next((c for c in children if _normalize_name_for_match(c.name) == target), None)
        if matched is None:
            return None
        current = matched

    return current if current.exists() else None


def resolve_image_path(raw_path: str, workspace_root: Path) -> Path | None:
    cleaned = raw_path.strip().strip("[]() \t,，。;；")
    if not cleaned:
        return None

    # 先尝试原始路径（支持绝对路径）
    p = Path(cleaned)
    if p.exists():
        return p

    normalized = cleaned.replace("\\", "/")

    # 处理 Windows 绝对路径，如 E:\python_code\langchain\plc\...
    normalized_no_drive = re.sub(r"^[A-Za-z]:/+", "", normalized)
    candidates = [
        workspace_root / normalized,
        workspace_root / normalized.lstrip("./"),
        workspace_root / _drop_leading_rel(normalized),
        workspace_root / normalized_no_drive,
    ]

    if "/plc/" in normalized:
        plc_suffix = normalized.split("/plc/", 1)[1]
        candidates.append(workspace_root / "plc" / plc_suffix)
    if "plc/" in normalized_no_drive:
        plc_suffix = normalized_no_drive.split("plc/", 1)[1]
        candidates.append(workspace_root / "plc" / plc_suffix)

    for item in candidates:
        resolved = item.resolve()
        if resolved.exists():
            return resolved

    # 回退：在 plc 根目录下做“空格不敏感”的逐级路径匹配
    plc_root = workspace_root / "plc"
    if plc_root.exists():
        suffix_candidates: list[str] = []
        if "/plc/" in normalized:
            suffix_candidates.append(normalized.split("/plc/", 1)[1])
        if normalized.lower().startswith("plc/"):
            suffix_candidates.append(normalized.split("/", 1)[1])
        if "plc/" in normalized_no_drive:
            suffix_candidates.append(normalized_no_drive.split("plc/", 1)[1])
        if "/知识库/" in normalized:
            suffix_candidates.append("知识库/" + normalized.split("/知识库/", 1)[1])
        if normalized.startswith("知识库/"):
            suffix_candidates.append(normalized)

        for suffix in suffix_candidates:
            maybe = _resolve_under_root_space_insensitive(plc_root, suffix)
            if maybe is not None:
                return maybe
    return None


def _looks_like_path_prefix(text: str) -> bool:
    """判断是否像“未结束的路径前半段”（无图片后缀）。"""
    stripped = text.strip()
    if not stripped or IMAGE_EXT_PATTERN.search(stripped):
        return False
    return bool(re.match(r"^(?:[A-Za-z]:[\\/]|(?:\.\./|/)?plc[\\/]).+", stripped))


def _looks_like_rel_image_fragment(text: str) -> bool:
    """判断是否像“相对路径后半段”，例如：连不上了\\image001.png"""
    stripped = text.strip().lstrip("\\/")
    if not stripped:
        return False
    if re.match(r"^[A-Za-z]:[\\/]", stripped):
        return False
    return bool(re.search(r"[\\/][^\\/\r\n]+\.(?:png|jpg|jpeg|gif|webp)$", stripped, flags=re.IGNORECASE))


def render_mixed_answer(answer: str, workspace_root: Path) -> bool:
    rendered_image = False
    lines = answer.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            st.write("")
            idx += 1
            continue

        # 兼容模型将图片路径断成两行：
        # 行1: E:\...\改了域密码后手机 wifi
        # 行2: 连不上了\image001.png
        if idx + 1 < len(lines):
            next_line = lines[idx + 1].strip()
            if _looks_like_path_prefix(line) and _looks_like_rel_image_fragment(next_line):
                left = line.rstrip("\\/")
                right = next_line.lstrip("\\/")
                line = f"{left}\\{right}"
                idx += 1

        text_part, image_paths = parse_line_text_and_images(line)
        if text_part:
            st.markdown(text_part)

        for image_path in image_paths:
            resolved = resolve_image_path(image_path, workspace_root)
            if resolved is not None:
                st.image(str(resolved), use_container_width=True)
                rendered_image = True
            else:
                st.caption(f"图片路径（本地未找到）: `{image_path}`")
        idx += 1
    return rendered_image


@st.cache_resource
def get_rag_chain():
    db = load_existing_vectorstore()
    if not db:
        raise RuntimeError("向量库加载失败，请检查 CHROMA_PATH 与 embedding 配置。")

    retriever = create_hybrid_retriever(db)
    if not retriever:
        raise RuntimeError("检索器构建失败，请检查依赖与环境变量配置。")

    rag_chain = create_rag_chain(retriever, _create_llm())
    return rag_chain


@st.cache_resource
def _create_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model_name="qwen3.5-flash",
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.7,
        max_tokens=1024,
    )


def _render_assistant_message(msg: dict, workspace_root: Path, show_docs: list[Any] | None) -> None:
    """渲染单条助手消息：正文、以及可选的参考片段（仅最后一条）。"""
    content = msg.get("content", "")
    render_mixed_answer(content, workspace_root)

    if not show_docs:
        return
    with st.expander("检索到的参考片段", expanded=False):
        for idx, doc in enumerate(show_docs, start=1):
            metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source", "未知来源")
            vector_score = metadata.get("vector_relevance")
            rerank_score = metadata.get("rerank_score")
            score_parts = []
            if vector_score is not None:
                score_parts.append(f"向量置信度: `{vector_score}`")
            if rerank_score is not None:
                score_parts.append(f"Rerank置信度: `{rerank_score}`")
            score_text = " | ".join(score_parts) if score_parts else "无分数信息"
            st.markdown(f"**[{idx}] {source}** | {score_text}")
            st.text((getattr(doc, "page_content", "") or "").strip())


def main() -> None:
    st.set_page_config(page_title="Chatbox 图文问答", page_icon=":robot_face:")
    st.title("Chatbox 图文问答")
    st.caption("基于 chatbox.py 的图文展示前端，对话历史会保留在本次会话中。")

    if not LLM_API_KEY or not LLM_BASE_URL:
        st.error("未配置 QIAN_API_KEY / QIAN_BASE_URL，请先检查 .env。")
        st.stop()
    if not create_embeddings():
        st.error("Embedding 初始化失败，请检查 .env 中的 embedding 配置。")
        st.stop()

    try:
        rag_chain = get_rag_chain()
    except Exception as e:
        st.error(f"初始化失败: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_docs" not in st.session_state:
        st.session_state.last_docs = None

    workspace_root = Path(__file__).resolve().parent

    with st.sidebar:
        if st.button("清空对话", type="secondary"):
            st.session_state.messages = []
            st.session_state.last_docs = None
            st.rerun()

    for i, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "user")
        with st.chat_message(role):
            if role == "user":
                st.markdown(msg.get("content", ""))
            else:
                is_last = i == len(st.session_state.messages) - 1
                show_docs = st.session_state.last_docs if is_last else None
                _render_assistant_message(msg, workspace_root, show_docs)

    if prompt := st.chat_input("输入问题，例如：堡垒机卡顿怎么处理？"):
        prompt = prompt.strip()
        if not prompt:
            st.warning("请输入问题后再发送。")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        # 立即在当前轮次渲染用户消息，避免等回答完成后才显示。
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索并生成回答中..."):
                answer = rag_chain.invoke(prompt)
            _render_assistant_message({"role": "assistant", "content": answer}, workspace_root, None)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_docs = None


if __name__ == "__main__":
    main()
