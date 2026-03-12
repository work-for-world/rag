import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from chatbox import (
    LLM_API_KEY as QWEN_API_KEY,
    LLM_BASE_URL as QWEN_BASE_URL,
    LLM_MODEL_NAME as QWEN_MODEL_NAME,
    build_kb_keyword_index,
    create_hybrid_retriever,
    create_rag_chain,
    load_existing_vectorstore,
)

DEFAULT_JUDGE_MODEL = "gpt-5.1-codex-mini"
NO_INFO_REPLY = "抱歉，当前招联IT数据库中不存在您要搜索的信息，我们会尽力添加"


def classify_skip_case(skip_retrieval: bool, ref_has_source: bool) -> str:
    """按是否应检索(source 是否为空)与是否被 skip 做四分类。"""
    if skip_retrieval and not ref_has_source:
        return "true_skip"
    if (not skip_retrieval) and (not ref_has_source):
        return "false_non_skip"
    if skip_retrieval and ref_has_source:
        return "false_skip"
    return "true_non_skip"


def is_no_info_answer(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    target = re.sub(r"\s+", "", NO_INFO_REPLY)
    return (target in normalized) or normalized.startswith("抱歉，当前招联IT数据库中不存在您要搜索的信息")


def get_judge_config() -> tuple[str, str, str]:
    model_name = os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    return model_name, api_key, base_url


def normalize_path_text(p: str) -> str:
    s = (p or "").strip().replace("\\", "/").lower()
    s = re.sub(r"/+", "/", s)
    return s


def canonical_source_path(path_text: str) -> str:
    s = normalize_path_text(path_text).lstrip("./").strip()
    if s.startswith("pic_text/"):
        s = s[len("pic_text/") :]
    if "/pic_text/" in s:
        s = s.split("/pic_text/", 1)[1]
    return s


def to_relative_path_text(path_text: str, root: Path) -> str:
    raw = (path_text or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        try:
            return p.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            return p.as_posix()
    return raw.replace("\\", "/").lstrip("./")


def extract_answer_and_docs(rag_result: Any) -> tuple[str, list[dict[str, Any]], bool]:
    if isinstance(rag_result, dict):
        answer = str(rag_result.get("answer", "")).strip()
        docs_raw = rag_result.get("docs") or []
        docs = [d for d in docs_raw if isinstance(d, dict)]
        skip_retrieval = bool(rag_result.get("skip_retrieval", False))
        return answer, docs, skip_retrieval
    return str(rag_result).strip(), [], False


def load_dataset(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                {
                    "question": str(item.get("question") or item.get("q", "")).strip(),
                    "answer": str(item.get("answer") or item.get("a", "")).strip(),
                    "source_file": str(item.get("source_file", "")).strip(),
                }
            )
    return [r for r in rows if r["question"] and r["answer"]]


def llm_judge_score(judge_llm: ChatOpenAI, question: str, reference: str, prediction: str) -> tuple[float, str]:
    prompt = f"""你是RAG评测裁判。请根据“问题、参考答案、模型答案”进行打分。
只输出严格 JSON：{{"score": 0到1之间小数, "reason": "一句话原因"}}。

评分标准：
- 1.0：关键信息完整且正确；
- 0.7~0.9：基本正确，少量细节缺失；
- 0.4~0.6：部分相关，但关键步骤/结论缺失或有错误；
- 0.0~0.3：基本错误、答非所问或幻觉明显。

问题：
{question}

参考答案：
{reference}

模型答案：
{prediction}
"""
    raw = judge_llm.invoke(prompt)
    text = getattr(raw, "content", str(raw)).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return 0.0, f"judge parse failed: {text[:120]}"
        data = json.loads(m.group(0))
    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    reason = str(data.get("reason", "")).strip()
    return score, reason


def build_rag(timeout_s: int) -> Any:
    db = load_existing_vectorstore()
    if not db:
        raise RuntimeError("向量库加载失败，请先确认索引已构建。")
    retriever = create_hybrid_retriever(db)
    if not retriever:
        raise RuntimeError("检索器构建失败。")
    kb_keywords = build_kb_keyword_index(db)
    if not QWEN_API_KEY or not QWEN_BASE_URL:
        raise RuntimeError("未配置 QIAN_API_KEY 或 QIAN_BASE_URL，请检查 .env")
    llm = ChatOpenAI(
        model_name=QWEN_MODEL_NAME,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        temperature=0.0,
        max_tokens=1024,
        timeout=timeout_s,
        request_timeout=timeout_s,
        max_retries=2,
    )
    return create_rag_chain(retriever, llm, kb_keywords=kb_keywords, use_memory=False)


def build_rag_vector_only(timeout_s: int, vector_k: int) -> Any:
    db = load_existing_vectorstore()
    if not db:
        raise RuntimeError("向量库加载失败，请先确认索引已构建。")
    retriever = db.as_retriever(search_kwargs={"k": max(1, vector_k)})
    kb_keywords = build_kb_keyword_index(db)
    if not QWEN_API_KEY or not QWEN_BASE_URL:
        raise RuntimeError("未配置 QIAN_API_KEY 或 QIAN_BASE_URL，请检查 .env")
    llm = ChatOpenAI(
        model_name=QWEN_MODEL_NAME,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        temperature=0.0,
        max_tokens=1024,
        timeout=timeout_s,
        request_timeout=timeout_s,
        max_retries=2,
    )
    return create_rag_chain(retriever, llm, kb_keywords=kb_keywords, use_memory=False)


def get_existing_csv_state(csv_path: Path) -> tuple[int, int]:
    if not csv_path.exists():
        return 0, 0
    max_id = 0
    count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            count += 1
            try:
                max_id = max(max_id, int(str(row.get("id", "0")).strip() or "0"))
            except Exception:
                continue
    return max_id, count


def main() -> None:
    parser = argparse.ArgumentParser(description="简化版评测：仅输出 skip/hit/judge")
    parser.add_argument("--dataset", default="qa_testset_merged_shuffled.jsonl", help="测试集路径（jsonl）")
    parser.add_argument("--offset", type=int, default=0, help="手动跳过前 N 条")
    parser.add_argument("--limit", type=int, default=0, help="仅评测前 N 条，0 表示全部")
    parser.add_argument("--out-prefix", default="rag_eval_merged", help="输出文件前缀")
    parser.add_argument("--append-csv", action="store_true", help="追加写入 detail.csv")
    parser.add_argument("--auto-resume", action="store_true", help="按已有 detail.csv 条数自动续跑")
    parser.add_argument("--kb-root", default=r"E:\python_code\langchain\pic_text")
    parser.add_argument("--judge-model", default="", help="裁判模型（默认 gpt-5.1-codex-mini）")
    parser.add_argument("--judge-threshold", type=float, default=0.8)
    parser.add_argument("--request-timeout", type=int, default=180, help="单次模型请求超时（秒）")
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "vector_only"],
        default="hybrid",
        help="检索模式：hybrid=混合检索，vector_only=仅向量检索",
    )
    parser.add_argument("--vector-k", type=int, default=4, help="仅向量检索模式下的 top-k")
    parser.add_argument(
        "--hit-mode",
        choices=["any", "top1"],
        default="any",
        help="hit 判定方式：any=检索结果任意命中；top1=仅首条命中",
    )
    args = parser.parse_args()

    load_dotenv()
    judge_model_name, judge_api_key, judge_base_url = get_judge_config()
    if not judge_api_key or not judge_base_url:
        raise RuntimeError("未配置 OPENAI_API_KEY 或 OPENAI_BASE_URL，请检查 .env")

    dataset_path = Path(args.dataset)
    out_csv = Path(f"{args.out_prefix}_detail.csv")
    out_summary = Path(f"{args.out_prefix}_summary.json")

    rows = load_dataset(dataset_path)
    id_base, existing_count = (get_existing_csv_state(out_csv) if args.append_csv else (0, 0))
    total_offset = max(0, args.offset) + (existing_count if args.auto_resume and args.append_csv else 0)
    if args.limit > 0:
        rows = rows[: args.limit]
    if total_offset > 0:
        rows = rows[total_offset:]
    if not rows:
        raise RuntimeError("测试集为空（或已全部跑完）。")

    kb_root = Path(args.kb_root)
    if args.retrieval_mode == "vector_only":
        rag_chain = build_rag_vector_only(
            timeout_s=max(5, args.request_timeout),
            vector_k=args.vector_k,
        )
    else:
        rag_chain = build_rag(timeout_s=max(5, args.request_timeout))
    judge_llm = ChatOpenAI(
        model_name=(args.judge_model.strip() or judge_model_name),
        api_key=judge_api_key,
        base_url=judge_base_url,
        temperature=0.0,
        max_tokens=500,
        timeout=max(5, args.request_timeout),
        request_timeout=max(5, args.request_timeout),
        max_retries=2,
    )

    details: list[dict[str, Any]] = []
    skip_flags: list[int] = []
    skip_cases: list[str] = []
    hits: list[int] = []
    judge_scores: list[float] = []

    try:
        for i, row in enumerate(rows, start=1):
            q = row["question"]
            ref = row["answer"]
            src = row["source_file"]

            rag_result = rag_chain.invoke(q)
            pred, docs, skip_retrieval = extract_answer_and_docs(rag_result)
            no_info_answer = is_no_info_answer(pred)
            eval_skip_retrieval = bool(skip_retrieval) or no_info_answer

            retrieved_sources = [to_relative_path_text(str(d.get("source", "")).strip(), kb_root) for d in docs if d.get("source")]
            retrieved_sources_for_eval = [] if eval_skip_retrieval else retrieved_sources
            output_source_file = "" if eval_skip_retrieval else (retrieved_sources_for_eval[0] if retrieved_sources_for_eval else "")
            ref_has_source = bool(src.strip())
            out_has_source = bool(output_source_file.strip())
            # skip 指标仅评估门控原始输出，不掺入 no_info_answer
            raw_skip_retrieval = bool(skip_retrieval)
            skip_case = classify_skip_case(skip_retrieval=raw_skip_retrieval, ref_has_source=ref_has_source)
            if not ref_has_source and not out_has_source:
                hit = 1
            elif ref_has_source != out_has_source:
                hit = 0
            else:
                src_norm = canonical_source_path(to_relative_path_text(src, kb_root))
                retrieved_norms = [canonical_source_path(s) for s in retrieved_sources_for_eval]
                if args.hit_mode == "top1":
                    hit = 1 if (retrieved_norms and retrieved_norms[0] == src_norm) else 0
                else:
                    hit = 1 if any(s == src_norm for s in retrieved_norms) else 0

            score, reason = llm_judge_score(judge_llm, q, ref, pred)

            skip_flags.append(int(raw_skip_retrieval))
            skip_cases.append(skip_case)
            hits.append(hit)
            judge_scores.append(score)
            details.append(
                {
                    "id": id_base + i,
                    "question": q,
                    "reference_answer": ref,
                    "pred_answer": pred,
                    "retrieved_sources": " | ".join(retrieved_sources),
                    "source_file_norm": canonical_source_path(output_source_file),
                    "reference_source_file_norm": canonical_source_path(to_relative_path_text(src, kb_root)),
                    "skip_retrieval": int(raw_skip_retrieval),
                    "raw_skip_retrieval": int(raw_skip_retrieval),
                    "no_info_answer": int(no_info_answer),
                    "skip_case": skip_case,
                    "retrieval_hit": hit,
                    "judge_score": round(score, 4),
                    "judge_reason": reason,
                }
            )

            print(
                f"[{i}/{len(rows)}|id={id_base + i}] "
                f"skip={int(raw_skip_retrieval)}(oos={int(no_info_answer)},{skip_case}) "
                f"hit={hit} judge={score:.3f} "
                f"q={q[:24]}{'...' if len(q) > 24 else ''}"
            )
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000)
    except KeyboardInterrupt:
        print("\n⚠️ 检测到 Ctrl+C，已停止并保存当前进度。")

    mode = "a" if args.append_csv and out_csv.exists() else "w"
    fieldnames = [
        "id",
        "question",
        "reference_answer",
        "pred_answer",
        "retrieved_sources",
        "source_file_norm",
        "reference_source_file_norm",
        "skip_retrieval",
        "raw_skip_retrieval",
        "no_info_answer",
        "skip_case",
        "retrieval_hit",
        "judge_score",
        "judge_reason",
    ]
    with out_csv.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for item in details:
            writer.writerow(item)

    total = len(details)
    true_skip_cnt = sum(1 for x in skip_cases if x == "true_skip")
    false_non_skip_cnt = sum(1 for x in skip_cases if x == "false_non_skip")
    false_skip_cnt = sum(1 for x in skip_cases if x == "false_skip")
    true_non_skip_cnt = sum(1 for x in skip_cases if x == "true_non_skip")
    gate_accuracy = ((true_skip_cnt + true_non_skip_cnt) / total) if total else 0.0
    # 以“应 skip”作为正类：recall = TP / (TP + FN)
    gate_recall = (true_skip_cnt / (true_skip_cnt + false_non_skip_cnt)) if (true_skip_cnt + false_non_skip_cnt) else 0.0
    summary = {
        "dataset": str(dataset_path),
        "retrieval_mode": args.retrieval_mode,
        "vector_k": int(args.vector_k),
        "hit_mode": args.hit_mode,
        "total": total,
        "skip_rate": round((sum(skip_flags) / total), 4) if total else 0.0,
        "gate_accuracy": round(gate_accuracy, 4),
        "gate_recall": round(gate_recall, 4),
        "hit_rate": round((sum(hits) / total), 4) if total else 0.0,
        "avg_judge_score": round(statistics.mean(judge_scores), 4) if judge_scores else 0.0,
        "judge_threshold": args.judge_threshold,
        "judge_pass_rate": round((sum(1 for s in judge_scores if s >= args.judge_threshold) / total), 4) if total else 0.0,
        "request_timeout_sec": args.request_timeout,
        "auto_resume": bool(args.auto_resume),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 评估完成 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细(CSV):  {out_csv.resolve()}")
    print(f"汇总(JSON):  {out_summary.resolve()}")


if __name__ == "__main__":
    main()

