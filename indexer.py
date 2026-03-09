import os
import shutil
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 加载环境变量
load_dotenv()

# ================= 配置区域 =================
BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "pic_text"))  # 知识库源文件路径
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db_qwen3"))  # 向量库存储路径

# 阿里百炼 Embedding 配置（indexer.py 使用）
QIAN_API_KEY = os.getenv("QIAN_API_KEY")
QIAN_BASE_URL = os.getenv("QIAN_BASE_URL")
QIAN_EMBED_MODEL = os.getenv("QIAN_EMBED_MODEL", "text-embedding-v4")
CLEAR_EXISTING_DB = os.getenv("CLEAR_EXISTING_DB", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)


# ================= 阿里百炼 Embedding 类（OpenAI 兼容接口）=================
class DashScopeEmbeddings(Embeddings):
    """使用阿里百炼 text-embedding-v4，通过 OpenAI 兼容 base_url 调用。"""

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v4"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # 百炼单次请求最多 10 条
    BATCH_SIZE = 10

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = []
        try:
            for i in range(0, len(texts), self.BATCH_SIZE):
                batch = texts[i : i + self.BATCH_SIZE]
                completion = self.client.embeddings.create(model=self.model, input=batch)
                sorted_data = sorted(completion.data, key=lambda x: x.index)
                result.extend([item.embedding for item in sorted_data])
            return result
        except Exception as e:
            print(f"❌ 百炼 Embedding API 失败: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ================= 数据加载与处理逻辑 =================
def load_documents():
    print(f"📂 正在从 {DATA_PATH} 加载文档...")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"⚠️ 目录不存在，已创建：{DATA_PATH}\n请将您的 .txt 文件放入此目录后重新运行。")
        return []

    class FlexibleTextLoader(TextLoader):
        def __init__(self, file_path, encoding_attempts=["utf-8", "gbk", "gb2312"]):
            self.file_path = file_path
            self.encoding_attempts = encoding_attempts

        def lazy_load(self):
            for encoding in self.encoding_attempts:
                try:
                    with open(self.file_path, "r", encoding=encoding) as f:
                        text = f.read()
                        from langchain_core.documents import Document

                        yield Document(
                            page_content=text, metadata={"source": self.file_path, "encoding": encoding}
                        )
                        return
                except UnicodeDecodeError:
                    continue
            raise RuntimeError(f"无法读取文件: {self.file_path}")

    loader = DirectoryLoader(
        DATA_PATH,
        loader_cls=lambda path: FlexibleTextLoader(path),
        glob="**/*.txt",
        show_progress=True,
    )

    try:
        docs = loader.load()
        print(f"✅ 成功加载 {len(docs)} 个文档。")
        return docs
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return []


# 分隔符：按此切分文档
CHUNK_SEP = "###"


def split_documents(docs):
    """按 "###" 对每个文档切分，每个非空块为一个 Document。"""
    if not docs:
        return []
    splits = []
    for doc in docs:
        parts = doc.page_content.split(CHUNK_SEP)
        for part in parts:
            text = part.strip()
            if not text:
                continue
            splits.append(Document(page_content=text, metadata=dict(doc.metadata)))
    print(f"✅ 按「{CHUNK_SEP}」分割完成，共 {len(splits)} 个片段。")
    return splits


def build_vector_store(splits, embeddings):
    print(f"🚀 正在构建向量库并保存至: {CHROMA_PATH}")

    if CLEAR_EXISTING_DB and os.path.exists(CHROMA_PATH):
        print(f"🧹 检测到 CLEAR_EXISTING_DB=true，正在清理旧向量库: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)

    # persist_directory 指向本地文件夹
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    if len(splits) > 0:
        # 添加文档：CLEAR_EXISTING_DB=false 时为追加模式，true 时为全新重建
        db.add_documents(splits)
        print(f"✅ 向量库构建完成！当前库中总片段数: {db._collection.count()}")
    else:
        print("⚠️ 没有文档片段可写入。")

    return db


if __name__ == "__main__":
    # 1. 检查配置（阿里百炼）
    if not QIAN_API_KEY or not QIAN_BASE_URL:
        print("❌ 错误: 请在 .env 中配置 QIAN_API_KEY 和 QIAN_BASE_URL")
        exit(1)

    # 2. 执行流程
    docs = load_documents()
    if docs:
        splits = split_documents(docs)

        embeddings = DashScopeEmbeddings(
            api_key=QIAN_API_KEY,
            base_url=QIAN_BASE_URL,
            model=QIAN_EMBED_MODEL,
        )

        build_vector_store(splits, embeddings)
        print("\n🎉 索引构建完毕（百炼 embedding）。可运行 chatbox 进行问答。")
    else:
        print("⚠️ 未找到任何文档，跳过构建。")
