import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from utils.path_tool import get_abs_root
from langchain_milvus import Milvus
from utils.config_handler import chroma_config, rag_config
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

load_dotenv()


class VectorStoreService:
    def __init__(self):
        # ===================== 【唯一正确写法】全新 Milvus 客户端连接 =====================
        self.vector_store = Milvus(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            # 直接用 uri 连接 Docker Milvus，无任何旧API
            connection_args={
                "uri": "http://localhost:19530",
            },
            # 自动创建集合（关键）
            auto_id=True,
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

    def get_retriever(self):
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        # 正确获取文档（兼容服务端 Milvus）
        try:
            results = self.vector_store.get()
            docs = []
            for item in results:
                # 兼容所有 Milvus 返回格式
                if "text" in item:
                    docs.append(Document(page_content=item["text"]))
                elif "page_content" in item:
                    docs.append(Document(page_content=item["page_content"]))

            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 10

            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.8, 0.2]
            )
        except:
            # 降级保护
            ensemble_retriever = vector_retriever

        # 重排
        rerank_model = HuggingFaceCrossEncoder(
            model_name=rag_config["rerank_model_name"],
            model_kwargs={"device": "cpu"}
        )
        compressor = CrossEncoderReranker(model=rerank_model, top_n=2)

        return ContextualCompressionRetriever(
            base_retriever=ensemble_retriever,
            base_compressor=compressor
        )

    def load_document(self):
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_root(chroma_config["md5_hex_store"])):
                open(get_abs_root(chroma_config["md5_hex_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_root(chroma_config["md5_hex_store"]), "r", encoding="utf-8") as f:
                return md5_for_check in [line.strip() for line in f]

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_root(chroma_config["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path, None)
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            return []

        allowed_file_path = listdir_with_allowed_type(
            get_abs_root(chroma_config["data_path"]),
            tuple(chroma_config["allow_konwledge_file_type"])
        )

        for path in allowed_file_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path} 已存在")
                continue

            try:
                documents = get_file_documents(path)
                if not documents:
                    continue

                splits = self.spliter.split_documents(documents)
                self.vector_store.add_documents(splits)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 加载成功")

            except Exception as e:
                logger.error(f"[加载知识库]{path} 失败: {str(e)}", exc_info=True)
                continue


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()

    retriever = vs.get_retriever()
    res = retriever.invoke("扫地机器人坏了怎么办")

    for r in res:
        print("-" * 50)
        print(r.page_content)