import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from utils.path_tool import get_abs_root
from langchain_chroma import Chroma
from utils.config_handler import chroma_config,rag_config
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
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_config["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

    # def get_retriever(self):
    #     return self.vector_store.as_retriever(search_kwargs={"k":chroma_config["k"]})

    def get_retriever(self):
        # 1. 基础向量检索
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        # 2. 从向量库拿到所有文档，构建 BM25
        docs_data = self.vector_store.get()
        doc_objects = [Document(page_content=d) for d in docs_data["documents"]]

        bm25_retriever = BM25Retriever.from_documents(doc_objects)
        bm25_retriever.k = 10

        # 3. 混合检索
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.8, 0.2]
        )

        # 4. 重排模型
        rerank_model = HuggingFaceCrossEncoder(
            model_name=rag_config["rerank_model_name"],
            model_kwargs={"device": "cpu"}
        )
        compressor = CrossEncoderReranker(model=rerank_model, top_n=2)

        # 5. 最终检索器
        return ContextualCompressionRetriever(
            base_retriever=ensemble_retriever,
            base_compressor=compressor
        )

    def load_document(self):
        """
        从数据文件中读取数据文件，转为向量存入向量数据库中
        要计算文件的md5值去重
        ：:return:None
        """

        def check_md5_hex(md5_for_check:str):
            if not os.path.exists(get_abs_root(chroma_config["md5_hex_store"])):
                #创建文件
                open(get_abs_root(chroma_config["md5_hex_store"]), "w",encoding="utf-8").close()
                return False
            with open(get_abs_root(chroma_config["md5_hex_store"]),"r",encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line==md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check:str):
            with open(get_abs_root(chroma_config["md5_hex_store"]),"a",encoding="utf-8") as f:
                f.write(md5_for_check+"\n")

        def get_file_documents(read_path:str):
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path,None)

            if read_path.endswith(".txt"):
                return txt_loader(read_path)

            return []
        allowed_file_path = listdir_with_allowed_type(
            get_abs_root(chroma_config["data_path"]),
            tuple(chroma_config["allow_konwledge_file_type"])
        )

        for path in allowed_file_path:
            #获取md5
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库中")
                continue

            try:
                documents:list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue
                spliter_document = self.spliter.split_documents(documents)

                if not spliter_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(spliter_document)

                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path}内容加载成功")
            except Exception as e:
                #exc_info为true会记录详细报错堆栈，为False则仅记录报错信息本身
                logger.error(f"[加载知识库]{path}内容加载失败：{str(e)}",exc_info=True)
                continue


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("扫地机器人如何保养")
    for r in res:
        print(r.page_content)
        print("-"*30)

