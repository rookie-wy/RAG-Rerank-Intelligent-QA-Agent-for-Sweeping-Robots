import os,hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import  PyPDFLoader,TextLoader



def get_file_md5_hex(file_path:str):
    if not os.path.exists(file_path):
        logger.error(f"[mde计算]文件{file_path}不存在")
        return

    if not os.path.isfile(file_path):
        logger.error(f"[mde计算]路径{file_path}不是文件")
        return

    md5_obj = hashlib.md5()
    chunk_size = 4096 #固定读取大小4B
    try:
        with open(file_path, "rb") as f:
            while chunk:= f.read(chunk_size):
                md5_obj.update(chunk)
                """
                chunk = f.read(chunk_size)
                while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
                """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e :
        logger.error(f"计算文件{file_path}md5失败，{str(e)}")
        return None

def listdir_with_allowed_type(path:str,allowed_type:tuple[str]):
    file_list = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return  allowed_type

    for f in os.listdir(path):
        if f.endswith(allowed_type):
            file_list.append(os.path.join(path, f))
    return tuple(file_list)


def pdf_loader(file_path:str,password)->Document:
    return  PyPDFLoader(file_path,password).load()

def txt_loader(file_path:str)->Document:
    return TextLoader(file_path,encoding="utf-8").load()
