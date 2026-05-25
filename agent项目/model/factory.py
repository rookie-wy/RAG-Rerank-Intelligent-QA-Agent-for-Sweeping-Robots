from abc import ABC, abstractmethod
from typing import Optional
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_config
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv
load_dotenv()

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self)->Embeddings | BaseChatModel:
        return init_chat_model(model=rag_config["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self)->Embeddings | BaseChatModel:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()