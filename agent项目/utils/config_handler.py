"""
yaml
"""

import yaml
from .path_tool import get_abs_root

def load_rag_config(config_path:str = get_abs_root("config/rag.yml"),encoding='utf-8'):
    with open(config_path,'r',encoding=encoding) as f:
        return  yaml.load(f,Loader=yaml.FullLoader)


def load_chroma_config(config_path:str = get_abs_root("config/chroma.yml"),encoding='utf-8'):
    with open(config_path,'r',encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)


def load_prompts_config(config_path:str = get_abs_root("config/prompts.yml"),encoding='utf-8'):
    with open(config_path,'r',encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)


def load_agent_config(config_path:str = get_abs_root("config/agent.yml"),encoding='utf-8'):
    with open(config_path,'r',encoding=encoding) as f:
       return yaml.load(f,Loader=yaml.FullLoader)


rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()

if __name__ == '__main__':
    print(rag_config["embedding_model_name"])