from utils.path_tool import get_abs_root
from utils.config_handler import prompts_config
from utils.logger_handler import logger


def load_main_prompt():
    try:
        system_prompt_path = get_abs_root(prompts_config["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_main_prompt]配置项中没有main_prompt_path配置项")
        raise  e

    try:
        return  open(system_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_main_prompt]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompt():
    try:
        rag_prompt_path = get_abs_root(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompt]配置项中没有rag_summarize_prompt_path配置项")
        raise  e

    try:
        return  open(rag_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompt]解析RAG总结提示词出错，{str(e)}")
        raise e

def load_report_prompt():
    try:
        report_prompt_path = get_abs_root(prompts_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompt]配置项中没有report_prompt_path 配置项")
        raise  e

    try:
        return  open(report_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompt]解析报告生成提示词出错，{str(e)}")
        raise e


if __name__ == "__main__":
   print(load_report_prompt())
