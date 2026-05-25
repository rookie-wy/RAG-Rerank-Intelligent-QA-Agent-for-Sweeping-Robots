from langchain.agents import create_agent

from model.factory import chat_model
from utils.prompt_loader import load_main_prompt
from tools.agent_tools import (rag_summarize,get_weather,get_month_arr,get_user_id,get_user_location,
                               fill_context_for_report,fetch_external_data)
from tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_main_prompt(),
            tools=[rag_summarize,get_weather,get_month_arr,get_user_id,get_user_location,
                               fill_context_for_report,fetch_external_data],
            middleware=[monitor_tool,log_before_model,report_prompt_switch],
        )

    def execute_stream(self,query):
        input_dict =  {
            "messages":[
            {"role":"user","content":query
             },
            ]}

        for chunk in   self.agent.stream(input_dict,stream_mode="values",context={"report":False}):
            last_messages = chunk["messages"][-1]
            yield last_messages.content.strip()+"\n"


if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk,end="",flush=True)