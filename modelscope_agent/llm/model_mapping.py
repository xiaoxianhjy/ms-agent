from modelscope_agent.llm.modelscope_llm import ModelScope
from modelscope_agent.llm.openai_llm import OpenAI

all_services_mapping = {
    'modelscope': ModelScope,
    'openai': OpenAI,
}