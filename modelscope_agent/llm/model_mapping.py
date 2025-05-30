from .modelscope import ModelScope
from .openai import OpenAI

all_services_mapping = {
    'modelscope': ModelScope,
    'openai': OpenAI,
}