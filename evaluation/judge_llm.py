import os

cuda_bin = r"D:\uni-chatbot\.venv\Lib\site-packages\nvidia\cu13\bin\x86_64"

os.add_dll_directory(cuda_bin)
os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]

from typing import Type
from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_community.chat_models import ChatLlamaCpp


class LocalQwenJudge(DeepEvalBaseLLM):

    def __init__(self):
        self.model = ChatLlamaCpp(
            model_path = r"D:\uni-chatbot\models\Qwen3.5-9B-Q4_0.gguf"
,
            temperature=0.0,
            max_tokens=1024,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )

    def load_model(self):
        return self.model

    def generate(
        self,
        prompt: str,
        schema: Type[BaseModel] | None = None
    ):
        if schema is None:
            response = self.model.invoke(prompt)
            return response.content

        structured_model = self.model.with_structured_output(
            schema
        )

        return structured_model.invoke(prompt)

    async def a_generate(
        self,
        prompt: str,
        schema: Type[BaseModel] | None = None
    ):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Qwen3.5-9B-Q4_0.gguf"