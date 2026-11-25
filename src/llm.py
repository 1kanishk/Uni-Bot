import os
from langchain_community.chat_models import ChatLlamaCpp
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

cuda_bin = r"D:\uni-chatbot\.venv\Lib\site-packages\nvidia\cu13\bin\x86_64"

os.add_dll_directory(cuda_bin)
os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]

LLM_MODEL = r"D:\uni-chatbot\models\Qwen3.5-9B-Q4_0.gguf"

def get_llm():
    return ChatLlamaCpp(
        model_path=LLM_MODEL,
        temperature=0.0,
        max_tokens=1536,
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False
    )

def get_groq_llm():
    return ChatGroq(
        model_name="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=2048
    )