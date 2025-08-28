import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load NVIDIA API key from environment
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not set in environment variables.")

# Initialize NVIDIA LLM wrapper
llm = ChatNVIDIA(
    model="meta/llama3-8b-instruct",  # or "llama3-70b-instruct", etc.
    api_key=NVIDIA_API_KEY
)

# Optional: LangChain prompt and chain
prompt = PromptTemplate.from_template("Explain the benefits of {topic}")
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
response = chain.run("NVIDIA LLMs")
print(response)
