from fastapi import FastAPI
from routes import ollama_router, openai_router

app = FastAPI(
    title="HF SmolVLM256 API Service",
    description="OpenAI-compatible SmolVLM API Service",
    version="1.0.0",
)
app.include_router(openai_router, prefix="/v1")
app.include_router(ollama_router, prefix="")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
