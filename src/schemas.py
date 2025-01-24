from pydantic import BaseModel, Field
from typing import Optional, Union, List


class OpenAiRequest(BaseModel):
    image: Union[str, bytes]  # base64 or URL
    prompt: Optional[str] = None


class OpenAiResponse(BaseModel):
    description: str
    model: str = Field(
        default="SmolVLM-256M-Instruct"
    )  # HuggingFaceTB/SmolVLM-256M-Instruct


class OllamaMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]
    images: Optional[List[str]] = None


class OllamaRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OllamaMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class OllamaResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaMessage
    done: bool
