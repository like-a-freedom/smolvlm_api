import base64
import io
from datetime import datetime, timezone

import requests
from fastapi import APIRouter, HTTPException
from PIL import Image

from config import settings
from exceptions import VisionServiceException
from schemas import (
    OllamaMessage,
    OllamaRequest,
    OllamaResponse,
    OpenAiRequest,
    OpenAiResponse,
)
from vision_service import SmolvlmVisionService

openai_router = APIRouter()
ollama_router = APIRouter()
vision_service = SmolvlmVisionService()


@openai_router.post("/chat/completions", response_model=OpenAiResponse)
async def analyze_image(request: OpenAiRequest):
    """
    OpenAI-compatible endpoint for image analysis
    Supports base64 encoded images and image URLs.
    """
    try:
        # Handle image input (base64 or URL)
        if isinstance(request.image, str):
            if request.image.startswith(("http://", "https://")):
                try:
                    response = requests.get(request.image)
                    image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    raise HTTPException(
                        status_code=400, detail=f"Invalid image URL: {e}"
                    ) from e
            else:
                # Assume base64 encoded image
                try:
                    image_data = base64.b64decode(request.image)
                    image = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    raise HTTPException(
                        status_code=400, detail=f"Invalid base64 image: {e}"
                    ) from e
        elif isinstance(request.image, bytes):
            # Handle raw image bytes
            image = Image.open(io.BytesIO(request.image))
        else:
            raise HTTPException(status_code=400, detail="Unsupported image format")

        prompt = request.prompt or settings.DEFAULT_PROMPT
        description = vision_service.describe_image(image, prompt)

        return OpenAiResponse(description=description)

    except VisionServiceException as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@openai_router.post("/images/generations")
async def generate_image(request: OpenAiRequest):
    raise HTTPException(status_code=501, detail="Image generation not implemented")


@ollama_router.post("/api/chat", response_model=OllamaResponse)
async def ollama_chat_completion(request: OllamaRequest):
    try:
        last_message = request.messages[-1]

        # Handle both image formats
        image_data = None
        prompt = ""

        if isinstance(last_message.content, list):
            # Standard Ollama format with type/image
            for item in last_message.content:
                if item.get("type") == "image":
                    image_data = item.get("image")
                elif item.get("type") == "text":
                    prompt = item.get("text", "")
        else:
            # Handle direct images array format
            prompt = last_message.content
            if hasattr(last_message, "images") and last_message.images:
                image_data = last_message.images[0]

        if not image_data:
            raise HTTPException(status_code=400, detail="No image provided")

        # Convert base64 to PIL Image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image data: {e}"
            ) from e

        text_answer = vision_service.describe_image(image, user_prompt=prompt)

        return OllamaResponse(
            model=settings.MODEL_NAME,
            created_at=datetime.now(timezone.utc).isoformat(),
            message=OllamaMessage(role="assistant", content=text_answer),
            done=True,
        )

    except VisionServiceException as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
