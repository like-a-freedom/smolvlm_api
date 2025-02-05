import os


class Settings:
    MODEL_NAME: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    DEFAULT_PROMPT: str = "Analyze the objects in these images from the security camera. Focus on the actions, behavior, and potential intent of the objects, rather than just describing its appearance."
    BASE_MODEL_DIR: str = os.getenv("MODEL_CACHE_DIR", "../model_cache")
    MAX_IMAGE_SIZE: int = 2048  # Longest edge
    QUANTIZE_MODEL: str = os.getenv("QUANTIZE_MODEL", "True")


settings = Settings()
