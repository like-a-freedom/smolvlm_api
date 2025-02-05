import os

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, QuantoConfig

from config import settings
from exceptions import ImageAnalysisError, ModelDownloadError, ModelLoadError


class SmolvlmVisionService:
    def __init__(self, base_dir=settings.BASE_MODEL_DIR):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.MODEL_NAME

        try:
            self.quanto_config = QuantoConfig(weights="int8")
            self.model_path = self._download_model(self.model_name)
            self.processor = self._load_processor()
            self.model = self._load_model()
            print(
                f"Model memory consumption: {self._get_inmemory_model_size(self.model):.2f} MB"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize vision service: {e}")

    def _download_model(self, model_name, model_type="model") -> str:
        """
        Download a specific model locally
        """
        try:
            model_dir = os.path.join(self.base_dir, model_name.replace("/", "_"))
            os.makedirs(model_dir, exist_ok=True)

            print(f"Starting download of {model_type} {model_name}")

            local_path = snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
            )

            print(f"{model_type.capitalize()} downloaded successfully to {local_path}")
            return local_path

        except Exception as e:
            raise ModelDownloadError(f"Error downloading model: {e}")

    def _load_processor(self):
        """
        Load the tokenizer from local path
        """
        try:
            return AutoProcessor.from_pretrained(self.model_name)
        except Exception as e:
            raise Exception(f"Error loading processor: {e}")

    def _load_model(self):
        """
        Load the model from local path
        """
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                # torch_dtype=torch.bfloat16,
                quantization_config=self.quanto_config
                if settings.QUANTIZE_MODEL == "True"
                else None,
                _attn_implementation="flash_attention_2"
                if self.device == "cuda"
                else "eager",
            ).to(self.device)
            return model
        except Exception as e:
            raise ModelLoadError(f"Error loading model: {e}")

    def _get_inmemory_model_size(self, model):
        return sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        longest_edge = max(image.size)
        if longest_edge > settings.MAX_IMAGE_SIZE:
            scale = settings.MAX_IMAGE_SIZE / longest_edge
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def describe_image(self, image, user_prompt: str) -> str:
        """
        Analyze an image using the VLM model
        Args:
            image: The image to analyze
            messages: Chat template messages
        Returns:
            Generated text description
        """
        image = self._resize_image(image)
        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": settings.DEFAULT_PROMPT
                        if not user_prompt
                        else user_prompt,
                    },
                ],
            },
        ]

        try:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(self.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=500)
            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            # Extract only the assistant's response by splitting on "Assistant:"
            # and taking everything after it
            raw_response = generated_texts[0]
            if "Assistant:" in raw_response:
                return raw_response.split("Assistant:")[-1].strip()
            return raw_response.strip()
        except Exception as e:
            raise ImageAnalysisError(f"Error describing image: {e}")
