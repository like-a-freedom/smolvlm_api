"""
This script quantizes the model and processor and saves them to the model directory.
It's just service optional, but it's a good idea to quantize the model to reduce memory consumption.
"""

from optimum.quanto import (
    QuantizedModelForCausalLM,
    qint8,
)

from config import settings
from vision_service import SmolvlmVisionService

vision_service = SmolvlmVisionService()
model = vision_service._load_model()
processor = vision_service._load_processor()


qmodel = QuantizedModelForCausalLM.quantize(
    model, weights=qint8, activations=None, exclude="lm_head"
)
qmodel.save_pretrained(settings.BASE_MODEL_DIR + "/smolvlm_quantized")
processor.save_pretrained(settings.BASE_MODEL_DIR + "/smolvlm_quantized")
