from typing import Dict
from fastapi import FastAPI

from transformers import ViTImageProcessor, ViTModel

from utils import url_to_image

feature_extraction_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
feature_extraction_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


app = FastAPI()

@app.post("/vector")
def predict(input: Dict):
    url = input.get("data")
    image = url_to_image(url)
    if not image:
        return None
    inputs = feature_extraction_processor(images=image, return_tensors="pt")
    output = feature_extraction_model(**inputs).pooler_output
    return output

