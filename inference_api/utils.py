import io
import requests
from PIL import Image


def ensure_3dim(img):
    if len(img.size)==2:
        img = img.convert('RGB')
    return img


def url_to_image(input: str):
    try:
        response = requests.get(input, stream=True)
    except:
        return None
    image = Image.open(io.BytesIO(response.content))
    image = ensure_3dim(image)
    return image
