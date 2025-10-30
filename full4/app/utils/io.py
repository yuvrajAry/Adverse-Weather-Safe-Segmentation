from fastapi import UploadFile
from PIL import Image
import io
import base64
import os


async def read_image_from_upload(file: UploadFile) -> Image.Image:
    content = await file.read()
    return Image.open(io.BytesIO(content)).convert("RGB")


def image_to_base64_png(img) -> str:
    if img is None:
        return ""
    if isinstance(img, Image.Image):
        pil_img = img
    else:
        # assume numpy array BGR or RGB
        import numpy as np
        if isinstance(img, np.ndarray):
            from PIL import Image
            if img.ndim == 2:
                pil_img = Image.fromarray(img.astype('uint8'))
            else:
                pil_img = Image.fromarray(img.astype('uint8'))
        else:
            raise ValueError("Unsupported image type for base64 encoding")
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return b64


def read_image_from_path(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")

