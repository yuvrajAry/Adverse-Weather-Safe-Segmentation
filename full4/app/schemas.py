from pydantic import BaseModel
from typing import Optional


class InferResponse(BaseModel):
    mask_png: str
    overlay_png: Optional[str] = None
    message: str

