from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import base64
import io

from .config import AppConfig
from .schemas import InferResponse
from .utils.io import read_image_from_upload, image_to_base64_png, read_image_from_path
from .utils.preprocess import preprocess_pair
from .utils.heatmap import overlay_heatmap_on_image, colorize_mask
from .utils.uncertainty import compute_uncertainty
from .models import build_model


app = FastAPI(title="IDD-AW Safety Heatmaps Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer", response_model=InferResponse)
async def infer(
    rgb: UploadFile | None = File(None),
    nir: UploadFile | None = File(None),
    model: str = Form("fast_scnn"),
    fusion: str = Form("early"),
    uncertainty: str = Form("softmax_entropy"),
    mc_iters: int = Form(8),
    weights_path: Optional[str] = Form(None),
    output_size: Optional[str] = Form(None),  # e.g., "512x512" or "512x1024"
    rgb_path: Optional[str] = Form(None),
    nir_path: Optional[str] = Form(None),
):
    if rgb is not None and nir is not None:
        rgb_img = await read_image_from_upload(rgb)
        nir_img = await read_image_from_upload(nir)
    else:
        # Read from disk under dataset root
        base = AppConfig.DATASET_ROOT
        if not rgb_path or not nir_path:
            return JSONResponse({"detail": "Provide files or rgb_path and nir_path."}, status_code=400)
        rgb_img = read_image_from_path(rgb_path if rgb_path.startswith('D:') else f"{base}\\{rgb_path}")
        nir_img = read_image_from_path(nir_path if nir_path.startswith('D:') else f"{base}\\{nir_path}")

    width, height = AppConfig.DEFAULT_INFER_SIZE
    if output_size:
        try:
            w_str, h_str = output_size.lower().split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            pass

    input_tensor, rgb_resized = preprocess_pair(
        rgb_img,
        nir_img,
        (width, height),
        fusion_mode=fusion,
        mean=AppConfig.NORMALIZE_MEAN,
        std=AppConfig.NORMALIZE_STD,
    )

    net = build_model(model_name=model, fusion_mode=fusion, num_classes=AppConfig.NUM_CLASSES, weights_path=weights_path)
    net.eval()

    mask, _ = net.predict(input_tensor)

    conf_map = None
    if uncertainty != "none":
        conf_map = compute_uncertainty(
            net,
            input_tensor,
            method=uncertainty,
            mc_iters=mc_iters,
        )

    colored_mask = colorize_mask(mask, AppConfig.NUM_CLASSES)
    overlay = overlay_heatmap_on_image(rgb_resized, conf_map) if conf_map is not None else None

    return InferResponse(
        mask_png=image_to_base64_png(colored_mask),
        overlay_png=image_to_base64_png(overlay) if overlay is not None else None,
        message="ok",
    )


@app.post("/infer-batch")
async def infer_batch():
    return JSONResponse({"detail": "Not implemented in this demo. Use /infer."})

