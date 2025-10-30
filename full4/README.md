## Safety-Critical Confidence Heatmaps Backend (FastAPI)

Runnable FastAPI backend for semantic segmentation with RGB+NIR fusion, uncertainty estimation (MC Dropout, Softmax Entropy), and heatmap overlays.

### Features
- Models: Fast-SCNN (stub), MobileNetV3 + Lite Decoder (stub)
- Fusion: Early (4-channel) and Mid-level (dual encoder)
- Uncertainty: Monte Carlo Dropout, Softmax Entropy
- Endpoints: `/health`, `/infer`, `/infer-batch`
- Outputs: Segmentation mask (colored), optional confidence heatmap overlay

### Local Run
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/docs`.

### Docker
```bash
docker build -t iddaw-backend:latest .
docker run --rm -p 8000:8000 iddaw-backend:latest
```

### Inference Notes
-
### Training on IDD-AW
Dataset root: `app/config.py` → `DATASET_ROOT` (default `D:\iddaw\IDDAW`). Expected layout:
```
IDDAW/
  train/
    FOG|RAIN|SNOW|LOWLIGHT/
      rgb/<city>/<frame>_rgb.png
      nir/<city>/<frame>_nir.png
      gtSeg/<city>/<frame>_mask.json
  val/
    ... same structure ...
```

Run training:
```powershell
cd D:\iddaw\full4
. .venv\Scripts\Activate.ps1  # if not active
python -m app.train --model fast_scnn --fusion early --width 512 --height 512 --batch-size 4 --epochs 20 --lr 1e-3 --workers 4 --out-dir checkpoints
```

Notes:
- Checkpoints saved in `checkpoints/` as `{model}_{fusion}_latest.pt` and `..._best.pt`.
- To use trained weights in the API, pass `weights_path` to `/infer` (absolute path or mount into container).
- If `weights_path` is not supplied, models use random weights (demo only).
- Configure `NUM_CLASSES` and `SAFETY_CLASS_IDS` in `app/config.py`.
- Input size can be controlled via `output_size` like `512x512`.
- Dataset paths: set `DATASET_ROOT` in `app/config.py` (default `D:\iddaw\IDDAW`).
- You can call `/infer` with either uploaded files `rgb`, `nir` or with form fields `rgb_path`, `nir_path` relative to `DATASET_ROOT` (or absolute Windows paths).

