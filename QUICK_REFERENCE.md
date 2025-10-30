# AW-SafeSeg Quick Reference Card

## 🚀 Essential Commands

### Setup & Installation
```bash
# Complete setup (run once)
setup.bat

# Install backend only
pip install -r requirements.txt

# Install frontend only
cd project\frontend && npm install
```

### Start Application
```bash
# Start everything
scripts\start_fullstack.bat

# Backend only (port 8000)
cd project && python start_backend.py

# Frontend only (port 5173)
cd project\frontend && npm run dev
```

### Access URLs
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/ping

---

## 📂 Key Files & Directories

| Path | Purpose |
|------|---------|
| `project/backend_api.py` | Main Flask API server |
| `project/start_backend.py` | Backend launcher |
| `project/models.py` | Model architectures |
| `project/train.py` | Training script |
| `project/eval.py` | Evaluation script |
| `project/ckpts/` | Model checkpoints |
| `project/frontend/` | React application |
| `IDDAW/` | Training dataset |
| `docs/` | All documentation |
| `scripts/` | Utility scripts |

---

## 🧠 Model Checkpoints

| Model | File | Purpose |
|-------|------|---------|
| RGB | `best_rgb_mbv3.pt` | RGB-only segmentation |
| NIR | `best_nir_fastscnn.pt` | NIR-only segmentation |
| Early Fusion | `best_early4_mbv3.pt` | 4-channel input fusion |
| Mid Fusion | `best_mid_mbv3.pt` | Feature-level fusion |

Location: `project/ckpts/`

---

## 🔌 API Endpoints

### Authentication
```
POST /api/auth/signup    # Register new user
POST /api/auth/login     # User login
GET  /api/me             # Get current user
```

### Prediction
```
POST /api/predict        # Upload RGB+NIR images
GET  /api/results        # List user results
GET  /api/results/<id>   # Get specific result
POST /api/results/save   # Save result to profile
```

---

## 🎓 Training Commands

### Basic Training
```bash
cd project

# Train RGB model
python train.py --model mobilenetv3 --modality rgb --epochs 100

# Train NIR model
python train.py --model fastscnn --modality nir --epochs 100

# Train fusion model
python train.py --model mobilenetv3 --modality early4 --epochs 100
```

### Advanced Training
```bash
# With custom learning rate
python train.py --model fastscnn --modality rgb --lr 0.001

# With data augmentation
python train_improved.py --model mobilenetv3 --augment

# Resume from checkpoint
python train.py --resume ckpts/checkpoint.pt
```

---

## 🧪 Testing & Evaluation

### Integration Tests
```bash
python scripts\test_integration.py
```

### Model Evaluation
```bash
cd project

# Evaluate specific model
python eval.py --checkpoint ckpts/best_rgb_mbv3.pt

# Evaluate on validation set
python eval.py --checkpoint ckpts/best_early4_mbv3.pt --split val
```

### Demo
```bash
cd project
python demo.py --rgb path/to/rgb.png --nir path/to/nir.png
```

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r requirements.txt

# Check model checkpoints
dir project\ckpts
```

### Frontend won't start
```bash
# Check Node.js version
node --version  # Should be 16+

# Reinstall dependencies
cd project\frontend
rm -rf node_modules
npm install

# Check .env file
type .env
```

### Models not loading
```bash
# Verify checkpoint files exist
dir project\ckpts\*.pt

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check CUDA availability (optional)
python -c "import torch; print(torch.cuda.is_available())"
```

### Database errors
```bash
# Reset database
del project\iddaw.db

# Restart backend (will recreate DB)
cd project
python start_backend.py
```

---

## 📦 Dependencies

### Python (Backend)
- PyTorch >= 2.0.0
- Flask >= 2.3.0
- OpenCV >= 4.8.0
- NumPy >= 1.24.0

### Node.js (Frontend)
- React 18
- Vite 5
- TailwindCSS 3
- TypeScript 5

---

## 🔧 Configuration

### Backend Config
File: `project/backend_api.py`
```python
PORT = 8000
SECRET_KEY = 'your-secret-key'
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

### Frontend Config
File: `project/frontend/.env`
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCKS=false
VITE_API_TIMEOUT=30000
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Inference Time | <100ms (GPU) |
| Model Size | <15MB per model |
| mIoU | 75-80% |
| Input Size | 512x512 |
| Batch Size | 8 (training) |

---

## 🔗 Useful Links

- **Main README**: [README.md](README.md)
- **Quick Start**: [docs/QUICK_START.md](docs/QUICK_START.md)
- **Deployment**: [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
- **Training**: [docs/TRAIN_ON_COLAB.md](docs/TRAIN_ON_COLAB.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING_GUIDE.md](docs/TROUBLESHOOTING_GUIDE.md)

---

## 💡 Tips

1. **Always activate virtual environment** before running commands
2. **Check logs** in terminal for error messages
3. **Use GPU** for faster training and inference
4. **Backup checkpoints** before training
5. **Test locally** before deploying to production
6. **Read documentation** in `docs/` folder
7. **Check CHANGELOG.md** for version updates

---

## 🆘 Getting Help

1. Check [TROUBLESHOOTING_GUIDE.md](docs/TROUBLESHOOTING_GUIDE.md)
2. Review error logs in terminal
3. Verify all dependencies are installed
4. Check system requirements
5. Review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

---

**Version**: 1.0.0  
**Last Updated**: October 28, 2025
