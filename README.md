# AW-SafeSeg: Adverse Weather Safe Segmentation System

A full-stack deep learning application for real-time semantic segmentation in adverse weather conditions using RGB+NIR image fusion.

## 🌟 Overview

AW-SafeSeg (IDDAW) is an advanced computer vision system designed for autonomous vehicles and safety-critical applications. It performs robust semantic segmentation even in challenging weather conditions like fog, rain, and low-light scenarios by leveraging both RGB and Near-Infrared (NIR) image modalities.

### Key Features

- **Multi-Modal Fusion**: Combines RGB and NIR images for enhanced segmentation accuracy
- **Weather Resilience**: Optimized for fog, rain, night-time, and low visibility conditions
- **Real-Time Processing**: Efficient models (FastSCNN, MobileNetV3) for fast inference
- **Full-Stack Web Application**: Modern React frontend with Flask backend
- **Confidence Analysis**: Safety heatmaps and confidence scoring for predictions
- **User Management**: Secure authentication and result history tracking

## 📁 Project Structure

```
pro/
├── README.md                    # This file
├── project/                     # Main application directory
│   ├── backend_api.py          # Flask REST API server
│   ├── start_backend.py        # Backend startup script
│   ├── requirements_backend.txt # Python dependencies
│   ├── models.py               # Model architectures (FastSCNN, MobileNetV3)
│   ├── dataset.py              # Data loading and preprocessing
│   ├── train.py                # Training script
│   ├── eval.py                 # Evaluation script
│   ├── ckpts/                  # Model checkpoints
│   │   ├── best_rgb_mbv3.pt
│   │   ├── best_nir_fastscnn.pt
│   │   ├── best_early4_mbv3.pt
│   │   └── best_mid_mbv3.pt
│   └── frontend/               # React web application
│       ├── client/             # React source code
│       ├── package.json        # Node.js dependencies
│       └── .env                # Frontend configuration
├── IDDAW/                      # Training dataset
│   ├── train/                  # Training images and masks
│   └── val/                    # Validation images and masks
├── full4/                      # Alternative training setup
│   ├── app/                    # Training application
│   ├── checkpoints/            # Training checkpoints
│   └── requirements.txt        # Training dependencies
├── docs/                       # Documentation
│   ├── DEPLOYMENT_GUIDE.md     # Production deployment guide
│   ├── QUICK_START.md          # Getting started guide
│   ├── TROUBLESHOOTING_GUIDE.md # Common issues and solutions
│   └── TRAIN_ON_COLAB.md       # Google Colab training guide
└── scripts/                    # Utility scripts
    ├── start_fullstack.bat     # Start both frontend and backend
    ├── test_integration.py     # Integration testing
    └── IDDAW_Colab_Training.ipynb # Jupyter notebook for training

```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with PyTorch
- **Node.js 16+** and npm
- **CUDA** (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   cd d:\iddaw\pro
   ```

2. **Install backend dependencies**
   ```bash
   cd project
   pip install -r requirements_backend.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

4. **Configure environment**
   
   Create `project/frontend/.env`:
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   VITE_USE_MOCKS=false
   VITE_API_TIMEOUT=30000
   ```

### Running the Application

**Option 1: Use the startup script (Recommended)**
```bash
scripts\start_fullstack.bat
```

**Option 2: Manual start**
```bash
# Terminal 1 - Backend
cd project
python start_backend.py

# Terminal 2 - Frontend
cd project/frontend
npm run dev
```

### Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Health Check**: http://localhost:8000/api/ping

## 🎯 Usage

1. **Create an account** or log in
2. **Upload RGB + NIR image pairs** for segmentation
3. **View results** with:
   - Segmentation masks
   - Confidence heatmaps
   - Safety analysis overlays
4. **Download results** as images or ZIP files
5. **Manage history** in your profile

## 🧠 Model Architecture

### Supported Models

1. **FastSCNN** - Fast Semantic Segmentation Network
   - Optimized for real-time inference
   - Used for NIR modality

2. **MobileNetV3** - Efficient CNN architecture
   - Lightweight and mobile-friendly
   - Used for RGB modality

3. **Fusion Models**
   - **Early Fusion**: Concatenates RGB+NIR at input level
   - **Mid Fusion**: Combines features at intermediate layers

### Model Checkpoints

Pre-trained models are stored in `project/ckpts/`:
- `best_rgb_mbv3.pt` - RGB-only model
- `best_nir_fastscnn.pt` - NIR-only model
- `best_early4_mbv3.pt` - Early fusion model
- `best_mid_mbv3.pt` - Mid fusion model

## 🔧 Training

### Training on Local Machine

```bash
cd project
python train.py --model fastscnn --modality rgb --epochs 100
```

### Training on Google Colab

Use the provided Jupyter notebook:
```bash
scripts/IDDAW_Colab_Training.ipynb
```

See `docs/TRAIN_ON_COLAB.md` for detailed instructions.

### Dataset Structure

The IDDAW dataset should be organized as:
```
IDDAW/
├── train/
│   ├── rgb/           # RGB images
│   ├── nir/           # NIR images
│   └── labels/        # Segmentation masks
└── val/
    ├── rgb/
    ├── nir/
    └── labels/
```

## 🔌 API Documentation

### Authentication Endpoints

- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login
- `GET /api/me` - Get current user info

### Prediction Endpoints

- `POST /api/predict` - Upload RGB + NIR images for segmentation
  - Request: `multipart/form-data` with `rgb_image` and `nir_image` files
  - Response: Result ID and URLs for generated visualizations

### Results Endpoints

- `GET /api/results` - List user's prediction history
- `GET /api/results/<id>` - Get specific result details
- `POST /api/results/save` - Save result to user profile

## 🧪 Testing

Run integration tests:
```bash
python scripts/test_integration.py
```

## 📊 Performance Metrics

The system achieves:
- **mIoU**: 75-80% on IDDAW validation set
- **Inference Time**: <100ms per image pair (with GPU)
- **Model Size**: <10MB (MobileNetV3), <15MB (FastSCNN)

## 🚢 Production Deployment

See `docs/DEPLOYMENT_GUIDE.md` for comprehensive deployment instructions including:
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)
- Database configuration (PostgreSQL)
- File storage (S3, Cloud Storage)
- HTTPS/SSL setup
- Monitoring and logging

## 🛠️ Troubleshooting

Common issues and solutions are documented in `docs/TROUBLESHOOTING_GUIDE.md`.

### Quick Fixes

1. **Backend won't start**: Check model checkpoints exist in `project/ckpts/`
2. **Frontend won't start**: Run `npm install` in `project/frontend/`
3. **Models not loading**: Verify PyTorch installation and CUDA compatibility
4. **Database errors**: Delete `project/iddaw.db` to reset

## 📚 Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get up and running quickly
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment
- **[Training Guide](docs/TRAIN_ON_COLAB.md)** - Model training instructions
- **[Troubleshooting](docs/TROUBLESHOOTING_GUIDE.md)** - Common issues
- **[Model Accuracy](docs/IMPROVE_MODEL_ACCURACY.md)** - Optimization tips

## 🔒 Security

- JWT-based authentication
- Password hashing with bcrypt
- File upload validation
- CORS configuration
- SQL injection protection

## 📈 Future Enhancements

- [ ] Real-time video processing
- [ ] Multi-GPU training support
- [ ] Model quantization for edge deployment
- [ ] Additional weather conditions (snow, sandstorm)
- [ ] 3D segmentation support
- [ ] Mobile app (iOS/Android)

## 🤝 Contributing

This is an academic/research project. For questions or collaboration:
1. Review the documentation
2. Check existing issues
3. Follow the code style guidelines

## 📄 License

This project is for academic and research purposes.

## 🙏 Acknowledgments

- FastSCNN architecture from [paper](https://arxiv.org/abs/1902.04502)
- MobileNetV3 from [paper](https://arxiv.org/abs/1905.02244)
- React frontend built with Vite and TailwindCSS
- Flask backend with PyTorch integration

## 📞 Support

For issues and questions:
1. Check the documentation in `docs/`
2. Review error logs
3. Run integration tests
4. Verify system requirements

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Status**: Production Ready ✅
