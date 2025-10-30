# AW-SafeSeg Project Structure

## Directory Layout

```
d:\iddaw\pro\
│
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CHANGELOG.md                 # Version history and updates
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 PROJECT_SUMMARY.md           # Cleanup report and summary
├── 📄 QUICK_REFERENCE.md           # Developer quick reference
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.bat                    # Automated setup script
├── 📄 .gitignore                   # Git ignore rules
│
├── 📁 project/                     # Main Application
│   ├── 🐍 backend_api.py          # Flask REST API server
│   ├── 🐍 start_backend.py        # Backend startup script
│   ├── 🐍 models.py                # Model architectures (FastSCNN, MobileNetV3)
│   ├── 🐍 dataset.py               # Dataset loading and preprocessing
│   ├── 🐍 train.py                 # Training script
│   ├── 🐍 train_improved.py        # Enhanced training with augmentation
│   ├── 🐍 eval.py                  # Model evaluation script
│   ├── 🐍 demo.py                  # Demo inference script
│   ├── 🐍 augment.py               # Data augmentation utilities
│   ├── 🐍 preprocess.py            # Image preprocessing
│   ├── 🐍 viz.py                   # Visualization utilities
│   ├── 🐍 metrics.py               # Evaluation metrics
│   ├── 🐍 labels.py                # Label definitions
│   ├── 🐍 launch.py                # Launch utilities
│   ├── 📄 requirements_backend.txt # Backend-specific dependencies
│   ├── 🗄️ iddaw.db                 # SQLite database
│   │
│   ├── 📁 ckpts/                   # Model Checkpoints
│   │   ├── best_rgb_mbv3.pt
│   │   ├── best_nir_fastscnn.pt
│   │   ├── best_early4_mbv3.pt
│   │   └── best_mid_mbv3.pt
│   │
│   ├── 📁 configs/                 # Configuration files
│   ├── 📁 splits/                  # Dataset split definitions
│   ├── 📁 output/                  # Training outputs
│   ├── 📁 outputs/                 # Prediction outputs
│   │
│   └── 📁 frontend/                # React Web Application
│       ├── 📁 client/              # React source code
│       │   ├── src/
│       │   │   ├── components/     # React components
│       │   │   ├── pages/          # Page components
│       │   │   ├── stores/         # State management
│       │   │   ├── lib/            # Utilities
│       │   │   └── App.tsx         # Main app component
│       │   └── index.html
│       │
│       ├── 📁 server/              # Server utilities
│       ├── 📁 public/              # Static assets
│       ├── 📁 shared/              # Shared code
│       │
│       ├── 📄 package.json         # Node.js dependencies
│       ├── 📄 vite.config.ts       # Vite configuration
│       ├── 📄 tsconfig.json        # TypeScript config
│       ├── 📄 tailwind.config.ts   # TailwindCSS config
│       ├── 📄 .env                 # Environment variables
│       └── 📄 .env.example         # Example env file
│
├── 📁 IDDAW/                       # Training Dataset
│   ├── 📁 train/                   # Training data (3430 items)
│   │   ├── rgb/                    # RGB images
│   │   ├── nir/                    # NIR images
│   │   └── labels/                 # Segmentation masks
│   │
│   └── 📁 val/                     # Validation data (475 items)
│       ├── rgb/
│       ├── nir/
│       └── labels/
│
├── 📁 full4/                       # Alternative Training Setup
│   ├── 📁 app/                     # Training application
│   │   ├── models/
│   │   ├── utils/
│   │   └── train.py
│   │
│   ├── 📁 checkpoints/             # Training checkpoints
│   ├── 🐍 efficient_train.py       # Efficient training script
│   ├── 🐍 quick_train.py           # Quick training script
│   ├── 🐍 simple_train.py          # Simple training script
│   ├── 🐍 train_comprehensive.py   # Comprehensive training
│   ├── 🐍 train_fscnn_full.py      # FastSCNN training
│   ├── 🐍 train_step_by_step.py    # Step-by-step training
│   ├── 🐍 start_training.py        # Training launcher
│   ├── 🐍 check_progress.py        # Progress monitoring
│   ├── 🐍 monitor_training.py      # Training monitor
│   ├── 🐍 test_dataset.py          # Dataset testing
│   ├── 📄 requirements.txt         # Training dependencies
│   ├── 📄 Dockerfile               # Docker configuration
│   ├── 📄 README.md                # Training documentation
│   └── 📄 TRAINING_SUMMARY.md      # Training summary
│
├── 📁 docs/                        # Documentation
│   ├── 📄 DEPLOYMENT_GUIDE.md      # Production deployment guide
│   ├── 📄 QUICK_START.md           # Getting started guide
│   ├── 📄 QUICK_START_TRAINING.md  # Training quick start
│   ├── 📄 TRAIN_ON_COLAB.md        # Google Colab training
│   ├── 📄 TROUBLESHOOTING_GUIDE.md # Common issues and solutions
│   ├── 📄 UPLOAD_TO_COLAB_GUIDE.md # Colab upload instructions
│   ├── 📄 IMPROVE_MODEL_ACCURACY.md # Model optimization tips
│   ├── 📄 testing_validation_report.md # Test report
│   └── 📄 PROJECT_STRUCTURE.md     # This file
│
└── 📁 scripts/                     # Utility Scripts
    ├── 🔧 start_fullstack.bat      # Start both frontend and backend
    ├── 🐍 test_integration.py      # Integration testing script
    └── 📓 IDDAW_Colab_Training.ipynb # Jupyter notebook for Colab
```

## Component Descriptions

### Root Level Files

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation with overview, features, and usage |
| `LICENSE` | MIT License for the project |
| `CHANGELOG.md` | Version history and release notes |
| `CONTRIBUTING.md` | Guidelines for contributing to the project |
| `PROJECT_SUMMARY.md` | Cleanup report and project status |
| `QUICK_REFERENCE.md` | Quick reference card for developers |
| `requirements.txt` | Python dependencies for the entire project |
| `setup.bat` | Automated setup script for Windows |
| `.gitignore` | Git ignore rules for clean repository |

### project/ - Main Application

**Backend Files**:
- `backend_api.py` - Flask REST API with authentication and image processing
- `start_backend.py` - Backend startup script with configuration
- `models.py` - PyTorch model architectures (FastSCNN, MobileNetV3)
- `dataset.py` - Custom dataset class for loading RGB+NIR pairs
- `train.py` - Training script with logging and checkpointing
- `train_improved.py` - Enhanced training with advanced augmentation
- `eval.py` - Model evaluation with metrics calculation
- `demo.py` - Demo script for single image inference

**Utility Files**:
- `augment.py` - Data augmentation transformations
- `preprocess.py` - Image preprocessing and normalization
- `viz.py` - Visualization utilities for results
- `metrics.py` - IoU, Dice, and other metrics
- `labels.py` - Class label definitions
- `launch.py` - Launch utilities and helpers

**Data & Config**:
- `ckpts/` - Pre-trained model checkpoints
- `configs/` - Configuration files for models
- `splits/` - Train/val/test split definitions
- `iddaw.db` - SQLite database for users and results

**Frontend**:
- `frontend/client/` - React application source code
- `frontend/server/` - Server-side utilities
- `frontend/public/` - Static assets (images, icons)
- `frontend/package.json` - Node.js dependencies
- `frontend/vite.config.ts` - Vite build configuration

### IDDAW/ - Training Dataset

Contains the complete dataset with RGB and NIR image pairs plus segmentation masks:
- **train/** - 3430 training samples
- **val/** - 475 validation samples

Each split contains:
- `rgb/` - RGB images
- `nir/` - Near-infrared images
- `labels/` - Ground truth segmentation masks

### full4/ - Alternative Training

Alternative training setup with different approaches:
- Multiple training scripts with varying strategies
- Checkpoint management
- Progress monitoring tools
- Docker support for containerized training

### docs/ - Documentation

Comprehensive documentation covering:
- **Deployment** - Production deployment guide
- **Quick Start** - Getting started quickly
- **Training** - Local and Colab training guides
- **Troubleshooting** - Common issues and solutions
- **Optimization** - Model accuracy improvement tips

### scripts/ - Utilities

Helper scripts for common tasks:
- `start_fullstack.bat` - Start entire application
- `test_integration.py` - Integration testing
- `IDDAW_Colab_Training.ipynb` - Colab training notebook

## File Counts

| Directory | Files | Purpose |
|-----------|-------|---------|
| `project/` | 33 | Main application code |
| `project/frontend/` | 103 | React web application |
| `IDDAW/` | 3905 | Training dataset |
| `full4/` | 47 | Alternative training |
| `docs/` | 8 | Documentation |
| `scripts/` | 3 | Utility scripts |

## Key Technologies

### Backend
- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **Flask** - Web framework
- **OpenCV** - Image processing
- **SQLite** - Database

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool
- **TailwindCSS** - Styling
- **Zustand** - State management

### Models
- **FastSCNN** - Fast semantic segmentation
- **MobileNetV3** - Efficient CNN
- **Custom Fusion** - RGB+NIR fusion architectures

## Data Flow

```
User Upload (Frontend)
    ↓
Flask API (Backend)
    ↓
Image Preprocessing
    ↓
Model Inference (PyTorch)
    ↓
Post-processing & Visualization
    ↓
Save to Database
    ↓
Return Results (Frontend)
```

## Development Workflow

1. **Setup**: Run `setup.bat`
2. **Development**: Use `scripts/start_fullstack.bat`
3. **Training**: Use scripts in `project/` or `full4/`
4. **Testing**: Run `scripts/test_integration.py`
5. **Deployment**: Follow `docs/DEPLOYMENT_GUIDE.md`

## Important Paths

| Purpose | Path |
|---------|------|
| Backend API | `project/backend_api.py` |
| Frontend Entry | `project/frontend/client/src/App.tsx` |
| Model Checkpoints | `project/ckpts/` |
| Training Data | `IDDAW/train/` |
| Documentation | `docs/` |
| Startup Script | `scripts/start_fullstack.bat` |

---

**Last Updated**: October 28, 2025  
**Version**: 1.0.0
