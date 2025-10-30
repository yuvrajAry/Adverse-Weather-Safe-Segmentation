# IDDAW Project - Quick Start Guide

## 🚀 Complete Integration Setup

Your IDDAW project has been fully integrated with the React frontend! Here's how to get everything running:

### Prerequisites
- Python 3.8+ ✅
- Node.js 16+ (for frontend)
- Your trained models in `project/ckpts/` ✅

### Step 1: Install Backend Dependencies
```bash
cd project
pip install flask flask-cors PyJWT opencv-python pillow
```

### Step 2: Install Frontend Dependencies
```bash
cd project/frontend
npm install
```

### Step 3: Configure Frontend
Create `project/frontend/.env`:
```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCKS=false
```

### Step 4: Start the Application

**Option A: Use the startup scripts (Recommended)**
```bash
# Start both backend and frontend
start_fullstack.bat
```

**Option B: Manual start**
```bash
# Terminal 1 - Backend
cd project
python start_backend.py

# Terminal 2 - Frontend
cd project/frontend
npm run dev
```

### Step 5: Access Your Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Test**: http://localhost:8000/api/ping

## 🎯 What You Get

### Complete Full-Stack Application
- ✅ **React Frontend** with modern UI
- ✅ **Flask Backend** with JWT authentication
- ✅ **IDDAW Models** integration (RGB, NIR, Early4, Mid fusion)
- ✅ **Image Processing** with confidence heatmaps
- ✅ **Result Management** with download capabilities
- ✅ **User Authentication** system
- ✅ **Database** for user and result storage

### Key Features
1. **User Registration/Login** - Secure JWT-based authentication
2. **Image Upload** - Upload RGB + NIR image pairs
3. **AI Segmentation** - Uses your trained IDDAW models
4. **Visualization** - Segmentation masks, confidence heatmaps, safety analysis
5. **Result Management** - Save, download, and share results
6. **Modern UI** - Responsive design with dark/light themes

## 🔧 API Endpoints

Your backend provides these endpoints:

- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login
- `GET /api/me` - Get current user
- `POST /api/predict` - Image segmentation (RGB + NIR)
- `GET /api/results` - List user results
- `GET /api/results/<id>` - Get specific result
- `POST /api/results/save` - Save result to profile

## 🧪 Testing the Integration

Run the integration test:
```bash
python test_integration.py
```

## 📁 Project Structure

```
project/
├── backend_api.py              # Main Flask API server
├── start_backend.py            # Backend startup script
├── requirements_backend.txt   # Backend dependencies
├── ckpts/                      # Your model checkpoints
│   ├── best_rgb_mbv3.pt
│   ├── best_nir_fastscnn.pt
│   ├── best_early4_mbv3.pt
│   └── best_mid_mbv3.pt
├── uploads/                    # Temporary uploads
├── results/                    # Generated results
├── iddaw.db                    # SQLite database
└── frontend/                   # React application
    ├── client/                 # React app source
    ├── package.json            # Frontend dependencies
    └── .env                    # Frontend configuration
```

## 🎨 Frontend Features

### Pages
- **Home** - Image upload interface
- **Results** - View segmentation results
- **Profile** - User account management
- **About** - Project information

### Components
- **FileUpload** - Drag-and-drop image selection
- **ResultsDisplay** - Grid layout for result visualization
- **AuthForm** - User authentication
- **Navbar** - Navigation with theme toggle

### State Management
- **Auth Store** - User authentication state
- **Results Store** - Result management
- **Theme Store** - Dark/light theme

## 🔍 How It Works

1. **User uploads RGB + NIR images** through the React frontend
2. **Frontend sends images** to Flask backend via multipart form data
3. **Backend processes images** using your trained IDDAW models:
   - RGB model (MobileNetV3)
   - NIR model (FastSCNN)
   - Early4 fusion model
   - Mid fusion model
4. **Backend generates visualizations**:
   - Segmentation mask
   - Confidence heatmap
   - Safety analysis
   - Overlay visualization
5. **Results are stored** in SQLite database
6. **Frontend displays results** with download capabilities

## 🚀 Production Deployment

For production deployment, see `DEPLOYMENT_GUIDE.md` for detailed instructions including:
- Environment configuration
- Database setup
- File storage
- Security considerations
- Scaling options

## 🆘 Troubleshooting

### Common Issues

1. **Backend won't start**
   - Check if models exist in `project/ckpts/`
   - Install dependencies: `pip install -r requirements_backend.txt`

2. **Frontend won't start**
   - Install dependencies: `npm install`
   - Check `.env` file configuration

3. **Models not loading**
   - Ensure checkpoints exist and are accessible
   - Check PyTorch installation

4. **Database errors**
   - Delete `iddaw.db` to reset database
   - Check file permissions

## 🎉 You're Ready!

Your IDDAW project is now fully integrated with a modern web interface! Users can:

1. **Create accounts** and log in securely
2. **Upload RGB + NIR images** for segmentation
3. **View AI-generated results** with confidence analysis
4. **Download results** as images or ZIP files
5. **Manage their results** in their profile

The integration preserves all your IDDAW model capabilities while providing a user-friendly web interface for real-world deployment.

