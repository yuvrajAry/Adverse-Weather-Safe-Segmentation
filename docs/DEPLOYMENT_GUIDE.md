# IDDAW Project - Full Stack Deployment Guide

This guide covers deploying the complete IDDAW (Adverse Weather Safe Segmentation) application with both backend and frontend components.

## 🚀 Quick Start (Development)

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### 1. Setup Integration
```bash
# Run the integration setup script
python setup_integration.py
```

### 2. Start Full Stack Application
```bash
# Option 1: Use the startup script
start_fullstack.bat

# Option 2: Manual start
# Terminal 1 - Backend
cd project
python start_backend.py

# Terminal 2 - Frontend  
cd project/frontend
npm run dev
```

### 3. Access Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/api/ping

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │  Flask Backend  │    │  IDDAW Models   │
│   (Port 5173)   │◄──►│   (Port 8000)   │◄──►│   (PyTorch)    │
│                 │    │                 │    │                 │
│ • Authentication│    │ • JWT Auth      │    │ • RGB Model     │
│ • File Upload   │    │ • Image Process │    │ • NIR Model     │
│ • Results View  │    │ • SQLite DB     │    │ • Early4 Fusion │
│ • Download      │    │ • File Storage  │    │ • Mid Fusion    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
D:\iddaw\pro\
├── project/                          # Main project directory
│   ├── backend_api.py               # Flask API server
│   ├── start_backend.py             # Backend startup script
│   ├── requirements_backend.txt     # Backend dependencies
│   ├── ckpts/                       # Model checkpoints
│   │   ├── best_rgb_mbv3.pt
│   │   ├── best_nir_fastscnn.pt
│   │   ├── best_early4_mbv3.pt
│   │   └── best_mid_mbv3.pt
│   ├── uploads/                     # Temporary uploads
│   ├── results/                     # Generated results
│   ├── iddaw.db                     # SQLite database
│   └── frontend/                    # React application
│       ├── client/                 # React app source
│       ├── package.json            # Frontend dependencies
│       ├── .env                     # Frontend config
│       └── vite.config.ts          # Vite configuration
├── setup_integration.py             # Integration setup script
├── test_integration.py              # Integration test script
├── start_fullstack.bat              # Full stack startup
├── start_backend.bat                # Backend startup
├── start_frontend.bat               # Frontend startup
└── INTEGRATION_README.md            # Integration documentation
```

## 🔧 Configuration

### Backend Configuration
The backend uses environment variables for configuration:

```python
# In backend_api.py
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

### Frontend Configuration
The frontend uses Vite environment variables:

```bash
# In project/frontend/.env
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCKS=false
VITE_API_TIMEOUT=30000
```

## 🗄️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Results Table
```sql
CREATE TABLE results (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    original_url TEXT NOT NULL,
    mask_url TEXT NOT NULL,
    heatmap_url TEXT NOT NULL,
    overlay_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login
- `GET /api/me` - Get current user info

### Image Processing
- `POST /api/predict` - Upload RGB + NIR images for segmentation
- `GET /api/results` - List user's results
- `GET /api/results/<id>` - Get specific result details
- `POST /api/results/save` - Save result to profile

### File Serving
- `GET /api/results/<id>/<filename>` - Serve result images

## 🧪 Testing

### Run Integration Tests
```bash
# Test backend API
python test_integration.py
```

### Manual Testing
1. Start backend: `start_backend.bat`
2. Start frontend: `start_frontend.bat`
3. Open http://localhost:5173
4. Create account and upload images

## 🚀 Production Deployment

### Backend Production Setup

1. **Environment Variables**
```bash
export SECRET_KEY="your-production-secret-key"
export FLASK_ENV="production"
export DATABASE_URL="postgresql://user:pass@host:port/db"
```

2. **Database Migration**
```python
# Use PostgreSQL for production
pip install psycopg2-binary
# Update database connection in backend_api.py
```

3. **File Storage**
```python
# Use cloud storage (AWS S3, etc.)
# Update file paths in backend_api.py
```

4. **WSGI Server**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 backend_api:app
```

### Frontend Production Build

1. **Build for Production**
```bash
cd project/frontend
npm run build
```

2. **Serve Static Files**
```bash
# Use nginx or serve with backend
# Update VITE_API_BASE_URL to production backend URL
```

### Docker Deployment

Create `Dockerfile` for backend:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_backend.txt .
RUN pip install -r requirements_backend.txt
COPY . .
EXPOSE 8000
CMD ["python", "start_backend.py"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./results:/app/results
    environment:
      - SECRET_KEY=your-secret-key
  
  frontend:
    build: ./project/frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_BASE_URL=http://localhost:8000
```

## 🔍 Monitoring & Logging

### Backend Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Health Checks
```bash
# Check backend health
curl http://localhost:8000/api/ping

# Check frontend
curl http://localhost:5173
```

## 🛠️ Troubleshooting

### Common Issues

1. **Backend won't start**
   - Check Python version (3.8+)
   - Install dependencies: `pip install -r requirements_backend.txt`
   - Check model checkpoints exist

2. **Frontend won't start**
   - Check Node.js version (16+)
   - Install dependencies: `npm install`
   - Check .env file configuration

3. **Models not loading**
   - Ensure checkpoints exist in `project/ckpts/`
   - Check file permissions
   - Verify PyTorch installation

4. **Database errors**
   - Check SQLite file permissions
   - Delete `iddaw.db` to reset database
   - Check disk space

5. **Image processing fails**
   - Check image file formats (PNG/JPG)
   - Verify file sizes (< 16MB)
   - Check OpenCV installation

### Performance Optimization

1. **Backend**
   - Use GPU if available
   - Implement model caching
   - Add request rate limiting
   - Use production WSGI server

2. **Frontend**
   - Enable code splitting
   - Use CDN for static assets
   - Implement image compression
   - Add loading states

## 📊 Monitoring

### Key Metrics
- API response times
- Model inference time
- Memory usage
- Disk space usage
- User registration/login rates

### Logs to Monitor
- Authentication attempts
- Image processing requests
- Error rates
- Performance metrics

## 🔒 Security Considerations

1. **Authentication**
   - Use strong JWT secrets
   - Implement token expiration
   - Add rate limiting

2. **File Upload**
   - Validate file types
   - Check file sizes
   - Scan for malware

3. **Database**
   - Use parameterized queries
   - Implement backup strategy
   - Encrypt sensitive data

4. **API**
   - Add CORS configuration
   - Implement request validation
   - Use HTTPS in production

## 📈 Scaling

### Horizontal Scaling
- Use load balancer
- Implement session storage
- Use shared file storage
- Database clustering

### Vertical Scaling
- Increase server resources
- Optimize model inference
- Use GPU acceleration
- Implement caching

## 🆘 Support

For issues and questions:
1. Check this deployment guide
2. Review error logs
3. Test with integration script
4. Check system requirements

## 📝 Changelog

- **v1.0.0** - Initial integration setup
- **v1.1.0** - Added authentication system
- **v1.2.0** - Implemented image processing API
- **v1.3.0** - Added result management
- **v1.4.0** - Production deployment support
