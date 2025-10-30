#!/usr/bin/env python3
"""
IDDAW Backend Startup Script
Starts the backend API server with proper configuration
"""

import os
import sys
from pathlib import Path

# Add project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Set environment variables
os.environ.setdefault('SECRET_KEY', 'iddaw-secret-key-change-in-production')
os.environ.setdefault('FLASK_ENV', 'development')

# Import and run the backend
if __name__ == '__main__':
    try:
        from backend_api import app, init_db, model_manager
        
        print("IDDAW Backend API Server")
        print("=" * 50)
        print(f"Project directory: {project_dir}")
        print(f"Device: {model_manager.device}")
        print("Initializing database...")
        init_db()
        print("Database initialized successfully")
        print("Starting server on http://localhost:8000")
        print("API Documentation:")
        print("  POST /api/auth/signup - User registration")
        print("  POST /api/auth/login - User login")
        print("  GET  /api/me - Get current user")
        print("  POST /api/predict - Image segmentation")
        print("  GET  /api/results - List user results")
        print("  GET  /api/results/<id> - Get specific result")
        print("=" * 50)
        
        app.run(host='0.0.0.0', port=8001, debug=True)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required dependencies:")
        print("pip install -r requirements_backend.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

