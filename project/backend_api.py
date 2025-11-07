#!/usr/bin/env python3
"""
IDDAW Backend API Server
Integrates the IDDAW project with the frontend application
"""

import os
import sys
import json
import uuid
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import jwt
from functools import wraps

# Import IDDAW project modules
from models import build_model
from labels import LabelMapper, SafetyGroups
from metrics import entropy_map
from viz import color_map, overlay_segmentation, safety_heatmap, confidence_heatmap

class ModelManager:
    def __init__(self, model_path: str = 'ckpts/best_mid_mbv3.pt'):
        self.model_path = model_path
        self.model = None
        self.label_mapper = LabelMapper.default()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

    def load_model(self):
        try:
            # Get number of classes from label mapper
            num_classes = len(self.label_mapper.class_to_id)
            # Build model with correct architecture for mid-fusion checkpoint
            self.model = build_model('mid_mbv3', num_classes=num_classes)
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        # Assume the checkpoint is the state dict itself
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print(f"✓ Model loaded successfully from {self.model_path}")
                print(f"✓ Model architecture: mid_mbv3, Classes: {num_classes}, Device: {self.device}")
            else:
                print(f"⚠ Warning: Model checkpoint not found at {self.model_path}")
                print(f"⚠ Model predictions will not be available")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            print(f"✗ Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
            print(f"⚠ Continuing without model - authentication will still work")
            self.model = None

    def predict(self, image):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert PIL Image to tensor
        input_tensor = torch.from_numpy(np.array(image)).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor / 255.0
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1)
            confidence = torch.max(output, dim=1)[0]

        return prediction[0].cpu().numpy(), confidence[0].cpu().numpy(), output[0].cpu().numpy()

# Initialize model manager
model_manager = ModelManager()

# Configuration
app = Flask(__name__)
# Allow all origins for deployment (configure specific domains in production)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Allow all origins for now, update with specific Vercel domain after deployment
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Additional CORS configuration for other routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Database setup
def init_db():
    """Initialize SQLite database for users and results"""
    conn = sqlite3.connect('iddaw.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            original_url TEXT NOT NULL,
            mask_url TEXT NOT NULL,
            heatmap_url TEXT NOT NULL,
            overlay_url TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create or update default test user
    test_user = {
        'id': "test123",
        'name': "Test User",
        'email': "test@example.com",
        'password': "password123"
    }
    
    try:
        cursor.execute('SELECT id FROM users WHERE email = ?', (test_user['email'],))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # Update existing user
            cursor.execute(
                'UPDATE users SET name = ?, password_hash = ? WHERE email = ?',
                (test_user['name'], hash_password(test_user['password']), test_user['email'])
            )
            print("Updated existing test user")
        else:
            # Create new test user
            cursor.execute(
                'INSERT INTO users (id, name, email, password_hash) VALUES (?, ?, ?, ?)',
                (test_user['id'], test_user['name'], test_user['email'], 
                 hash_password(test_user['password']))
            )
            print("Created new test user")
            
        print("\nTest User Credentials:")
        print("Email:", test_user['email'])
        print("Password:", test_user['password'])
    except Exception as e:
        print(f"Error managing test user: {str(e)}")
        raise
    
    conn.commit()
    conn.close()

# Model loading and caching
class ModelManager:
    def __init__(self):
        self.models = {}
        self.label_mapper = LabelMapper.default()
        self.safety_groups = SafetyGroups.default().group_to_classes
        self.color_map = color_map(len(self.label_mapper.id_to_class))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_type: str):
        """Load and cache a model"""
        if model_type not in self.models:
            print(f"Loading model: {model_type}")
            
            
            # Map frontend model types to our model variants 
            model_mapping = {
                'rgb_mbv3': 'rgb_mbv3',
                'nir_fastscnn': 'nir_fastscnn',
                'early4_mbv3': 'early4_mbv3', 
                'mid_mbv3': 'mid_mbv3'
            }
            
            variant = model_mapping.get(model_type, 'rgb_mbv3')
            
            try:
                print(f"Building model {variant}...")
                model = build_model(variant, num_classes=len(self.label_mapper.id_to_class))
                
                # Load checkpoint
                checkpoint_path = os.path.join("ckpts", f"best_{variant}.pt")
                if os.path.exists(checkpoint_path):
                    print(f"Loading checkpoint from {checkpoint_path}")
                    try:
                        state = torch.load(checkpoint_path, map_location=self.device)
                        model.load_state_dict(state["model"])
                    except Exception as e:
                        print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(f"Failed to load model checkpoint: {str(e)}")
                else:
                    print(f"Checkpoint not found: {checkpoint_path}")
                    raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
                
                print("Moving model to device:", self.device)
                model = model.to(self.device)
                model.eval()
                self.models[model_type] = model
                print(f"Successfully loaded {variant} model")
                
            except Exception as e:
                print(f"Failed to load model {variant}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            return self.models[model_type]# Global model manager instance with error handling
def get_model_manager() -> ModelManager:
    """Get or create global model manager instance"""
    global _model_manager
    if not hasattr(get_model_manager, '_model_manager'):
        try:
            get_model_manager._model_manager = ModelManager()
            print(f"Created new ModelManager instance - Using device: {get_model_manager._model_manager.device}")
        except Exception as e:
            print("Error creating ModelManager:")
            print(str(e))
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize ModelManager: {str(e)}")
    return get_model_manager._model_manager

# Authentication helpers
def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

def generate_token(user_id: str) -> str:
    """Generate JWT token"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow().timestamp() + 86400  # 24 hours
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return user_id"""
    try:
        # Handle various token formats
        if token.startswith('Bearer '):
            token = token.replace('Bearer ', '')
        
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        if not payload or 'user_id' not in payload:
            return None
            
        # Check token expiration
        exp = payload.get('exp', 0)
        if exp < datetime.utcnow().timestamp():
            return None
            
        return payload['user_id']
    except jwt.InvalidTokenError:
        return None
    except Exception as e:
        print(f"Token verification error: {str(e)}")
        return None

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'error': 'Invalid token'}), 401
        
        request.user_id = user_id
        return f(*args, **kwargs)
    return decorated_function

# API Routes
@app.route('/api/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({'message': 'IDDAW API is running'})

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """User registration"""
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not all([name, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    conn = sqlite3.connect('iddaw.db')
    cursor = conn.cursor()
    
    # Check if user already exists
    cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({'error': 'User already exists'}), 400
    
    # Create user
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    
    cursor.execute(
        'INSERT INTO users (id, name, email, password_hash) VALUES (?, ?, ?, ?)',
        (user_id, name, email, password_hash)
    )
    
    conn.commit()
    conn.close()
    
    token = generate_token(user_id)
    return jsonify({
        'token': token,
        'user': {
            'id': user_id,
            'name': name,
            'email': email
        }
    })

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not all([email, password]):
        return jsonify({'error': 'Missing credentials'}), 400
    
    conn = sqlite3.connect('iddaw.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT id, name, email, password_hash FROM users WHERE email = ?',
        (email,)
    )
    user = cursor.fetchone()
    conn.close()
    
    if not user or not verify_password(password, user[3]):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    token = generate_token(user[0])
    return jsonify({
        'token': token,
        'user': {
            'id': user[0],
            'name': user[1],
            'email': user[2]
        }
    })

@app.route('/api/auth/check', methods=['GET'])
def check_auth():
    """Check if token is valid and return user info"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({'error': 'No token provided'}), 401
        
    user_id = verify_token(token)
    if not user_id:
        return jsonify({'error': 'Invalid token'}), 401
        
    conn = sqlite3.connect('iddaw.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT id, name, email FROM users WHERE id = ?',
        (user_id,)
    )
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    return jsonify({
        'user': {
            'id': user[0],
            'name': user[1],
            'email': user[2]
        }
    })

@app.route('/api/me', methods=['GET'])
@require_auth
def get_user():
    """Get current user info"""
    conn = sqlite3.connect('iddaw.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT id, name, email FROM users WHERE id = ?',
        (request.user_id,)
    )
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user[0],
        'name': user[1],
        'email': user[2]
    })

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler to log all exceptions"""
    print(f"Exception in {request.path}: {str(e)}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
@require_auth
def predict():
    """Image segmentation prediction endpoint"""
    try:
        print("\n=== New Prediction Request ===")
        print("Content-Type:", request.content_type)
        print("Files in request:", list(request.files.keys()) if request.files else "No files")
        print("Headers:", dict(request.headers))
        
        # Get model manager instance
        model_manager = get_model_manager()
        print(f"Using ModelManager - Device: {model_manager.device}")
        print(f"Available models: {list(model_manager.models.keys())}")
        
        # Validate request
        if not request.files:
            raise ValueError('No files uploaded')
            
        if 'rgb' not in request.files or 'nir' not in request.files:
            raise ValueError('Both RGB and NIR images are required')
            
        rgb_file = request.files['rgb']
        nir_file = request.files['nir']
        
        print(f"RGB file: {rgb_file.filename} ({rgb_file.content_type})")
        print(f"NIR file: {nir_file.filename} ({nir_file.content_type})")
        
        if not rgb_file.filename or not nir_file.filename:
            raise ValueError('Empty filenames not allowed')
        
        # Save uploaded files with unique names
        rgb_filename = secure_filename(rgb_file.filename)
        nir_filename = secure_filename(nir_file.filename)
        
        rgb_path = os.path.join(app.config['UPLOAD_FOLDER'], f"rgb_{uuid.uuid4()}_{rgb_filename}")
        nir_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nir_{uuid.uuid4()}_{nir_filename}")
        
        try:
            # Create upload directory and save files
            print("\nSaving uploaded files...")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            rgb_file.save(rgb_path)
            nir_file.save(nir_path)
            
            # Verify files were saved correctly
            if not os.path.exists(rgb_path) or not os.path.exists(nir_path):
                raise IOError("Failed to save uploaded files")
                
            print(f"Files saved successfully:")
            print(f"RGB: {rgb_path} ({os.path.getsize(rgb_path)} bytes)")
            print(f"NIR: {nir_path} ({os.path.getsize(nir_path)} bytes)")
            
        except Exception as e:
            print(f"Error saving files: {str(e)}")
            import traceback
            traceback.print_exc()
            # Clean up partial files
            for path in [rgb_path, nir_path]:
                if os.path.exists(path):
                    os.remove(path)
            raise RuntimeError(f"Failed to save uploaded files: {str(e)}")
            if os.path.exists(rgb_path):
                os.remove(rgb_path)
            if os.path.exists(nir_path):
                os.remove(nir_path)
            raise
        
        # Process images
        result_id = str(uuid.uuid4())
        result_urls = process_images(rgb_path, nir_path, result_id)
        
        # Save result to database
        conn = sqlite3.connect('iddaw.db')
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO results (id, user_id, original_url, mask_url, heatmap_url, overlay_url) VALUES (?, ?, ?, ?, ?, ?)',
            (result_id, request.user_id, result_urls['original'], result_urls['mask'], 
             result_urls['heatmap'], result_urls['overlay'])
        )
        
        conn.commit()
        conn.close()
        
        # Clean up uploaded files
        os.remove(rgb_path)
        os.remove(nir_path)
        
        return jsonify({
            'id': result_id,
            'originalUrl': result_urls['original'],
            'maskUrl': result_urls['mask'],
            'heatmapUrl': result_urls['heatmap'],
            'overlayUrl': result_urls['overlay'],
            'createdAt': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def process_images(rgb_path: str, nir_path: str, result_id: str, model_manager: ModelManager = None) -> Dict[str, str]:
    """Process RGB and NIR images to generate segmentation results"""
    try:
        if model_manager is None:
            model_manager = ModelManager()
            
        print(f"\n=== Processing Images ===")
        print(f"RGB Path: {rgb_path}")
        print(f"NIR Path: {nir_path}")
        print(f"Result ID: {result_id}")
        print(f"Using device: {model_manager.device}")
        
        # Verify file existence
        if not os.path.exists(rgb_path):
            raise ValueError(f"RGB image file not found: {rgb_path}")
        if not os.path.exists(nir_path):
            raise ValueError(f"NIR image file not found: {nir_path}")
            
        # Check file sizes
        rgb_size = os.path.getsize(rgb_path)
        nir_size = os.path.getsize(nir_path)
        print(f"File sizes - RGB: {rgb_size} bytes, NIR: {nir_size} bytes")
        
        try:
            # Load and preprocess images
            print("\nLoading and preprocessing images...")
            rgb_img = cv2.imread(rgb_path)
            if rgb_img is None:
                raise ValueError(f"Could not load RGB image: {rgb_path}")
            print("RGB image shape:", rgb_img.shape)
            
            nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
            if nir_img is None:
                raise ValueError(f"Could not load NIR image: {nir_path}")
            print("NIR image shape:", nir_img.shape)
            
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Store original RGB for visualization
            original_rgb = rgb_img.copy()
            
            # Resize images to model input size
            target_size = (512, 512)
            print(f"\nResizing images to {target_size}...")
            rgb_img = cv2.resize(rgb_img, target_size)
            nir_img = cv2.resize(nir_img, target_size)
            
            # Convert to tensors with proper normalization
            print("\nConverting images to tensors...")
            rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
            rgb_tensor = rgb_tensor.contiguous()
            print("RGB tensor shape:", rgb_tensor.shape)
            
            nir_tensor = torch.from_numpy(nir_img).unsqueeze(0).float() / 255.0
            nir_tensor = nir_tensor.contiguous()
            print("NIR tensor shape:", nir_tensor.shape)
            
            # Create early4 fusion (RGB + NIR as 4-channel input)
            print("\nCreating early4 fusion tensor...")
            early4_tensor = torch.cat([rgb_tensor, nir_tensor], dim=0)
            print("Early4 tensor shape:", early4_tensor.shape)
            
        except Exception as e:
            print("Error during image preprocessing:")
            print(str(e))
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to preprocess images: {str(e)}")
            
    except Exception as e:
        print(f"Error in process_images initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Load and run models
    results = {}
    
    # RGB model inference
    try:
        print("\nRunning RGB model inference...")
        rgb_model = model_manager.load_model('rgb_mbv3')
        with torch.no_grad():
            rgb_input = rgb_tensor.unsqueeze(0).to(model_manager.device)
            print("RGB input tensor shape:", rgb_input.shape)
            rgb_logits = rgb_model(rgb_input)
            print("RGB logits shape:", rgb_logits.shape)
            rgb_pred = torch.argmax(rgb_logits, dim=1)[0].cpu().numpy()
            print("RGB prediction shape:", rgb_pred.shape)
            rgb_entropy = entropy_map(rgb_logits)[0].cpu().numpy()
            print("RGB entropy map shape:", rgb_entropy.shape)
    except Exception as e:
        print(f"Error during RGB model inference: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"RGB model inference failed: {str(e)}")
    
    # NIR model inference
    try:
        print("\nRunning NIR model inference...")
        nir_model = model_manager.load_model('nir_fastscnn')
        with torch.no_grad():
            nir_input = nir_tensor.unsqueeze(0).to(model_manager.device)
            print("NIR input tensor shape:", nir_input.shape)
            nir_logits = nir_model(nir_input)
            print("NIR logits shape:", nir_logits.shape)
            nir_pred = torch.argmax(nir_logits, dim=1)[0].cpu().numpy()
            print("NIR prediction shape:", nir_pred.shape)
    except Exception as e:
        print(f"Error during NIR model inference: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"NIR model inference failed: {str(e)}")
    
    # Calculate NIR entropy
    try:
        nir_entropy = entropy_map(nir_logits)[0].cpu().numpy()
        print("NIR entropy map shape:", nir_entropy.shape)
    except Exception as e:
        print(f"Error calculating NIR entropy: {str(e)}")
        nir_entropy = None
    
    # Early4 fusion model
    try:
        print("\nRunning Early4 fusion model inference...")
        early4_model = model_manager.load_model('early4_mbv3')
        with torch.no_grad():
            early4_input = early4_tensor.unsqueeze(0).to(model_manager.device)
            print("Early4 input tensor shape:", early4_input.shape)
            early4_logits = early4_model(early4_input)
            print("Early4 logits shape:", early4_logits.shape)
            early4_pred = torch.argmax(early4_logits, dim=1)[0].cpu().numpy()
            print("Early4 prediction shape:", early4_pred.shape)
            early4_entropy = entropy_map(early4_logits)[0].cpu().numpy()
            print("Early4 entropy map shape:", early4_entropy.shape)
    except Exception as e:
        print(f"Error during Early4 model inference: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Early4 model inference failed: {str(e)}")
    
    # Mid fusion model
    try:
        print("\nRunning Mid fusion model inference...")
        mid_model = model_manager.load_model('mid_mbv3')
        with torch.no_grad():
            rgb_input = rgb_tensor.unsqueeze(0).to(model_manager.device)
            nir_input = nir_tensor.unsqueeze(0).to(model_manager.device)
            print("Mid fusion input shapes - RGB:", rgb_input.shape, "NIR:", nir_input.shape)
            mid_logits = mid_model(rgb_input, nir_input)
            print("Mid fusion logits shape:", mid_logits.shape)
            mid_pred = torch.argmax(mid_logits, dim=1)[0].cpu().numpy()
            print("Mid fusion prediction shape:", mid_pred.shape)
            mid_entropy = entropy_map(mid_logits)[0].cpu().numpy()
            print("Mid fusion entropy map shape:", mid_entropy.shape)
    except Exception as e:
        print(f"Error during Mid fusion model inference: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Mid fusion model inference failed: {str(e)}")
        mid_pred = torch.argmax(mid_logits, dim=1)[0].cpu().numpy()
        mid_entropy = entropy_map(mid_logits)[0].cpu().numpy()
    
    try:
        print("\nGenerating ensemble prediction...")
        # Use the best performing model (early4_mbv3 based on validation results)
        final_pred = early4_pred
        final_entropy = early4_entropy
        print("Final prediction shape:", final_pred.shape)
        print("Final entropy map shape:", final_entropy.shape)
        
        # Save model predictions and entropy maps
        results = {
            'rgb_pred': rgb_pred,
            'rgb_entropy': rgb_entropy,
            'nir_pred': nir_pred,
            'nir_entropy': nir_entropy,
            'early4_pred': early4_pred,
            'early4_entropy': early4_entropy,
            'mid_pred': mid_pred,
            'mid_entropy': mid_entropy,
            'final_pred': final_pred,
            'final_entropy': final_entropy
        }
        
        # Generate visualizations
        try:
            print("\nGenerating visualizations...")
            # Use original resolution RGB for visualization
            rgb_vis = cv2.resize(original_rgb, target_size)  # Resize to match prediction size
            print("RGB visualization shape:", rgb_vis.shape)
            
            # Segmentation mask with proper coloring
            print("Generating segmentation mask...")
            mask_vis = np.zeros_like(rgb_vis)
            unique_classes = np.unique(final_pred)
            print(f"Found {len(unique_classes)} unique classes in prediction")
            for class_id in unique_classes:
                mask_vis[final_pred == class_id] = model_manager.color_map[class_id]
            print("Mask visualization shape:", mask_vis.shape)
            
            # Overlay with adjusted alpha for better visibility
            print("Generating segmentation overlay...")
            overlay_vis = overlay_segmentation(rgb_vis, final_pred, model_manager.color_map, alpha=0.7)
            print("Overlay visualization shape:", overlay_vis.shape)
            
            # Safety heatmap with stronger highlighting
            print("Generating safety heatmap...")
            safety_vis = safety_heatmap(rgb_vis, final_pred, model_manager.safety_groups, 
                                      model_manager.label_mapper.id_to_class, alpha=0.7)
            print("Safety heatmap shape:", safety_vis.shape)
            
            # Confidence heatmap with enhanced contrast
            print("Generating confidence heatmap...")
            conf_vis = confidence_heatmap(rgb_vis, final_entropy, alpha=0.7)
            print("Confidence heatmap shape:", conf_vis.shape)
            
            # Save results
            print("\nSaving visualization results...")
            results_dir = Path(app.config['RESULTS_FOLDER']) / result_id
            results_dir.mkdir(exist_ok=True, parents=True)
            print(f"Created results directory: {results_dir}")
            
            # Convert to BGR for saving (cv2.imwrite expects BGR)
            rgb_vis_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
            overlay_vis_bgr = cv2.cvtColor(overlay_vis, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error during visualization generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate visualizations: {str(e)}")
    except Exception as e:
        print(f"Error during model prediction processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    safety_vis_bgr = cv2.cvtColor(safety_vis, cv2.COLOR_RGB2BGR)
    conf_vis_bgr = cv2.cvtColor(conf_vis, cv2.COLOR_RGB2BGR)
    
    # Save images
    cv2.imwrite(str(results_dir / 'original.png'), rgb_vis_bgr)
    cv2.imwrite(str(results_dir / 'mask.png'), mask_vis)  # Mask is already in correct format
    cv2.imwrite(str(results_dir / 'overlay.png'), overlay_vis_bgr)
    cv2.imwrite(str(results_dir / 'safety.png'), safety_vis_bgr)
    cv2.imwrite(str(results_dir / 'confidence.png'), conf_vis_bgr)
    
    # Return absolute URLs for the results
    base_url = f"/api/results/{result_id}"
    return {
        'id': result_id,
        'originalUrl': f"{base_url}/original.png",
        'maskUrl': f"{base_url}/mask.png", 
        'overlayUrl': f"{base_url}/overlay.png",
        'heatmapUrl': f"{base_url}/safety.png",
        'confidenceUrl': f"{base_url}/confidence.png",
        'createdAt': datetime.utcnow().isoformat()
    }

@app.route('/api/results/<result_id>/<filename>')
@require_auth
def get_result_image(result_id: str, filename: str):
    """Serve result images"""
    # Verify user owns this result
    conn = sqlite3.connect('iddaw.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT id FROM results WHERE id = ? AND user_id = ?',
        (result_id, request.user_id)
    )
    
    if not cursor.fetchone():
        conn.close()
        return jsonify({'error': 'Result not found'}), 404
    
    conn.close()
    
    # Serve file
    file_path = Path(app.config['RESULTS_FOLDER']) / result_id / filename
    if file_path.exists():
        response = send_file(str(file_path), mimetype='image/png')
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Authorization'
        return response
    return jsonify({'error': 'Image not found'}), 404

@app.route('/api/results', methods=['GET'])
@require_auth
def list_results():
    """List user's results"""
    conn = sqlite3.connect('iddaw.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT id, overlay_url, created_at FROM results WHERE user_id = ? ORDER BY created_at DESC',
        (request.user_id,)
    )
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row[0],
            'thumbnailUrl': row[1],
            'createdAt': row[2]
        })
    
    conn.close()
    return jsonify(results)

@app.route('/api/results/<result_id>', methods=['GET'])
@require_auth
def get_result(result_id: str):
    """Get specific result details"""
    conn = sqlite3.connect('iddaw.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT id, original_url, mask_url, heatmap_url, overlay_url, created_at FROM results WHERE id = ? AND user_id = ?',
        (result_id, request.user_id)
    )
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'error': 'Result not found'}), 404
    
    return jsonify({
        'id': result[0],
        'originalUrl': result[1],
        'maskUrl': result[2],
        'heatmapUrl': result[3],
        'overlayUrl': result[4],
        'createdAt': result[5]
    })

@app.route('/api/results/save', methods=['POST'])
@require_auth
def save_result():
    """Save result to user's profile (already saved, just confirm)"""
    data = request.get_json()
    result_id = data.get('resultId')
    
    if not result_id:
        return jsonify({'error': 'Result ID required'}), 400
    
    # Verify result exists and belongs to user
    conn = sqlite3.connect('iddaw.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT id FROM results WHERE id = ? AND user_id = ?',
        (result_id, request.user_id)
    )
    
    if not cursor.fetchone():
        conn.close()
        return jsonify({'error': 'Result not found'}), 404
    
    conn.close()
    return jsonify({'ok': True})

if __name__ == '__main__':
    import socket
    
    try:
        init_db()
        print("\nIDDAW Backend API Server")
        print("=" * 50)
        
        # Initialize model manager
        model_manager = get_model_manager()
        print(f"Device: {model_manager.device}")
        print(f"Models available: {list(model_manager.models.keys())}")
        print("=" * 50)
        
        port = 8001
        
        # Try to find an available port
        while port < 8010:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('0.0.0.0', port))
                sock.close()
                break
            except socket.error:
                print(f"Port {port} is in use, trying next port...")
                port += 1
                sock.close()
                continue
        
        if port >= 8010:
            print("ERROR: Could not find an available port")
            sys.exit(1)
            
        print(f"Starting server on http://localhost:{port}")
        from waitress import serve
        print(f"Serving on http://0.0.0.0:{port}")
        serve(app, host='0.0.0.0', port=port)
        
    except Exception as e:
        print("Failed to start server:")
        print(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
