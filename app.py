#!/usr/bin/env python3
"""
AgriMitra - Multi-Crop Disease Classification API
Supports Cotton and Sugarcane disease classification
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

class ModelManager:
    def __init__(self):
        self.cotton_model = None
        self.sugarcane_model = None
        self.cotton_class_names = None
        self.sugarcane_class_names = None
        self.cotton_model_info = None
        self.sugarcane_model_info = None
        
    def load_models(self):
        """Load both cotton and sugarcane models"""
        print("🔄 Loading models...")
        
        # Load Cotton Model
        try:
            self.cotton_model = keras.models.load_model('cotton_disease_final_model.h5')
            with open('cotton_model_info.json', 'r') as f:
                self.cotton_model_info = json.load(f)
            self.cotton_class_names = self.cotton_model_info['class_names']
            print("✅ Cotton model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading cotton model: {e}")
            self.cotton_model = None
            
        # Load Sugarcane Model
        try:
            self.sugarcane_model = keras.models.load_model('sugarcane_disease_final_model.h5')
            with open('sugarcane_model_info.json', 'r') as f:
                self.sugarcane_model_info = json.load(f)
            self.sugarcane_class_names = self.sugarcane_model_info['class_names']
            print("✅ Sugarcane model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading sugarcane model: {e}")
            self.sugarcane_model = None
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess image for model prediction"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_cotton(self, image):
        """Predict cotton disease"""
        if self.cotton_model is None:
            return {"error": "Cotton model not loaded"}
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.cotton_model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            predicted_class = self.cotton_class_names[predicted_class_idx]
            
            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.cotton_class_names):
                class_probabilities[class_name] = float(predictions[0][i])
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "model_info": self.cotton_model_info
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_sugarcane(self, image):
        """Predict sugarcane disease"""
        if self.sugarcane_model is None:
            return {"error": "Sugarcane model not loaded"}
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.sugarcane_model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            predicted_class = self.sugarcane_class_names[predicted_class_idx]
            
            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.sugarcane_class_names):
                class_probabilities[class_name] = float(predictions[0][i])
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "model_info": self.sugarcane_model_info
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize model manager
model_manager = ModelManager()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with model selection"""
    return render_template('index.html')

@app.route('/cotton')
def cotton_page():
    """Cotton disease classification page"""
    return render_template('cotton.html')

@app.route('/sugarcane')
def sugarcane_page():
    """Sugarcane disease classification page"""
    return render_template('sugarcane.html')

@app.route('/api/cotton/predict', methods=['POST'])
def predict_cotton_api():
    """API endpoint for cotton disease prediction"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Open image
            image = Image.open(io.BytesIO(file.read()))
            
            # Make prediction
            result = model_manager.predict_cotton(image)
            
            if "error" in result:
                return jsonify(result), 500
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP"}), 400

@app.route('/api/sugarcane/predict', methods=['POST'])
def predict_sugarcane_api():
    """API endpoint for sugarcane disease prediction"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Open image
            image = Image.open(io.BytesIO(file.read()))
            
            # Make prediction
            result = model_manager.predict_sugarcane(image)
            
            if "error" in result:
                return jsonify(result), 500
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP"}), 400

@app.route('/api/models/status')
def models_status():
    """API endpoint to check model status"""
    status = {
        "cotton_model": {
            "loaded": model_manager.cotton_model is not None,
            "info": model_manager.cotton_model_info
        },
        "sugarcane_model": {
            "loaded": model_manager.sugarcane_model is not None,
            "info": model_manager.sugarcane_model_info
        }
    }
    return jsonify(status)

@app.route('/api/cotton/classes')
def cotton_classes():
    """API endpoint to get cotton disease classes"""
    if model_manager.cotton_class_names:
        return jsonify({"classes": model_manager.cotton_class_names})
    else:
        return jsonify({"error": "Cotton model not loaded"}), 500

@app.route('/api/sugarcane/classes')
def sugarcane_classes():
    """API endpoint to get sugarcane disease classes"""
    if model_manager.sugarcane_class_names:
        return jsonify({"classes": model_manager.sugarcane_class_names})
    else:
        return jsonify({"error": "Sugarcane model not loaded"}), 500

if __name__ == '__main__':
    print("🌾 AgriMitra - Multi-Crop Disease Classification API")
    print("=" * 60)
    
    # Load models
    model_manager.load_models()
    
    # Get port from environment variable (for Render)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n🚀 Starting Flask server...")
    print(f"📱 Server will be available on port {port}")
    print("🛑 Press CTRL+C to stop the server")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=port)

