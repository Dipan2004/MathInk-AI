from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import cv2
import os
import base64
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
from flask_cors import CORS
import logging
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Enhanced class mapping with more symbols
index_by_directory = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '-': 11, 'x': 12, '=': 13, 'a': 14, 'b': 15,
    'c': 16, 'd': 17, 'e': 18, 'f': 19, 'g': 20, 'h': 21,
    'i': 22, 'j': 23, 'k': 24, 'l': 25, 'm': 26, 'n': 27,
    'o': 28, 'p': 29, 'q': 30, 'r': 31, 's': 32, 't': 33,
    'u': 34, 'v': 35, 'w': 36, 'y': 37, 'z': 38
}

def solve_linear_equation(equation, variable):
    """Enhanced equation solver with better error handling"""
    try:
        n = len(equation)
        if '=' not in equation:
            return f"No equation found (missing '=')"
        
        # Split equation at '='
        left_side, right_side = equation.split('=')
        
        # Initialize coefficients
        var_coeff = 0
        constant = 0
        
        def parse_side(side, sign):
            nonlocal var_coeff, constant
            current_num = ""
            current_sign = sign
            i = 0
            
            while i < len(side):
                char = side[i]
                
                if char in ['+', '-']:
                    if current_num:
                        constant += current_sign * float(current_num)
                        current_num = ""
                    current_sign = sign * (1 if char == '+' else -1)
                elif char == variable:
                    if current_num == "":
                        var_coeff += current_sign
                    else:
                        var_coeff += current_sign * float(current_num)
                    current_num = ""
                    current_sign = sign
                elif char.isdigit() or char == '.':
                    current_num += char
                
                i += 1
            
            # Handle remaining number
            if current_num:
                constant += current_sign * float(current_num)
        
        # Parse both sides
        parse_side(left_side.strip(), 1)
        parse_side(right_side.strip(), -1)
        
        # Solve for variable
        if var_coeff == 0:
            if constant == 0:
                return "Infinite solutions"
            else:
                return "No solution"
        
        result = -constant / var_coeff
        return round(result, 6)
        
    except Exception as e:
        logger.error(f"Error solving equation: {e}")
        return f"Error solving equation: {str(e)}"

class EnhancedCNN:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.load_model()
    
    def load_model(self):
        """Load model with multiple format support"""
        try:
            model_dir = Path('models')
            
            # Try loading the enhanced Keras model first
            keras_model_path = model_dir / 'enhanced_math_equation_model.keras'
            if keras_model_path.exists():
                logger.info('Loading enhanced Keras model...')
                self.model = load_model(str(keras_model_path))
                logger.info('Enhanced model loaded successfully!')
            
            # Fallback to legacy format
            elif (model_dir / 'model.json').exists() and (model_dir / 'model_weights.h5').exists():
                logger.info('Loading legacy model format...')
                with open(model_dir / 'model.json', 'r') as json_file:
                    model_json = json_file.read()
                self.model = model_from_json(model_json)
                self.model.load_weights(str(model_dir / 'model_weights.h5'))
                logger.info('Legacy model loaded successfully!')
            
            else:
                raise FileNotFoundError("No model files found!")
            
            # Load label encoder if available
            label_encoder_path = model_dir / 'label_encoder_classes.npy'
            if label_encoder_path.exists():
                classes = np.load(str(label_encoder_path), allow_pickle=True)
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = classes
                logger.info(f'Label encoder loaded with classes: {classes}')
            
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            raise e
    
    def preprocess_image_enhanced(self, img):
        """Enhanced image preprocessing"""
        try:
            # Resize with aspect ratio preservation
            h, w = img.shape
            if h > w:
                new_h, new_w = 32, int(32 * w / h)
            else:
                new_h, new_w = int(32 * h / w), 32
            
            img = cv2.resize(img, (new_w, new_h))
            
            # Pad to 32x32
            pad_h = (32 - new_h) // 2
            pad_w = (32 - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_h, 32-new_h-pad_h, 
                                   pad_w, 32-new_w-pad_w, cv2.BORDER_CONSTANT, value=255)
            
            # Apply bilateral filter for noise reduction
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
            # Adaptive thresholding
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations
            kernel = np.ones((2,2), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None
    
    def extract_characters_enhanced(self, img):
        """Enhanced character extraction with better segmentation"""
        try:
            # Invert image for processing
            img_inv = cv2.bitwise_not((img * 255).astype(np.uint8))
            
            # Find contours
            contours, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return []
            
            # Filter and sort contours
            valid_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter based on size and area
                if w > 5 and h > 5 and area > 50:
                    valid_contours.append((x, y, w, h, contour))
            
            # Sort by x-coordinate (left to right)
            valid_contours.sort(key=lambda x: x[0])
            
            # Extract character images
            char_images = []
            for x, y, w, h, contour in valid_contours:
                # Add padding
                padding = 5
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(img.shape[1], x + w + padding)
                y_end = min(img.shape[0], y + h + padding)
                
                char_img = img_inv[y_start:y_end, x_start:x_end]
                
                if char_img.size > 0:
                    # Preprocess individual character
                    char_img = self.preprocess_image_enhanced(char_img)
                    if char_img is not None:
                        char_images.append(char_img)
            
            return char_images
            
        except Exception as e:
            logger.error(f"Error extracting characters: {e}")
            return []
    
    def predict(self, img_bytes):
        """Enhanced prediction with better error handling"""
        try:
            # Save and load image
            temp_path = '_temp_image_.png'
            Image.open(img_bytes).save(temp_path)
            
            img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            os.remove(temp_path)
            
            if img is None:
                return "Error: Could not process image"
            
            # Extract characters
            char_images = self.extract_characters_enhanced(img)
            
            if not char_images:
                return "Error: No characters detected"
            
            # Predict each character
            predicted_chars = []
            confidence_scores = []
            
            for char_img in char_images:
                # Prepare input
                char_input = np.expand_dims(char_img, axis=0)
                if len(char_input.shape) == 3:
                    char_input = np.expand_dims(char_input, axis=-1)
                
                # Predict
                prediction = self.model.predict(char_input, verbose=0)
                
                if prediction is not None and len(prediction) > 0:
                    predicted_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    
                    # Map prediction to character
                    if self.label_encoder:
                        try:
                            char = self.label_encoder.inverse_transform([predicted_class])[0]
                        except:
                            char = str(predicted_class)
                    else:
                        # Fallback mapping
                        char_map = {
                            0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                            5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                            10: '+', 11: '-', 12: 'x', 13: '=', 14: 'a'
                        }
                        char = char_map.get(predicted_class, str(predicted_class))
                    
                    predicted_chars.append(char)
                    confidence_scores.append(confidence)
            
            if not predicted_chars:
                return "Error: No valid predictions"
            
            # Combine characters into equation
            equation = ''.join(predicted_chars)
            
            # Post-process equation (fix common OCR errors)
            equation = self.post_process_equation(equation)
            
            logger.info(f"Predicted equation: {equation}")
            logger.info(f"Confidence scores: {confidence_scores}")
            
            return equation
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}"
    
    def post_process_equation(self, equation):
        """Post-process equation to fix common recognition errors"""
        try:
            # Common corrections
            corrections = {
                'X': 'x',  # Uppercase X to multiplication
                '*': 'x',  # Asterisk to x
                '×': 'x',  # Times symbol to x
                '÷': '/',  # Division symbol
                'o': '0',  # Letter o to zero
                'O': '0',  # Uppercase O to zero
                'l': '1',  # Lowercase l to one
                'I': '1',  # Uppercase I to one
            }
            
            for old, new in corrections.items():
                equation = equation.replace(old, new)
            
            # Remove extra spaces
            equation = ''.join(equation.split())
            
            # Ensure proper equation format
            if '=' not in equation and any(var in equation for var in 'abcdefghijklmnopqrstuvwxyz'):
                # Try to add equals sign if missing in linear equation
                pass  # Keep as is for now
            
            return equation
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return equation

def preprocess_uploaded_image(img):
    """Preprocess uploaded image for better recognition"""
    try:
        # Resize maintaining aspect ratio
        height, width = img.shape
        if width > 800:
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Convert to binary using Otsu's thresholding
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return img

# Initialize the enhanced CNN model
try:
    cnn = EnhancedCNN()
    model_status = "Enhanced model loaded successfully!"
    logger.info(model_status)
except Exception as e:
    model_status = f"Error loading model: {str(e)}"
    logger.error(model_status)
    cnn = None

@app.route('/')
def index():
    """Main page with model status"""
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if cnn is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get processing mode
        mode = request.form.get('mode', 'basic')
        
        # Process image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Preprocess image
        processed_img = preprocess_uploaded_image(img)
        
        # Convert to BytesIO for model prediction
        _, buffer = cv2.imencode('.png', processed_img)
        img_io = BytesIO(buffer.tobytes())
        
        # Get prediction
        equation = cnn.predict(img_io)
        
        if equation.startswith("Error"):
            return jsonify({'error': equation}), 400
        
        # Process based on mode
        if mode == 'basic':
            try:
                # Basic arithmetic evaluation
                eval_equation = equation.replace('x', '*').replace('=', '==')
                
                # Check if it's an equation or expression
                if '==' in eval_equation:
                    # It's an equation, check if it's true
                    result = eval(eval_equation)
                    result = "True" if result else "False"
                else:
                    # It's an expression, evaluate it
                    result = eval(equation.replace('x', '*'))
                    
            except Exception as e:
                logger.error(f"Error evaluating expression: {e}")
                result = f"Could not evaluate: {str(e)}"
                
        elif mode == 'linear':
            # Linear equation solving
            result = {}
            variables = set(char for char in equation if char.isalpha() and char not in ['x'])
            
            if not variables:
                result = "No variables found to solve"
            else:
                for var in variables:
                    try:
                        solution = solve_linear_equation(equation, var)
                        result[var] = solution
                    except Exception as e:
                        result[var] = f"Error: {str(e)}"
        
        else:
            result = "Invalid mode"

        # Encode processed image for response
        retval, buffer = cv2.imencode('.png', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        response = {
            'equation': equation,
            'result': result,
            'processed_image': f'data:image/png;base64,{img_base64}',
            'confidence': 'High' if isinstance(result, (int, float, dict)) else 'Low'
        }
        
        logger.info(f"Successful prediction: {equation} -> {result}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': cnn is not None,
        'model_status': model_status
    })

@app.route('/api/supported_symbols', methods=['GET'])
def get_supported_symbols():
    """Get list of supported symbols"""
    symbols = {
        'digits': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'operators': ['+', '-', 'x', '='],
        'variables': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                     'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z']
    }
    return jsonify(symbols)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Configure for production
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Model status: {model_status}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
