#!/usr/bin/env python3
"""
Compatible with Python 3.11+
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is 3.11 or higher"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        logger.error("Python 3.11 or higher is required!")
        logger.error("Please upgrade your Python installation.")
        return False
    
    logger.info("✓ Python version check passed")
    return True

def install_requirements():
    """Install required packages"""
    logger.info("Installing required packages...")
    
    requirements = [
        "flask>=2.3.0",
        "tensorflow>=2.13.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
        "flask-cors>=4.0.0",
        "albumentations>=1.3.0"
    ]
    
    for requirement in requirements:
        try:
            logger.info(f"Installing {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            logger.info(f"✓ {requirement} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install {requirement}: {e}")
            return False
    
    logger.info("✓ All requirements installed successfully")
    return True

def create_directory_structure():
    """Create necessary directories"""
    logger.info("Creating directory structure...")
    
    directories = [
        "models",
        "data",
        "data/extracted_images",
        "logs",
        "templates",
        "static",
        "static/css",
        "static/js",
        "uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    logger.info("✓ Directory structure created")
    return True

def create_sample_html_template():
    """Create a basic HTML template for the Flask app"""
    logger.info("Creating HTML template...")
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Math Equation Recognition</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .status {
            padding: 10px;
            margin: 20px 0;
            border-radius: 5px;
            font-weight: bold;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #007bff;
        }
        .upload-area.dragover {
            border-color: #007bff;
            background-color: #f8f9fa;
        }
        input[type="file"] {
            margin: 20px 0;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        .mode-selector {
            margin: 20px 0;
        }
        .mode-selector label {
            margin-right: 20px;
            font-weight: normal;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .result h3 {
            margin-top: 0;
            color: #007bff;
        }
        .equation {
            font-family: 'Courier New', monospace;
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
        }
        .processed-image {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧮 Math Equation Recognition</h1>
        
        <div class="status {{ 'success' if 'success' in model_status else 'error' }}">
            Model Status: {{ model_status }}
        </div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" id="uploadArea">
                <p>📁 Drag and drop an image here or click to select</p>
                <input type="file" id="fileInput" name="file" accept="image/*" required>
            </div>
            
            <div class="mode-selector">
                <strong>Processing Mode:</strong><br>
                <label>
                    <input type="radio" name="mode" value="basic" checked>
                    Basic Arithmetic (e.g., 2+3, 5*7)
                </label>
                <label>
                    <input type="radio" name="mode" value="linear">
                    Linear Equations (e.g., 2x+3=7, a+5=12)
                </label>
            </div>
            
            <button type="submit" id="submitBtn">🔍 Analyze Equation</button>
            <button type="button" id="clearBtn">🗑️ Clear</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing your equation...</p>
        </div>
        
        <div id="results" style="display: none;">
            <div class="result">
                <h3>📝 Recognition Results</h3>
                <div>
                    <strong>Detected Equation:</strong>
                    <div class="equation" id="detectedEquation"></div>
                </div>
                <div>
                    <strong>Result:</strong>
                    <div id="calculationResult"></div>
                </div>
                <div>
                    <strong>Processed Image:</strong><br>
                    <img id="processedImage" class="processed-image" alt="Processed image">
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadForm = document.getElementById('uploadForm');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const submitBtn = document.getElementById('submitBtn');
        const clearBtn = document.getElementById('clearBtn');

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
            }
        });

        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Form submission
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(uploadForm);
            const mode = document.querySelector('input[name="mode"]:checked').value;
            formData.append('mode', mode);
            
            // Show loading
            loading.style.display = 'block';
            results.style.display = 'none';
            submitBtn.disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Display results
                    document.getElementById('detectedEquation').textContent = data.equation;
                    
                    let resultText;
                    if (typeof data.result === 'object') {
                        resultText = JSON.stringify(data.result, null, 2);
                    } else {
                        resultText = data.result;
                    }
                    document.getElementById('calculationResult').textContent = resultText;
                    
                    if (data.processed_image) {
                        document.getElementById('processedImage').src = data.processed_image;
                    }
                    
                    results.style.display = 'block';
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });

        // Clear functionality
        clearBtn.addEventListener('click', () => {
            uploadForm.reset();
            results.style.display = 'none';
        });
    </script>
</body>
</html>"""
    
    template_path = Path("templates/index.html")
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("✓ HTML template created")
    return True

def check_gpu_support():
    """Check for GPU support"""
    logger.info("Checking GPU support...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"✓ Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Enable memory growth to avoid taking all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            return True
        else:
            logger.info("No GPU found, will use CPU")
            return False
    except Exception as e:
        logger.warning(f"Could not check GPU support: {e}")
        return False

def create_run_script():
    """Create a run script for easy execution"""
    logger.info("Creating run script...")
    
    if platform.system() == "Windows":
        script_content = """@echo off
echo Starting Math Equation Recognition System...
python app.py
pause
"""
        script_path = "run.bat"
    else:
        script_content = """#!/bin/bash
echo "Starting Math Equation Recognition System..."
python3 app.py
"""
        script_path = "run.sh"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    if platform.system() != "Windows":
        os.chmod(script_path, 0o755)
    
    logger.info(f"✓ Run script created: {script_path}")
    return True

def create_sample_data():
    """Create sample data directory structure"""
    logger.info("Creating sample data structure...")
    
    sample_dir = Path("data/extracted_images")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample directories for digits 0-9
    for digit in range(10):
        digit_dir = sample_dir / str(digit)
        digit_dir.mkdir(exist_ok=True)
    
    # Create directories for basic operators
    operators = ['plus', 'minus', 'multiply', 'equals']
    for op in operators:
        op_dir = sample_dir / op
        op_dir.mkdir(exist_ok=True)
    
    logger.info("✓ Sample data structure created")
    return True

def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("Enhanced Math Equation Recognition Setup")
    logger.info("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        return False
    
    # Create directory structure
    if not create_directory_structure():
        logger.error("Failed to create directory structure")
        return False
    
    # Create HTML template
    if not create_sample_html_template():
        logger.error("Failed to create HTML template")
        return False
    
    # Check for GPU support
    check_gpu_support()
    
    # Create run script
    if not create_run_script():
        logger.error("Failed to create run script")
        return False
    
    # Create sample data structure
    if not create_sample_data():
        logger.warning("Failed to create sample data structure (non-critical)")
    
    logger.info("\nSetup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Place your training images in data/extracted_images/")
    logger.info("2. Run train.py to train the model")
    logger.info("3. Run app.py to start the web application")
    logger.info("4. Access the application at http://localhost:5000")
    
    return True

if __name__ == "__main__":
    main()