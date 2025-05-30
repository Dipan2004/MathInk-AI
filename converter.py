"""
Model Format Converter
Converts between different Keras model formats for compatibility with Python 3.11+
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_legacy_to_modern(model_json_path, weights_path, output_path):
    """
    Convert legacy Keras model (JSON + H5) to modern Keras format
    """
    try:
        logger.info("Loading legacy model...")
        
        # Load model architecture from JSON
        with open(model_json_path, 'r') as json_file:
            model_json = json_file.read()
        
        # Create model from JSON
        model = model_from_json(model_json)
        
        # Load weights
        model.load_weights(weights_path)
        
        logger.info("Legacy model loaded successfully")
        
        # Update model for modern TensorFlow
        # Fix any compatibility issues
        model = update_model_for_tf2(model)
        
        # Save in modern format
        model.save(output_path)
        logger.info(f"Modern model saved to: {output_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error converting model: {e}")
        raise e

def update_model_for_tf2(model):
    """
    Update model architecture for TensorFlow 2.x compatibility
    """
    try:
        # Get model config
        config = model.get_config()
        
        # Update any deprecated layer configurations
        for layer_config in config.get('layers', []):
            layer_type = layer_config.get('class_name', '')
            
            # Fix common compatibility issues
            if layer_type == 'Dense':
                # Ensure proper activation format
                activation = layer_config.get('config', {}).get('activation', 'linear')
                if activation and not isinstance(activation, str):
                    layer_config['config']['activation'] = 'linear'
            
            elif layer_type == 'Conv2D':
                # Ensure proper padding format
                padding = layer_config.get('config', {}).get('padding', 'valid')
                if padding not in ['valid', 'same']:
                    layer_config['config']['padding'] = 'valid'
        
        # Rebuild model with updated config
        from tensorflow.keras.models import Model
        rebuilt_model = tf.keras.models.model_from_config(config)
        
        # Copy weights
        rebuilt_model.set_weights(model.get_weights())
        
        return rebuilt_model
        
    except Exception as e:
        logger.warning(f"Could not update model architecture: {e}")
        return model

def create_compatible_model_structure():
    """
    Create a model structure compatible with the original app
    """
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    from tensorflow.keras.models import Sequential
    
    model = Sequential([
        Conv2D(30, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(15, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(50, activation='relu'),
        Dense(15, activation='softmax')  # Adjust based on your number of classes
    ])
    
    return model

def fix_model_json(json_path, output_json_path=None):
    """
    Fix common issues in model JSON files for TF 2.x compatibility
    """
    try:
        with open(json_path, 'r') as f:
            model_config = json.load(f)
        
        # Fix common issues
        if 'config' in model_config:
            layers = model_config['config'].get('layers', [])
            
            for layer in layers:
                layer_config = layer.get('config', {})
                
                # Fix kernel_initializer format
                if 'kernel_initializer' in layer_config:
                    if isinstance(layer_config['kernel_initializer'], dict):
                        if 'class_name' in layer_config['kernel_initializer']:
                            # Already in correct format
                            pass
                        else:
                            # Convert old format
                            layer_config['kernel_initializer'] = {
                                'class_name': 'GlorotUniform',
                                'config': {'seed': None}
                            }
                
                # Fix bias_initializer format
                if 'bias_initializer' in layer_config:
                    if isinstance(layer_config['bias_initializer'], dict):
                        if 'class_name' not in layer_config['bias_initializer']:
                            layer_config['bias_initializer'] = {
                                'class_name': 'Zeros',
                                'config': {}
                            }
        
        # Save fixed JSON
        output_path = output_json_path or json_path
        with open(output_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        logger.info(f"Fixed model JSON saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error fixing model JSON: {e}")
        raise e

def main():
    """
    Main conversion function
    """
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Paths for legacy model files
    json_path = model_dir / "model.json"
    weights_path = model_dir / "model_weights.h5"
    
    # Output path for modern model
    modern_model_path = model_dir / "enhanced_math_equation_model.keras"
    
    if json_path.exists() and weights_path.exists():
        logger.info("Converting legacy model to modern format...")
        
        try:
            # Fix JSON first
            fix_model_json(json_path)
            
            # Convert to modern format
            model = convert_legacy_to_modern(
                str(json_path), 
                str(weights_path), 
                str(modern_model_path)
            )
            
            logger.info("Conversion completed successfully!")
            
            # Test loading the converted model
            test_model = load_model(str(modern_model_path))
            logger.info("Converted model loads successfully!")
            
            # Print model summary
            print("\nModel Summary:")
            test_model.summary()
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            
            # Create a compatible model structure as fallback
            logger.info("Creating compatible model structure...")
            fallback_model = create_compatible_model_structure()
            fallback_model.save(str(modern_model_path))
            logger.info("Fallback model created. You'll need to retrain it.")
    
    else:
        logger.warning("Legacy model files not found. Creating new model structure...")
        
        # Create new model structure
        new_model = create_compatible_model_structure()
        new_model.save(str(modern_model_path))
        
        logger.info("New model structure created. Train it using the enhanced training script.")

if __name__ == "__main__":
    main()