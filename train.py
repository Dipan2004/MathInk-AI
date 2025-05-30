# Enhanced Training Script for Handwritten Math Equation Recognition
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Activation, MaxPool2D, GlobalAveragePooling2D,
    Flatten, Dense, Dropout, BatchNormalization, Add, 
    DepthwiseConv2D, SeparableConv2D
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    LearningRateScheduler, ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.mixed_precision import set_global_policy

import albumentations as A
import seaborn as sns

# Set mixed precision for better performance on modern GPUs
set_global_policy('mixed_float16')

# Configure for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class EnhancedDataLoader:
    def __init__(self, data_dir, target_size=(32, 32), augment=True):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augment = augment
        
        # Enhanced augmentation pipeline
        self.transform = A.Compose([
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        ])
    
    def preprocess_image(self, img):
        """Enhanced preprocessing with better noise reduction"""
        # Resize to target size
        img = cv2.resize(img, self.target_size)
        
        # Apply bilateral filter for noise reduction while preserving edges
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Adaptive thresholding for better binarization
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = np.ones((2,2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_data(self):
        """Load and preprocess data with enhanced augmentation"""
        images_data = []
        images_label = []
        
        print("Loading data from:", self.data_dir)
        
        for folder in os.listdir(self.data_dir):
            folder_path = self.data_dir / folder
            if not folder_path.is_dir():
                continue
                
            print(f"Processing folder: {folder}")
            for img_file in os.listdir(folder_path):
                try:
                    img = cv2.imread(str(folder_path / img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Preprocess image
                    img = self.preprocess_image(img)
                    
                    images_data.append(img)
                    images_label.append(folder)
                    
                    # Add augmented versions if augmentation is enabled
                    if self.augment and len(images_data) % 100 == 0:  # Add augmented data periodically
                        for _ in range(2):  # Add 2 augmented versions
                            img_uint8 = (img * 255).astype(np.uint8)
                            augmented = self.transform(image=img_uint8)['image']
                            augmented = augmented.astype(np.float32) / 255.0
                            images_data.append(augmented)
                            images_label.append(folder)
                            
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue
        
        # Shuffle data
        combined = list(zip(images_data, images_label))
        random.shuffle(combined)
        images_data, images_label = zip(*combined)
        
        return np.array(images_data), np.array(images_label)

class EnhancedModel:
    def __init__(self, input_shape=(32, 32, 1), num_classes=15):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def create_mobilenet_based_model(self):
        """Create a MobileNet-based model for better efficiency"""
        base_model = MobileNetV2(
            input_shape=(*self.input_shape[:2], 3),  # MobileNet expects 3 channels
            include_top=False,
            weights=None  # No pretrained weights for our custom task
        )
        
        # Input layer for grayscale images
        inputs = Input(shape=self.input_shape)
        
        # Convert grayscale to RGB by repeating the channel
        x = tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(inputs)
        
        # Pass through MobileNet
        x = base_model(x)
        
        # Global average pooling
        x = GlobalAveragePooling2D()(x)
        
        # Classification head
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        model = Model(inputs, outputs)
        return model
    
    def create_enhanced_cnn(self):
        """Create an enhanced CNN with residual connections"""
        inputs = Input(shape=self.input_shape)
        
        # First block
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Second block with residual connection
        residual = x
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Adjust residual connection dimensions
        residual = Conv2D(64, (1, 1), padding='same')(residual)
        x = Add()([x, residual])
        
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Third block
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = SeparableConv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Global average pooling instead of flatten
        x = GlobalAveragePooling2D()(x)
        
        # Classification head
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        model = Model(inputs, outputs)
        return model

class EnhancedTrainer:
    def __init__(self, model, train_data, val_data, model_name="enhanced_math_model"):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.model_name = model_name
        
        # Create directories
        self.model_dir = Path("models")
        self.log_dir = Path("logs")
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
    
    def get_callbacks(self):
        """Enhanced callbacks for better training"""
        callbacks = [
            # Model checkpoint
            ModelCheckpoint(
                filepath=str(self.model_dir / f"{self.model_name}_best.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                verbose=1,
                restore_best_weights=True,
                mode='max'
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
    
    def cosine_annealing_schedule(self, epoch, epochs=100):
        """Cosine annealing learning rate schedule"""
        initial_lr = 0.001
        min_lr = 1e-6
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / epochs)) / 2
    
    def train(self, epochs=100, batch_size=64):
        """Enhanced training with better optimization"""
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        # Compile model with AdamW optimizer
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Enhanced data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Don't flip math symbols
            fill_mode='nearest'
        )
        
        # Add cosine annealing to callbacks
        callbacks = self.get_callbacks()
        callbacks.append(
            LearningRateScheduler(lambda epoch: self.cosine_annealing_schedule(epoch, epochs))
        )
        
        # Train the model
        history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model_for_deployment(self):
        """Save model in format compatible with the Flask app"""
        # Save in Keras format
        self.model.save(str(self.model_dir / f"{self.model_name}.keras"))
        
        # Save in legacy format for compatibility
        model_json = self.model.to_json()
        with open(str(self.model_dir / "model.json"), "w") as json_file:
            json_file.write(model_json)
        
        # Save weights
        self.model.save_weights(str(self.model_dir / "model_weights.h5"))
        
        print(f"Model saved in multiple formats in {self.model_dir}")

def main():
    # Configuration
    DATA_DIR = "data/extracted_images"
    INPUT_SHAPE = (32, 32, 1)  # Increased from 28x28 for better recognition
    BATCH_SIZE = 64
    EPOCHS = 100
    
    print("Starting enhanced training pipeline...")
    
    # Load and preprocess data
    print("Loading data...")
    data_loader = EnhancedDataLoader(DATA_DIR, target_size=(32, 32), augment=True)
    X, y = data_loader.load_data()
    
    print(f"Loaded {len(X)} images")
    print(f"Unique classes: {np.unique(y)}")
    
    # Prepare data
    X = np.expand_dims(X, axis=-1) if len(X.shape) == 3 else X
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Save label encoder for later use
    np.save("models/label_encoder_classes.npy", label_encoder.classes_)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create model
    print("Creating enhanced model...")
    model_creator = EnhancedModel(INPUT_SHAPE, len(label_encoder.classes_))
    
    # Try MobileNet-based model first, fallback to enhanced CNN
    try:
        model = model_creator.create_mobilenet_based_model()
        print("Using MobileNet-based model")
    except Exception as e:
        print(f"MobileNet model failed: {e}")
        print("Using enhanced CNN model")
        model = model_creator.create_enhanced_cnn()
    
    print(f"Model summary:")
    model.summary()
    
    # Train model
    print("Starting training...")
    trainer = EnhancedTrainer(
        model, 
        (X_train, y_train), 
        (X_test, y_test),
        "enhanced_math_equation_model"
    )
    
    history = trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Save model
    trainer.save_model_for_deployment()
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc, test_top3_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test top-3 accuracy: {test_top3_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()