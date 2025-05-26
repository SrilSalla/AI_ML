import cv2
import numpy as np
import pytesseract
import xgboost as xgb
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Tuple
import os

import pytesseract

# Windows (adjust path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Mac/Linux (usually auto-detected)
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

class HybridCheckValidator:
    def __init__(self, cnn_model_path: str = None, xgb_model_path: str = None, scaler_path: str = None):
        """
        Initialize the hybrid validator with CNN and XGBoost models
        
        Args:
            cnn_model_path: Path to CNN feature extractor (.h5 file)
            xgb_model_path: Path to XGBoost classifier (.pkl file)
            scaler_path: Path to feature scaler (.pkl file)
        """
        # Initialize models
        self.cnn_feature_extractor = self._load_cnn_model(cnn_model_path)
        self.xgb_classifier = self._load_xgb_model(xgb_model_path)
        self.scaler = self._load_scaler(scaler_path)
        
        # OCR configuration
        self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:'
        
        # Image parameters
        self.target_size = (224, 224)
    
    def validate_check(self, image_path: str) -> Tuple[bool, float, Dict]:
        """
        Validate a bank check image
        
        Args:
            image_path: Path to the check image file
            
        Returns:
            tuple: (is_valid, confidence, features)
        """
        try:
            # Step 1: Preprocess image
            processed_img = self._preprocess_image(image_path)
            
            # Step 2: Extract features
            cnn_features = self._extract_cnn_features(processed_img)
            tabular_features = self._extract_tabular_features(processed_img)
            
            # Step 3: Combine and scale features
            combined_features = np.concatenate([
                cnn_features,
                np.array(list(tabular_features.values()))
            ])
            
            scaled_features = self.scaler.transform(combined_features.reshape(1, -1))
            
            # Step 4: Predict
            confidence = self.xgb_classifier.predict_proba(scaled_features)[0][1]
            is_valid = confidence >= 0.5
            
            return is_valid, confidence, tabular_features
            
        except Exception as e:
            raise RuntimeError(f"Check validation failed: {str(e)}")
    
    def _load_cnn_model(self, model_path: str) -> Model:
        """Load or create CNN feature extractor"""
        if model_path and os.path.exists(model_path):
            return load_model(model_path)
        return self._create_feature_extractor()
    
    def _load_xgb_model(self, model_path: str) -> xgb.XGBClassifier:
        """Load XGBoost classifier"""
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
        raise ValueError("XGBoost model path is required")
    
    def _load_scaler(self, scaler_path: str) -> StandardScaler:
        """Load feature scaler"""
        if scaler_path and os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        raise ValueError("Scaler path is required")
    
    def _create_feature_extractor(self) -> Model:
        """Create a new CNN feature extractor"""
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze convolutional layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        
        return Model(inputs=base_model.input, outputs=x)
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess check image for feature extraction"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        # Resize and convert to RGB
        img = cv2.resize(img, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _extract_cnn_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep features using CNN"""
        img_array = np.expand_dims(image, axis=0)
        img_array = preprocess_input(img_array)
        features = self.cnn_feature_extractor.predict(img_array)
        return features.flatten()
    
    def _extract_tabular_features(self, image: np.ndarray) -> Dict:
        """Extract traditional computer vision features"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # OCR Features
        ocr_text = pytesseract.image_to_string(gray, config=self.ocr_config).strip()
        
        # Edge Features
        edges = cv2.Canny(gray, 50, 150)
        
        # Texture Features
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'micr_present': 1.0 if any(c.isdigit() for c in ocr_text) else 0.0,
            'micr_length': len(ocr_text),
            'edge_density': np.mean(edges)/255,
            'blur_metric': blur,
            'aspect_ratio': image.shape[1]/image.shape[0],
            'brightness': np.mean(gray)/255
        }

# Example usage
if __name__ == "__main__":
    # Initialize validator (replace with actual model paths)
    validator = HybridCheckValidator(
        cnn_model_path="model/cnn_features.h5",
        xgb_model_path="model/xgboost_model.pkl",
        scaler_path="model/scaler.pkl"
    )
    
    # Validate a sample check
    try:
        is_valid, confidence, features = validator.validate_check("sample_check.jpg")
        print(f"Check is {'VALID' if is_valid else 'INVALID'}")
        print(f"Confidence: {confidence:.2%}")
        print("Detected features:")
        for feature, value in features.items():
            print(f"- {feature}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")