import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import pytesseract
import argparse

# Add this right after imports
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path
    # OR for Linux/Mac:
    # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    
    # Test Tesseract
    pytesseract.get_tesseract_version()
except Exception as e:
    raise RuntimeError(
        "Tesseract not found. Install it first:\n"
        "Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "Mac: brew install tesseract\n"
        "Linux: sudo apt install tesseract-ocr"
    )

def create_feature_extractor():
    """Create CNN feature extractor"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=base_model.input, outputs=x)

def extract_tabular_features(image):
    """Extract traditional computer vision features"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ocr_text = pytesseract.image_to_string(gray, config=r'--oem 3 --psm 6')
    
    edges = cv2.Canny(gray, 50, 150)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'micr_present': 1.0 if any(c.isdigit() for c in ocr_text) else 0.0,
        'micr_length': len(ocr_text),
        'edge_density': np.mean(edges)/255,
        'blur_metric': blur,
        'aspect_ratio': image.shape[1]/image.shape[0],
        'brightness': np.mean(gray)/255
    }

def extract_features(image_paths, labels):
    """Extract features from dataset with error handling"""
    cnn_model = create_feature_extractor()
    cnn_features = []
    tabular_features = []
    valid_labels = []
    
    for path, label in zip(image_paths, labels):
        try:
            img = cv2.imread(str(path))
            if img is None:
                print(f"Warning: Could not read {path}, skipping")
                continue
                
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # CNN features
            cnn_feat = cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)
            cnn_features.append(cnn_feat.flatten())
            
            # Tabular features
            tabular_feat = extract_tabular_features(img)
            tabular_features.append(list(tabular_feat.values()))
            
            valid_labels.append(label)
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
            
    if not cnn_features:
        raise ValueError("No images were successfully processed")
        
    return np.array(cnn_features), np.array(tabular_features), np.array(valid_labels)

def train_hybrid_model(data_dir, output_dir, epochs=20, batch_size=32):
    """Complete training workflow with error handling"""
    try:
        # Get image paths
        valid_paths = list(Path(data_dir).glob('valid/*.jpg'))
        invalid_paths = list(Path(data_dir).glob('invalid/*.jpg'))
        
        if not valid_paths and not invalid_paths:
            raise ValueError(f"No images found in {data_dir}")
            
        image_paths = valid_paths + invalid_paths
        labels = [1]*len(valid_paths) + [0]*len(invalid_paths)
        
        print(f"\nFound {len(image_paths)} images ({len(valid_paths)} valid, {len(invalid_paths)} invalid)")

        # Extract features
        print("\nExtracting features...")
        cnn_feats, tabular_feats, y = extract_features(image_paths, labels)
        
        # Debug shapes
        print(f"\nFeature Shapes:")
        print(f"CNN Features: {cnn_feats.shape}")
        print(f"Tabular Features: {tabular_feats.shape}")
        print(f"Labels: {y.shape}")

        # Combine features
        X = np.hstack([cnn_feats, tabular_feats])
        print(f"\nCombined Features Shape: {X.shape}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train XGBoost
        print("\nTraining XGBoost model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = xgb_model.score(X_train, y_train)
        test_acc = xgb_model.score(X_test, y_test)
        print(f"\nTraining Complete:")
        print(f"Train Accuracy: {train_acc:.2%}")
        print(f"Test Accuracy: {test_acc:.2%}")

        # Save models
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cnn_model = create_feature_extractor()
        cnn_model.save(f'{output_dir}/cnn_features.h5')
        joblib.dump(xgb_model, f'{output_dir}/xgboost_model.pkl')
        joblib.dump(scaler, f'{output_dir}/scaler.pkl')
        
        print(f"\nModels saved to {output_dir}/")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/checks', help='Path to training data')
    parser.add_argument('--output_dir', default='model', help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    args = parser.parse_args()

    # Run training
    print("Starting Hybrid Check Validator Training")
    print("=======================================")
    
    train_hybrid_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )