import streamlit as st
import os
from pathlib import Path
from check_validator import HybridCheckValidator

# Set page config
st.set_page_config(page_title="Bank Check Validator", layout="wide")

@st.cache_resource
def load_validator():
    """Load models with absolute paths"""
    model_dir = Path(__file__).parent / "model"
    
    if not (model_dir / "cnn_features.h5").exists():
        st.error("Missing model files! Please ensure you have:")
        st.error("1. cnn_features.h5")
        st.error("2. xgboost_model.pkl") 
        st.error("3. scaler.pkl")
        st.error(f"In directory: {model_dir}")
        raise FileNotFoundError("Model files missing")

    return HybridCheckValidator(
        cnn_model_path=str(model_dir / "cnn_features.h5"),
        xgb_model_path=str(model_dir / "xgboost_model.pkl"), 
        scaler_path=str(model_dir / "scaler.pkl")
    )

def main():
    st.title("Bank Check Validator")
    st.write("Upload a check image to validate its authenticity")

    validator = load_validator()
    
    uploaded_file = st.file_uploader("Choose check image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            # Save temp file
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Validate
            with st.spinner("Validating check..."):
                valid, confidence, features = validator.validate_check(str(temp_path))
            
            # Show results
            st.success(f"Check is {'VALID' if valid else 'INVALID'}")
            st.metric("Confidence", f"{confidence:.2%}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(temp_path, caption="Uploaded Check")
            with col2:
                st.subheader("Detected Features")
                for k, v in features.items():
                    st.write(f"{k.replace('_', ' ').title()}: {v:.4f}")
            
        except Exception as e:
            st.error(f"Validation failed: {str(e)}")
        finally:
            if temp_path.exists():
                temp_path.unlink()

if __name__ == "__main__":
    main()