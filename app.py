from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import os
import base64

# LIME and SHAP
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load AI Model
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'crop_disease_model.h5')
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

CLASS_NAMES = [
    'Bacterial Blight',
    'Brown Spot',
    'Healthy',
    'Leaf Blast',
    'Powdery Mildew',
    'Rust',
    'Yellow Leaf Curl',
]

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def figure_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='PNG', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return b64


# ─────────────────────────────────────────────
# Severity Calculation using OpenCV
# ─────────────────────────────────────────────
def calculate_severity(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_disease = np.array([5, 50, 50])
    upper_disease = np.array([35, 255, 200])
    disease_mask = cv2.inRange(hsv, lower_disease, upper_disease)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    combined_mask = cv2.bitwise_or(disease_mask, dark_mask)
    total_pixels = img.shape[0] * img.shape[1]
    diseased_pixels = cv2.countNonZero(combined_mask)
    return round(min((diseased_pixels / total_pixels) * 100, 100.0), 2)


def get_severity_level(pct):
    if pct < 25:   return "Mild"
    if pct < 50:   return "Moderate"
    if pct < 75:   return "Severe"
    return "Critical"


def preprocess_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    return np.array(img) / 255.0


# ─────────────────────────────────────────────
# LIME Explanation
# ─────────────────────────────────────────────
def generate_lime_explanation(img_array, predicted_class_index):
    """
    Highlights which regions of the leaf caused the disease prediction.
    Green border = regions that increased disease confidence.
    """
    try:
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            return model.predict(images, verbose=0)

        explanation = explainer.explain_instance(
            img_array.astype('double'),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=500
        )

        temp, mask = explanation.get_image_and_mask(
            predicted_class_index,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#F0FFF4')

        axes[0].imshow(img_array)
        axes[0].set_title('Original Leaf', fontsize=12, color='#1B4332')
        axes[0].axis('off')

        axes[1].imshow(mark_boundaries(temp, mask))
        axes[1].set_title('LIME — Diseased Regions\n(outlined areas caused prediction)', fontsize=11, color='#1B4332')
        axes[1].axis('off')

        fig.suptitle('LIME: Why the AI detected this disease', fontsize=13,
                     fontweight='bold', color='#1B4332')
        plt.tight_layout()
        return figure_to_base64(fig)

    except Exception as e:
        print(f"LIME error: {e}")
        return None


# ─────────────────────────────────────────────
# SHAP Explanation
# ─────────────────────────────────────────────
def generate_shap_explanation(img_array, predicted_class_index):
    """
    Shows which pixel regions had highest impact on the AI decision.
    Bright/hot colors = high influence on result.
    """
    try:
        background = np.zeros((1, 224, 224, 3))
        input_tensor = np.expand_dims(img_array, axis=0)

        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(input_tensor)

        shap_img = shap_values[predicted_class_index][0]
        shap_sum = np.sum(np.abs(shap_img), axis=-1)
        shap_norm = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor('#F0FFF4')

        axes[0].imshow(img_array)
        axes[0].set_title('Original Leaf', fontsize=11, color='#1B4332')
        axes[0].axis('off')

        im = axes[1].imshow(shap_norm, cmap='hot')
        axes[1].set_title('SHAP Heatmap\n(Brighter = more influence)', fontsize=11, color='#1B4332')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(img_array)
        axes[2].imshow(shap_norm, cmap='hot', alpha=0.5)
        axes[2].set_title('SHAP Overlay\non Original Image', fontsize=11, color='#1B4332')
        axes[2].axis('off')

        fig.suptitle(
            f'SHAP: Pixel influence for "{CLASS_NAMES[predicted_class_index]}"',
            fontsize=12, fontweight='bold', color='#1B4332'
        )
        plt.tight_layout()
        return figure_to_base64(fig)

    except Exception as e:
        print(f"SHAP error: {e}")
        return None


# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Form data: image (file)
    Query param: ?explain=true (default) or ?explain=false (faster, no LIME/SHAP)

    Returns JSON:
    {
        disease, confidence, severity_percentage, severity_level,
        all_predictions, lime_image (base64), shap_image (base64)
    }
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    include_explain = request.args.get('explain', 'true').lower() == 'true'

    try:
        image_bytes = request.files['image'].read()

        # Preprocess
        img_array = preprocess_image(image_bytes)
        input_batch = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(input_batch, verbose=0)
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index]) * 100
        disease_name = CLASS_NAMES[predicted_class_index]

        # Severity
        severity_pct = calculate_severity(image_bytes)
        severity_level = get_severity_level(severity_pct)

        # LIME + SHAP
        lime_b64 = None
        shap_b64 = None
        if include_explain:
            print("⏳ Generating LIME explanation...")
            lime_b64 = generate_lime_explanation(img_array, predicted_class_index)
            print("⏳ Generating SHAP explanation...")
            shap_b64 = generate_shap_explanation(img_array, predicted_class_index)

        # All class probabilities
        all_preds = {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            'disease': disease_name,
            'confidence': round(confidence, 1),
            'severity_percentage': severity_pct,
            'severity_level': severity_level,
            'all_predictions': all_preds,
            'lime_image': lime_b64,
            'shap_image': shap_b64,
        }), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
