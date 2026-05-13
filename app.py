"""
Plant Disease Classifier - Flask Backend
Connects directly to the trained EfficientNetB0 model from the notebook.

Usage:
    pip install flask flask-cors tensorflow pillow numpy opencv-python
    python app.py

The model file 'plant_disease_efficientnet_final.keras' must be in the same directory.
"""

import os
import io
import cv2
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import json

app = Flask(__name__)
CORS(app)  # Allow requests from the frontend HTML file

# ─────────────────────────────────────────────
# CONFIG — mirrors CFG class from notebook
# ─────────────────────────────────────────────
IMG_SIZE     = 224
MODEL_PATH   = "plant_disease_efficientnet_final.keras"

# 38 classes — same order as train_gen.class_indices from notebook
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ─────────────────────────────────────────────
# LOAD MODEL (once at startup)
# ─────────────────────────────────────────────
print("Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. "
        "Download it from Colab: model.save('plant_disease_efficientnet_final.keras')"
    )
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded. Input shape: {model.input_shape}")


# ─────────────────────────────────────────────
# GRAD-CAM — matches notebook's make_gradcam_heatmap()
# ─────────────────────────────────────────────
def make_gradcam_heatmap(img_array):
    """
    Smooth Grad-CAM using EfficientNetB0's top_conv layer.
    Matches the get_smooth_gradcam() function from the notebook exactly.
    """
    base_model = model.layers[0]  # EfficientNetB0 base

    try:
        last_conv_layer = base_model.get_layer("top_conv")
    except ValueError:
        return None  # Gracefully skip if layer name differs

    grad_model = tf.keras.models.Model(
        [base_model.inputs],
        [last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_output, base_output = grad_model(img_array)

        # Run the top layers (GAP → Dense → BN → Dropout → Softmax)
        x = base_output
        for layer in model.layers[1:]:
            x = layer(x)
        preds = x

        pred_idx = tf.argmax(preds[0])
        loss = preds[:, pred_idx]

    grads = tape.gradient(loss, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.squeeze(last_conv_output[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    # Bilinear upscale + Gaussian smooth — same as notebook
    heatmap_np = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    heatmap_smooth = cv2.GaussianBlur(heatmap_np, (11, 11), 0)
    return heatmap_smooth


def overlay_gradcam(original_rgb, heatmap):
    """Overlay jet colormap heatmap on original image at 40% alpha."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * original_rgb + 0.4 * jet_rgb).astype(np.uint8)
    return overlay


def ndarray_to_base64_png(arr):
    """Convert a numpy RGB array to a base64-encoded PNG string."""
    pil_img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────
# PREPROCESS — mirrors val_test_datagen (rescale=1./255)
# ─────────────────────────────────────────────
def preprocess_image(file_bytes):
    """Resize to 224×224 and normalise to [0, 1]."""
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0   # rescale=1./255
    return img_array, np.array(pil_img)   # (normalised, original uint8)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: multipart/form-data with field 'image' (any common image format)

    Returns JSON:
    {
      "prediction":  "Tomato___Early_blight",
      "plant":       "Tomato",
      "disease":     "Early blight",
      "confidence":  0.9741,
      "is_healthy":  false,
      "top4": [
        {"class": "...", "plant": "...", "disease": "...", "confidence": 0.97},
        ...
      ],
      "gradcam_original": "<base64 PNG>",
      "gradcam_overlay":  "<base64 PNG>"
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file in request. Send field name 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        file_bytes = file.read()
        img_norm, img_orig = preprocess_image(file_bytes)

        # Batch dimension → (1, 224, 224, 3)
        img_tensor = np.expand_dims(img_norm, axis=0)

        # ── PREDICTION ──
        preds = model.predict(img_tensor, verbose=0)[0]   # shape (38,)
        top4_idx = np.argsort(preds)[-4:][::-1]

        pred_idx  = int(top4_idx[0])
        pred_cls  = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])

        parts   = pred_cls.split("___")
        plant   = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ") if len(parts) > 1 else ""

        top4 = []
        for i in top4_idx:
            p = CLASS_NAMES[i].split("___")
            top4.append({
                "class":      CLASS_NAMES[i],
                "plant":      p[0].replace("_", " "),
                "disease":    p[1].replace("_", " ") if len(p) > 1 else "",
                "confidence": float(preds[i]),
            })

        # ── GRAD-CAM ──
        gradcam_original_b64 = ndarray_to_base64_png(img_orig)
        gradcam_overlay_b64  = None

        heatmap = make_gradcam_heatmap(img_tensor)
        if heatmap is not None:
            overlay = overlay_gradcam(img_orig, heatmap)
            gradcam_overlay_b64 = ndarray_to_base64_png(overlay)

        return jsonify({
            "prediction":         pred_cls,
            "plant":              plant,
            "disease":            disease,
            "confidence":         round(confidence, 4),
            "is_healthy":         "healthy" in pred_cls.lower(),
            "top4":               top4,
            "gradcam_original":   gradcam_original_b64,
            "gradcam_overlay":    gradcam_overlay_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH, "classes": len(CLASS_NAMES)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
