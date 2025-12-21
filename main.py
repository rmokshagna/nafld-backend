from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = FastAPI(title="NAFLD Detection API")

# ================= LOAD MODELS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nafld_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "nafld_model.h5"),
    compile=False
)

unet_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "liver_unet.keras"),
    compile=False
)

# ================= HELPERS =================

def preprocess_for_cnn(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    gray = gray / 255.0
    return gray.reshape(1, 224, 224, 1)

def segment_liver(image):
    # U-Net expects 1-channel input
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    gray = gray / 255.0
    gray = gray.reshape(1, 256, 256, 1)

    mask = unet_model.predict(gray, verbose=0)[0, :, :, 0]
    mask = (mask > 0.3).astype(np.uint8)  # relaxed threshold
    return mask

def compute_mean_intensity(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    liver_pixels = gray[mask == 1]

    if liver_pixels.size == 0:
        return None

    return float(np.mean(liver_pixels))

def nafld_stage_from_intensity(value):
    # NON-CLINICAL, RELATIVE STAGING
    if value >= 140:
        return "Mild Steatosis"
    elif 110 <= value < 140:
        return "Moderate Steatosis"
    else:
        return "Severe Steatosis"

def encode_image(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

# ================= API =================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    # ---- CNN Prediction ----
    cnn_input = preprocess_for_cnn(image)
    prob = float(nafld_model.predict(cnn_input, verbose=0)[0][0])

    # ---------- HEALTHY ----------
    if prob < 0.5:
        return {
            "Diagnosis": "Healthy Liver",
            "Probability": round(prob, 3),
            "Stage": "Normal Liver"
        }

    # ---------- NAFLD ----------
    mask = segment_liver(image)

    # Resize mask to original image size
    mask_resized = cv2.resize(
        mask, (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    mean_intensity = compute_mean_intensity(image, mask_resized)

    if mean_intensity is None:
        stage = "NAFLD Detected"
        note = "Segmentation low confidence – intensity not computed"
    else:
        stage = nafld_stage_from_intensity(mean_intensity)
        note = "Relative intensity value (non-clinical)"

    # ROI
    roi = image.copy()
    roi[mask_resized == 0] = 0

    # Heatmap
    heatmap = cv2.applyColorMap(mask_resized * 255, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return {
        "Diagnosis": "NAFLD",
        "Probability": round(prob, 3),
        "Relative_Intensity": round(mean_intensity, 2) if mean_intensity else None,
        "Stage": stage,
        "Note": note,
        "ROI_Image": encode_image(roi),
        "Heatmap_Image": encode_image(heatmap),
        "Segmentation_Mask": encode_image(mask_resized * 255)
    }
