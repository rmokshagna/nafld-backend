from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = FastAPI()

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

# ================= HELPER FUNCTIONS =================
def preprocess_for_cnn(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img.reshape(1, 224, 224, 1)

def segment_liver(image):
    img = cv2.resize(image, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    mask = unet_model.predict(img)[0, :, :, 0]
    return (mask > 0.5).astype(np.uint8)

def compute_mean_hu(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    liver_pixels = gray[mask == 1]
    if liver_pixels.size == 0:
        return None
    return float(np.mean(liver_pixels))

def nafld_stage(mean_hu):
    if mean_hu >= 55:
        return "Early NAFLD"
    elif 45 <= mean_hu < 55:
        return "Mild Steatosis"
    elif 35 <= mean_hu < 45:
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

    # ---------- CNN CLASSIFICATION ----------
    cnn_input = preprocess_for_cnn(image)
    prob = float(nafld_model.predict(cnn_input)[0][0])

    # ---------- HEALTHY ----------
    if prob < 0.5:
        return {
            "Diagnosis": "Healthy Liver",
            "Probability": round(prob, 3),
            "Stage": "Normal Liver"
        }

    # ---------- NAFLD ----------
    mask = segment_liver(image)

    # Resize mask to original size
    mask_resized = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    mean_hu = compute_mean_hu(image, mask_resized)

    # ---------- SAFE FALLBACK ----------
    if mean_hu is None:
        return {
            "Diagnosis": "NAFLD",
            "Probability": round(prob, 3),
            "Stage": "NAFLD (Segmentation Failed)",
            "Note": "Liver segmentation failed for this image"
        }

    stage = nafld_stage(mean_hu)

    # ---------- ROI ----------
    roi = image.copy()
    roi[mask_resized == 0] = 0

    # ---------- HEATMAP ----------
    heatmap = cv2.applyColorMap(mask_resized * 255, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return {
        "Diagnosis": "NAFLD",
        "Probability": round(prob, 3),
        "Mean_HU": round(mean_hu, 2),
        "Stage": stage,
        "ROI_Image": encode_image(roi),
        "Heatmap_Image": encode_image(heatmap),
        "Segmentation_Mask": encode_image(mask_resized * 255)
    }
