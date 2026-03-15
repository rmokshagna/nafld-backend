from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# LOAD TFLITE MODELS
# ===============================

cnn_interpreter = tf.lite.Interpreter(
    model_path=os.path.join(BASE_DIR, "nafld_model_quant.tflite")
)
cnn_interpreter.allocate_tensors()
cnn_input = cnn_interpreter.get_input_details()
cnn_output = cnn_interpreter.get_output_details()

unet_interpreter = tf.lite.Interpreter(
    model_path=os.path.join(BASE_DIR, "unet_fat_segmentation_quant.tflite")
)
unet_interpreter.allocate_tensors()
unet_input = unet_interpreter.get_input_details()
unet_output = unet_interpreter.get_output_details()

# ===============================
# IMAGE PREPROCESSING
# ===============================

def preprocess_cnn(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 224, 224, 1).astype(np.float32)
    return img

def preprocess_unet(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (256, 256))
    img = img / 255.0
    img = img.reshape(1, 256, 256, 1).astype(np.float32)
    return img

# ===============================
# BASE64 ENCODER
# ===============================

def encode(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

# ===============================
# HU / MEAN DENSITY CALCULATION
# ===============================

def calculate_mean_hu(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    liver_pixels = gray[mask == 1]

    if liver_pixels.size == 0:
        return 0.0

    mean_hu = np.mean(liver_pixels)
    return float(mean_hu)

# ===============================
# STAGE DETERMINATION
# ===============================

def determine_stage(mean_hu):
    if mean_hu > 75:
        return "Normal Liver"
    elif 65 <= mean_hu <= 75:
        return "Mild NAFLD"
    elif 55 <= mean_hu < 65:
        return "Moderate NAFLD"
    else:
        return "Severe NAFLD"

# ===============================
# EXPLAINABLE AI TEXT
# ===============================

def get_explainable_text(diagnosis, stage, mean_hu):
    if diagnosis == "Healthy Liver":
        return (
            "The AI model analyzed the CT image and did not detect significant "
            "fat accumulation in the liver. The liver region appears within the "
            "normal density range."
        )

    return (
        f"The AI model detected features suggestive of fatty liver disease. "
        f"The liver region was segmented first, and analysis was restricted to "
        f"the segmented liver area. The mean liver density value is {round(mean_hu, 2)}, "
        f"which corresponds to {stage}. In the heatmap, red regions indicate "
        f"lower-density suspicious fatty areas, yellow indicates intermediate regions, "
        f"and green indicates relatively normal liver tissue."
    )

# ===============================
# SYMPTOMS TEXT
# ===============================

def get_symptoms_text(stage):
    if stage == "Normal Liver":
        return [
            "No major symptoms expected",
            "Early fatty liver may still be asymptomatic",
            "Routine screening is recommended if risk factors are present"
        ]
    elif stage == "Mild NAFLD":
        return [
            "Often no symptoms in early stage",
            "Mild fatigue",
            "General weakness",
            "Mild abdominal discomfort"
        ]
    elif stage == "Moderate NAFLD":
        return [
            "Fatigue",
            "Upper right abdominal discomfort",
            "Weakness",
            "Reduced physical activity tolerance"
        ]
    else:
        return [
            "Persistent fatigue",
            "Abdominal discomfort",
            "Weakness",
            "Possible metabolic risk association",
            "Medical consultation is strongly recommended"
        ]

# ===============================
# REMEDIES TEXT
# ===============================

def get_remedies_text(stage):
    if stage == "Normal Liver":
        return [
            "Maintain a balanced diet",
            "Exercise regularly",
            "Continue periodic health checkups"
        ]
    elif stage == "Mild NAFLD":
        return [
            "Reduce oily and sugary foods",
            "Exercise regularly",
            "Maintain healthy body weight",
            "Avoid alcohol"
        ]
    elif stage == "Moderate NAFLD":
        return [
            "Follow a low-fat and low-sugar diet",
            "Increase physical activity",
            "Weight reduction is recommended",
            "Avoid alcohol and junk food",
            "Consult a doctor for further evaluation"
        ]
    else:
        return [
            "Immediate medical consultation is recommended",
            "Strict dietary modification is needed",
            "Avoid alcohol completely",
            "Weight management is essential",
            "Regular liver monitoring is recommended"
        ]

# ===============================
# GENERATE LIVER ROI
# ===============================

def generate_roi(image, mask):
    roi = image.copy()
    roi[mask == 0] = 0
    return roi

# ===============================
# GENERATE FAT-AWARE HEATMAP
# ===============================

def generate_fatty_liver_heatmap(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    liver_pixels = gray[mask == 1]

    if liver_pixels.size == 0:
        return np.zeros_like(image)

    min_val = np.min(liver_pixels)
    max_val = np.max(liver_pixels)

    if max_val - min_val == 0:
        norm_gray = gray.copy()
    else:
        norm_gray = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    heatmap = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Only inside liver mask
    fatty_region = (mask == 1) & (norm_gray < 85)
    moderate_region = (mask == 1) & (norm_gray >= 85) & (norm_gray < 170)
    normal_region = (mask == 1) & (norm_gray >= 170)

    # BGR colors for OpenCV
    heatmap[fatty_region] = [0, 0, 255]       # Red
    heatmap[moderate_region] = [0, 255, 255]  # Yellow
    heatmap[normal_region] = [0, 255, 0]      # Green

    overlay = cv2.addWeighted(image, 0.65, heatmap, 0.35, 0)
    return overlay

# ===============================
# PREDICTION API
# ===============================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

        # ===============================
        # CNN PREDICTION
        # ===============================
        cnn_img = preprocess_cnn(image)
        cnn_interpreter.set_tensor(cnn_input[0]["index"], cnn_img)
        cnn_interpreter.invoke()
        prob = float(cnn_interpreter.get_tensor(cnn_output[0]["index"])[0][0])

        # ===============================
        # HEALTHY CASE
        # ===============================
        if prob < 0.5:
            stage = "Normal Liver"

            return {
                "Diagnosis": "Healthy Liver",
                "Probability": round(prob, 6),
                "Mean_HU": None,
                "Stage": stage,
                "Segmentation_Mask": None,
                "ROI_Image": None,
                "Heatmap_Image": None,
                "Explainable_AI": get_explainable_text("Healthy Liver", stage, 0.0),
                "Symptoms": get_symptoms_text(stage),
                "Remedies": get_remedies_text(stage),
                "Status": "Success"
            }

        # ===============================
        # U-NET SEGMENTATION
        # ===============================
        unet_img = preprocess_unet(image)
        unet_interpreter.set_tensor(unet_input[0]["index"], unet_img)
        unet_interpreter.invoke()

        mask = unet_interpreter.get_tensor(unet_output[0]["index"])[0, :, :, 0]
        mask = (mask > 0.5).astype(np.uint8)

        mask_resized = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # ===============================
        # ROI
        # ===============================
        roi = generate_roi(image, mask_resized)

        # ===============================
        # HEATMAP
        # ===============================
        overlay = generate_fatty_liver_heatmap(image, mask_resized)

        # ===============================
        # MEAN HU + STAGE
        # ===============================
        mean_hu = calculate_mean_hu(image, mask_resized)
        stage = determine_stage(mean_hu)

        # ===============================
        # API RESPONSE
        # ===============================
        return {
            "Diagnosis": "NAFLD",
            "Probability": round(prob, 6),
            "Mean_HU": round(mean_hu, 2),
            "Stage": stage,
            "Segmentation_Mask": encode(mask_resized * 255),
            "ROI_Image": encode(roi),
            "Heatmap_Image": encode(overlay),
            "Explainable_AI": get_explainable_text("NAFLD", stage, mean_hu),
            "Symptoms": get_symptoms_text(stage),
            "Remedies": get_remedies_text(stage),
            "Status": "Success"
        }

    except Exception as e:
        return {
            "Diagnosis": "Error",
            "Probability": 0.0,
            "Mean_HU": None,
            "Stage": "Unknown",
            "Segmentation_Mask": None,
            "ROI_Image": None,
            "Heatmap_Image": None,
            "Explainable_AI": "Prediction failed.",
            "Symptoms": [],
            "Remedies": [],
            "Status": f"Failed: {str(e)}"
        }