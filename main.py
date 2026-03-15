from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io, base64, os

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
# PREPROCESSING
# ===============================
def preprocess_cnn(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (224, 224)) / 255.0
    img = img.reshape(1, 224, 224, 1).astype(np.float32)
    return img

def preprocess_unet(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (256, 256)) / 255.0
    img = img.reshape(1, 256, 256, 1).astype(np.float32)
    return img

# ===============================
# ENCODER
# ===============================
def encode(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()

# ===============================
# MASK POST-PROCESSING
# ===============================
def clean_liver_mask(mask, original_shape):
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)

    return (clean_mask > 0).astype(np.uint8)

# ===============================
# ROI WITH LIVER BOUNDARY
# ===============================
def create_roi_with_boundary(image, liver_mask):
    roi = image.copy()

    # darken outside liver slightly
    dark = (roi * 0.2).astype(np.uint8)
    roi[liver_mask == 0] = dark[liver_mask == 0]

    contours, _ = cv2.findContours(
        (liver_mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)   # green boundary
    return roi

# ===============================
# FAT MAP INSIDE LIVER ONLY
# ===============================
def create_fat_heatmap(image, liver_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    overlay = image.copy()

    # Liver pixels only
    liver_pixels = gray[liver_mask == 1]

    if liver_pixels.size == 0:
        return overlay

    # Example threshold rule for demo visualization
    # lower intensity in liver can indicate fat-like darker areas in plain CT
    threshold = np.percentile(liver_pixels, 40)

    fatty_region = np.zeros_like(gray, dtype=np.uint8)
    fatty_region[(liver_mask == 1) & (gray <= threshold)] = 1

    normal_region = np.zeros_like(gray, dtype=np.uint8)
    normal_region[(liver_mask == 1) & (gray > threshold)] = 1

    # Apply colors only inside liver
    # Blue for normal liver
    overlay[normal_region == 1] = (
        0.6 * overlay[normal_region == 1] + 0.4 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # Yellow for fatty areas
    overlay[fatty_region == 1] = (
        0.6 * overlay[fatty_region == 1] + 0.4 * np.array([0, 255, 255])
    ).astype(np.uint8)

    # draw liver boundary
    contours, _ = cv2.findContours(
        (liver_mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay

# ===============================
# HU CALCULATION
# ===============================
def calculate_mean_hu(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    liver_pixels = gray[mask == 1]

    if liver_pixels.size == 0:
        return 0.0

    return float(np.mean(liver_pixels))

# ===============================
# STAGE
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
# EXPLAINABLE TEXT
# ===============================
def generate_explainable_text(stage, diagnosis):
    if diagnosis == "Healthy Liver":
        explain = "The model did not detect significant fatty infiltration in the liver region."
        symptoms = ["Usually no major liver-related symptoms", "General health monitoring is advised"]
        remedies = ["Maintain balanced diet", "Exercise regularly", "Avoid alcohol and junk food"]
    else:
        explain = f"The model detected fatty liver changes in the segmented liver region. The highlighted yellow regions indicate suspected fatty infiltration, while blue regions indicate comparatively normal liver tissue. Stage detected: {stage}."
        symptoms = [
            "Fatigue",
            "Upper abdominal discomfort",
            "Mild liver enlargement",
            "Often asymptomatic in early stages"
        ]
        remedies = [
            "Reduce oily and sugary foods",
            "Exercise daily",
            "Weight management",
            "Regular liver checkup",
            "Consult doctor for clinical confirmation"
        ]

    return explain, symptoms, remedies

# ===============================
# API
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # CNN classification
    cnn_img = preprocess_cnn(image)
    cnn_interpreter.set_tensor(cnn_input[0]['index'], cnn_img)
    cnn_interpreter.invoke()
    prob = float(cnn_interpreter.get_tensor(cnn_output[0]['index'])[0][0])

    # Healthy case
    if prob < 0.5:
        explain, symptoms, remedies = generate_explainable_text("Normal Liver", "Healthy Liver")
        return {
            "Diagnosis": "Healthy Liver",
            "Probability": round(prob, 6),
            "Mean_HU": None,
            "Stage": "Normal Liver",
            "ROI_Image": None,
            "Heatmap_Image": None,
            "Segmentation_Mask": None,
            "Explainable_AI": explain,
            "Symptoms": symptoms,
            "Remedies": remedies,
            "Status": "Success"
        }

    # U-Net segmentation
    unet_img = preprocess_unet(image)
    unet_interpreter.set_tensor(unet_input[0]['index'], unet_img)
    unet_interpreter.invoke()
    raw_mask = unet_interpreter.get_tensor(unet_output[0]['index'])[0, :, :, 0]

    liver_mask = clean_liver_mask(raw_mask, image.shape)

    # outputs
    roi = create_roi_with_boundary(image, liver_mask)
    heatmap = create_fat_heatmap(image, liver_mask)
    segmentation_mask = (liver_mask * 255).astype(np.uint8)

    mean_hu = calculate_mean_hu(image, liver_mask)
    stage = determine_stage(mean_hu)

    explain, symptoms, remedies = generate_explainable_text(stage, "NAFLD")

    return {
        "Diagnosis": "NAFLD",
        "Probability": round(prob, 6),
        "Mean_HU": round(mean_hu, 2),
        "Stage": stage,
        "ROI_Image": encode(roi),
        "Heatmap_Image": encode(heatmap),
        "Segmentation_Mask": encode(segmentation_mask),
        "Explainable_AI": explain,
        "Symptoms": symptoms,
        "Remedies": remedies,
        "Status": "Success"
    }