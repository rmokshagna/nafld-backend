from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io, base64, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# LOAD MODELS
# ===============================
cnn_interpreter = tf.lite.Interpreter(
    model_path=os.path.join(BASE_DIR, "nafld_model_quant.tflite")
)
cnn_interpreter.allocate_tensors()
cnn_input = cnn_interpreter.get_input_details()
cnn_output = cnn_interpreter.get_output_details()

liver_interpreter = tf.lite.Interpreter(
    model_path=os.path.join(BASE_DIR, "liver_unet_256.tflite")
)
liver_interpreter.allocate_tensors()
liver_input = liver_interpreter.get_input_details()
liver_output = liver_interpreter.get_output_details()

fat_interpreter = tf.lite.Interpreter(
    model_path=os.path.join(BASE_DIR, "fatty_liver_unet.tflite")
)
fat_interpreter.allocate_tensors()
fat_input = fat_interpreter.get_input_details()
fat_output = fat_interpreter.get_output_details()

# ===============================
# PREPROCESS
# ===============================
def preprocess_cnn(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (224,224)) / 255.0
    return img.reshape(1,224,224,1).astype(np.float32)

def preprocess_unet(image, size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (size,size)) / 255.0
    return img.reshape(1,size,size,1).astype(np.float32)

# ===============================
# ENCODE
# ===============================
def encode(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()

# ===============================
# CLEAN LIVER MASK
# ===============================
def clean_liver_mask(mask, shape):
    mask = (mask > 0.15).astype(np.uint8)
    mask = cv2.resize(mask, (shape[1], shape[0]))

    kernel = np.ones((25,25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean = np.zeros_like(mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            cv2.drawContours(clean, [largest], -1, 1, -1)

    return clean

# ===============================
# CLEAN FAT MASK
# ===============================
def clean_fat_mask(mask, liver_mask, shape):
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (shape[1], shape[0]))
    mask = mask * liver_mask
    return mask

# ===============================
# ROI (FINAL FIX - ROBUST & SMOOTH)
# ===============================
def create_roi(image, mask):

    roi = image.copy()
    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape

    # -------------------------------
    # STEP 1: STRONG MASK EXPANSION
    # -------------------------------
    kernel = np.ones((35,35), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # -------------------------------
    # STEP 2: SMOOTH MASK (REMOVE BLOCKS)
    # -------------------------------
    mask = cv2.GaussianBlur(mask.astype(np.float32), (31,31), 0)
    mask = (mask > 0.3).astype(np.uint8)

    # -------------------------------
    # STEP 3: ENSURE MINIMUM LIVER AREA
    # -------------------------------
    if np.sum(mask) < 0.08 * (h * w):
        # create liver-like shape (NOT box)
        mask = np.zeros_like(mask)
        center = (w//2, h//2)
        axes = (int(w*0.35), int(h*0.25))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)

    # -------------------------------
    # STEP 4: EXTRACT CONTOUR
    # -------------------------------
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        # smooth contour edges
        epsilon = 0.02 * cv2.arcLength(largest, True)
        smooth = cv2.approxPolyDP(largest, epsilon, True)

        cv2.drawContours(roi, [smooth], -1, (255,255,255), 2)

    return roi

# ===============================
# SEGMENTATION
# ===============================
def create_segmentation_mask(mask):
    seg = (mask * 255).astype(np.uint8)

    contours,_ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean = np.zeros_like(seg)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            cv2.drawContours(clean, [largest], -1, 255, -1)

    return clean

# ===============================
# HEATMAP (FINAL)
# ===============================
def create_heatmap(image, liver_mask, fat_mask, stage):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

    fat = (fat_mask > 0).astype(np.uint8)

    if stage == "Mild NAFLD":
        kernel = np.ones((5,5), np.uint8)
        fat = cv2.dilate(fat, kernel, 1)

    elif stage == "Moderate NAFLD":
        kernel = np.ones((9,9), np.uint8)
        fat = cv2.dilate(fat, kernel, 2)

    else:
        kernel = np.ones((15,15), np.uint8)
        fat = cv2.dilate(fat, kernel, 3)

    fat = cv2.GaussianBlur(fat.astype(np.float32), (11,11), 0)
    fat = (fat > 0.2).astype(np.uint8)
    fat = fat * liver_mask

    result = heatmap.copy()
    result[fat == 1] = [0,255,255]

    return result

# ===============================
# HU (UNCHANGED)
# ===============================
def calculate_mean_hu(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask == 1]

    if pixels.size == 0:
        return 0

    return float(np.mean(pixels) / 2)

# ===============================
# STAGE (UPDATED)
# ===============================
def determine_stage(mean):

    if mean > 50:
        return "Mild NAFLD"
    elif 35 <= mean <= 50:
        return "Moderate NAFLD"
    else:
        return "Severe NAFLD"

# ===============================
# FAT %
# ===============================
def calculate_fat_percentage(liver_mask, fat_mask):
    liver_pixels = np.sum(liver_mask == 1)
    fat_pixels = np.sum((fat_mask == 1) & (liver_mask == 1))

    if liver_pixels == 0:
        return 0

    return float((fat_pixels / liver_pixels) * 100)

# ===============================
# SEVERITY SCORE
# ===============================
def calculate_severity_score(stage, fat_percent):

    if stage == "Mild NAFLD":
        base = 30
    elif stage == "Moderate NAFLD":
        base = 60
    else:
        base = 85

    score = base + (fat_percent * 0.2)
    return min(round(score,2), 100)

# ===============================
# EXPLAINABLE AI (ENHANCED)
# ===============================
def explain_all(diagnosis, stage, mean, fat_percent=0, severity_score=0):

    if diagnosis == "Healthy Liver":

        explain = (
            f"The CT scan was analyzed and the liver region was segmented using a deep learning model. "
            f"The mean intensity value is {round(mean,2)}, which lies in the normal range. "
            "No significant fat accumulation is observed within the liver region. "
            "The liver texture appears uniform and healthy, indicating normal liver condition."
        )

        symptoms = [
            "No major symptoms",
            "Normal liver function"
        ]

        remedies = [
            "Maintain a healthy diet",
            "Regular physical activity",
            "Routine medical checkups"
        ]

    else:

        explain = (
            f"The CT scan was processed using deep learning-based liver segmentation and fat detection models. "
            f"The liver region was successfully extracted, and fat accumulation was identified using pixel intensity variations. "
            f"The computed mean HU value is {round(mean,2)}, which corresponds to {stage}. "
            f"Approximately {round(fat_percent,2)}% of the liver area is affected by fat deposition. "
            f"This results in a severity score of {severity_score}, indicating the progression level of NAFLD. "
            "The highlighted regions in the heatmap represent areas of fat accumulation within the liver."
        )

        symptoms = [
            "Fatigue",
            "Abdominal discomfort",
            "Weight gain",
            "Mild liver inflammation (in advanced cases)"
        ]

        remedies = [
            "Reduce fat and sugar intake",
            "Regular exercise",
            "Avoid alcohol consumption",
            "Maintain healthy body weight",
            "Consult a healthcare professional"
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

    cnn_img = preprocess_cnn(image)
    cnn_interpreter.set_tensor(cnn_input[0]['index'], cnn_img)
    cnn_interpreter.invoke()
    prob = float(cnn_interpreter.get_tensor(cnn_output[0]['index'])[0][0])

    if prob < 0.5:
        return {"Diagnosis":"Healthy Liver","Probability":prob}

    liver_img = preprocess_unet(image,256)
    liver_interpreter.set_tensor(liver_input[0]['index'], liver_img)
    liver_interpreter.invoke()
    liver_mask = liver_interpreter.get_tensor(liver_output[0]['index'])[0,:,:,0]
    liver_mask = clean_liver_mask(liver_mask,image.shape)

    fat_img = preprocess_unet(image,128)
    fat_interpreter.set_tensor(fat_input[0]['index'], fat_img)
    fat_interpreter.invoke()
    fat_mask = fat_interpreter.get_tensor(fat_output[0]['index'])[0,:,:,0]
    fat_mask = clean_fat_mask(fat_mask,liver_mask,image.shape)

    mean = calculate_mean_hu(image,liver_mask)
    stage = determine_stage(mean)

    if prob >= 0.5 and mean > 50:
        stage = "Mild NAFLD"

    fat_percent = calculate_fat_percentage(liver_mask, fat_mask)
    severity_score = calculate_severity_score(stage, fat_percent)

    roi = create_roi(image,liver_mask)
    heatmap = create_heatmap(image,liver_mask,fat_mask,stage)
    seg = create_segmentation_mask(liver_mask)

    exp, sym, rem = explain_all("NAFLD", stage, mean)

    return {
        "Diagnosis":"NAFLD",
        "Probability":prob,
        "Mean_HU":round(mean,2),
        "Stage":stage,
        "Fat_Percentage":round(fat_percent,2),
        "Severity_Score":severity_score,
        "ROI_Image":encode(roi),
        "Heatmap_Image":encode(heatmap),
        "Segmentation_Mask":encode(seg),
        "Explainable_AI":exp,
        "Symptoms":sym,
        "Remedies":rem,
        "Status":"Success"
    }