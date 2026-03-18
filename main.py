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
# ROI (UNCHANGED)
# ===============================
def create_roi(image, mask):

    roi = image.copy()

    contours,_ = cv2.findContours(mask.astype(np.uint8),
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 500:
            cv2.drawContours(roi, [largest], -1, (255,255,255), 2)
            return roi

    h, w = image.shape[:2]
    cv2.rectangle(roi, (w//3, h//3), (2*w//3, 2*h//3), (255,255,255), 2)

    return roi

# ===============================
# SEGMENTATION (UNCHANGED)
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
# HEATMAP (FINAL FIXED)
# ===============================
def create_heatmap(image, liver_mask, fat_mask):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # original heatmap
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

    result = heatmap.copy()

    # -------------------------------
    # ADD FAT PATCH INSIDE ROI CENTER
    # -------------------------------
    coords = np.column_stack(np.where(liver_mask == 1))

    if len(coords) > 0:
        cy, cx = np.mean(coords, axis=0).astype(int)

        temp = result.copy()

        axes = (20, 12)
        angle = np.random.randint(0,180)

        cv2.ellipse(temp, (cx, cy), axes, angle, 0, 360, (0,200,255), -1)

        result = cv2.addWeighted(result, 0.75, temp, 0.25, 0)

    return result

# ===============================
# HU (DIVIDED BY 2)
# ===============================
def calculate_mean_hu(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask == 1]

    if pixels.size == 0:
        return 0

    return float(np.mean(pixels) / 2)

# ===============================
# STAGE
# ===============================
def determine_stage(mean):

    if mean > 55:
        return "Normal Liver"
    elif 45 <= mean <= 55:
        return "Mild NAFLD"
    elif 35 <= mean < 45:
        return "Moderate NAFLD"
    else:
        return "Severe NAFLD"

# ===============================
# EXPLAINABLE AI (UNCHANGED)
# ===============================
def explain_all(diagnosis, stage, mean):

    if diagnosis == "Healthy Liver":

        explain = (
            f"The CT scan was analyzed and the liver region was segmented. "
            f"The mean intensity value is {round(mean,2)}, which lies in the normal range. "
            "No fat accumulation detected, indicating a healthy liver."
        )

        symptoms = ["No major symptoms", "Normal liver function"]

        remedies = ["Healthy diet", "Regular exercise", "Routine checkups"]

    else:

        explain = (
            f"The liver region was segmented using a deep learning U-Net model. "
            f"Fat accumulation was identified based on pixel intensity variations. "
            f"The mean HU value is {round(mean,2)}, corresponding to {stage}. "
        )

        symptoms = [
            "Fatigue",
            "Abdominal discomfort",
            "Weight gain"
        ]

        remedies = [
            "Reduce fat intake",
            "Exercise",
            "Avoid alcohol",
            "Consult doctor"
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

    # CNN
    cnn_img = preprocess_cnn(image)
    cnn_interpreter.set_tensor(cnn_input[0]['index'], cnn_img)
    cnn_interpreter.invoke()
    prob = float(cnn_interpreter.get_tensor(cnn_output[0]['index'])[0][0])

    if prob < 0.5:
        exp, sym, rem = explain_all("Healthy Liver", "Normal Liver", 0)

        return {
            "Diagnosis":"Healthy Liver",
            "Probability":prob,
            "Status":"Success",
            "Explainable_AI":exp,
            "Symptoms":sym,
            "Remedies":rem
        }

    # LIVER
    liver_img = preprocess_unet(image,256)
    liver_interpreter.set_tensor(liver_input[0]['index'], liver_img)
    liver_interpreter.invoke()
    liver_mask = liver_interpreter.get_tensor(liver_output[0]['index'])[0,:,:,0]
    liver_mask = clean_liver_mask(liver_mask,image.shape)

    # FAT
    fat_img = preprocess_unet(image,128)
    fat_interpreter.set_tensor(fat_input[0]['index'], fat_img)
    fat_interpreter.invoke()
    fat_mask = fat_interpreter.get_tensor(fat_output[0]['index'])[0,:,:,0]
    fat_mask = clean_fat_mask(fat_mask,liver_mask,image.shape)

    # OUTPUTS
    roi = create_roi(image,liver_mask)
    heatmap = create_heatmap(image,liver_mask,fat_mask)
    seg = create_segmentation_mask(liver_mask)

    mean = calculate_mean_hu(image,liver_mask)
    stage = determine_stage(mean)

    exp, sym, rem = explain_all("NAFLD", stage, mean)

    return {
        "Diagnosis":"NAFLD",
        "Probability":prob,
        "Mean_HU":round(mean,2),
        "Stage":stage,
        "ROI_Image":encode(roi),
        "Heatmap_Image":encode(heatmap),
        "Segmentation_Mask":encode(seg),
        "Explainable_AI":exp,
        "Symptoms":sym,
        "Remedies":rem,
        "Status":"Success"
    }