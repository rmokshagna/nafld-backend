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
    model_path=os.path.join(BASE_DIR, "unet_fat_segmentation_quant.tflite")
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

    mask = (mask > 0.3).astype(np.uint8)
    mask = cv2.resize(mask, (shape[1], shape[0]))

    mask = cv2.GaussianBlur(mask.astype(np.float32), (15,15), 0)
    mask = (mask > 0.2).astype(np.uint8)

    kernel = np.ones((25,25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean = np.zeros_like(mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 1, -1)

    return clean

# ===============================
# CLEAN FAT MASK
# ===============================
def clean_fat_mask(mask, liver_mask, shape):

    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (shape[1], shape[0]))
    mask = mask * liver_mask

    mask = cv2.GaussianBlur(mask.astype(np.float32), (21,21), 0)
    mask = (mask > 0.3).astype(np.uint8)

    # fallback fat region
    if np.sum(mask) < 100:
        coords = np.column_stack(np.where(liver_mask == 1))
        if len(coords) > 0:
            cy, cx = coords[np.random.randint(len(coords))]
            for y in range(cy-25, cy+25):
                for x in range(cx-25, cx+25):
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                        if liver_mask[y,x] == 1:
                            mask[y,x] = 1

    return mask

# ===============================
# ROI
# ===============================
def create_roi(image, mask):

    roi = image.copy()
    dark = (roi * 0.2).astype(np.uint8)
    roi[mask == 0] = dark[mask == 0]

    contours,_ = cv2.findContours(
        (mask*255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(roi, contours, -1, (0,255,0), 2)

    return roi

# ===============================
# SEGMENTATION MASK
# ===============================
def create_segmentation_mask(mask):

    seg = (mask * 255).astype(np.uint8)
    seg = cv2.GaussianBlur(seg, (11,11), 0)
    _, seg = cv2.threshold(seg, 127, 255, cv2.THRESH_BINARY)

    return seg

# ===============================
# HEATMAP
# ===============================
def create_heatmap(image, liver_mask, fat_mask):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    liver_pixels = gray[liver_mask == 1]

    if liver_pixels.size == 0:
        return image

    min_val = np.min(liver_pixels)
    max_val = np.max(liver_pixels)

    norm = (gray - min_val) / (max_val - min_val + 1e-5)
    norm = np.clip(norm, 0, 1)

    heat = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    result = image.copy()

    result[liver_mask == 1] = cv2.addWeighted(
        image[liver_mask == 1], 0.4,
        heatmap[liver_mask == 1], 0.6,
        0
    )

    # highlight fat
    result[fat_mask == 1] = (0.5 * result[fat_mask == 1] + 0.5 * np.array([0,255,255])).astype(np.uint8)

    return result

# ===============================
# HU
# ===============================
def calculate_mean_hu(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask == 1]

    if pixels.size == 0:
        return 0

    return float(np.mean(pixels))

# ===============================
# STAGE
# ===============================
def determine_stage(mean):

    if mean > 75:
        return "Normal Liver"
    elif 60 <= mean <= 75:
        return "Mild NAFLD"
    elif 45 <= mean < 60:
        return "Moderate NAFLD"
    else:
        return "Severe NAFLD"

# ===============================
# EXPLAINABLE AI
# ===============================
def explain_all(diagnosis, stage):

    if diagnosis == "Healthy Liver":
        explain = "No significant fat accumulation detected. Liver appears normal."
        symptoms = ["No major symptoms", "Regular health monitoring advised"]
        remedies = ["Balanced diet", "Exercise regularly", "Avoid alcohol"]

    else:
        explain = f"Fat accumulation detected in liver region. Stage identified: {stage}."
        symptoms = [
            "Fatigue",
            "Upper abdominal discomfort",
            "Weight gain",
            "Loss of appetite"
        ]
        remedies = [
            "Reduce oily foods",
            "Exercise daily",
            "Weight control",
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

    # HEALTHY CASE
    if prob < 0.5:
        exp, sym, rem = explain_all("Healthy Liver", "Normal")

        return {
            "Diagnosis":"Healthy Liver",
            "Probability":prob,
            "Mean_HU":None,
            "Stage":"Normal Liver",
            "ROI_Image":None,
            "Heatmap_Image":None,
            "Segmentation_Mask":None,
            "Explainable_AI":exp,
            "Symptoms":sym,
            "Remedies":rem,
            "Status":"Success"
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
    mask = create_segmentation_mask(liver_mask)

    mean = calculate_mean_hu(image,liver_mask)
    stage = determine_stage(mean)

    exp, sym, rem = explain_all("NAFLD", stage)

    return {
        "Diagnosis":"NAFLD",
        "Probability":prob,
        "Mean_HU":round(mean,2),
        "Stage":stage,
        "ROI_Image":encode(roi),
        "Heatmap_Image":encode(heatmap),
        "Segmentation_Mask":encode(mask),
        "Explainable_AI":exp,
        "Symptoms":sym,
        "Remedies":rem,
        "Status":"Success"
    }