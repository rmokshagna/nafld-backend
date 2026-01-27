from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import base64
import tensorflow as tf
from PIL import Image
import io
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cnn_interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "nafld_model_quant.tflite"))
unet_interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "unet_fat_segmentation_quant.tflite"))

cnn_interpreter.allocate_tensors()
unet_interpreter.allocate_tensors()

cnn_input = cnn_interpreter.get_input_details()[0]["index"]
cnn_output = cnn_interpreter.get_output_details()[0]["index"]

unet_input = unet_interpreter.get_input_details()[0]["index"]
unet_output = unet_interpreter.get_output_details()[0]["index"]


def preprocess_cnn(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224)) / 255.0
    return img.reshape(1, 224, 224, 1).astype(np.float32)


def preprocess_unet(img):
    img = cv2.resize(img, (256, 256)) / 255.0
    return img.reshape(1, 256, 256, 1).astype(np.float32)


def encode_image(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    # ---- CNN Prediction ----
    cnn_input_data = preprocess_cnn(image)
    cnn_interpreter.set_tensor(cnn_input, cnn_input_data)
    cnn_interpreter.invoke()
    prob = float(cnn_interpreter.get_tensor(cnn_output)[0][0])

    if prob < 0.5:
        return {
            "Diagnosis": "Normal Liver",
            "Probability": round(prob, 5),
            "Stage": "Normal",
            "Status": "Success"
        }

    # ---- U-Net Segmentation ----
   # Prepare image for UNet (must be grayscale, 1 channel)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (256, 256))
gray = gray.astype(np.float32) / 255.0
gray = np.expand_dims(gray, axis=-1)      # (256,256,1)
gray = np.expand_dims(gray, axis=0)       # (1,256,256,1)

unet_input = unet_interpreter.get_input_details()[0]['index']
unet_interpreter.set_tensor(unet_input, gray)
unet_interpreter.invoke()


    # ---- Fat Ratio ----
    fat_pixels = np.sum(mask_resized == 255)
    total_pixels = mask_resized.size
    fat_ratio = fat_pixels / total_pixels

    if fat_ratio < 0.10:
        stage = "Mild NAFLD"
    elif fat_ratio < 0.30:
        stage = "Moderate NAFLD"
    else:
        stage = "Severe NAFLD"

    # ---- HU Mean (grayscale proxy) ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hu_mean = float(np.mean(gray[mask_resized == 255])) if fat_pixels > 0 else None

    # ---- ROI & Heatmap ----
    roi = image.copy()
    roi[mask_resized == 0] = 0

    heatmap = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return {
        "Diagnosis": "NAFLD",
        "Probability": round(prob, 5),
        "Fat_Ratio": round(fat_ratio, 3),
        "Mean_HU": round(hu_mean, 2) if hu_mean else None,
        "Stage": stage,
        "Segmentation_Mask": encode_image(mask_resized),
        "ROI_Image": encode_image(roi),
        "Heatmap_Image": encode_image(overlay),
        "Status": "Success"
    }
