from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io, base64, os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load TFLite models
cnn_interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "nafld_model_quant.tflite"))
cnn_interpreter.allocate_tensors()
cnn_input = cnn_interpreter.get_input_details()
cnn_output = cnn_interpreter.get_output_details()

unet_interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "unet_fat_segmentation_quant.tflite"))
unet_interpreter.allocate_tensors()
unet_input = unet_interpreter.get_input_details()
unet_output = unet_interpreter.get_output_details()

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

def encode(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    # CNN
    cnn_img = preprocess_cnn(image)
    cnn_interpreter.set_tensor(cnn_input[0]['index'], cnn_img)
    cnn_interpreter.invoke()
    prob = float(cnn_interpreter.get_tensor(cnn_output[0]['index'])[0][0])

    # Healthy
    if prob < 0.5:
        return {
            "Diagnosis": "Healthy Liver",
            "Probability": round(prob, 6),
            "Stage": "Normal Liver",
            "Status": "Success"
        }

    # NAFLD segmentation
    unet_img = preprocess_unet(image)
    unet_interpreter.set_tensor(unet_input[0]['index'], unet_img)
    unet_interpreter.invoke()
    mask = unet_interpreter.get_tensor(unet_output[0]['index'])[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8)

    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    roi = image.copy()
    roi[mask_resized == 0] = 0

    heatmap = cv2.applyColorMap(mask_resized * 255, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    stage = "Severe NAFLD" if prob > 0.8 else "Moderate NAFLD"

    return {
        "Diagnosis": "NAFLD",
        "Probability": round(prob, 6),
        "Stage": stage,
        "Segmentation_Mask": encode(mask_resized * 255),
        "ROI_Image": encode(roi),
        "Heatmap_Image": encode(overlay),
        "Status": "Success"
    }
