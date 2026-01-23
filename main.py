from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from PIL import Image
import io
import base64
import tensorflow as tf

app = FastAPI()

# ===== Load Quantized TFLite Model =====
interpreter = tf.lite.Interpreter(model_path="nafld_model_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== Helpers =====
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224))
    norm = resized / 255.0
    return norm.reshape(1, 224, 224, 1).astype(np.float32)

def predict_nafld(image):
    input_data = preprocess(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return float(output[0][0])

def encode_image(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

# ===== API =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    prob = predict_nafld(image)

    if prob < 0.5:
        return {
            "Diagnosis": "Healthy Liver",
            "Probability": round(prob, 3),
            "Stage": "Normal"
        }

    # Simple heatmap (no heavy segmentation to avoid memory crash)
    heatmap = cv2.applyColorMap(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)

    return {
        "Diagnosis": "NAFLD",
        "Probability": round(prob, 3),
        "Stage": "Fatty Liver Detected",
        "Heatmap": encode_image(heatmap)
    }
