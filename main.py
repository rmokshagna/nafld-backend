from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import base64

app = FastAPI()

# Load TFLite model once at startup
interpreter = tf.lite.Interpreter(model_path="nafld_model_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_heatmap(original, prediction_map):
    heatmap = cv2.applyColorMap(np.uint8(255 * prediction_map), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return overlay

def encode_image(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        stage_index = int(np.argmax(output))
        stages = ["Normal", "Mild", "Moderate", "Severe"]
        stage = stages[stage_index]

        # Fake segmentation map (replace later with real U-Net output)
        seg_map = np.random.rand(224, 224)

        original = cv2.resize(np.array(image), (224, 224))
        heatmap_img = generate_heatmap(original, seg_map)

        response = {
            "stage": stage,
            "heatmap": encode_image(heatmap_img),
            "segmentation": encode_image(heatmap_img)
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
