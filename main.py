from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load models
interpreter = tf.lite.Interpreter(model_path="nafld_model_quant.tflite")
interpreter.allocate_tensors()

seg_model = tf.keras.models.load_model("liver_unet.keras", compile=False)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224

def preprocess(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image.astype(np.float32)

def to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def gradcam(image):
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return heatmap

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("L")
        img = np.array(img)

        input_data = preprocess(img)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        if prediction > 0.5:
            stage = "Severe NAFLD"
            diagnosis = "NAFLD"
        else:
            stage = "Normal Liver"
            diagnosis = "Normal"

        # Segmentation
        seg_input = cv2.resize(img, (256, 256)) / 255.0
        seg_input = np.expand_dims(seg_input, axis=(0, -1))
        seg_mask = seg_model.predict(seg_input)[0]
        seg_mask = np.argmax(seg_mask, axis=-1)
        seg_mask = (seg_mask * 85).astype(np.uint8)
        seg_mask = cv2.applyColorMap(seg_mask, cv2.COLORMAP_JET)

        # Heatmap
        heatmap = gradcam(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))

        return JSONResponse({
            "Status": "Success",
            "Diagnosis": diagnosis,
            "Probability": float(prediction),
            "Stage": stage,
            "Heatmap": to_base64(heatmap),
            "Segmentation": to_base64(seg_mask)
        })

    except Exception as e:
        return JSONResponse({"Status": "Error", "Message": str(e)})
