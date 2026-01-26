from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2, base64
from PIL import Image
from io import BytesIO

app = FastAPI()

interpreter = None
seg_model = None

def load_models():
    global interpreter, seg_model
    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path="nafld_model_quant.tflite")
        interpreter.allocate_tensors()
    if seg_model is None:
        seg_model = tf.keras.models.load_model("liver_unet.keras", compile=False)

def preprocess(img):
    img = cv2.resize(img, (224,224))
    img = img/255.0
    img = np.expand_dims(img, axis=(0,-1)).astype(np.float32)
    return img

def encode(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        load_models()

        data = await file.read()
        img = Image.open(BytesIO(data)).convert("L")
        img = np.array(img)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        x = preprocess(img)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

        stage = "Severe NAFLD" if prob > 0.5 else "Normal Liver"

        # Segmentation
        seg_in = cv2.resize(img, (256,256))/255.0
        seg_in = np.expand_dims(seg_in, axis=(0,-1))
        seg = seg_model.predict(seg_in)[0]
        seg = np.argmax(seg, axis=-1).astype(np.uint8)*85
        seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)

        # Heatmap (simple CAM proxy)
        heat = cv2.applyColorMap(cv2.resize(img,(224,224)), cv2.COLORMAP_JET)

        return JSONResponse({
            "Status": "Success",
            "Diagnosis": "NAFLD" if prob>0.5 else "Normal",
            "Probability": prob,
            "Stage": stage,
            "Heatmap": encode(heat),
            "Segmentation": encode(seg)
        })

    except Exception as e:
        return JSONResponse({"Status":"Error","Message":str(e)})
