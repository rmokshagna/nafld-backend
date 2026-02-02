from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load models once (important for Render stability)
clf_interpreter = tf.lite.Interpreter(model_path="nafld_model_quant.tflite")
clf_interpreter.allocate_tensors()

seg_interpreter = tf.lite.Interpreter(model_path="unet_fat_segmentation_quant.tflite")
seg_interpreter.allocate_tensors()

clf_input = clf_interpreter.get_input_details()
clf_output = clf_interpreter.get_output_details()

seg_input = seg_interpreter.get_input_details()
seg_output = seg_interpreter.get_output_details()


def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img


def to_base64(image):
    pil = Image.fromarray(image)
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    img = np.array(Image.open(BytesIO(image_bytes)).convert("L"))

    input_tensor = preprocess(img)

    clf_interpreter.set_tensor(clf_input[0]['index'], input_tensor)
    clf_interpreter.invoke()
    prob = float(clf_interpreter.get_tensor(clf_output[0]['index'])[0][0])

    if prob < 0.5:
        return {
            "Diagnosis": "Normal",
            "Probability": prob,
            "Stage": "Normal Liver",
            "Status": "Success"
        }

    # ---- Fat Segmentation ----
    seg_interpreter.set_tensor(seg_input[0]['index'], input_tensor)
    seg_interpreter.invoke()
    mask = seg_interpreter.get_tensor(seg_output[0]['index'])[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8) * 255

    # ---- Heatmap ----
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)

    # ---- HU Estimation (Fat region mean) ----
    hu_region = img[mask == 255]
    hu_value = float(np.mean(hu_region)) if len(hu_region) > 0 else -100

    # ---- Stage ----
    if prob < 0.65:
        stage = "Mild NAFLD"
    elif prob < 0.85:
        stage = "Moderate NAFLD"
    else:
        stage = "Severe NAFLD"

    return {
        "Diagnosis": "NAFLD",
        "Probability": prob,
        "Stage": stage,
        "HU_Fat_Mean": hu_value,
        "SegmentationMask": to_base64(mask),
        "Heatmap": to_base64(overlay),
        "Status": "Success"
    }
