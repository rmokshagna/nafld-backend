from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

# Allow Android app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="nafld_model_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert("L")  # Convert to grayscale
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (224,224,1)
    img_array = np.expand_dims(img_array, axis=0)   # (1,224,224,1)
    return img_array

def classify_stage(prob):
    if prob < 0.25:
        return "Normal Liver"
    elif prob < 0.5:
        return "Mild NAFLD"
    elif prob < 0.75:
        return "Moderate NAFLD"
    else:
        return "Severe NAFLD"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        input_data = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        stage = classify_stage(prediction)

        return {
            "Diagnosis": "NAFLD",
            "Probability": float(prediction),
            "Stage": stage,
            "Status": "Success"
        }

    except Exception as e:
        return {
            "error": "NAFLD pipeline crashed",
            "details": str(e)
        }

@app.get("/")
def root():
    return {"message": "NAFLD Detection API Running"}
