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

# CNN classifier
cnn_interpreter = tf.lite.Interpreter(
    model_path=os.path.join(BASE_DIR, "nafld_model_quant.tflite")
)
cnn_interpreter.allocate_tensors()
cnn_input = cnn_interpreter.get_input_details()
cnn_output = cnn_interpreter.get_output_details()

# OLD MODEL (liver segmentation)
liver_interpreter = tf.lite.Interpreter(
    model_path=os.path.join(BASE_DIR, "unet_fat_segmentation_quant.tflite")
)
liver_interpreter.allocate_tensors()
liver_input = liver_interpreter.get_input_details()
liver_output = liver_interpreter.get_output_details()

# NEW MODEL (fat segmentation)
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
    img = img.reshape(1,224,224,1).astype(np.float32)

    return img


def preprocess_unet(image, size):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (size,size)) / 255.0
    img = img.reshape(1,size,size,1).astype(np.float32)

    return img


# ===============================
# BASE64 ENCODE
# ===============================

def encode(img):

    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


# ===============================
# CLEAN LIVER MASK
# ===============================

def clean_liver_mask(mask, shape):

    mask = (mask > 0.5).astype(np.uint8)*255
    mask = cv2.resize(mask,(shape[1],shape[0]))

    kernel = np.ones((5,5),np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    clean = np.zeros_like(mask)

    if contours:
        largest = max(contours,key=cv2.contourArea)
        cv2.drawContours(clean,[largest],-1,255,-1)

    return (clean>0).astype(np.uint8)


# ===============================
# ROI IMAGE
# ===============================

def create_roi(image,mask):

    roi=image.copy()

    dark=(roi*0.3).astype(np.uint8)

    roi[mask==0]=dark[mask==0]

    contours,_=cv2.findContours(
        (mask*255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(roi,contours,-1,(0,255,0),2)

    return roi


# ===============================
# HEATMAP
# ===============================

def create_heatmap(image,liver_mask,fat_mask):

    overlay=image.copy()

    normal=(liver_mask==1)&(fat_mask==0)
    fat=(fat_mask==1)

    # BLUE = normal liver
    overlay[normal]=(255,0,0)

    # YELLOW = fat
    overlay[fat]=(0,255,255)

    contours,_=cv2.findContours(
        (liver_mask*255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(overlay,contours,-1,(0,255,0),2)

    return overlay


# ===============================
# HU CALCULATION
# ===============================

def calculate_mean_hu(image,mask):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    pixels=gray[mask==1]

    if pixels.size==0:
        return 0

    return float(np.mean(pixels))


# ===============================
# STAGE
# ===============================

def determine_stage(mean):

    if mean>75:
        return "Normal Liver"

    elif 65<=mean<=75:
        return "Mild NAFLD"

    elif 55<=mean<65:
        return "Moderate NAFLD"

    else:
        return "Severe NAFLD"


# ===============================
# API
# ===============================

@app.post("/predict")
async def predict(file:UploadFile=File(...)):

    contents=await file.read()

    image=np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


    # ===============================
    # CNN
    # ===============================

    cnn_img=preprocess_cnn(image)

    cnn_interpreter.set_tensor(cnn_input[0]['index'],cnn_img)

    cnn_interpreter.invoke()

    prob=float(cnn_interpreter.get_tensor(cnn_output[0]['index'])[0][0])


    if prob<0.5:

        return{
            "Diagnosis":"Healthy Liver",
            "Probability":prob,
            "Status":"Success"
        }


    # ===============================
    # LIVER SEGMENTATION
    # ===============================

    liver_img=preprocess_unet(image,256)

    liver_interpreter.set_tensor(liver_input[0]['index'],liver_img)

    liver_interpreter.invoke()

    liver_mask=liver_interpreter.get_tensor(liver_output[0]['index'])[0,:,:,0]

    liver_mask=clean_liver_mask(liver_mask,image.shape)


    # ===============================
    # FAT SEGMENTATION
    # ===============================

    fat_img=preprocess_unet(image,128)

    fat_interpreter.set_tensor(fat_input[0]['index'],fat_img)

    fat_interpreter.invoke()

    fat_mask=fat_interpreter.get_tensor(fat_output[0]['index'])[0,:,:,0]

    fat_mask=cv2.resize((fat_mask>0.5).astype(np.uint8),(image.shape[1],image.shape[0]))


    # ===============================
    # VISUALS
    # ===============================

    roi=create_roi(image,liver_mask)

    heatmap=create_heatmap(image,liver_mask,fat_mask)

    mask=(liver_mask*255).astype(np.uint8)


    # ===============================
    # STAGE
    # ===============================

    mean=calculate_mean_hu(image,liver_mask)

    stage=determine_stage(mean)


    return{

        "Diagnosis":"NAFLD",
        "Probability":prob,
        "Mean_HU":round(mean,2),
        "Stage":stage,

        "ROI_Image":encode(roi),
        "Heatmap_Image":encode(heatmap),
        "Segmentation_Mask":encode(mask),

        "Status":"Success"

    }