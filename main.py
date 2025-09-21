from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Literal, Annotated
from fastapi.responses import JSONResponse

import tensorflow as tf
import pickle
import pandas as pd
import cv2
import numpy as np

# import the ml model
from tensorflow.keras.models import load_model

model = load_model("model.h5")


app = FastAPI()

app.title = "Object detetion using CNN API"
MODEL_VERSION=app.version = "1.0.0"

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    # Convert bytes to numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)

    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess
    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    return img.reshape(1, 128, 128, 3).astype(np.float32)

# Human-readable labels for the classes
@app.get("/")
async def root():
    return {"message": "Welcome to the Object Detection API"}
# Machine health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "OK", "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        input_array = preprocess_image(file_bytes)
        class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
        
        prediction = model.predict(input_array)
        probs = tf.nn.softmax(prediction).numpy()[0]*100
        pred_class_idx = np.argmax(probs)
        pred_class = class_names[pred_class_idx]
        
        return JSONResponse(status_code=200, content={
            "Filename": file.filename,
            "Input_shape": str(input_array.shape),
            "Prediction": pred_class,
            "Probabilities": {class_names[i]: round(float(probs[i]),3) for i in range(len(class_names))}
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

