from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import uvicorn

app = FastAPI()

# TensorFlow Serving endpoint
endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        raise e

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)

        # Prepare the data for TensorFlow Serving
        json_data = {
            "instances": img_batch.tolist()
        }

        # Send request to TensorFlow Serving
        response = requests.post(endpoint, json=json_data)
        
        if response.status_code != 200:
            print(f"Error from TensorFlow Serving: {response.content}")
            return {"error": "Failed to get prediction from model"}

        predictions = np.array(response.json()["predictions"][0])

        # Log the raw predictions
        print(f"Raw predictions: {predictions}")

        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }

    except Exception as e:
        # Catch and log any exception
        print(f"Error during prediction: {e}")
        return {"error": "An error occurred during prediction"}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
