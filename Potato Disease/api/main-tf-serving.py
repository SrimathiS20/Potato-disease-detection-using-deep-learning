from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import uvicorn

app = FastAPI()

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
        print(f"Error reading image: {e}")
        raise e

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
       
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)

       
        print(f"Image shape: {img_batch.shape}")

       
        json_data = {
            "instances": img_batch.tolist()
        }

       
        response = requests.post(endpoint, json=json_data)
        
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.content}")

        if response.status_code != 200:
            return {"error": f"Error from TensorFlow Serving: {response.content}"}

        predictions = np.array(response.json()["predictions"][0])

        
        print(f"Predictions: {predictions}")

        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }

    except Exception as e:
       
        print(f"Error during prediction: {e}")
        return {"error": "An error occurred during prediction"}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
