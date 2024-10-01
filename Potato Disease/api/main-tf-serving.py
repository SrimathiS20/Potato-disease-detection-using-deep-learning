from fastapi import FastAPI, File, UploadFile # type: ignore
import numpy as np # type: ignore
from io import BytesIO
from PIL import Image # type: ignore
import httpx # type: ignore
import uvicorn # type: ignore

app = FastAPI()

# TensorFlow Serving endpoint
endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

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
        # Read the uploaded image
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)

        # Log the image shape for debugging
        print(f"Image shape: {img_batch.shape}")

        # Prepare the data for TensorFlow Serving
        json_data = {
            "instances": img_batch.tolist()
        }

        # Send request to TensorFlow Serving using httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=json_data)
        
        # Log the response status and content for debugging
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.content}")

        if response.status_code != 200:
            return {"error": f"Error from TensorFlow Serving: {response.content}"}

        predictions = np.array(response.json()["predictions"][0])

        # Log the raw predictions
        print(f"Predictions: {predictions}")

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
