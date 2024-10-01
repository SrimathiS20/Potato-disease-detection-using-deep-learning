from fastapi import FastAPI, File, UploadFile # type: ignore
import numpy as np # type: ignore
from io import BytesIO
from PIL import Image # type: ignore
import tensorflow as tf # type: ignore
import uvicorn # type: ignore

# Load the model from a local path
MODEL_PATH = r"C:\Users\shailaja\Music\Potato Disease Project\Potato-disease-detection-using-deep-learning\Potato Disease\saved_models\potatoes_model_tf\1"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI()

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)

        # Get predictions from the model
        predictions = model.predict(img_batch)

        # Log the raw predictions
        print(f"Raw predictions: {predictions}")

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

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
