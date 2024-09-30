from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn


MODEL_PATH = r"C:\Users\shailaja\Music\Potato Disease Project\Potato-disease-detection-using-deep-learning\Potato Disease\saved_models\potatoes_model_tf\1"
model_layer = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    
    predictions = model_layer(img_batch)
    
    
    print(f"Raw predictions: {predictions}")

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
