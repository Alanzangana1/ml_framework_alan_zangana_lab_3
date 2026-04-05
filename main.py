import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

session = ort.InferenceSession("model.onnx")

def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_data  = preprocess(image_bytes)
    inputs      = {session.get_inputs()[0].name: input_data}
    outputs     = session.run(None, inputs)
    predicted_index = int(np.argmax(outputs[0], axis=1)[0])
    predicted_class = CLASSES[predicted_index]
    return JSONResponse({"prediction": predicted_class, "class_index": predicted_index})