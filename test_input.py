import tensorflow
import keras
import numpy as np
import cv2
from PIL import Image

model = tensorflow.keras.models.load_model("pneumonia_detection_model.keras")

def process(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    resized_img = cv2.resize(img_array, (150,150))
    normalized = resized_img / 255.0
    reshaped = normalized.reshape(1,150,150,1)
    return reshaped

def predict(image_path):
    processed_img = process(image_path)
    prediction = model.predict(processed_img)

    if np.argmax(prediction) == 0:
        result = 'NORMAL'
    else:
        result = 'PNEUMONIA'
    return result

#image_path = r'C:\Users\...\pic.jpeg'
image_path = r''
result = predict(image_path)
print(f'The model predicts: {result}')