import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the model
model = load_model('brainTumor.h5')

# Load and process the image
image = cv2.imread('D:\\Major Project\\AI-based-Medical-Web-Application\\medical_app\\pred\\pred0.jpg')

img = Image.fromarray(image)
img = img.resize((64,64)) # Resize to the input size expected by the model
img = np.array(img)
input_img = np.expand_dims(img, axis=0) # Add batch dimension

# print(img)

result = model.predict(input_img)
# print(result)

# Assuming a binary classification, interpret the result as a class label
predicted_class = np.argmax(result, axis = 1) # Gets the index of the highest probability
print(predicted_class)

