import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from main import load_data
from network import Network
# import os


def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def read_image(path):
    img = cv2.imread(path)[:,:,0]
    return img

def show_image(img):
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

def process_image(img):
    processed_img = img.reshape(784, 1) / 255
    return processed_img

def load_random_image(x_test):
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    return img

def test_samples(x_test, num_samples):
    for _ in range(num_samples):
        img = load_random_image(x_test)
        show_image(img)
        processed_img = process_image(img)
        prediction = model.predict(processed_img)
        print(f"Prediction is: {np.argmax(prediction)}")

(x_train, y_train), (x_test, y_test) = load_data()
PATH = '../models/model_google_colab.pkl'
model = load_model(PATH)
test_samples(x_test, 10)

# image_number = 1
# while os.path.isfile(f"../tests/digit{image_number}.png"):
#     try:
#         img = read_image(f"../tests/digit{image_number}.png")
#         img = np.invert(np.array(img))
#         show_image(img)
#         processed_img = process_image(img)
#         prediction = model.predict(processed_img)
#         print(f"Prediction is: {np.argmax(prediction)}")
#     except:
#         break
#     finally:
#         image_number += 1
