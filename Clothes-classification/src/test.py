import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from main import load_data, CLASS_NAMES

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

def show_img(img):
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

def process_img(img):
    processed_img = np.expand_dims(img, axis=0) / 255
    return processed_img

def load_random_img(x_test, y_test):
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    label = CLASS_NAMES[y_test[idx]]
    return img, label

def test_samples(model, x_test, y_test, num_samples):
    for _ in range(num_samples):
        img, true_label = load_random_img(x_test, y_test)
        show_img(img)
        processed_img = process_img(img)
        prediction = model.predict(processed_img, verbose=0)
        predicted_label = CLASS_NAMES[np.argmax(prediction)]
        print(f"The clothe is a: {true_label}.")
        print(f"The model predicted a: {predicted_label}.\n")

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = load_model('../models/model_1.h5')
    test_samples(model, x_test, y_test, 10)


if __name__ == "__main__":
    main()