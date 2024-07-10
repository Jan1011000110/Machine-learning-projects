import tensorflow as tf
import pickle
from network import Network


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def process_data(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(-1, 784, 1) / 255
    x_test = x_test.reshape(-1, 784, 1) / 255
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_train = y_train.reshape(-1, 10, 1)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    y_test = y_test.reshape(-1, 10, 1)
    train_data = list(zip(x_train, y_train))
    test_data = list(zip(x_test, y_test))
    return train_data, test_data

def train_model(model, train_data, epochs, batch_size, learning_rate):
    model.fit(train_data, epochs, batch_size, learning_rate)

def test_model(model, test_data):
    model.test(test_data)

def save_model(model):
    with open('../models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

def main():
    model = Network([784, 100, 10])
    (x_train, y_train), (x_test, y_test) = load_data()
    train_data, test_data = process_data(x_train, y_train, x_test, y_test)
    train_model(model, train_data, 30, 10, 0.01)
    test_model(model, test_data)
    save_model(model)


if __name__ == "__main__":
    main()
