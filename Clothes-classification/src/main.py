import tensorflow as tf

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def process_data(x_train, x_test):
    x_train = x_train / 255
    x_test = x_test / 255
    return x_train, x_test

def save_model(model):
    model.save('../models/model_1.h5')

def train_model(model, x_train, y_train, epochs):
    model.fit(x_train, y_train, epochs=epochs)

def test_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy={test_acc}.\n")


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_test = process_data(x_train, x_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy']
    )

    train_model(model, x_train, y_train, 10)
    test_model(model, x_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()