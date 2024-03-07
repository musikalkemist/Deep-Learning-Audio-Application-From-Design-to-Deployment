import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

DATA_PATH = 'data.json'
SAVED_MODEL_PATH = 'model.h5'
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 5
NUM_KEYWORDS = 10


def get_data_splits(data_path: str, test_size: float = 0.05, val_size: float = 0.1):
    # read data from .json file
    with open(data_path, 'rb') as f:
        data = json.load(f)
    X, y, files = np.array(data['MFCCs']), np.array(data['labels']), np.array(data['files'])

    # create train/validation/test splits
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X, y, files, test_size=test_size, random_state=23)
    X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
        X_train, y_train, files_train, test_size=val_size, random_state=23)

    # convert inputs from 2D to 3D arrays (number of segments, 13) -> (number of segments, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test, files_train, files_val, files_test


def plot_history(history):
    r"""Plots accuracy/loss for training/validation set as a function of the epochs.

    Args:
        history:
            Training history of the model.

    Returns:
        : matplotlib figure
    """
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history['acc'] if isinstance(history, dict) else history.history['acc'], label='acc')
    axs[0].plot(history['val_acc'] if isinstance(history, dict) else history.history['val_acc'], label='val_acc')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Accuracy evaluation')

    # create loss subplot
    axs[1].plot(history['loss'] if isinstance(history, dict) else history.history['loss'], label='loss')
    axs[1].plot(history['val_loss'] if isinstance(history, dict) else history.history['val_loss'], label='val_loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Loss evaluation')
    plt.show()


########################################################################
# 1. TensorFlow implementation functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def build_tf_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape, name='MFCC')

    # conv layer 1
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    # conv layer 2
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    # conv layer 3
    x = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)

    # flatten output
    x = tf.keras.layers.Flatten()(x)

    # dense layer
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # last layer
    outputs = tf.keras.layers.Dense(NUM_KEYWORDS)(x)

    model = tf.keras.Model(inputs, outputs, name='cnn')

    # print model overview
    print(model.summary())

    return model


def train_tf_model(
        model: tf.keras.Model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 40,
        patience: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc'])

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=patience)

    # train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop_callback]
    )

    return model, history


def eval_tf_model(model: tf.keras.Model, X_test, y_test, files_test, mapping=None):
    # get accumulated statistics
    test_loss, test_acc = model.evaluate(X_test, y_test)  # evaluate model on test data
    print(f'\ntest_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}\n')

    # see specific examples
    y_test_pred = tf.argmax(model.predict(X_test), 1).numpy()
    for file, y, yhat in zip(files_test, y_test, y_test_pred):
        if y == yhat:
            print(f'{file}: true: {y if mapping is None else mapping[y]}, '
                  f'pred: {yhat if mapping is None else mapping[yhat]}')
        else:
            print(f'    WRONG!!! {file}: true: {y if mapping is None else mapping[y]}, '
                  f'pred: {yhat if mapping is None else mapping[yhat]}')


def main():
    with open(DATA_PATH, 'rb') as f:
        data = json.load(f)
    mapping = data['mapping']

    X_train, X_val, X_test, y_train, y_val, y_test, files_train, files_val, files_test = get_data_splits(DATA_PATH)

    # build the model
    tf_model = build_tf_model(X_train.shape[1:])

    # train the model
    tf_model, tf_history = train_tf_model(
        tf_model, X_train, y_train, X_val=X_val, y_val=y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE
    )

    # plot history
    plot_history(tf_history)

    # evaluate the model on test data
    eval_tf_model(tf_model, X_test, y_test, files_test, mapping=mapping)

    # save the model
    tf_model.save(SAVED_MODEL_PATH)


if __name__ == '__main__':
    main()






