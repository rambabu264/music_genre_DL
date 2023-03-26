import numpy as np
import os
import CRNN_model
from MFCC import Feature_Extractor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


# To extract MFCC data from WAV format files
def load_data(path):
    mfcc = []
    labels = []

    for i in os.listdir(path):
        for song in os.listdir(path + '/' + i):
            feature_extractor = Feature_Extractor(path + '/' + i + '/' + song)
            if path + '/' + i + '/' + song != '../Data/genres_original/jazz/jazz.00054.wav':
                cepstral_coefficents, audio, sample_rate = feature_extractor.MFCC()
                mfcc.append(cepstral_coefficents)
                labels.append(i)

    for i in range(len(mfcc)):
        mfcc[i] = mfcc[i][:997]
    X = np.asarray(mfcc)
    y = np.asarray(labels)

    # y` is a list of textual labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y


# Train test dataset split
def prepare_datasets(inputs, targets):
    
    # Creating a validation set and a test set.
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2)
    return inputs_train, inputs_test, targets_train, targets_test


# Retreive data
def get_data(path):
    X, y = load_data(path)

    Xtrain, Xtest, ytrain, ytest = prepare_datasets(X, y, 0.2)

    return Xtrain, Xtest, ytrain, ytest


# Building the model
def build_model(path):
    Xtrain, Xtest, ytrain, ytest = get_data(path)

    input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)

    model_c = CRNN_model.Model(input_shape)

    model = model_c.fit(Xtrain, ytrain)

    return model_c


# To load the saved model
def load_model(checkpoint_path):
    # Load the saved model
    loaded_model = tf.keras.models.load_model(checkpoint_path)

    # Verify that the model was loaded successfully
    print(loaded_model.summary())

    return loaded_model


# Plotting the performance of the trained model
def plot_performance(hist):
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
