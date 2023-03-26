import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import KFold


class Model:

    def __init__(self, input):
        self.num_frames = input[1]
        self.num_mfcc = input[2]
        self.num_classes = 10
        self.checkpoint_path = "checkpoint/cp.ckpt"
        self.model = 0

    def model(self, y):
        model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.num_frames, self.num_mfcc, 1)),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
        ])

        return model

    def checkpoint(self):
        checkpoint_dir = os.path.dirname(self.checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        return cp_callback

    def fit(self, inputs_train, targets_train):
        model = self.model(targets_train)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cp_callback = self.checkpoint()

        kf = KFold(n_splits=5, shuffle=True)

        # Initialize a list to store the model's performance on each fold
        scores = []

        # Perform k-fold cross-validation
        for train_index, val_index in kf.split(inputs_train):
            # Split the data into training and validation sets for this fold
            X_train_fold, X_val_fold = inputs_train[train_index], inputs_train[val_index]
            y_train_fold, y_val_fold = targets_train[train_index], targets_train[val_index]

            # Train the model on the training set for this fold

            model.fit(x=X_train_fold, y=y_train_fold, batch_size=16, epochs=30,
                      validation_data=(X_val_fold, y_val_fold), callbacks=[cp_callback])

            # Evaluate the model on the validation set for this fold
            score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            scores.append(score[1])

        self.model = model

    def model_evaluation(self, model, inputs_test, targets_test):

        # Evaluate the model
        loss, acc = model.evaluate(inputs_test, targets_test)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))