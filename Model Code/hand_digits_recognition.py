
# **Importing Libraries**
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler

"""# **Importing Dataset**"""

data = pd.read_csv('/content/train.csv')

data.head()

"""# **Splitting into input features and output lable**"""

x = data.iloc[:,1:].values
y = data.iloc[:,:1]["label"]

x.shape,y.shape

"""# **Regularization**"""

x_train = x/255

plt.imshow(x[30].reshape(28,28))

"""# **Splitting into train and test**"""

from sklearn.model_selection import train_test_split

x_train,x_test , y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=20,stratify=y)

"""# **Creating the architecture of the model**"""

def convolutional_model(input_shape):
    """
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """

    # Define input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional layers
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout1 = tf.keras.layers.Dropout(0.25)(pool1)

    conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(dropout1)
    conv4 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(conv3)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    dropout2 = tf.keras.layers.Dropout(0.25)(pool2)

    conv5 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(dropout2)
    conv6 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(conv5)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv6)
    dropout3 = tf.keras.layers.Dropout(0.25)(pool3)

    # Flatten layer
    flatten = tf.keras.layers.Flatten()(dropout3)

    # Fully connected layers
    dense1 = tf.keras.layers.Dense(256, activation='relu')(flatten)
    dropout4 = tf.keras.layers.Dropout(0.5)(dense1)
    output = tf.keras.layers.Dense(10, activation='softmax')(dropout4)

    # Define model
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model

conv_model = convolutional_model((28,28,1))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)

"""# **Reshaping the data to be fit for CNN input**"""

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

"""# **Fitting the model**"""

history = conv_model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test),)

"""# **Evaluating the model**"""

conv_model.evaluate(x_train,y_train) , conv_model.evaluate(x_test,y_test)

history.history.keys()

loss = history.history["loss"]
accuracy = history.history["accuracy"]
val_loss = history.history["val_loss"]
val_accuracy = history.history["val_accuracy"]

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(loss,label="Traning Loss")
plt.plot(val_loss,label="Validation Loss")
plt.title("Traning Loss Vs Validation Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(accuracy,label="Traning Accuracy")
plt.plot(val_accuracy,label="Validation Accuracy")
plt.title("Traning Accuracy Vs Validation Accuracy")
plt.legend()

