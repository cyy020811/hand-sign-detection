import pandas as pd
from tensorflow import keras
from keras.layers import Dense, Input, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import numpy as np
from matplotlib import pyplot as plt

# Setup
DATA_PATH = 'DATA'
EPOCHS = 100
BS = 256

# Load Data
df = pd.DataFrame()
classNames = []
label = 0
labelsFile = open('Model/labels.txt', 'w')
for file in (name for name in os.listdir(DATA_PATH) if name != '.DS_Store'):
    # Load data
    filename = file.split('.')[0]
    classNames.append(filename)
    currDf = pd.read_csv(f'{DATA_PATH}/{file}')
    currDf['label'] = label
    df = pd.concat([df, currDf])
    label += 1
    labelsFile.write(file.rsplit('.', 1)[0]+'\n')
labelsFile.close()

# Train-test-split 80/20
X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(X_train[0].shape)

# Build a model
inputLayer = Input(shape=X_train[0].shape, name='input')
x = Dense(units=64, activation='relu', name='fc1')(inputLayer)
x = Dense(units=64, activation='relu', name='fc2')(x)
x = Dense(units=64, activation='relu', name='fc3')(x)
x = Dropout(0.2)(x)
predictions = Dense(units=len(classNames), activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputLayer, outputs=predictions)
model.summary()

# Compile the model
model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# Add callbacks
filePath = 'Model/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filePath, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
callbacks = [checkpoint]

# Fit the model
hist = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BS, shuffle=True,
                 callbacks=callbacks)

# Save the model
model.save('Model/hand_sign_detection.h5')

# Visualise the accuracy and loss
plt.subplot(211)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.subplot(212)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Validation Loss'])
plt.show()
