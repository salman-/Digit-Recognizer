import pandas as pd
import numpy as np
from Dataset import Dataset
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential

dt = Dataset("../datasets/train.csv")

train_x, test_x, train_y, test_y = dt.get_test_train_datasets()

model = Sequential()
model.add(Conv2D(filters=10, input_shape=(28, 28, 1), strides=1, kernel_size=3, name='input_layer', activation='relu'))
model.add(Conv2D(filters=10, kernel_size=3, name='Conv_1', activation='relu'))
model.add(MaxPool2D(name='MaxPool_1'))
model.add(Conv2D(filters=10, kernel_size=3, name='Conv_2', activation='relu'))
model.add(Conv2D(filters=10, kernel_size=3, name='Conv_3', activation='relu'))
model.add(MaxPool2D(name='MaxPool_2'))


model.add(Flatten(name='Flatten_layer'))

model.add(Dense(10, name='output_layer', activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=train_x, y=train_y, epochs=40, validation_data=(test_x, test_y))

#Predict
predict_dt = Dataset("../datasets/test.csv")
predictions = model.predict(predict_dt.dt.reshape(28000, 28, 28))
predictions = np.round(predictions)
predictions = pd.DataFrame(predictions).idxmax(1)
predictions = predictions.reset_index().rename(columns={'index':'ImageId', 0:'Label'})
predictions['ImageId'] = np.arange(1, len(predictions)+1)

predictions.to_csv("./../datasets/final.csv",index=False)